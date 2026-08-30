//! Querying a moving 3D scene two ways: rebuild the 3D tree every frame, or build
//! one 4D spacetime tree and query slices of it.
//!
//! A cloud of spheres moves over a 5-second, 60fps window (300 frames). We ask one
//! raycast per frame, at that frame's time, so 300 queries in total. The 3D way has
//! to put the tree in the right state for each query, so it rebuilds (or refits)
//! once per frame. The 4D way treats time as a fourth axis: each sphere's
//! trajectory is sliced into short spacetime segments, one BVH is built over all of
//! them once, and a query at time t is a 4D ray fixed at that t. The time axis in
//! the slab test prunes every segment that does not straddle t, so one structure
//! answers all 300 frames.
//!
//! Both ways compute each sphere's position from the same closed-form path, so they
//! return the same object every frame; the run asserts that before reporting times.
//!
//! Run with: cargo run --release --example spacetime_query

use std::time::Instant;

use spatial_query::{Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray};

const FPS: usize = 60;
const SECS: usize = 5;
const FRAMES: usize = FPS * SECS; // 300
const T_END: f32 = SECS as f32;

const N: usize = 3000; // moving spheres
const SEG: usize = 20; // spacetime segments per sphere (each 0.25s)
const REPS: usize = 5; // repeat each timed section, keep the fastest

// ---------------------------------------------------------------------------
// The moving scene: each sphere follows a closed-form sinusoidal path so both
// the 3D and 4D sides evaluate exactly the same position at a given time.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Mover {
    base: [f32; 3],
    amp: [f32; 3],
    freq: [f32; 3],
    phase: [f32; 3],
    radius: f32,
}

impl Mover {
    #[inline]
    fn position(&self, t: f32) -> [f32; 3] {
        let mut p = [0.0; 3];
        for k in 0..3 {
            p[k] = self.base[k] + self.amp[k] * (self.freq[k] * t + self.phase[k]).sin();
        }
        p
    }
}

/// A tiny deterministic PRNG so the scene is identical every run (no dependency).
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn build_movers() -> Vec<Mover> {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    (0..N)
        .map(|_| Mover {
            base: [
                rng.range(-40.0, 40.0),
                rng.range(-40.0, 40.0),
                rng.range(-40.0, 40.0),
            ],
            amp: [
                rng.range(4.0, 14.0),
                rng.range(4.0, 14.0),
                rng.range(4.0, 14.0),
            ],
            freq: [
                rng.range(0.5, 2.0),
                rng.range(0.5, 2.0),
                rng.range(0.5, 2.0),
            ],
            phase: [
                rng.range(0.0, 6.28),
                rng.range(0.0, 6.28),
                rng.range(0.0, 6.28),
            ],
            radius: rng.range(0.6, 1.4),
        })
        .collect()
}

/// The 300 query rays, one per frame, sweeping around the cloud and aimed through
/// the origin so they cross plenty of spheres.
fn query_rays() -> Vec<([f32; 3], [f32; 3])> {
    (0..FRAMES)
        .map(|f| {
            let a = f as f32 * 0.06;
            let origin = [80.0 * a.cos(), 80.0 * a.sin(), 18.0 * (f as f32 * 0.031).sin()];
            let dir = normalise([-origin[0], -origin[1], -origin[2]]);
            (origin, dir)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 3D provider: spheres at their positions for one instant. Re-time it and
// rebuild (or refit) the tree per frame.
// ---------------------------------------------------------------------------

struct Scene3 {
    centres: Vec<[f32; 3]>,
    radii: Vec<f32>,
}

impl Scene3 {
    fn at(movers: &[Mover], t: f32) -> Self {
        Scene3 {
            centres: movers.iter().map(|m| m.position(t)).collect(),
            radii: movers.iter().map(|m| m.radius).collect(),
        }
    }
    fn retime(&mut self, movers: &[Mover], t: f32) {
        for (c, m) in self.centres.iter_mut().zip(movers) {
            *c = m.position(t);
        }
    }
}

impl QueryGeometry<3> for Scene3 {
    type Id = usize;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        self.centres.len()
    }
    fn id(&self, leaf: usize) -> usize {
        leaf
    }
    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        let c = self.centres[leaf];
        let r = self.radii[leaf];
        Aabb::new(
            Point([c[0] - r, c[1] - r, c[2] - r]),
            Point([c[0] + r, c[1] + r, c[2] + r]),
        )
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
        let o = [ray.origin[0], ray.origin[1], ray.origin[2]];
        let d = [ray.dir[0], ray.dir[1], ray.dir[2]];
        ray_sphere(o, d, self.centres[leaf], self.radii[leaf], max_toi)
            .map(|(t, n)| LeafHit::new(t, Point(n)))
    }
}

// ---------------------------------------------------------------------------
// 4D provider: time is the fourth axis. Each sphere's [0, T_END] trajectory is
// cut into SEG segments; a leaf is one segment, its 4D box spanning the sphere's
// spatial extent over that sub-interval and the sub-interval on the time axis.
// ---------------------------------------------------------------------------

struct Scene4 {
    movers: Vec<Mover>,
}

impl Scene4 {
    /// The (object index, time range) a leaf stands for.
    #[inline]
    fn decode(&self, leaf: usize) -> (usize, f32, f32) {
        let i = leaf / SEG;
        let s = leaf % SEG;
        let dt = T_END / SEG as f32;
        (i, s as f32 * dt, (s + 1) as f32 * dt)
    }
}

impl QueryGeometry<4> for Scene4 {
    type Id = usize;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        self.movers.len() * SEG
    }
    fn id(&self, leaf: usize) -> usize {
        leaf / SEG
    }
    fn world_aabb(&self, leaf: usize) -> Aabb<4> {
        let (i, t0, t1) = self.decode(leaf);
        let m = &self.movers[i];
        // Sample the path across the segment for a tight-ish spatial box, then pad
        // by the radius and a small margin so the true position never escapes it.
        let samples = 8;
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for s in 0..=samples {
            let t = t0 + (t1 - t0) * (s as f32 / samples as f32);
            let p = m.position(t);
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let pad = m.radius + 0.05;
        Aabb::new(
            Point([lo[0] - pad, lo[1] - pad, lo[2] - pad, t0]),
            Point([hi[0] + pad, hi[1] + pad, hi[2] + pad, t1]),
        )
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<4>, max_toi: f32) -> Option<LeafHit<4>> {
        let t_query = ray.origin[3];
        let i = leaf / SEG;
        let c = self.movers[i].position(t_query);
        let o = [ray.origin[0], ray.origin[1], ray.origin[2]];
        let d = [ray.dir[0], ray.dir[1], ray.dir[2]];
        ray_sphere(o, d, c, self.movers[i].radius, max_toi)
            .map(|(t, n)| LeafHit::new(t, Point([n[0], n[1], n[2], 0.0])))
    }
}

// ---------------------------------------------------------------------------
// Shared narrow-phase and small vector helpers.
// ---------------------------------------------------------------------------

/// Nearest ray-sphere entry with `t` in `[0, max_toi]`; `d` must be unit length.
#[inline]
fn ray_sphere(o: [f32; 3], d: [f32; 3], c: [f32; 3], r: f32, max_toi: f32) -> Option<(f32, [f32; 3])> {
    let oc = [o[0] - c[0], o[1] - c[1], o[2] - c[2]];
    let b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2];
    let cc = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - r * r;
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let mut t = -b - s;
    if t < 0.0 {
        t = -b + s; // origin inside the sphere: take the far root
    }
    if t < 0.0 || t > max_toi {
        return None;
    }
    let hit = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
    let n = normalise([hit[0] - c[0], hit[1] - c[1], hit[2] - c[2]]);
    Some((t, n))
}

#[inline]
fn normalise(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

const MAX_TOI: f32 = 300.0;

fn main() {
    let movers = build_movers();
    let rays = query_rays();
    let filter = QueryFilter::default();

    // --- Correctness: both ways must resolve the same object every frame. ---
    let scene4 = Scene4 {
        movers: movers.clone(),
    };
    let bvh4 = Bvh::build(&scene4);

    let mut mismatches = 0usize;
    let mut hits = 0usize;
    for (f, (o, d)) in rays.iter().enumerate() {
        let t = f as f32 / FPS as f32;

        let scene3 = Scene3::at(&movers, t);
        let bvh3 = Bvh::build(&scene3);
        let h3 = bvh3.raycast_nearest(&scene3, &Ray::new(Point(*o), Point(*d)), MAX_TOI, &filter);

        let ray4 = Ray::new(
            Point([o[0], o[1], o[2], t]),
            Point([d[0], d[1], d[2], 0.0]),
        );
        let h4 = bvh4.raycast_nearest(&scene4, &ray4, MAX_TOI, &filter);

        match (h3.map(|h| h.id), h4.map(|h| h.id)) {
            (a, b) if a == b => {
                if a.is_some() {
                    hits += 1;
                }
            }
            (a, b) => {
                if mismatches < 5 {
                    eprintln!("frame {f}: 3D picked {a:?}, 4D picked {b:?}");
                }
                mismatches += 1;
            }
        }
    }
    assert_eq!(mismatches, 0, "3D and 4D disagreed on {mismatches} frames");
    println!(
        "scene: {N} spheres, {FRAMES} frames ({FPS}fps x {SECS}s), one query per frame",
    );
    println!("both methods agree on all {FRAMES} frames ({hits} hits, {} misses)\n", FRAMES - hits);

    // --- Timing. Keep the fastest of REPS runs for each method. ---
    let mut best_rebuild = f64::INFINITY;
    let mut best_refit = f64::INFINITY;
    let mut best_4d_build = f64::INFINITY;
    let mut best_4d_query = f64::INFINITY;

    for _ in 0..REPS {
        // 3D, rebuild the tree every frame.
        let mut scene3 = Scene3::at(&movers, 0.0);
        let start = Instant::now();
        for (f, (o, d)) in rays.iter().enumerate() {
            let t = f as f32 / FPS as f32;
            scene3.retime(&movers, t);
            let bvh3 = Bvh::build(&scene3);
            let _ = bvh3.raycast_nearest(&scene3, &Ray::new(Point(*o), Point(*d)), MAX_TOI, &filter);
        }
        best_rebuild = best_rebuild.min(start.elapsed().as_secs_f64() * 1e3);

        // 3D, refit the tree every frame (keep topology, update bounds).
        let mut scene3 = Scene3::at(&movers, 0.0);
        let mut bvh3 = Bvh::build(&scene3);
        let start = Instant::now();
        for (f, (o, d)) in rays.iter().enumerate() {
            let t = f as f32 / FPS as f32;
            scene3.retime(&movers, t);
            bvh3.refit(&scene3);
            let _ = bvh3.raycast_nearest(&scene3, &Ray::new(Point(*o), Point(*d)), MAX_TOI, &filter);
        }
        best_refit = best_refit.min(start.elapsed().as_secs_f64() * 1e3);

        // 4D, build once then query each frame's time slice.
        let start = Instant::now();
        let bvh4 = Bvh::build(&scene4);
        best_4d_build = best_4d_build.min(start.elapsed().as_secs_f64() * 1e3);

        let start = Instant::now();
        for (f, (o, d)) in rays.iter().enumerate() {
            let t = f as f32 / FPS as f32;
            let ray4 = Ray::new(Point([o[0], o[1], o[2], t]), Point([d[0], d[1], d[2], 0.0]));
            let _ = bvh4.raycast_nearest(&scene4, &ray4, MAX_TOI, &filter);
        }
        best_4d_query = best_4d_query.min(start.elapsed().as_secs_f64() * 1e3);
    }

    let total_4d = best_4d_build + best_4d_query;
    println!("3D, rebuild per frame : {best_rebuild:7.2} ms");
    println!("3D, refit per frame   : {best_refit:7.2} ms");
    println!(
        "4D, build once        : {best_4d_build:7.2} ms build + {best_4d_query:.2} ms query = {total_4d:.2} ms",
    );
    println!();
    println!("4D vs 3D-rebuild : {:.1}x", best_rebuild / total_4d);
    println!("4D vs 3D-refit   : {:.1}x", best_refit / total_4d);
}
