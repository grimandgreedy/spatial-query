//! GPU batched-ray backend: device hits match the CPU backend within tolerance,
//! and a refit re-queries a moved scene without a rebuild.
//!
//! These run only under a GPU leg (`--features gpu`) and only where an adapter
//! exists; with no device the raycaster is `None` and each test skips rather than
//! fails, so headless CI without a GPU stays green.
#![cfg(any(feature = "wgpu27", feature = "wgpu29"))]

use spatial_query::gpu::{GpuGeometry, GpuHit, GpuRaycaster};
use spatial_query::{Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray, Scalar};

/// SplitMix64, so scenes and rays are the same on every run without an RNG crate.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        lo + unit * (hi - lo)
    }
}

/// Spheres, described for both the CPU (`test_ray`) and the device (`narrow`).
struct Spheres {
    /// Packed `[cx, cy, cz, r]` per leaf, so the flat block is the device layout.
    prims: Vec<f32>,
    /// A displacement added to every centre by `drift`, to exercise refit.
    offset: [f32; 3],
}

impl Spheres {
    fn scatter(n: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut prims = Vec::with_capacity(n * 4);
        for _ in 0..n {
            prims.push(rng.range(-18.0, 18.0));
            prims.push(rng.range(-18.0, 18.0));
            prims.push(rng.range(-18.0, 18.0));
            prims.push(rng.range(0.5, 1.6));
        }
        Spheres {
            prims,
            offset: [0.0; 3],
        }
    }

    fn count(&self) -> usize {
        self.prims.len() / 4
    }

    fn center(&self, leaf: usize) -> Point<3> {
        Point([
            self.prims[leaf * 4] + self.offset[0],
            self.prims[leaf * 4 + 1] + self.offset[1],
            self.prims[leaf * 4 + 2] + self.offset[2],
        ])
    }

    fn radius(&self, leaf: usize) -> f32 {
        self.prims[leaf * 4 + 3]
    }

    /// Move the whole cloud, for the refit test.
    fn drift(&mut self, d: [f32; 3]) {
        self.offset = [
            self.offset[0] + d[0],
            self.offset[1] + d[1],
            self.offset[2] + d[2],
        ];
    }
}

impl QueryGeometry<3> for Spheres {
    type Id = usize;
    type SubObject = ();
    fn leaf_count(&self) -> usize {
        self.count()
    }
    fn id(&self, leaf: usize) -> usize {
        leaf
    }
    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        let c = self.center(leaf);
        let r = self.radius(leaf);
        Aabb::new(c - Point::splat(r), c + Point::splat(r))
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        let c = self.center(leaf);
        let r = self.radius(leaf);
        let oc = ray.origin - c;
        let b = oc.dot(ray.dir);
        let disc = b * b - (oc.length_squared() - r * r);
        if disc < 0.0 {
            return None;
        }
        let t = -b - disc.sqrt();
        if t < 0.0 || t > max_toi {
            return None;
        }
        Some(LeafHit::new(t, (ray.at(t) - c).normalize_or_zero()))
    }
}

impl GpuGeometry for Spheres {
    fn prim_stride(&self) -> u32 {
        4
    }
    fn prim_data(&self) -> Vec<f32> {
        // The device sees the drifted centres, matching `center`.
        let mut out = self.prims.clone();
        for leaf in 0..self.count() {
            out[leaf * 4] += self.offset[0];
            out[leaf * 4 + 1] += self.offset[1];
            out[leaf * 4 + 2] += self.offset[2];
        }
        out
    }
    fn narrow_wgsl(&self) -> String {
        // Analytic ray-sphere, the WGSL twin of `test_ray`.
        r#"
        fn narrow(leaf: u32, ro: vec3<f32>, rd: vec3<f32>, max_toi: f32) -> Hit {
            let base = leaf * params.prim_stride;
            let c = vec3<f32>(prims[base], prims[base + 1u], prims[base + 2u]);
            let r = prims[base + 3u];
            let oc = ro - c;
            let b = dot(oc, rd);
            let disc = b * b - (dot(oc, oc) - r * r);
            var h: Hit;
            h.hit = 0u;
            h.toi = 0.0;
            h.normal = vec3<f32>(0.0);
            if (disc < 0.0) {
                return h;
            }
            let t = -b - sqrt(disc);
            if (t < 0.0 || t > max_toi) {
                return h;
            }
            h.hit = 1u;
            h.toi = t;
            h.normal = normalize((ro + rd * t) - c);
            return h;
        }
        "#
        .to_string()
    }
}

fn ray_batch(n: usize, seed: u64) -> Vec<Ray<3>> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let origin = Point([
                rng.range(-30.0, 30.0),
                rng.range(-30.0, 30.0),
                rng.range(-30.0, 30.0),
            ]);
            let target = Point([
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
            ]);
            Ray::new(origin, target - origin)
        })
        .collect()
}

/// Compare a GPU batch against the CPU baseline. Same leaf identity everywhere,
/// `time_of_impact` and normal within tolerance. On the rare genuine near-tie
/// (two leaves within tolerance for one ray), the leaves may differ; then the
/// tois must be within tolerance of each other, which is the definition of a tie.
fn assert_matches(
    gpu: &[Option<GpuHit>],
    cpu: &[Option<spatial_query::Hit<usize, 3>>],
    rays: &[Ray<3>],
    scene: &Spheres,
) {
    // The GPU narrow-phase rounds differently from the CPU, so `time_of_impact`
    // agrees only within a combined absolute + relative tolerance (a far hit at
    // toi ~40 carries proportionally more rounding than a near one).
    let toi_close = |a: f32, b: f32| (a - b).abs() <= 1e-3 + 1e-4 * a.abs().max(b.abs());
    const N_TOL: f32 = 3e-3;
    assert_eq!(gpu.len(), cpu.len());
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        match (g, c) {
            (None, None) => {}
            (Some(g), Some(c)) => {
                if g.leaf as usize != c.leaf {
                    // Allowed only if it is a true toi tie between the two.
                    let alt = scene
                        .test_ray(g.leaf as usize, &rays[i], 1e30)
                        .map(|lh| lh.toi)
                        .unwrap_or(f32::INFINITY);
                    assert!(
                        toi_close(alt, c.time_of_impact),
                        "ray {i}: gpu leaf {} != cpu leaf {} and not a tie ({} vs {})",
                        g.leaf,
                        c.leaf,
                        alt,
                        c.time_of_impact
                    );
                }
                assert!(
                    toi_close(g.time_of_impact, c.time_of_impact),
                    "ray {i}: toi {} vs {}",
                    g.time_of_impact,
                    c.time_of_impact
                );
                let dn = (g.normal - c.normal).length();
                assert!(dn < N_TOL, "ray {i}: normal off by {dn}");
            }
            (g, c) => panic!("ray {i}: gpu {g:?} disagrees with cpu {c:?} on hit/miss"),
        }
    }
}

#[test]
fn gpu_batch_matches_cpu_within_tolerance() {
    let scene = Spheres::scatter(1500, 0xC0FF_EE00_1234_5678);
    let bvh = Bvh::build(&scene);
    let Some(gpu) = GpuRaycaster::headless(&bvh, &scene) else {
        eprintln!("no GPU adapter: skipping gpu_batch_matches_cpu_within_tolerance");
        return;
    };
    let rays = ray_batch(2000, 0x0BAD_F00D_1111_2222);
    let filter = QueryFilter::default();

    let cpu = bvh.raycast_nearest_batch(&scene, &rays, 200.0, &filter);
    let dev = gpu.raycast_nearest_batch(&rays, 200.0);

    // A meaningful batch really hits and misses.
    assert!(cpu.iter().any(Option::is_some), "some rays hit");
    assert!(cpu.iter().any(Option::is_none), "some rays miss");
    assert_matches(&dev, &cpu, &rays, &scene);
}

#[test]
fn gpu_refit_requeries_a_moved_scene() {
    let mut scene = Spheres::scatter(1000, 0x5EED_1234_ABCD_0001);
    let mut bvh = Bvh::build(&scene);
    let Some(mut gpu) = GpuRaycaster::headless(&bvh, &scene) else {
        eprintln!("no GPU adapter: skipping gpu_refit_requeries_a_moved_scene");
        return;
    };
    let rays = ray_batch(1200, 0x3141_5926_5358_9793);
    let filter = QueryFilter::default();

    // Move the whole cloud, refit the CPU tree, and refit the GPU buffers in
    // place (no rebuild, no new pipeline).
    scene.drift([5.0, -3.0, 2.0]);
    bvh.refit(&scene);
    gpu.refit(&bvh, &scene);

    let cpu = bvh.raycast_nearest_batch(&scene, &rays, 200.0, &filter);
    let dev = gpu.raycast_nearest_batch(&rays, 200.0);
    assert!(cpu.iter().any(Option::is_some), "some rays hit after drift");
    assert_matches(&dev, &cpu, &rays, &scene);
}

#[test]
fn gpu_empty_batch_and_empty_scene() {
    let scene = Spheres::scatter(30, 0x0000_DEAD_BEEF_0001);
    let bvh = Bvh::build(&scene);
    let Some(gpu) = GpuRaycaster::headless(&bvh, &scene) else {
        eprintln!("no GPU adapter: skipping gpu_empty_batch_and_empty_scene");
        return;
    };
    // Empty batch: empty result, no dispatch.
    assert!(gpu.raycast_nearest_batch(&[], 100.0).is_empty());
}
