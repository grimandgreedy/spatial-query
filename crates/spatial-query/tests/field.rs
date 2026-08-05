//! Phase 13: implicit-surface / field primitive leaves.
//!
//! These prove the `QueryGeometry` trait is already general enough to answer
//! queries against *field* leaves -- a leaf whose surface is an implicit
//! iso-surface marched inside `test_ray`, not an analytic primitive -- with no
//! change to the core. A field leaf marches a density field along the ray,
//! brackets each sign change, bisects to the crossing, and reports the field
//! gradient as the normal. The core builds a BVH over the leaves' bounding boxes
//! and traverses exactly as it does for triangles or spheres.
//!
//! The load-bearing checks: the accelerated BVH agrees with the naive scan on
//! field leaves (so broad-phase never wrongly prunes a leaf the field would hit),
//! the march converges to the true surface on a case with a known analytic answer,
//! and the gradient normal points back toward the ray.

use spatial_query::{naive, Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray, Scalar};

/// A tiny deterministic RNG (SplitMix64), matching the other tests.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: Scalar, hi: Scalar) -> Scalar {
        let u = (self.next_u64() >> 11) as Scalar / (1u64 << 53) as Scalar;
        lo + u * (hi - lo)
    }
    fn point(&mut self, lo: Scalar, hi: Scalar) -> Point<3> {
        Point([self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)])
    }
}

/// One implicit blob: a wobbly sphere. The field is negative inside and zero on
/// the surface. With `amp = 0` it is exactly a sphere of radius `r0`; a non-zero
/// `amp` adds smooth bumps, so the surface is genuinely implicit and must be
/// found by marching rather than a closed-form root.
#[derive(Clone, Copy)]
struct Blob {
    center: Point<3>,
    r0: Scalar,
    amp: Scalar,
    freq: Scalar,
}

impl Blob {
    fn field(&self, p: Point<3>) -> Scalar {
        let d = p - self.center;
        let base = d.dot(d) - self.r0 * self.r0;
        let bump = self.amp
            * ((self.freq * d[0]).sin() + (self.freq * d[1]).sin() + (self.freq * d[2]).sin());
        base + bump
    }

    fn gradient(&self, p: Point<3>) -> Point<3> {
        let d = p - self.center;
        let k = self.amp * self.freq;
        Point([
            2.0 * d[0] + k * (self.freq * d[0]).cos(),
            2.0 * d[1] + k * (self.freq * d[1]).cos(),
            2.0 * d[2] + k * (self.freq * d[2]).cos(),
        ])
    }

    /// A conservative half-extent for the AABB: the bump moves the surface by at
    /// most a bounded amount, so `r0` plus a fixed margin encloses it. A too-large
    /// box is always safe for broad-phase (it never wrongly prunes); it only
    /// costs a few extra narrow tests.
    fn half_extent(&self) -> Scalar {
        self.r0 + self.amp + 1.0
    }
}

/// A set of implicit blobs, one field leaf each.
struct Blobs {
    blobs: Vec<Blob>,
}

impl QueryGeometry<3> for Blobs {
    type Id = usize;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        self.blobs.len()
    }
    fn id(&self, leaf: usize) -> usize {
        leaf
    }
    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        let b = &self.blobs[leaf];
        Aabb::new(
            b.center - Point::splat(b.half_extent()),
            b.center + Point::splat(b.half_extent()),
        )
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        // The nearest crossing the march emits is the nearest hit.
        let mut nearest = None;
        self.test_ray_crossings(leaf, ray, max_toi, &mut |lh| {
            if nearest.is_none() {
                nearest = Some(lh);
            }
        });
        nearest
    }
    fn test_ray_crossings(
        &self,
        leaf: usize,
        ray: &Ray<3>,
        max_toi: Scalar,
        out: &mut dyn FnMut(LeafHit<3>),
    ) {
        let b = &self.blobs[leaf];
        const STEP: Scalar = 0.05;
        let mut t: Scalar = 0.0;
        let mut f_prev = b.field(ray.at(t));
        while t < max_toi {
            let t_next = (t + STEP).min(max_toi);
            let f_next = b.field(ray.at(t_next));
            if f_prev != 0.0 && (f_prev < 0.0) != (f_next < 0.0) {
                // Bracket [t, t_next]: bisect to the crossing.
                let (mut a, mut c) = (t, t_next);
                let mut fa = f_prev;
                for _ in 0..40 {
                    let m = 0.5 * (a + c);
                    let fm = b.field(ray.at(m));
                    if (fa < 0.0) != (fm < 0.0) {
                        c = m;
                    } else {
                        a = m;
                        fa = fm;
                    }
                }
                let root = 0.5 * (a + c);
                let normal = b.gradient(ray.at(root)).normalize_or_zero();
                out(LeafHit::new(root, normal));
            }
            if t_next >= max_toi {
                break;
            }
            t = t_next;
            f_prev = f_next;
        }
    }
}

fn random_blobs(rng: &mut Rng, n: usize) -> Blobs {
    let blobs = (0..n)
        .map(|_| Blob {
            center: rng.point(-40.0, 40.0),
            r0: rng.range(2.0, 5.0),
            amp: rng.range(0.0, 2.0),
            freq: rng.range(0.5, 1.5),
        })
        .collect();
    Blobs { blobs }
}

/// The BVH over field leaves must agree with the naive scan exactly: both call
/// the same marching `test_ray`, so any disagreement is broad-phase wrongly
/// pruning a leaf whose field the ray would have hit.
#[test]
fn bvh_matches_naive_on_field_leaves() {
    let filter = QueryFilter::default();
    let mut rng = Rng::new(0xF1E1_D000);
    for _ in 0..200 {
        let blobs = random_blobs(&mut rng, 40);
        let bvh = Bvh::build(&blobs);
        let origin = rng.point(-60.0, 60.0);
        let dir = rng.point(-1.0, 1.0);
        if dir.0 == [0.0; 3] {
            continue;
        }
        let ray = Ray::new(origin, dir);

        let n_naive = naive::raycast_nearest(&blobs, &ray, 500.0, &filter);
        let n_bvh = bvh.raycast_nearest(&blobs, &ray, 500.0, &filter);
        match (n_naive, n_bvh) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.leaf, b.leaf, "nearest leaf mismatch");
                assert_eq!(a.time_of_impact, b.time_of_impact, "nearest toi mismatch");
            }
            (a, b) => panic!(
                "nearest presence mismatch: {:?} vs {:?}",
                a.is_some(),
                b.is_some()
            ),
        }

        // All crossings of every blob along the ray, sorted, must match too.
        let c_naive = naive::raycast_crossings(&blobs, &ray, 500.0, &filter);
        let c_bvh = bvh.raycast_crossings(&blobs, &ray, 500.0, &filter);
        assert_eq!(c_naive.len(), c_bvh.len(), "crossings count mismatch");
        for (x, y) in c_naive.iter().zip(c_bvh.iter()) {
            assert_eq!(x.leaf, y.leaf, "crossings order mismatch");
            assert_eq!(x.time_of_impact, y.time_of_impact, "crossings toi mismatch");
        }
    }
}

/// With `amp = 0` a blob is exactly a sphere, so the march must converge to the
/// analytic entry and exit distances, and the gradient normal at the near hit
/// must point back toward the ray.
#[test]
fn march_converges_to_the_true_surface() {
    let filter = QueryFilter::default();
    // A pure sphere of radius 3 centred 20 units down the x axis.
    let blobs = Blobs {
        blobs: vec![Blob {
            center: Point([20.0, 0.0, 0.0]),
            r0: 3.0,
            amp: 0.0,
            freq: 1.0,
        }],
    };
    let bvh = Bvh::build(&blobs);
    let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));

    let hit = bvh
        .raycast_nearest(&blobs, &ray, 100.0, &filter)
        .expect("ray hits the sphere");
    // Entry at 20 - 3 = 17, within the march + bisection tolerance.
    assert!(
        (hit.time_of_impact - 17.0).abs() < 1e-2,
        "toi {} not near 17",
        hit.time_of_impact
    );
    // The outward normal at the near face points back along -x, toward the ray.
    assert!(
        hit.normal.dot(ray.dir) < 0.0,
        "normal faces away from the ray"
    );
    assert!(
        (hit.normal[0] + 1.0).abs() < 1e-2,
        "near-face normal is not -x: {:?}",
        hit.normal
    );

    // Both crossings: entry at 17, exit at 23.
    let crossings = bvh.raycast_crossings(&blobs, &ray, 100.0, &filter);
    assert_eq!(crossings.len(), 2, "a ray through the centre crosses twice");
    assert!((crossings[0].time_of_impact - 17.0).abs() < 1e-2);
    assert!((crossings[1].time_of_impact - 23.0).abs() < 1e-2);
}

/// A ray that misses every blob's bounding box returns nothing, and the empty
/// field scene is well-behaved.
#[test]
fn field_misses_and_empty() {
    let filter = QueryFilter::default();
    let blobs = Blobs {
        blobs: vec![Blob {
            center: Point([0.0, 0.0, 0.0]),
            r0: 2.0,
            amp: 0.5,
            freq: 1.0,
        }],
    };
    let bvh = Bvh::build(&blobs);
    // Parallel to x, offset far in z: never enters the AABB.
    let miss = Ray::new(Point([-10.0, 0.0, 50.0]), Point([1.0, 0.0, 0.0]));
    assert!(bvh.raycast_nearest(&blobs, &miss, 100.0, &filter).is_none());

    let empty = Blobs { blobs: vec![] };
    let bvh = Bvh::build(&empty);
    let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
    assert!(bvh.raycast_nearest(&empty, &ray, 100.0, &filter).is_none());
    assert!(bvh
        .raycast_crossings(&empty, &ray, 100.0, &filter)
        .is_empty());
}
