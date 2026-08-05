//! Determinism conformance: the CPU query path is bitwise-repeatable across
//! runs, and fixed-order parallel batching reproduces the sequential result
//! exactly. These are the guarantees a replay-based consumer relies on.

use spatial_query::{Aabb, Bvh, Hit, LeafHit, Point, QueryFilter, QueryGeometry, Ray, Scalar};

/// A deterministic value source: SplitMix64, so scenes and rays are the same on
/// every run without pulling in an RNG crate.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        lo + unit * (hi - lo)
    }
}

/// Scattered balls of varied radius. `SubObject = usize` exercises a non-`()`
/// sub-object through the parallel `Send` bounds.
struct Balls {
    centers: Vec<Point<3>>,
    radii: Vec<f32>,
}

impl Balls {
    fn scatter(n: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut centers = Vec::with_capacity(n);
        let mut radii = Vec::with_capacity(n);
        for _ in 0..n {
            centers.push(Point([
                rng.range(-20.0, 20.0),
                rng.range(-20.0, 20.0),
                rng.range(-20.0, 20.0),
            ]));
            radii.push(rng.range(0.3, 1.4));
        }
        Balls { centers, radii }
    }
}

impl QueryGeometry<3> for Balls {
    type Id = usize;
    type SubObject = usize;

    fn leaf_count(&self) -> usize {
        self.centers.len()
    }
    fn id(&self, leaf: usize) -> usize {
        leaf
    }
    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        let c = self.centers[leaf];
        let r = self.radii[leaf];
        Aabb::new(c - Point::splat(r), c + Point::splat(r))
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3, usize>> {
        let c = self.centers[leaf];
        let r = self.radii[leaf];
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
        Some(LeafHit::new(t, (ray.at(t) - c).normalize_or_zero()).with_sub_object(leaf))
    }
}

/// A fan of rays from scattered origins toward the cloud centre, so many hit and
/// some miss.
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
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
            ]);
            Ray::new(origin, target - origin)
        })
        .collect()
}

type Batch = Vec<Option<Hit<usize, 3, usize>>>;

#[test]
fn sequential_batch_is_bitwise_repeatable() {
    let balls = Balls::scatter(600, 0xC0FF_EE12_3456_789A);
    let bvh = Bvh::build(&balls);
    let rays = ray_batch(256, 0x0BAD_F00D_DEAD_BEEF);
    let filter = QueryFilter::default();

    let reference: Batch = bvh.raycast_nearest_batch(&balls, &rays, 200.0, &filter);
    // A nontrivial batch really does hit and miss, so the equality is meaningful.
    assert!(reference.iter().any(Option::is_some), "some rays hit");
    assert!(reference.iter().any(Option::is_none), "some rays miss");

    // Same scene, same rays, many times: identical every time, to the field.
    for run in 0..1000 {
        let again: Batch = bvh.raycast_nearest_batch(&balls, &rays, 200.0, &filter);
        assert!(again == reference, "run {run} diverged from the reference");
    }
}

#[test]
fn parallel_batch_matches_sequential_across_thread_counts() {
    let balls = Balls::scatter(800, 0x1234_5678_9ABC_DEF0);
    let bvh = Bvh::build(&balls);
    let rays = ray_batch(500, 0xFACE_B00C_1357_9BDF);
    let filter = QueryFilter::default();

    let sequential: Batch = bvh.raycast_nearest_batch(&balls, &rays, 200.0, &filter);

    // Every worker count, including ones that do not divide the batch evenly and
    // one larger than the batch, yields the identical vector: fixed-order
    // parallelism, never an unordered reduction.
    for threads in [1, 2, 3, 4, 7, 8, 16, 999] {
        let parallel: Batch =
            bvh.raycast_nearest_batch_parallel(&balls, &rays, 200.0, &filter, threads);
        assert!(
            parallel == sequential,
            "parallel with {threads} threads diverged from sequential"
        );
    }
}

#[test]
fn parallel_batch_handles_empty_and_single_ray() {
    let balls = Balls::scatter(50, 0x0000_0000_DEAD_C0DE);
    let bvh = Bvh::build(&balls);
    let filter = QueryFilter::default();

    // Empty batch: no panic, empty result, whatever the requested worker count.
    let empty: Batch = bvh.raycast_nearest_batch_parallel(&balls, &[], 200.0, &filter, 8);
    assert!(empty.is_empty());

    // Single ray: still matches the sequential answer.
    let rays = ray_batch(1, 0x5EED_5EED_5EED_5EED);
    let seq: Batch = bvh.raycast_nearest_batch(&balls, &rays, 200.0, &filter);
    let par: Batch = bvh.raycast_nearest_batch_parallel(&balls, &rays, 200.0, &filter, 8);
    assert!(par == seq);
}

#[test]
fn parallel_batch_matches_single_ray_nearest() {
    // The batch result for each ray is exactly the standalone nearest hit, so a
    // consumer can batch without changing any per-ray answer.
    let balls = Balls::scatter(400, 0xAAAA_5555_AAAA_5555);
    let bvh = Bvh::build(&balls);
    let rays = ray_batch(120, 0x3141_5926_5358_9793);
    let filter = QueryFilter::default();

    let batch: Batch = bvh.raycast_nearest_batch_parallel(&balls, &rays, 200.0, &filter, 4);
    for (i, ray) in rays.iter().enumerate() {
        let single = bvh.raycast_nearest(&balls, ray, 200.0, &filter);
        assert!(
            batch[i] == single,
            "ray {i} batch result differs from single"
        );
    }
}
