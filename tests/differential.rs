//! Differential tests: the accelerated BVH must match the naive baseline
//! bit-for-bit, in several dimensions, on randomized scenes.

use spatial_query::{naive, Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray, Scalar};

/// A tiny deterministic RNG (SplitMix64) so tests need no `rand` dependency and
/// are reproducible across runs and platforms.
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
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: Scalar, hi: Scalar) -> Scalar {
        let u = (self.next_u64() >> 11) as Scalar / (1u64 << 53) as Scalar;
        lo + u * (hi - lo)
    }
    fn point<const D: usize>(&mut self, lo: Scalar, hi: Scalar) -> Point<D> {
        let mut a = [0.0; D];
        for x in a.iter_mut() {
            *x = self.range(lo, hi);
        }
        Point(a)
    }
}

/// N-ball provider: analytic hyperspheres, whose ray test is dimension-generic.
struct Balls<const D: usize> {
    centers: Vec<Point<D>>,
    radii: Vec<Scalar>,
}

impl<const D: usize> QueryGeometry<D> for Balls<D> {
    type Id = usize;
    // A non-() sub-object, so the accelerated path is exercised threading a real
    // payload through at D = 2, 3, and 4.
    type SubObject = usize;

    fn leaf_count(&self) -> usize {
        self.centers.len()
    }

    fn id(&self, leaf: usize) -> usize {
        leaf
    }

    fn world_aabb(&self, leaf: usize) -> Aabb<D> {
        let c = self.centers[leaf];
        let r = self.radii[leaf];
        Aabb::new(c - Point::splat(r), c + Point::splat(r))
    }

    fn test_ray(&self, leaf: usize, ray: &Ray<D>, max_toi: Scalar) -> Option<LeafHit<D, usize>> {
        let c = self.centers[leaf];
        let r = self.radii[leaf];
        let oc = ray.origin - c;
        // dir is unit length, so the quadratic has a = 1.
        let b = oc.dot(ray.dir);
        let cc = oc.length_squared() - r * r;
        let disc = b * b - cc;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let mut t = -b - sq;
        if t < 0.0 {
            t = -b + sq; // ray origin inside the ball: take the exit
        }
        if t < 0.0 || t > max_toi {
            return None;
        }
        let normal = (ray.at(t) - c).normalize_or_zero();
        Some(LeafHit::new(t, normal).with_sub_object(leaf))
    }
}

fn random_balls<const D: usize>(rng: &mut Rng, n: usize) -> Balls<D> {
    let centers = (0..n).map(|_| rng.point::<D>(-50.0, 50.0)).collect();
    let radii = (0..n).map(|_| rng.range(0.5, 4.0)).collect();
    Balls { centers, radii }
}

/// Assert BVH nearest and all-hits agree with the naive baseline exactly.
fn assert_parity<const D: usize>(balls: &Balls<D>, ray: &Ray<D>, max_toi: Scalar) {
    let filter = QueryFilter::default();
    let bvh = Bvh::build(balls);

    let n_naive = naive::raycast_nearest(balls, ray, max_toi, &filter);
    let n_bvh = bvh.raycast_nearest(balls, ray, max_toi, &filter);
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

    let a_naive = naive::raycast_all(balls, ray, max_toi, &filter);
    let a_bvh = bvh.raycast_all(balls, ray, max_toi, &filter);
    assert_eq!(a_naive.len(), a_bvh.len(), "all-hits count mismatch");
    for (x, y) in a_naive.iter().zip(a_bvh.iter()) {
        assert_eq!(x.leaf, y.leaf, "all-hits order mismatch");
        assert_eq!(x.time_of_impact, y.time_of_impact, "all-hits toi mismatch");
    }
}

#[test]
fn bvh_matches_naive_3d() {
    let mut rng = Rng::new(0xABCD_1234);
    for _ in 0..200 {
        let balls = random_balls::<3>(&mut rng, 64);
        let origin = rng.point::<3>(-60.0, 60.0);
        let dir = rng.point::<3>(-1.0, 1.0);
        let ray = Ray::new(origin, dir);
        if ray.dir == Point::<3>::ZERO {
            continue;
        }
        assert_parity(&balls, &ray, 500.0);
    }
}

// Run the core at dimensions other than 3 to confirm it is genuinely
// N-dimensional and not accidentally hardcoded to 3.

#[test]
fn bvh_matches_naive_2d() {
    let mut rng = Rng::new(0x2222_2222);
    for _ in 0..200 {
        let balls = random_balls::<2>(&mut rng, 48);
        let ray = Ray::new(rng.point::<2>(-60.0, 60.0), rng.point::<2>(-1.0, 1.0));
        if ray.dir == Point::<2>::ZERO {
            continue;
        }
        assert_parity(&balls, &ray, 500.0);
    }
}

#[test]
fn bvh_matches_naive_4d() {
    let mut rng = Rng::new(0x4444_4444);
    for _ in 0..200 {
        let balls = random_balls::<4>(&mut rng, 48);
        let ray = Ray::new(rng.point::<4>(-40.0, 40.0), rng.point::<4>(-1.0, 1.0));
        if ray.dir == Point::<4>::ZERO {
            continue;
        }
        assert_parity(&balls, &ray, 500.0);
    }
}

#[test]
fn edge_cases() {
    let filter = QueryFilter::default();

    // Empty scene.
    let empty = Balls::<3> {
        centers: vec![],
        radii: vec![],
    };
    let bvh = Bvh::build(&empty);
    let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
    assert!(bvh.raycast_nearest(&empty, &ray, 100.0, &filter).is_none());
    assert!(bvh.raycast_all(&empty, &ray, 100.0, &filter).is_empty());

    // Single leaf, direct hit.
    let one = Balls::<3> {
        centers: vec![Point([5.0, 0.0, 0.0])],
        radii: vec![1.0],
    };
    let bvh = Bvh::build(&one);
    let hit = bvh.raycast_nearest(&one, &ray, 100.0, &filter).unwrap();
    assert_eq!(hit.leaf, 0);
    assert!((hit.time_of_impact - 4.0).abs() < 1e-4);

    // Ray points away: miss.
    let away = Ray::new(Point([0.0, 0.0, 0.0]), Point([-1.0, 0.0, 0.0]));
    assert!(bvh.raycast_nearest(&one, &away, 100.0, &filter).is_none());

    // max_toi clips the hit.
    assert!(bvh.raycast_nearest(&one, &ray, 3.0, &filter).is_none());
}

#[test]
fn all_hits_are_sorted_and_ordered_by_leaf_on_ties() {
    let filter = QueryFilter::default();
    // Two identical concentric balls -> equal toi -> tie broken by leaf index.
    let balls = Balls::<3> {
        centers: vec![Point([5.0, 0.0, 0.0]), Point([5.0, 0.0, 0.0])],
        radii: vec![1.0, 1.0],
    };
    let bvh = Bvh::build(&balls);
    let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
    let hits = bvh.raycast_all(&balls, &ray, 100.0, &filter);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].leaf, 0);
    assert_eq!(hits[1].leaf, 1);
    assert!(hits[0].time_of_impact <= hits[1].time_of_impact);
}
