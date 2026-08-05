//! Tests for shape-cast and overlap queries: accelerated-vs-naive parity on
//! random scenes, a swept sphere stopping at the nearest surface checked against
//! a fine-step raycast, and overlap on analytic cases with known answers.

use spatial_query::{
    naive, Aabb, Bvh, InstancedGeometry, Isometry, LeafHit, Overlap, Point, QueryFilter,
    QueryGeometry, Ray, Scalar, ShapeCast, Tlas,
};

// Small deterministic RNG (SplitMix64), so tests need no `rand` dependency.
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
    fn point<const D: usize>(&mut self, lo: Scalar, hi: Scalar) -> Point<D> {
        let mut a = [0.0; D];
        for x in a.iter_mut() {
            *x = self.range(lo, hi);
        }
        Point(a)
    }
}

// ---------------------------------------------------------------------------
// A ball provider for the single-level path.
// ---------------------------------------------------------------------------

/// The probe swept in these tests: a ball of radius `PROBE_RADIUS`.
const PROBE_RADIUS: Scalar = 0.75;

struct Balls {
    centers: Vec<Point<3>>,
    radii: Vec<Scalar>,
}

impl Balls {
    fn random(rng: &mut Rng, n: usize) -> Self {
        let centers = (0..n).map(|_| rng.point::<3>(-30.0, 30.0)).collect();
        let radii = (0..n).map(|_| rng.range(0.5, 2.0)).collect();
        Balls { centers, radii }
    }
}

/// The nearest distance from `p` to the surface of the box `[min, max]`, zero if
/// inside.
fn point_to_aabb(p: Point<3>, min: Point<3>, max: Point<3>) -> Scalar {
    let mut d2 = 0.0;
    for i in 0..3 {
        let v = if p[i] < min[i] {
            min[i] - p[i]
        } else if p[i] > max[i] {
            p[i] - max[i]
        } else {
            0.0
        };
        d2 += v * v;
    }
    d2.sqrt()
}

impl QueryGeometry<3> for Balls {
    type Id = usize;

    fn leaf_count(&self) -> usize {
        self.centers.len()
    }
    fn id(&self, leaf: usize) -> usize {
        leaf
    }
    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        let c = self.centers[leaf];
        Aabb::new(
            c - Point::splat(self.radii[leaf]),
            c + Point::splat(self.radii[leaf]),
        )
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        ray_ball(ray, self.centers[leaf], self.radii[leaf], max_toi)
    }
    // A ball probe swept against a static ball reduces to a ray against a ball of
    // the summed radius, centred on the static ball.
    fn test_shape_cast(
        &self,
        leaf: usize,
        cast: &ShapeCast<3>,
        max_toi: Scalar,
    ) -> Option<LeafHit<3>> {
        let c = self.centers[leaf];
        let sum = self.radii[leaf] + PROBE_RADIUS;
        let ray = Ray::new_unnormalized(cast.origin, cast.dir);
        let mut lh = ray_ball(&ray, c, sum, max_toi)?;
        // The contact normal points from the static ball's centre to the probe's
        // centre at the moment of contact.
        lh.normal = (cast.at(lh.toi) - c).normalize_or_zero();
        Some(lh)
    }
    // Exact ball-vs-box overlap: nearest point of the box to the centre within
    // the radius.
    fn test_overlap(&self, leaf: usize, region: &Aabb<3>) -> bool {
        point_to_aabb(self.centers[leaf], region.min, region.max) <= self.radii[leaf]
    }
}

/// Ray against a ball, returning the near entry point-of-impact and outward
/// normal, or `None`.
fn ray_ball(ray: &Ray<3>, center: Point<3>, radius: Scalar, max_toi: Scalar) -> Option<LeafHit<3>> {
    let oc = ray.origin - center;
    let b = oc.dot(ray.dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let mut t = -b - sq;
    if t < 0.0 {
        t = -b + sq;
    }
    if t < 0.0 || t > max_toi {
        return None;
    }
    let normal = (ray.at(t) - center).normalize_or_zero();
    Some(LeafHit::new(t, normal))
}

/// A ball-shaped probe from `origin` along `dir`.
fn ball_cast(origin: Point<3>, dir: Point<3>) -> ShapeCast<3> {
    ShapeCast::new(
        origin,
        dir,
        Aabb::new(Point::splat(-PROBE_RADIUS), Point::splat(PROBE_RADIUS)),
    )
}

// ---------------------------------------------------------------------------
// Shape cast: accelerated vs naive on random scenes.
// ---------------------------------------------------------------------------

#[test]
fn bvh_shapecast_matches_naive() {
    let mut rng = Rng::new(0x5EED_C0DE);
    for _ in 0..200 {
        let balls = Balls::random(&mut rng, 64);
        let bvh = Bvh::build(&balls);
        let filter = QueryFilter::default();
        let cast = ball_cast(rng.point::<3>(-50.0, 50.0), rng.point::<3>(-1.0, 1.0));
        if cast.dir == Point::<3>::ZERO {
            continue;
        }

        let n_near = naive::shapecast_nearest(&balls, &cast, 500.0, &filter);
        let b_near = bvh.shapecast_nearest(&balls, &cast, 500.0, &filter);
        assert_eq!(
            n_near.map(|h| (h.leaf, h.time_of_impact)),
            b_near.map(|h| (h.leaf, h.time_of_impact)),
            "shapecast nearest",
        );

        let n_all = naive::shapecast_all(&balls, &cast, 500.0, &filter);
        let b_all = bvh.shapecast_all(&balls, &cast, 500.0, &filter);
        assert_eq!(n_all.len(), b_all.len(), "shapecast all-hits count");
        for (x, y) in n_all.iter().zip(b_all.iter()) {
            assert_eq!(x.leaf, y.leaf, "shapecast all-hits order");
            assert_eq!(x.time_of_impact, y.time_of_impact, "shapecast all-hits toi");
        }
    }
}

#[test]
fn swept_ball_stops_at_nearest_surface() {
    // One ball at x = 10, radius 1. A radius-0.75 probe swept from the origin
    // along +x should first touch it when the probe centre is at x = 10 - 1.75,
    // so toi = 8.25.
    let balls = Balls {
        centers: vec![Point([10.0, 0.0, 0.0])],
        radii: vec![1.0],
    };
    let bvh = Bvh::build(&balls);
    let cast = ball_cast(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
    let hit = bvh
        .shapecast_nearest(&balls, &cast, 100.0, &QueryFilter::default())
        .expect("the swept ball should hit");
    assert!(
        (hit.time_of_impact - 8.25).abs() < 1e-4,
        "toi {}",
        hit.time_of_impact
    );
    // Contact normal faces back along the sweep (outward from the static ball).
    assert!(
        (hit.normal[0] - -1.0).abs() < 1e-4,
        "normal {:?}",
        hit.normal
    );

    // Cross-check the contact toi against a fine-step point raycast: march the
    // probe centre forward until it first comes within (radius + probe) of the
    // ball centre.
    let step = 0.001;
    let mut t = 0.0;
    let sum = 1.0 + PROBE_RADIUS;
    let approx = loop {
        let c = cast.at(t);
        if (c - balls.centers[0]).length() <= sum {
            break t;
        }
        t += step;
        if t > 100.0 {
            panic!("fine-step march found no contact");
        }
    };
    assert!(
        (hit.time_of_impact - approx).abs() < 2.0 * step,
        "swept toi {} vs fine-step {}",
        hit.time_of_impact,
        approx
    );
}

// ---------------------------------------------------------------------------
// Overlap: known analytic answers, and accelerated vs naive.
// ---------------------------------------------------------------------------

#[test]
fn overlap_returns_exactly_intersecting_balls() {
    // Three unit balls on the x axis at 0, 5, 100. A box region around the
    // origin spanning [-1.5, 1.5] touches only the ball at 0 (the ball at 5 is
    // 4 units away from the box face, further than its radius).
    let balls = Balls {
        centers: vec![
            Point([0.0, 0.0, 0.0]),
            Point([5.0, 0.0, 0.0]),
            Point([100.0, 0.0, 0.0]),
        ],
        radii: vec![1.0, 1.0, 1.0],
    };
    let bvh = Bvh::build(&balls);
    let region = Aabb::new(Point([-1.5, -1.5, -1.5]), Point([1.5, 1.5, 1.5]));
    let hits = bvh.overlap(&balls, &region, &QueryFilter::default());
    let ids: Vec<usize> = hits.iter().map(|o: &Overlap<usize>| o.leaf).collect();
    assert_eq!(ids, vec![0], "only the ball at the origin overlaps");

    // A wider box reaching x = 4 pulls in the ball at 5 (its surface reaches
    // x = 4), and never the far one.
    let region = Aabb::new(Point([-1.5, -1.5, -1.5]), Point([4.0, 1.5, 1.5]));
    let hits = bvh.overlap(&balls, &region, &QueryFilter::default());
    let ids: Vec<usize> = hits.iter().map(|o| o.leaf).collect();
    assert_eq!(ids, vec![0, 1], "the near two balls overlap");
}

#[test]
fn bvh_overlap_matches_naive() {
    let mut rng = Rng::new(0x0BEE_F00D);
    for _ in 0..200 {
        let balls = Balls::random(&mut rng, 96);
        let bvh = Bvh::build(&balls);
        let filter = QueryFilter::default();
        let c = rng.point::<3>(-30.0, 30.0);
        let h = rng.point::<3>(1.0, 6.0);
        let region = Aabb::new(c - h, c + h);

        let n = naive::overlap(&balls, &region, &filter);
        let b = bvh.overlap(&balls, &region, &filter);
        let nk: Vec<usize> = n.iter().map(|o| o.leaf).collect();
        let bk: Vec<usize> = b.iter().map(|o| o.leaf).collect();
        assert_eq!(nk, bk, "overlap leaves");
    }
}

// ---------------------------------------------------------------------------
// The two-level path over transformed ball instances.
// ---------------------------------------------------------------------------

// Rotation does not move a ball, so a ball instance is a clean case for the
// world-space instanced shape-cast and overlap tests.
struct BallInstances {
    transforms: Vec<Isometry<3>>,
    radii: Vec<Scalar>,
}

impl BallInstances {
    fn random(rng: &mut Rng, n: usize) -> Self {
        let transforms = (0..n)
            .map(|_| Isometry::from_translation(rng.point::<3>(-30.0, 30.0)))
            .collect();
        let radii = (0..n).map(|_| rng.range(0.5, 2.0)).collect();
        BallInstances { transforms, radii }
    }
    fn center(&self, i: usize) -> Point<3> {
        self.transforms[i].translation
    }
}

impl InstancedGeometry<3> for BallInstances {
    type Id = usize;

    fn instance_count(&self) -> usize {
        self.transforms.len()
    }
    fn id(&self, i: usize) -> usize {
        i
    }
    fn transform(&self, i: usize) -> Isometry<3> {
        self.transforms[i]
    }
    fn local_aabb(&self, i: usize) -> Aabb<3> {
        Aabb::new(Point::splat(-self.radii[i]), Point::splat(self.radii[i]))
    }
    fn test_ray_local(&self, i: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        ray_ball(ray, Point::<3>::ZERO, self.radii[i], max_toi)
    }
    fn test_shape_cast(
        &self,
        i: usize,
        cast: &ShapeCast<3>,
        max_toi: Scalar,
    ) -> Option<LeafHit<3>> {
        let c = self.center(i);
        let sum = self.radii[i] + PROBE_RADIUS;
        let ray = Ray::new_unnormalized(cast.origin, cast.dir);
        let mut lh = ray_ball(&ray, c, sum, max_toi)?;
        lh.normal = (cast.at(lh.toi) - c).normalize_or_zero();
        Some(lh)
    }
    fn test_overlap(&self, i: usize, region: &Aabb<3>) -> bool {
        point_to_aabb(self.center(i), region.min, region.max) <= self.radii[i]
    }
}

#[test]
fn tlas_shapecast_matches_naive_via_ball_bvh() {
    // Ground truth: the same balls tested through the single-level naive path.
    let mut rng = Rng::new(0x1234_ABCD);
    for _ in 0..150 {
        let n = 48;
        let inst = BallInstances::random(&mut rng, n);
        let tlas = Tlas::build(&inst);
        let filter = QueryFilter::default();

        // A plain Balls provider over the same centres/radii is the oracle.
        let balls = Balls {
            centers: (0..n).map(|i| inst.center(i)).collect(),
            radii: inst.radii.clone(),
        };

        let cast = ball_cast(rng.point::<3>(-50.0, 50.0), rng.point::<3>(-1.0, 1.0));
        if cast.dir == Point::<3>::ZERO {
            continue;
        }

        let want = naive::shapecast_nearest(&balls, &cast, 500.0, &filter);
        let got = tlas.shapecast_nearest(&inst, &cast, 500.0, &filter);
        assert_eq!(
            want.map(|h| (h.leaf, h.time_of_impact)),
            got.map(|h| (h.leaf, h.time_of_impact)),
            "tlas shapecast nearest",
        );

        let want_all = naive::shapecast_all(&balls, &cast, 500.0, &filter);
        let got_all = tlas.shapecast_all(&inst, &cast, 500.0, &filter);
        assert_eq!(want_all.len(), got_all.len(), "tlas shapecast all count");
        for (x, y) in want_all.iter().zip(got_all.iter()) {
            assert_eq!(x.leaf, y.leaf, "tlas shapecast all order");
            assert_eq!(x.time_of_impact, y.time_of_impact, "tlas shapecast all toi");
        }
    }
}

#[test]
fn tlas_overlap_matches_naive_via_ball_bvh() {
    let mut rng = Rng::new(0x9999_5555);
    for _ in 0..150 {
        let n = 64;
        let inst = BallInstances::random(&mut rng, n);
        let tlas = Tlas::build(&inst);
        let filter = QueryFilter::default();
        let balls = Balls {
            centers: (0..n).map(|i| inst.center(i)).collect(),
            radii: inst.radii.clone(),
        };

        let c = rng.point::<3>(-30.0, 30.0);
        let h = rng.point::<3>(1.0, 6.0);
        let region = Aabb::new(c - h, c + h);

        let want: Vec<usize> = naive::overlap(&balls, &region, &filter)
            .iter()
            .map(|o| o.leaf)
            .collect();
        let got: Vec<usize> = tlas
            .overlap(&inst, &region, &filter)
            .iter()
            .map(|o| o.leaf)
            .collect();
        assert_eq!(want, got, "tlas overlap leaves");
    }
}
