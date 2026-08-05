//! Tests for all-crossings raycast: a provider that reports both the entry and
//! exit of each sphere, checked analytically, against the naive scan, and for
//! the two-level tree. Also checks the default (single-hit) behaviour matches
//! `raycast_all`.

use spatial_query::{
    naive, Aabb, Bvh, InstancedGeometry, Isometry, LeafHit, Point, QueryFilter, QueryGeometry, Ray,
    Scalar, Tlas,
};

// Small deterministic RNG (SplitMix64).
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

/// Spheres that report both the entry and the exit of a ray as crossings.
struct Shells<const D: usize> {
    centers: Vec<Point<D>>,
    radii: Vec<Scalar>,
}

impl<const D: usize> Shells<D> {
    fn random(rng: &mut Rng, n: usize) -> Self {
        let centers = (0..n).map(|_| rng.point::<D>(-30.0, 30.0)).collect();
        let radii = (0..n).map(|_| rng.range(1.0, 4.0)).collect();
        Shells { centers, radii }
    }
}

/// The two ray parameters where an unbounded ray meets a sphere, if real.
fn sphere_roots<const D: usize>(
    ray: &Ray<D>,
    center: Point<D>,
    radius: Scalar,
) -> Option<(Scalar, Scalar)> {
    let oc = ray.origin - center;
    let b = oc.dot(ray.dir);
    let disc = b * b - (oc.length_squared() - radius * radius);
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    Some((-b - sq, -b + sq))
}

impl<const D: usize> QueryGeometry<D> for Shells<D> {
    type Id = usize;
    type SubObject = ();

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
    fn test_ray(&self, leaf: usize, ray: &Ray<D>, max_toi: Scalar) -> Option<LeafHit<D>> {
        let (t0, t1) = sphere_roots(ray, self.centers[leaf], self.radii[leaf])?;
        let t = if t0 >= 0.0 { t0 } else { t1 };
        if t < 0.0 || t > max_toi {
            return None;
        }
        Some(LeafHit::new(
            t,
            (ray.at(t) - self.centers[leaf]).normalize_or_zero(),
        ))
    }
    fn test_ray_crossings(
        &self,
        leaf: usize,
        ray: &Ray<D>,
        max_toi: Scalar,
        out: &mut dyn FnMut(LeafHit<D>),
    ) {
        let Some((t0, t1)) = sphere_roots(ray, self.centers[leaf], self.radii[leaf]) else {
            return;
        };
        for t in [t0, t1] {
            if t >= 0.0 && t <= max_toi {
                let n = (ray.at(t) - self.centers[leaf]).normalize_or_zero();
                out(LeafHit::new(t, n));
            }
        }
    }
}

#[test]
fn crossings_report_entry_and_exit() {
    // One sphere at x = 10, radius 2. A ray from the origin along +x enters at
    // x = 8 (toi 8, normal -x) and exits at x = 12 (toi 12, normal +x).
    let shells = Shells::<3> {
        centers: vec![Point([10.0, 0.0, 0.0])],
        radii: vec![2.0],
    };
    let bvh = Bvh::build(&shells);
    let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
    let hits = bvh.raycast_crossings(&shells, &ray, 100.0, &QueryFilter::default());
    assert_eq!(hits.len(), 2, "entry and exit");
    assert!((hits[0].time_of_impact - 8.0).abs() < 1e-4);
    assert!(
        (hits[0].normal[0] - -1.0).abs() < 1e-4,
        "entry normal faces -x"
    );
    assert!((hits[1].time_of_impact - 12.0).abs() < 1e-4);
    assert!(
        (hits[1].normal[0] - 1.0).abs() < 1e-4,
        "exit normal faces +x"
    );
}

fn assert_bvh_matches_naive<const D: usize>(shells: &Shells<D>, ray: &Ray<D>, max_toi: Scalar) {
    let bvh = Bvh::build(shells);
    let filter = QueryFilter::default();
    let n = naive::raycast_crossings(shells, ray, max_toi, &filter);
    let b = bvh.raycast_crossings(shells, ray, max_toi, &filter);
    assert_eq!(n.len(), b.len(), "crossing count");
    for (x, y) in n.iter().zip(b.iter()) {
        assert_eq!(x.leaf, y.leaf, "crossing order (leaf)");
        assert_eq!(x.time_of_impact, y.time_of_impact, "crossing toi");
    }
}

#[test]
fn crossings_match_naive_3d() {
    let mut rng = Rng::new(0xC0FF_EE11);
    for _ in 0..200 {
        let shells = Shells::<3>::random(&mut rng, 64);
        let ray = Ray::new(rng.point::<3>(-40.0, 40.0), rng.point::<3>(-1.0, 1.0));
        if ray.dir == Point::<3>::ZERO {
            continue;
        }
        assert_bvh_matches_naive(&shells, &ray, 500.0);
    }
}

#[test]
fn crossings_match_naive_2d() {
    let mut rng = Rng::new(0x2D2D_2D2D);
    for _ in 0..200 {
        let shells = Shells::<2>::random(&mut rng, 48);
        let ray = Ray::new(rng.point::<2>(-40.0, 40.0), rng.point::<2>(-1.0, 1.0));
        if ray.dir == Point::<2>::ZERO {
            continue;
        }
        assert_bvh_matches_naive(&shells, &ray, 500.0);
    }
}

/// A provider that reports only a single hit, to check the default
/// `test_ray_crossings` (which wraps `test_ray`) reproduces `raycast_all`.
struct SolidBalls {
    centers: Vec<Point<3>>,
    radii: Vec<Scalar>,
}

impl QueryGeometry<3> for SolidBalls {
    type Id = usize;
    type SubObject = ();

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
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        let (t0, t1) = sphere_roots(ray, self.centers[leaf], self.radii[leaf])?;
        let t = if t0 >= 0.0 { t0 } else { t1 };
        if t < 0.0 || t > max_toi {
            return None;
        }
        Some(LeafHit::new(
            t,
            (ray.at(t) - self.centers[leaf]).normalize_or_zero(),
        ))
    }
}

#[test]
fn default_crossings_reproduce_all_hits() {
    let mut rng = Rng::new(0x5011_D000);
    let balls = SolidBalls {
        centers: (0..80).map(|_| rng.point::<3>(-25.0, 25.0)).collect(),
        radii: (0..80).map(|_| rng.range(1.0, 3.0)).collect(),
    };
    let bvh = Bvh::build(&balls);
    let filter = QueryFilter::default();
    for _ in 0..100 {
        let ray = Ray::new(rng.point::<3>(-35.0, 35.0), rng.point::<3>(-1.0, 1.0));
        if ray.dir == Point::<3>::ZERO {
            continue;
        }
        let all = bvh.raycast_all(&balls, &ray, 500.0, &filter);
        let crossings = bvh.raycast_crossings(&balls, &ray, 500.0, &filter);
        assert_eq!(all.len(), crossings.len(), "default crossings count");
        for (a, c) in all.iter().zip(crossings.iter()) {
            assert_eq!(a.leaf, c.leaf);
            assert_eq!(a.time_of_impact, c.time_of_impact);
        }
    }
}

// ---------------------------------------------------------------------------
// Two-level crossings over sphere instances (rotation leaves a sphere fixed, so
// a single-level Shells over the same centres is the oracle).
// ---------------------------------------------------------------------------

struct ShellInstances {
    transforms: Vec<Isometry<3>>,
    radii: Vec<Scalar>,
}

impl InstancedGeometry<3> for ShellInstances {
    type Id = usize;
    type SubObject = ();

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
        let (t0, t1) = sphere_roots(ray, Point::<3>::ZERO, self.radii[i])?;
        let t = if t0 >= 0.0 { t0 } else { t1 };
        if t < 0.0 || t > max_toi {
            return None;
        }
        Some(LeafHit::new(t, ray.at(t).normalize_or_zero()))
    }
    fn test_ray_crossings_local(
        &self,
        i: usize,
        ray: &Ray<3>,
        max_toi: Scalar,
        out: &mut dyn FnMut(LeafHit<3>),
    ) {
        let Some((t0, t1)) = sphere_roots(ray, Point::<3>::ZERO, self.radii[i]) else {
            return;
        };
        for t in [t0, t1] {
            if t >= 0.0 && t <= max_toi {
                out(LeafHit::new(t, ray.at(t).normalize_or_zero()));
            }
        }
    }
}

#[test]
fn tlas_crossings_match_single_level_oracle() {
    let mut rng = Rng::new(0x7717_9999);
    for _ in 0..150 {
        let n = 48;
        let centers: Vec<Point<3>> = (0..n).map(|_| rng.point::<3>(-30.0, 30.0)).collect();
        let radii: Vec<Scalar> = (0..n).map(|_| rng.range(1.0, 4.0)).collect();

        let inst = ShellInstances {
            transforms: centers
                .iter()
                .map(|&c| Isometry::from_translation(c))
                .collect(),
            radii: radii.clone(),
        };
        let shells = Shells::<3> {
            centers: centers.clone(),
            radii: radii.clone(),
        };
        let tlas = Tlas::build(&inst);
        let filter = QueryFilter::default();

        let ray = Ray::new(rng.point::<3>(-40.0, 40.0), rng.point::<3>(-1.0, 1.0));
        if ray.dir == Point::<3>::ZERO {
            continue;
        }
        let want = naive::raycast_crossings(&shells, &ray, 500.0, &filter);
        let got = tlas.raycast_crossings(&inst, &ray, 500.0, &filter);
        assert_eq!(want.len(), got.len(), "tlas crossing count");
        for (x, y) in want.iter().zip(got.iter()) {
            assert_eq!(x.leaf, y.leaf, "tlas crossing order");
            assert_eq!(x.time_of_impact, y.time_of_impact, "tlas crossing toi");
        }
    }
}
