//! Tests for the two-level tree: transform maths, end-to-end hits, tree-vs-naive
//! parity, and refit-vs-rebuild on moving and deforming scenes.

#![allow(clippy::needless_range_loop)]

use spatial_query::{
    plane_rotation, Aabb, Bvh, InstancedGeometry, Isometry, LeafHit, Point, QueryFilter,
    QueryGeometry, Ray, Scalar, Tlas,
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
// An oriented-box instance provider.
// ---------------------------------------------------------------------------

struct OrientedBoxes<const D: usize> {
    transforms: Vec<Isometry<D>>,
    halves: Vec<Point<D>>,
}

impl<const D: usize> InstancedGeometry<D> for OrientedBoxes<D> {
    type Id = usize;

    fn instance_count(&self) -> usize {
        self.transforms.len()
    }
    fn id(&self, i: usize) -> usize {
        i
    }
    fn transform(&self, i: usize) -> Isometry<D> {
        self.transforms[i]
    }
    fn local_aabb(&self, i: usize) -> Aabb<D> {
        Aabb::new(-self.halves[i], self.halves[i])
    }
    fn test_ray_local(&self, i: usize, ray: &Ray<D>, max_toi: Scalar) -> Option<LeafHit<D>> {
        ray_local_box(ray, self.halves[i], max_toi)
    }
}

/// Ray against a local axis-aligned box `[-half, half]`. Returns the entry
/// point-of-impact and the outward face normal, or `None`.
fn ray_local_box<const D: usize>(
    ray: &Ray<D>,
    half: Point<D>,
    max_toi: Scalar,
) -> Option<LeafHit<D>> {
    let mut t_min: Scalar = 0.0;
    let mut t_max: Scalar = max_toi;
    let mut axis: Option<usize> = None;
    for i in 0..D {
        let o = ray.origin[i];
        let d = ray.dir[i];
        if d.abs() < 1.0e-9 {
            if o < -half[i] || o > half[i] {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut t1 = (-half[i] - o) * inv;
            let mut t2 = (half[i] - o) * inv;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            if t1 > t_min {
                t_min = t1;
                axis = Some(i);
            }
            if t2 < t_max {
                t_max = t2;
            }
            if t_min > t_max {
                return None;
            }
        }
    }
    let axis = match axis {
        Some(a) if t_min > 1.0e-6 && t_min <= max_toi => a,
        _ => return None,
    };
    // The ray enters the -face when travelling in +axis, and vice versa.
    let sign = if ray.dir[axis] > 0.0 { -1.0 } else { 1.0 };
    let mut n = [0.0; D];
    n[axis] = sign;
    Some(LeafHit {
        toi: t_min,
        normal: Point(n),
        sub_object: None,
    })
}

fn random_boxes<const D: usize>(rng: &mut Rng, n: usize) -> OrientedBoxes<D> {
    let mut transforms = Vec::with_capacity(n);
    let mut halves = Vec::with_capacity(n);
    for _ in 0..n {
        let t = rng.point::<D>(-40.0, 40.0);
        // A rotation in a random coordinate plane by a random angle.
        let a = (rng.next_u64() as usize) % D;
        let mut b = (rng.next_u64() as usize) % D;
        if b == a {
            b = (b + 1) % D;
        }
        let rot = plane_rotation::<D>(a, b, rng.range(0.0, std::f32::consts::TAU));
        transforms.push(Isometry::new(rot, t));
        halves.push(rng.point::<D>(0.5, 2.5).max(Point::splat(0.5)));
    }
    OrientedBoxes { transforms, halves }
}

// A ground-truth scan over all instances, matching the tree's tie-break.
fn naive_nearest<const D: usize, S: InstancedGeometry<D>>(
    scene: &S,
    ray: &Ray<D>,
    max_toi: Scalar,
) -> Option<(usize, Scalar)> {
    let mut best: Option<(Scalar, usize)> = None;
    for i in 0..scene.instance_count() {
        let iso = scene.transform(i);
        let local = Ray::new_unnormalized(
            iso.inverse_transform_point(ray.origin),
            iso.inverse_transform_vector(ray.dir),
        );
        if let Some(lh) = scene.test_ray_local(i, &local, max_toi) {
            let key = (lh.toi, i);
            if best.is_none_or(|b| key < b) {
                best = Some(key);
            }
        }
    }
    best.map(|(t, i)| (i, t))
}

fn naive_all<const D: usize, S: InstancedGeometry<D>>(
    scene: &S,
    ray: &Ray<D>,
    max_toi: Scalar,
) -> Vec<(usize, Scalar)> {
    let mut hits = Vec::new();
    for i in 0..scene.instance_count() {
        let iso = scene.transform(i);
        let local = Ray::new_unnormalized(
            iso.inverse_transform_point(ray.origin),
            iso.inverse_transform_vector(ray.dir),
        );
        if let Some(lh) = scene.test_ray_local(i, &local, max_toi) {
            hits.push((i, lh.toi));
        }
    }
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    hits
}

// ---------------------------------------------------------------------------
// Transform maths, checked independently of the tree.
// ---------------------------------------------------------------------------

#[test]
fn isometry_rotation_and_inverse() {
    let rot = plane_rotation::<3>(0, 1, std::f32::consts::FRAC_PI_2);
    let iso = Isometry::new(rot, Point([0.0, 0.0, 0.0]));
    // A quarter turn in the x-y plane sends +x to +y.
    let r = iso.transform_vector(Point([1.0, 0.0, 0.0]));
    assert!((r[0]).abs() < 1e-6 && (r[1] - 1.0).abs() < 1e-6 && r[2].abs() < 1e-6);
    // The inverse brings it back.
    let back = iso.inverse_transform_vector(r);
    assert!((back[0] - 1.0).abs() < 1e-6 && back[1].abs() < 1e-6);
}

#[test]
fn transform_aabb_of_rotated_box() {
    // A unit box turned 45 degrees in x-y grows to half-extent cos45 + sin45 on
    // x and y, and keeps 0.5 on z; translation shifts the centre.
    let rot = plane_rotation::<3>(0, 1, std::f32::consts::FRAC_PI_4);
    let iso = Isometry::new(rot, Point([10.0, 0.0, 0.0]));
    let world = iso.transform_aabb(Aabb::new(Point([-0.5; 3]), Point([0.5; 3])));
    let ext = world.extents();
    let diag = 2.0 * std::f32::consts::FRAC_1_SQRT_2; // 2 * 0.5 * (cos45 + sin45)
    assert!((ext[0] - diag).abs() < 1e-5, "x extent {}", ext[0]);
    assert!((ext[1] - diag).abs() < 1e-5, "y extent {}", ext[1]);
    assert!((ext[2] - 1.0).abs() < 1e-5, "z extent {}", ext[2]);
    assert!((world.center()[0] - 10.0).abs() < 1e-5);
}

#[test]
fn end_to_end_hit_point_and_normal() {
    // One unit box at x = 5, no rotation. A ray from the origin along +x enters
    // the near face at x = 4 with an outward normal of -x.
    let scene = OrientedBoxes::<3> {
        transforms: vec![Isometry::from_translation(Point([5.0, 0.0, 0.0]))],
        halves: vec![Point([1.0, 1.0, 1.0])],
    };
    let tlas = Tlas::build(&scene);
    let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
    let hit = tlas
        .raycast_nearest(&scene, &ray, 100.0, &QueryFilter::default())
        .unwrap();
    assert_eq!(hit.leaf, 0);
    assert!((hit.time_of_impact - 4.0).abs() < 1e-4);
    assert!((hit.point[0] - 4.0).abs() < 1e-4);
    assert!((hit.normal[0] - -1.0).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// Tree vs naive, and refit vs rebuild.
// ---------------------------------------------------------------------------

fn assert_tree_matches_naive<const D: usize>(
    scene: &OrientedBoxes<D>,
    ray: &Ray<D>,
    max_toi: Scalar,
) {
    let tlas = Tlas::build(scene);
    let filter = QueryFilter::default();

    match (
        naive_nearest(scene, ray, max_toi),
        tlas.raycast_nearest(scene, ray, max_toi, &filter),
    ) {
        (None, None) => {}
        (Some((li, lt)), Some(h)) => {
            assert_eq!(li, h.leaf, "nearest instance");
            assert_eq!(lt, h.time_of_impact, "nearest toi");
        }
        (a, b) => panic!(
            "nearest presence mismatch: {} vs {}",
            a.is_some(),
            b.is_some()
        ),
    }

    let na = naive_all(scene, ray, max_toi);
    let ta = tlas.raycast_all(scene, ray, max_toi, &filter);
    assert_eq!(na.len(), ta.len(), "all-hits count");
    for (x, y) in na.iter().zip(ta.iter()) {
        assert_eq!(x.0, y.leaf, "all-hits order");
        assert_eq!(x.1, y.time_of_impact, "all-hits toi");
    }
}

#[test]
fn tlas_matches_naive_3d() {
    let mut rng = Rng::new(0x7ABC_1234);
    for _ in 0..200 {
        let scene = random_boxes::<3>(&mut rng, 64);
        let ray = Ray::new(rng.point::<3>(-60.0, 60.0), rng.point::<3>(-1.0, 1.0));
        if ray.dir == Point::<3>::ZERO {
            continue;
        }
        assert_tree_matches_naive(&scene, &ray, 500.0);
    }
}

#[test]
fn tlas_matches_naive_2d() {
    let mut rng = Rng::new(0x2020_2020);
    for _ in 0..200 {
        let scene = random_boxes::<2>(&mut rng, 48);
        let ray = Ray::new(rng.point::<2>(-60.0, 60.0), rng.point::<2>(-1.0, 1.0));
        if ray.dir == Point::<2>::ZERO {
            continue;
        }
        assert_tree_matches_naive(&scene, &ray, 500.0);
    }
}

#[test]
fn refit_matches_rebuild_when_instances_move() {
    let mut rng = Rng::new(0xF00D_5EED);
    let mut scene = random_boxes::<3>(&mut rng, 200);
    let mut tlas = Tlas::build(&scene);
    let filter = QueryFilter::default();

    for frame in 0..10 {
        // Move every instance (rigid translation) and also deform a few (change
        // local half-sizes), then refit rather than rebuild.
        for i in 0..scene.transforms.len() {
            let d = rng.point::<3>(-1.5, 1.5);
            scene.transforms[i].translation = scene.transforms[i].translation + d;
            if i % 7 == 0 {
                scene.halves[i] = scene.halves[i] * 1.05;
            }
        }
        tlas.refit(&scene);
        let rebuilt = Tlas::build(&scene);

        // A batch of rays: the refit tree, the rebuilt tree, and the naive scan
        // must all agree exactly.
        for _ in 0..40 {
            let ray = Ray::new(rng.point::<3>(-60.0, 60.0), rng.point::<3>(-1.0, 1.0));
            if ray.dir == Point::<3>::ZERO {
                continue;
            }
            let a = tlas.raycast_nearest(&scene, &ray, 500.0, &filter);
            let b = rebuilt.raycast_nearest(&scene, &ray, 500.0, &filter);
            let n = naive_nearest(&scene, &ray, 500.0);
            let key = |h: &spatial_query::Hit<usize, 3>| (h.leaf, h.time_of_impact);
            assert_eq!(
                a.as_ref().map(key),
                b.as_ref().map(key),
                "refit vs rebuild, frame {frame}"
            );
            assert_eq!(
                a.as_ref().map(|h| (h.leaf, h.time_of_impact)),
                n,
                "refit vs naive, frame {frame}"
            );
        }
    }
}

#[test]
fn needs_rebuild_triggers_after_spreading_instances() {
    let mut rng = Rng::new(0xCAFE_1111);
    // Start tightly clustered so the initial tree is high quality.
    let n = 128;
    let scene0 = OrientedBoxes::<3> {
        transforms: (0..n)
            .map(|_| Isometry::from_translation(rng.point::<3>(-2.0, 2.0)))
            .collect(),
        halves: (0..n).map(|_| Point([0.5, 0.5, 0.5])).collect(),
    };
    let mut tlas = Tlas::build(&scene0);
    assert!(
        !tlas.needs_rebuild(2.0),
        "fresh tree should not want a rebuild"
    );

    // Fling the instances far apart without rebuilding: the refit bounds blow up
    // and the quality metric should cross the threshold.
    let mut scene = scene0;
    for i in 0..n {
        scene.transforms[i].translation = rng.point::<3>(-500.0, 500.0);
    }
    tlas.refit(&scene);
    assert!(
        tlas.needs_rebuild(2.0),
        "spread-out refit tree should want a rebuild"
    );
}

// ---------------------------------------------------------------------------
// Bvh refit (single level).
// ---------------------------------------------------------------------------

struct Balls {
    centers: Vec<Point<3>>,
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
        Aabb::new(c - Point::splat(1.0), c + Point::splat(1.0))
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        let oc = ray.origin - self.centers[leaf];
        let b = oc.dot(ray.dir);
        let c = oc.length_squared() - 1.0;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        let t = -b - disc.sqrt();
        if t < 0.0 || t > max_toi {
            return None;
        }
        Some(LeafHit {
            toi: t,
            normal: (ray.at(t) - self.centers[leaf]).normalize_or_zero(),
            sub_object: None,
        })
    }
}

#[test]
fn bvh_refit_matches_rebuild() {
    let mut rng = Rng::new(0x1357_9BDF);
    let mut balls = Balls {
        centers: (0..150).map(|_| rng.point::<3>(-30.0, 30.0)).collect(),
    };
    let mut bvh = Bvh::build(&balls);
    let filter = QueryFilter::default();

    for _ in 0..8 {
        for c in balls.centers.iter_mut() {
            *c = *c + rng.point::<3>(-2.0, 2.0);
        }
        bvh.refit(&balls);
        let rebuilt = Bvh::build(&balls);
        for _ in 0..40 {
            let ray = Ray::new(rng.point::<3>(-40.0, 40.0), rng.point::<3>(-1.0, 1.0));
            if ray.dir == Point::<3>::ZERO {
                continue;
            }
            let a = bvh.raycast_nearest(&balls, &ray, 300.0, &filter);
            let b = rebuilt.raycast_nearest(&balls, &ray, 300.0, &filter);
            assert_eq!(
                a.map(|h| (h.leaf, h.time_of_impact)),
                b.map(|h| (h.leaf, h.time_of_impact)),
            );
        }
    }
}
