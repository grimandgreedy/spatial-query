//! Phase 5 guarantees: a query filter is honoured through the provider's
//! `accepts` (a provider that rejects half its leaves yields exactly the
//! complement), and a typed sub-object survives from the narrow test to the
//! returned `Hit` unchanged, across the single-level and two-level trees.

use spatial_query::{
    naive, Aabb, Bvh, InstancedGeometry, Isometry, LeafHit, Point, QueryFilter, QueryGeometry, Ray,
    Scalar, ShapeCast, Tlas,
};

/// A provider-defined sub-object identity, deliberately not a bare integer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Face(u32);

/// The sub-object a leaf reports, distinct from the leaf index so a test cannot
/// pass by echoing the index.
fn face_of(leaf: usize) -> Face {
    Face(leaf as u32 * 10 + 1)
}

const PROBE_RADIUS: Scalar = 0.5;

/// Ray against a ball, returning the near entry `(toi, normal)`, or `None`.
fn ray_ball(
    ray: &Ray<3>,
    center: Point<3>,
    radius: Scalar,
    max_toi: Scalar,
) -> Option<(Scalar, Point<3>)> {
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
    Some((t, (ray.at(t) - center).normalize_or_zero()))
}

// ---------------------------------------------------------------------------
// Single-level provider: labelled balls in a line, with a layer-based filter.
// ---------------------------------------------------------------------------

struct Balls {
    centers: Vec<Point<3>>,
    radii: Vec<Scalar>,
}

impl Balls {
    /// `n` unit-ish balls strung along the x axis so a ray from the origin along
    /// +x pierces all of them in index order.
    fn line(n: usize) -> Self {
        let centers = (0..n)
            .map(|i| Point([2.0 * (i as f32 + 1.0), 0.0, 0.0]))
            .collect();
        let radii = (0..n).map(|_| 0.6).collect();
        Balls { centers, radii }
    }
}

impl QueryGeometry<3> for Balls {
    type Id = usize;
    type SubObject = Face;

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
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3, Face>> {
        let (t, n) = ray_ball(ray, self.centers[leaf], self.radii[leaf], max_toi)?;
        Some(LeafHit::new(t, n).with_sub_object(face_of(leaf)))
    }
    fn test_shape_cast(
        &self,
        leaf: usize,
        cast: &ShapeCast<3>,
        max_toi: Scalar,
    ) -> Option<LeafHit<3, Face>> {
        let sum = self.radii[leaf] + cast.aabb.max[0];
        let ray = Ray::new_unnormalized(cast.origin, cast.dir);
        let (t, _) = ray_ball(&ray, self.centers[leaf], sum, max_toi)?;
        let n = (cast.at(t) - self.centers[leaf]).normalize_or_zero();
        Some(LeafHit::new(t, n).with_sub_object(face_of(leaf)))
    }
    // Even leaves sit on layer bit 0, odd leaves on layer bit 1. A filter's
    // layer_mask then selects one parity or both.
    fn accepts(&self, leaf: usize, filter: &QueryFilter) -> bool {
        filter.layer_mask & (1 << (leaf % 2)) != 0
    }
}

fn axis_ray() -> Ray<3> {
    Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]))
}

fn axis_cast() -> ShapeCast<3> {
    ShapeCast::new(
        Point([0.0, 0.0, 0.0]),
        Point([1.0, 0.0, 0.0]),
        Aabb::new(Point::splat(-PROBE_RADIUS), Point::splat(PROBE_RADIUS)),
    )
}

// ---------------------------------------------------------------------------
// Filter round-trip.
// ---------------------------------------------------------------------------

#[test]
fn filter_selects_exactly_the_accepted_complement() {
    let balls = Balls::line(12);
    let bvh = Bvh::build(&balls);
    let ray = axis_ray();

    let all = QueryFilter::default();
    // Mutate after default because QueryFilter is #[non_exhaustive].
    let mut even = QueryFilter::default();
    even.layer_mask = 0b01;
    let mut odd = QueryFilter::default();
    odd.layer_mask = 0b10;

    let leaves = |f: &QueryFilter| -> Vec<usize> {
        bvh.raycast_all(&balls, &ray, 100.0, f)
            .iter()
            .map(|h| h.leaf)
            .collect()
    };

    let full = leaves(&all);
    assert_eq!(full, (0..12).collect::<Vec<_>>(), "all leaves pierced");

    let got_even = leaves(&even);
    let got_odd = leaves(&odd);
    assert_eq!(got_even, vec![0, 2, 4, 6, 8, 10], "even layer");
    assert_eq!(got_odd, vec![1, 3, 5, 7, 9, 11], "odd layer");

    // The two halves partition the whole: disjoint and covering.
    let mut union = got_even.clone();
    union.extend(got_odd.iter().copied());
    union.sort_unstable();
    assert_eq!(union, full, "accepted halves are the exact complement");

    // The accelerated result matches the naive scan under the same filter.
    let naive_even: Vec<usize> = naive::raycast_all(&balls, &ray, 100.0, &even)
        .iter()
        .map(|h| h.leaf)
        .collect();
    assert_eq!(got_even, naive_even, "bvh vs naive under filter");
}

#[test]
fn filter_applies_to_overlap_too() {
    let balls = Balls::line(12);
    let bvh = Bvh::build(&balls);
    // A slab spanning the whole line on x, thin on y/z, overlaps every ball.
    let region = Aabb::new(Point([0.0, -1.0, -1.0]), Point([30.0, 1.0, 1.0]));

    let mut even = QueryFilter::default();
    even.layer_mask = 0b01;
    let got: Vec<usize> = bvh
        .overlap(&balls, &region, &even)
        .iter()
        .map(|o| o.leaf)
        .collect();
    assert_eq!(got, vec![0, 2, 4, 6, 8, 10], "overlap honours accepts");
}

// ---------------------------------------------------------------------------
// Sub-object survival, single level.
// ---------------------------------------------------------------------------

#[test]
fn sub_object_survives_bvh_ray_and_shapecast() {
    let balls = Balls::line(8);
    let bvh = Bvh::build(&balls);
    let filter = QueryFilter::default();

    let near = bvh
        .raycast_nearest(&balls, &axis_ray(), 100.0, &filter)
        .unwrap();
    assert_eq!(near.sub_object, Some(face_of(near.leaf)), "nearest ray");

    for h in bvh.raycast_all(&balls, &axis_ray(), 100.0, &filter) {
        assert_eq!(h.sub_object, Some(face_of(h.leaf)), "all-hits ray");
    }

    let scnear = bvh
        .shapecast_nearest(&balls, &axis_cast(), 100.0, &filter)
        .unwrap();
    assert_eq!(
        scnear.sub_object,
        Some(face_of(scnear.leaf)),
        "nearest cast"
    );

    for h in bvh.shapecast_all(&balls, &axis_cast(), 100.0, &filter) {
        assert_eq!(h.sub_object, Some(face_of(h.leaf)), "all-hits cast");
    }

    // The naive scan carries the same sub-object.
    let n = naive::raycast_nearest(&balls, &axis_ray(), 100.0, &filter).unwrap();
    assert_eq!(n.sub_object, Some(face_of(n.leaf)), "naive ray");
}

// ---------------------------------------------------------------------------
// Sub-object survival, two level (the tree rotates the normal but must leave
// the sub-object untouched).
// ---------------------------------------------------------------------------

struct LabeledInstances {
    transforms: Vec<Isometry<3>>,
    radii: Vec<Scalar>,
}

impl InstancedGeometry<3> for LabeledInstances {
    type Id = usize;
    type SubObject = Face;

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
    fn test_ray_local(&self, i: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3, Face>> {
        let (t, n) = ray_ball(ray, Point::<3>::ZERO, self.radii[i], max_toi)?;
        Some(LeafHit::new(t, n).with_sub_object(face_of(i)))
    }
}

#[test]
fn sub_object_survives_tlas_ray() {
    let inst = LabeledInstances {
        transforms: (0..6)
            .map(|i| Isometry::from_translation(Point([2.0 * (i as f32 + 1.0), 0.0, 0.0])))
            .collect(),
        radii: (0..6).map(|_| 0.6).collect(),
    };
    let tlas = Tlas::build(&inst);
    let filter = QueryFilter::default();

    let near = tlas
        .raycast_nearest(&inst, &axis_ray(), 100.0, &filter)
        .unwrap();
    assert_eq!(near.sub_object, Some(face_of(near.leaf)), "tlas nearest");

    for h in tlas.raycast_all(&inst, &axis_ray(), 100.0, &filter) {
        assert_eq!(h.sub_object, Some(face_of(h.leaf)), "tlas all-hits");
    }
}
