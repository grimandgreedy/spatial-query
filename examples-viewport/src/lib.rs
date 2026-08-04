//! Shared code for the spatial-query viewport examples.
//!
//! The examples render with viewport-lib's `ViewportApp` and query with
//! `spatial-query`. This holds the one thing they share: a `QueryGeometry`
//! provider over a set of spheres, plus `glam` <-> `Point<3>` conversions. The
//! conversions live here so that `glam` never reaches the core.

use glam::Vec3;
use spatial_query::{Aabb, LeafHit, Point, QueryGeometry, Ray};

/// Core point -> glam vector (3D instantiation, at the crate boundary).
#[inline]
pub fn to_vec3(p: Point<3>) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// glam vector -> core point.
#[inline]
pub fn to_point(v: Vec3) -> Point<3> {
    Point([v.x, v.y, v.z])
}

/// A ray for the sweep animations. The origin orbits the sphere cloud on a
/// radius-12 circle (with a slow vertical bob) and always aims at the centre, so
/// the ray passes through the cloud every frame and the direction rotates
/// visibly over time. Returns `(origin, unit_direction)`.
pub fn sweep_ray(time: f32) -> (Vec3, Vec3) {
    let theta = time * 0.7;
    let origin = Vec3::new(
        12.0 * theta.cos(),
        12.0 * theta.sin(),
        3.5 * (time * 0.5).sin(),
    );
    let dir = (Vec3::ZERO - origin).normalize();
    (origin, dir)
}

/// A scene of spheres: the geometry both rendered by viewport-lib and queried by
/// spatial-query. Centres and radii are the single source of truth for both.
pub struct SphereScene {
    pub centers: Vec<Point<3>>,
    pub radii: Vec<f32>,
}

impl SphereScene {
    /// A deterministic scatter of `n` spheres, so runs are reproducible without
    /// a `rand` dependency (SplitMix64).
    pub fn scatter(n: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut unit = || (next() >> 11) as f32 / (1u64 << 53) as f32;

        // A fairly dense cluster near the origin, so a ray aimed through the
        // centre reliably intersects several spheres.
        let mut centers = Vec::with_capacity(n);
        let mut radii = Vec::with_capacity(n);
        for _ in 0..n {
            let x = (unit() - 0.5) * 9.0;
            let y = (unit() - 0.5) * 9.0;
            let z = (unit() - 0.5) * 4.0;
            centers.push(Point([x, y, z]));
            radii.push(0.45 + unit() * 0.45);
        }
        SphereScene { centers, radii }
    }

    /// Number of spheres.
    pub fn len(&self) -> usize {
        self.centers.len()
    }

    /// Whether the scene is empty.
    pub fn is_empty(&self) -> bool {
        self.centers.is_empty()
    }
}

impl QueryGeometry<3> for SphereScene {
    type Id = usize;

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

    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
        let c = self.centers[leaf];
        let r = self.radii[leaf];
        let oc = ray.origin - c;
        let b = oc.dot(ray.dir);
        let cc = oc.length_squared() - r * r;
        let disc = b * b - cc;
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
        let normal = (ray.at(t) - c).normalize_or_zero();
        Some(LeafHit {
            toi: t,
            normal,
            sub_object: None,
        })
    }
}
