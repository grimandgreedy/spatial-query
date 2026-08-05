//! Shape-cast query descriptor: a shape swept along a direction.
//!
//! A shape cast moves a probe shape from a start position along a direction and
//! reports where it first touches the scene, in the same nearest / all-hits
//! shapes as a ray cast. The core owns the broad-phase reject and the provider
//! owns the exact swept test, through
//! [`QueryGeometry::test_shape_cast`](crate::QueryGeometry::test_shape_cast) and
//! [`InstancedGeometry::test_shape_cast`](crate::InstancedGeometry::test_shape_cast).

use crate::maths::{Aabb, Point, Ray, Scalar};

/// A probe shape swept along a direction.
///
/// `origin` and `dir` describe the path of the probe's reference point; `aabb`
/// is the probe's bounding box as offsets from that reference point (so a
/// radius-`r` ball centred on the reference point has `aabb` from `-r` to `+r`
/// on every axis). The core uses `aabb` only for the conservative broad-phase
/// reject; the provider's exact test owns the real shape.
///
/// For a hit's `time_of_impact` to read in world units, `dir` should be unit
/// length; [`new`](Self::new) normalises it.
#[derive(Clone, Copy, Debug)]
pub struct ShapeCast<const D: usize> {
    /// Start position of the probe's reference point.
    pub origin: Point<D>,
    /// Sweep direction of the probe's reference point.
    pub dir: Point<D>,
    /// The probe's bounding box, as offsets from the reference point.
    pub aabb: Aabb<D>,
}

impl<const D: usize> ShapeCast<D> {
    /// A shape cast from `origin` along `dir`, normalising `dir` to unit length.
    #[inline]
    pub fn new(origin: Point<D>, dir: Point<D>, aabb: Aabb<D>) -> Self {
        ShapeCast {
            origin,
            dir: dir.normalize_or_zero(),
            aabb,
        }
    }

    /// A shape cast that trusts `dir` to already be unit length.
    #[inline]
    pub fn new_unnormalized(origin: Point<D>, dir: Point<D>, aabb: Aabb<D>) -> Self {
        ShapeCast { origin, dir, aabb }
    }

    /// The reference-point position at sweep parameter `t`.
    #[inline]
    pub fn at(&self, t: Scalar) -> Point<D> {
        self.origin + self.dir * t
    }

    /// The conservative broad-phase test against a target box: the sweep
    /// parameters `(t_enter, t_exit)` over which the probe's box could touch
    /// `target` within `max_toi`, or `None`.
    ///
    /// A `None` guarantees the probe misses `target`; a `Some` does not
    /// guarantee an exact hit (the exact test decides). The probe touches
    /// `target` exactly when its reference point enters `target` grown by the
    /// probe box, so this is a ray-slab test against that grown box (a Minkowski
    /// difference).
    #[inline]
    pub(crate) fn broadphase(&self, target: Aabb<D>, max_toi: Scalar) -> Option<(Scalar, Scalar)> {
        let grown = Aabb::new(target.min - self.aabb.max, target.max - self.aabb.min);
        grown.ray_intersect(&Ray::new_unnormalized(self.origin, self.dir), max_toi)
    }
}
