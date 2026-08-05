//! The provider trait for the two-level tree: a scene of rigidly-transformed
//! instances.
//!
//! Each instance has a rigid world transform and some local-space geometry. The
//! top-level tree is built over instance world AABBs and refit as instances
//! move. An instance's local geometry (its local AABB and its local ray test) is
//! untouched by rigid motion, so a moving scene refits the top level only.
//!
//! This is the model to use when objects move or rotate. For a static or simply
//! moving set of leaves a plain [`QueryGeometry`](super::geometry::QueryGeometry)
//! with a [`Bvh`](crate::accel::Bvh) is enough.

use super::filter::QueryFilter;
use super::hit::LeafHit;
use super::shapecast::ShapeCast;
use crate::maths::{Aabb, Isometry, Ray, Scalar};

/// A set of transformed instances the [`Tlas`](crate::accel::Tlas) queries.
pub trait InstancedGeometry<const D: usize> {
    /// The consumer's identity for an instance.
    type Id: Copy;

    /// The number of instances. Instance indices are `0..instance_count`.
    fn instance_count(&self) -> usize;

    /// The consumer identity of instance `i`.
    fn id(&self, i: usize) -> Self::Id;

    /// The rigid transform mapping instance `i`'s local space to world space.
    fn transform(&self, i: usize) -> Isometry<D>;

    /// The AABB of instance `i`'s geometry in its own local space. With the
    /// transform, this gives the world AABB the tree stores and refits.
    fn local_aabb(&self, i: usize) -> Aabb<D>;

    /// Test a local-space ray against instance `i`. `max_toi` is in local units,
    /// which equal world units for a rigid transform. The returned normal is in
    /// local space; the tree rotates it back to world. The tree has already
    /// established that the world ray meets the instance's world AABB.
    fn test_ray_local(&self, i: usize, local_ray: &Ray<D>, max_toi: Scalar) -> Option<LeafHit<D>>;

    /// Test the swept probe of `cast` against instance `i`. Return the nearest
    /// contact at or before `max_toi`, or `None`, with a world-space contact
    /// normal.
    ///
    /// Unlike [`test_ray_local`](Self::test_ray_local), the cast is given in
    /// world space and the normal is expected in world space: a swept box does
    /// not stay axis-aligned under rotation, so the core does not move the probe
    /// into local space for you. Use [`transform`](Self::transform) to do that
    /// inside the test if the exact narrow test needs it. The core has already
    /// established that the swept probe's box meets the instance's world AABB.
    ///
    /// The default returns `None`; providers override to answer shape casts.
    fn test_shape_cast(
        &self,
        _i: usize,
        _cast: &ShapeCast<D>,
        _max_toi: Scalar,
    ) -> Option<LeafHit<D>> {
        None
    }

    /// Whether instance `i`'s exact geometry intersects the world-space
    /// `region`.
    ///
    /// The default returns `true`: the core only calls this once the instance's
    /// world AABB is known to meet `region`, so the default reports AABB-level
    /// overlap. Providers override for an exact test, transforming `region` into
    /// local space themselves if needed.
    fn test_overlap(&self, _i: usize, _region: &Aabb<D>) -> bool {
        true
    }

    /// Whether instance `i` passes `filter`. The default accepts every instance.
    fn accepts(&self, _i: usize, _filter: &QueryFilter) -> bool {
        true
    }
}
