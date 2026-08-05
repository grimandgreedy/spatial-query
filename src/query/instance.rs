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

    /// Whether instance `i` passes `filter`. The default accepts every instance.
    fn accepts(&self, _i: usize, _filter: &QueryFilter) -> bool {
        true
    }
}
