//! The one thing the core knows about a consumer's data.
//!
//! A consumer implements [`QueryGeometry`] over its own objects: render-scene
//! items, physics bodies, or any set of things with a bounding box and a ray
//! test. The core owns the acceleration structure and traversal; the provider
//! owns what a leaf is.
//!
//! Leaves are a dense range `0..leaf_count`, tested against a world-space ray.
//! Per-leaf local transforms and borrowed geometry, which a two-level tree over
//! deforming objects needs, are not part of this trait yet.

use super::filter::QueryFilter;
use super::hit::LeafHit;
use crate::maths::{Aabb, Ray, Scalar};

/// A consumer's spatial data, described in the terms the core queries. Generic
/// over the dimension `D`.
pub trait QueryGeometry<const D: usize> {
    /// The consumer's identity for a leaf (a `pick_id`, a `BodyId`, ...).
    type Id: Copy;

    /// The number of leaves. Leaf indices are `0..leaf_count`.
    fn leaf_count(&self) -> usize;

    /// The world-space bounding box of `leaf`, read during tree build/refit.
    fn world_aabb(&self, leaf: usize) -> Aabb<D>;

    /// The consumer identity of `leaf`.
    fn id(&self, leaf: usize) -> Self::Id;

    /// Test `ray` against `leaf`'s exact geometry. Return the nearest hit at or
    /// before `max_toi`, or `None`. The core has already established that the
    /// ray meets the leaf's AABB.
    fn test_ray(&self, leaf: usize, ray: &Ray<D>, max_toi: Scalar) -> Option<LeafHit<D>>;

    /// Whether `leaf` passes `filter`. The default accepts every leaf; providers
    /// override to implement layer masks, exclusion, sensor rules, and so on.
    fn accepts(&self, _leaf: usize, _filter: &QueryFilter) -> bool {
        true
    }
}
