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
use super::shapecast::ShapeCast;
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

    /// Test the swept probe of `cast` against `leaf`'s exact geometry. Return the
    /// nearest contact at or before `max_toi`, or `None`. The returned `toi` is
    /// the sweep distance to contact, and the normal is the world-space contact
    /// normal. The core has already established that the swept probe's box meets
    /// the leaf's AABB.
    ///
    /// The default returns `None`; providers override to answer shape casts.
    fn test_shape_cast(
        &self,
        _leaf: usize,
        _cast: &ShapeCast<D>,
        _max_toi: Scalar,
    ) -> Option<LeafHit<D>> {
        None
    }

    /// Whether `leaf`'s exact geometry intersects the world-space `region`.
    ///
    /// The default returns `true`: the core only calls this once the leaf's AABB
    /// is known to meet `region`, so the default reports AABB-level overlap.
    /// Providers override for an exact test against their real geometry.
    fn test_overlap(&self, _leaf: usize, _region: &Aabb<D>) -> bool {
        true
    }

    /// Whether `leaf` passes `filter`. The default accepts every leaf; providers
    /// override to implement layer masks, exclusion, sensor rules, and so on.
    fn accepts(&self, _leaf: usize, _filter: &QueryFilter) -> bool {
        true
    }
}
