//! Query result types.

use crate::maths::{Point, Scalar};

/// What a provider returns from [`QueryGeometry::test_ray`](crate::QueryGeometry::test_ray)
/// for a single leaf.
///
/// The core fills in the world hit point (`ray.at(toi)`) and the leaf id; the
/// provider only reports the impact parameter, the surface normal, and an opaque
/// sub-object payload.
#[derive(Clone, Copy, Debug)]
pub struct LeafHit<const D: usize> {
    /// Distance along the ray direction to the hit (world units for a unit ray).
    pub toi: Scalar,
    /// Outward surface normal at the hit.
    pub normal: Point<D>,
    /// Opaque, provider-defined sub-object identity (e.g. a face or vertex
    /// index), stored and returned by the core without interpretation.
    pub sub_object: Option<u64>,
}

/// A single ray-query hit against the world.
#[derive(Clone, Copy, Debug)]
pub struct Hit<Id, const D: usize> {
    /// The provider's identity for the object that was hit.
    pub id: Id,
    /// The leaf index within the provider (`0..leaf_count`). Also the
    /// deterministic tie-breaker when two hits share a `time_of_impact`.
    pub leaf: usize,
    /// Distance along the ray direction to the hit.
    pub time_of_impact: Scalar,
    /// World-space hit position.
    pub point: Point<D>,
    /// Outward surface normal at the hit.
    pub normal: Point<D>,
    /// Opaque, provider-defined sub-object identity.
    pub sub_object: Option<u64>,
}

/// Total order used for nearest-selection and all-hits sorting: by
/// `time_of_impact`, then by leaf index. Deterministic regardless of traversal
/// order, which lets the accelerated path match the naive baseline exactly.
#[inline]
pub(crate) fn hit_ordering<Id, const D: usize>(
    a: &Hit<Id, D>,
    b: &Hit<Id, D>,
) -> core::cmp::Ordering {
    a.time_of_impact
        .total_cmp(&b.time_of_impact)
        .then(a.leaf.cmp(&b.leaf))
}
