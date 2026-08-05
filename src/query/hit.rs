//! Query result types.

use crate::maths::{Point, Scalar};

/// What a provider returns from [`QueryGeometry::test_ray`](crate::QueryGeometry::test_ray)
/// for a single leaf.
///
/// The core fills in the world hit point (`ray.at(toi)`) and the leaf id; the
/// provider only reports the impact parameter, the surface normal, and an
/// optional sub-object of its own type `S`.
///
/// `S` is the provider's [`SubObject`](crate::QueryGeometry::SubObject) type
/// (a face, vertex, cell, particle index, ...), stored and returned by the core
/// without interpretation. It defaults to `()` for providers that report only
/// the object, so `LeafHit<D>` means "no sub-object".
///
/// Build one with [`LeafHit::new`], then optionally [`with_sub_object`](Self::with_sub_object).
/// The struct is `#[non_exhaustive]` so later query kinds can add fields without
/// breaking providers.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct LeafHit<const D: usize, S = ()> {
    /// Distance along the ray direction to the hit (world units for a unit ray).
    pub toi: Scalar,
    /// Outward surface normal at the hit.
    pub normal: Point<D>,
    /// The provider-defined sub-object that was hit, if any. Stored and returned
    /// by the core without interpretation.
    pub sub_object: Option<S>,
}

impl<const D: usize, S> LeafHit<D, S> {
    /// A hit at `toi` with surface `normal` and no sub-object.
    #[inline]
    pub fn new(toi: Scalar, normal: Point<D>) -> Self {
        LeafHit {
            toi,
            normal,
            sub_object: None,
        }
    }

    /// Attach the provider-defined sub-object that was hit.
    #[inline]
    pub fn with_sub_object(mut self, sub_object: S) -> Self {
        self.sub_object = Some(sub_object);
        self
    }
}

/// A single ray-query hit against the world.
///
/// `S` is the provider's [`SubObject`](crate::QueryGeometry::SubObject) type,
/// carried through from the [`LeafHit`]. It defaults to `()`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Hit<Id, const D: usize, S = ()> {
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
    /// The provider-defined sub-object that was hit, if any.
    pub sub_object: Option<S>,
}

/// Total order used for nearest-selection and all-hits sorting: by
/// `time_of_impact`, then by leaf index. Deterministic regardless of traversal
/// order, which lets the accelerated path match the naive baseline exactly.
#[inline]
pub(crate) fn hit_ordering<Id, const D: usize, S>(
    a: &Hit<Id, D, S>,
    b: &Hit<Id, D, S>,
) -> core::cmp::Ordering {
    a.time_of_impact
        .total_cmp(&b.time_of_impact)
        .then(a.leaf.cmp(&b.leaf))
}
