//! Overlap query result.
//!
//! An overlap query returns every object whose exact geometry intersects a query
//! region (an [`Aabb`](crate::maths::Aabb)). There is no distance to sort by, so
//! results carry only identity. The core returns them in ascending leaf order,
//! which keeps the accelerated result identical to the naive scan.

/// A single overlap-query result: an object that intersects the query region.
#[derive(Clone, Copy, Debug)]
pub struct Overlap<Id> {
    /// The provider's identity for the object.
    pub id: Id,
    /// The leaf index within the provider (`0..leaf_count`). Also the order the
    /// results are returned in.
    pub leaf: usize,
}
