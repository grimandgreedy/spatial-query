//! Ray type for spatial queries.

use super::point::{Point, Scalar};

/// A ray with an origin and a direction in `D`-dimensional space.
///
/// For `time_of_impact` values to read in world-space units, `dir` should be
/// unit length; [`new`](Self::new) normalises for you.
#[derive(Clone, Copy, Debug)]
pub struct Ray<const D: usize> {
    pub origin: Point<D>,
    pub dir: Point<D>,
}

impl<const D: usize> Ray<D> {
    /// A ray from `origin` along `dir`, normalising `dir` to unit length.
    #[inline]
    pub fn new(origin: Point<D>, dir: Point<D>) -> Self {
        Ray {
            origin,
            dir: dir.normalize_or_zero(),
        }
    }

    /// A ray that trusts `dir` to already be unit length (no normalisation).
    #[inline]
    pub fn new_unnormalized(origin: Point<D>, dir: Point<D>) -> Self {
        Ray { origin, dir }
    }

    /// The point at parameter `t` along the ray: `origin + dir * t`.
    #[inline]
    pub fn at(&self, t: Scalar) -> Point<D> {
        self.origin + self.dir * t
    }
}
