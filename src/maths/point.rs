//! Const-generic vector maths over `[Scalar; D]`.
//!
//! `Point<D>` is the single vector type used for positions and directions
//! alike. It avoids `glam`, which cannot express an arbitrary `D`, so the crate
//! has no linear-algebra dependency.

// These are componentwise operations over two parallel `[Scalar; D]` arrays.
// Indexed loops are the clearest form here; rebuilding a `[Scalar; D]` from a
// zipped iterator without an allocation or extra crate is strictly worse.
#![allow(clippy::needless_range_loop)]

use core::ops::{Add, Index, Mul, Neg, Sub};

/// The scalar type for all query math. `f64` behind the `f64` feature.
#[cfg(not(feature = "f64"))]
pub type Scalar = f32;
#[cfg(feature = "f64")]
pub type Scalar = f64;

/// A point or vector in `D`-dimensional space.
///
/// The same type serves as a position and as a direction; ray directions are
/// expected to be unit-length where a query's `time_of_impact` is to be read in
/// world units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point<const D: usize>(pub [Scalar; D]);

impl<const D: usize> Point<D> {
    /// The origin / zero vector.
    pub const ZERO: Self = Point([0.0; D]);

    /// A vector with every component set to `v`.
    #[inline]
    pub fn splat(v: Scalar) -> Self {
        Point([v; D])
    }

    /// Componentwise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        let mut out = self.0;
        for i in 0..D {
            out[i] = out[i].min(other.0[i]);
        }
        Point(out)
    }

    /// Componentwise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        let mut out = self.0;
        for i in 0..D {
            out[i] = out[i].max(other.0[i]);
        }
        Point(out)
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> Scalar {
        let mut acc = 0.0;
        for i in 0..D {
            acc += self.0[i] * other.0[i];
        }
        acc
    }

    /// Squared Euclidean length.
    #[inline]
    pub fn length_squared(self) -> Scalar {
        self.dot(self)
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> Scalar {
        self.length_squared().sqrt()
    }

    /// This vector scaled to unit length, or `ZERO` if it is (near) zero.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len > Scalar::EPSILON {
            self * (1.0 / len)
        } else {
            Self::ZERO
        }
    }
}

impl<const D: usize> Index<usize> for Point<D> {
    type Output = Scalar;
    #[inline]
    fn index(&self, i: usize) -> &Scalar {
        &self.0[i]
    }
}

impl<const D: usize> Add for Point<D> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = self.0;
        for i in 0..D {
            out[i] += rhs.0[i];
        }
        Point(out)
    }
}

impl<const D: usize> Sub for Point<D> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = self.0;
        for i in 0..D {
            out[i] -= rhs.0[i];
        }
        Point(out)
    }
}

impl<const D: usize> Mul<Scalar> for Point<D> {
    type Output = Self;
    #[inline]
    fn mul(self, s: Scalar) -> Self {
        let mut out = self.0;
        for i in 0..D {
            out[i] *= s;
        }
        Point(out)
    }
}

impl<const D: usize> Neg for Point<D> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self * -1.0
    }
}
