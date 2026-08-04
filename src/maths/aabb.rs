//! Axis-aligned bounding box in `D` dimensions.

// Per-axis loops over parallel `[Scalar; D]` arrays read clearest as indexed
// loops; see the note in `point.rs`.
#![allow(clippy::needless_range_loop)]

use super::point::{Point, Scalar};
use super::ray::Ray;

/// An axis-aligned bounding box `[min, max]` in `D`-dimensional space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb<const D: usize> {
    pub min: Point<D>,
    pub max: Point<D>,
}

impl<const D: usize> Aabb<D> {
    /// An empty box (inverted bounds) that absorbs points via [`union_point`](Self::union_point).
    pub const EMPTY: Self = Aabb {
        min: Point([Scalar::INFINITY; D]),
        max: Point([Scalar::NEG_INFINITY; D]),
    };

    /// A box from explicit corners. `min` must be componentwise <= `max`.
    #[inline]
    pub fn new(min: Point<D>, max: Point<D>) -> Self {
        Aabb { min, max }
    }

    /// A zero-size box at a single point.
    #[inline]
    pub fn from_point(p: Point<D>) -> Self {
        Aabb { min: p, max: p }
    }

    /// The tightest box enclosing both `self` and `other`.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Grow to include `p`.
    #[inline]
    pub fn union_point(self, p: Point<D>) -> Self {
        Aabb {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    /// The box centre.
    #[inline]
    pub fn center(self) -> Point<D> {
        (self.min + self.max) * 0.5
    }

    /// Per-axis extents (`max - min`).
    #[inline]
    pub fn extents(self) -> Point<D> {
        self.max - self.min
    }

    /// The axis (0..D) along which the box is longest.
    #[inline]
    pub fn longest_axis(self) -> usize {
        let e = self.extents();
        let mut best = 0;
        for i in 1..D {
            if e[i] > e[best] {
                best = i;
            }
        }
        best
    }

    /// A dimension-generic SAH surface measure: the sum, over each axis `i`, of
    /// the product of the extents of all other axes. In 3D this is half the true
    /// surface area (`xy + yz + zx`); in 2D it is half the perimeter. Only its
    /// proportionality matters for split heuristics.
    #[inline]
    pub fn half_surface(self) -> Scalar {
        let e = self.extents();
        let mut sum = 0.0;
        for i in 0..D {
            let mut prod = 1.0;
            for j in 0..D {
                if j != i {
                    prod *= e[j];
                }
            }
            sum += prod;
        }
        sum
    }

    /// Slab test against a ray. Returns the entry/exit parameters
    /// `(t_min, t_max)` clamped so `t_min >= 0`, or `None` if the ray misses the
    /// box within `[0, max_toi]`.
    ///
    /// `ray.dir` need not be unit length, but the returned parameters are then in
    /// units of `dir` length; callers that want world-space `t` pass a unit
    /// direction.
    #[inline]
    pub fn ray_intersect(self, ray: &Ray<D>, max_toi: Scalar) -> Option<(Scalar, Scalar)> {
        let mut t_min: Scalar = 0.0;
        let mut t_max: Scalar = max_toi;
        for i in 0..D {
            let o = ray.origin[i];
            let d = ray.dir[i];
            if d.abs() < Scalar::EPSILON {
                // Ray parallel to this slab: miss unless the origin is inside it.
                if o < self.min[i] || o > self.max[i] {
                    return None;
                }
            } else {
                let inv = 1.0 / d;
                let mut t1 = (self.min[i] - o) * inv;
                let mut t2 = (self.max[i] - o) * inv;
                if t1 > t2 {
                    core::mem::swap(&mut t1, &mut t2);
                }
                if t1 > t_min {
                    t_min = t1;
                }
                if t2 < t_max {
                    t_max = t2;
                }
                if t_min > t_max {
                    return None;
                }
            }
        }
        Some((t_min, t_max))
    }
}
