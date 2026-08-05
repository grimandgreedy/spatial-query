//! Rigid transforms: an orthonormal rotation followed by a translation.
//!
//! An `Isometry` maps an instance's local coordinates to world coordinates. It
//! is the transform the two-level tree carries per instance. Rigid motion
//! changes only the isometry, which is why refitting the tree is cheap; changing
//! shape is a change to the local geometry, not to the isometry.

// Matrix-vector products are naturally written as indexed loops over two axes.
#![allow(clippy::needless_range_loop)]

use super::aabb::Aabb;
use super::point::{Point, Scalar};

/// A rigid transform in `D` dimensions.
///
/// `rotation` must be orthonormal. Because it is, the inverse is the transpose,
/// distances are preserved, and a `time_of_impact` measured in local space is
/// the same in world space.
#[derive(Clone, Copy, Debug)]
pub struct Isometry<const D: usize> {
    /// Row-major rotation. Element `[i][j]` is row `i`, column `j`.
    pub rotation: [[Scalar; D]; D],
    /// Translation applied after the rotation.
    pub translation: Point<D>,
}

impl<const D: usize> Isometry<D> {
    /// The identity transform.
    pub fn identity() -> Self {
        let mut rotation = [[0.0; D]; D];
        for i in 0..D {
            rotation[i][i] = 1.0;
        }
        Isometry {
            rotation,
            translation: Point::ZERO,
        }
    }

    /// A transform from an orthonormal `rotation` and a `translation`.
    pub fn new(rotation: [[Scalar; D]; D], translation: Point<D>) -> Self {
        Isometry {
            rotation,
            translation,
        }
    }

    /// A pure translation.
    pub fn from_translation(translation: Point<D>) -> Self {
        Isometry {
            translation,
            ..Self::identity()
        }
    }

    /// Map a local point to world space: `R * p + t`.
    #[inline]
    pub fn transform_point(&self, p: Point<D>) -> Point<D> {
        self.rotate(p) + self.translation
    }

    /// Map a local direction to world space: `R * v` (no translation).
    #[inline]
    pub fn transform_vector(&self, v: Point<D>) -> Point<D> {
        self.rotate(v)
    }

    /// Map a world point to local space: `R^T * (p - t)`.
    #[inline]
    pub fn inverse_transform_point(&self, p: Point<D>) -> Point<D> {
        self.rotate_transpose(p - self.translation)
    }

    /// Map a world direction to local space: `R^T * v`.
    #[inline]
    pub fn inverse_transform_vector(&self, v: Point<D>) -> Point<D> {
        self.rotate_transpose(v)
    }

    /// The world-space AABB of a local-space box under this transform.
    ///
    /// Uses the rotated-extents form: the world half-extent along axis `i` is the
    /// sum over `j` of `|R[i][j]|` times the local half-extent on axis `j`.
    pub fn transform_aabb(&self, local: Aabb<D>) -> Aabb<D> {
        let center = self.transform_point(local.center());
        let h = local.extents() * 0.5;
        let mut world_h = [0.0; D];
        for i in 0..D {
            let mut acc = 0.0;
            for j in 0..D {
                acc += self.rotation[i][j].abs() * h[j];
            }
            world_h[i] = acc;
        }
        let world_h = Point(world_h);
        Aabb::new(center - world_h, center + world_h)
    }

    #[inline]
    fn rotate(&self, v: Point<D>) -> Point<D> {
        let mut out = [0.0; D];
        for i in 0..D {
            let mut acc = 0.0;
            for j in 0..D {
                acc += self.rotation[i][j] * v[j];
            }
            out[i] = acc;
        }
        Point(out)
    }

    #[inline]
    fn rotate_transpose(&self, v: Point<D>) -> Point<D> {
        let mut out = [0.0; D];
        for i in 0..D {
            let mut acc = 0.0;
            for j in 0..D {
                acc += self.rotation[j][i] * v[j];
            }
            out[i] = acc;
        }
        Point(out)
    }
}

/// A rotation matrix that rotates by `angle` in the plane of axes `a` and `b`,
/// leaving all other axes fixed. Composing these builds any rotation, and it is
/// the simplest way to construct one in a dimension-generic way.
pub fn plane_rotation<const D: usize>(a: usize, b: usize, angle: Scalar) -> [[Scalar; D]; D] {
    let mut r = [[0.0; D]; D];
    for i in 0..D {
        r[i][i] = 1.0;
    }
    let (sin, cos) = angle.sin_cos();
    r[a][a] = cos;
    r[a][b] = -sin;
    r[b][a] = sin;
    r[b][b] = cos;
    r
}
