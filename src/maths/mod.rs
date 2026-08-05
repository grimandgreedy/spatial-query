//! The dimension-generic geometric primitives the whole crate is built on:
//! points, boxes, and rays over `[Scalar; D]`. No `glam`, no consumer types.

pub mod aabb;
pub mod isometry;
pub mod point;
pub mod ray;

pub use aabb::Aabb;
pub use isometry::{plane_rotation, Isometry};
pub use point::{Point, Scalar};
pub use ray::Ray;
