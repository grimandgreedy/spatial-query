//! The dimension-generic geometric primitives the whole crate is built on:
//! points, boxes, and rays over `[Scalar; D]`. No `glam`, no consumer types.

pub mod aabb;
pub mod point;
pub mod ray;

pub use aabb::Aabb;
pub use point::{Point, Scalar};
pub use ray::Ray;
