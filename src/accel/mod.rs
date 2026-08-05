//! Acceleration structures and traversal.
//!
//! [`Bvh`] is a single-level tree over a set of leaves. [`Tlas`] is a two-level
//! tree over transformed instances, built for scenes that move.

pub mod bvh;
pub mod tlas;
pub(crate) mod tree;

pub use bvh::Bvh;
pub use tlas::Tlas;
