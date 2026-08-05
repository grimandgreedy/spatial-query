//! Acceleration structures and traversal.
//!
//! [`Bvh`] is a single-level tree over a set of leaves. [`Tlas`] is a two-level
//! tree over transformed instances, built for scenes that move.

pub mod bvh;
pub mod stats;
pub mod tlas;
pub(crate) mod tree;

pub use bvh::Bvh;
pub use stats::{QueryStats, TreeStats};
pub use tlas::Tlas;
