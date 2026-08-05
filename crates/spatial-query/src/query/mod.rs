//! The query interface: the provider trait, the filter, the result types, and
//! the naive baseline.
//!
//! [`QueryGeometry`] is the one thing the core knows about a consumer's data.
//! [`Hit`] and [`LeafHit`] are what queries return. [`QueryFilter`] is passed
//! through to the provider. [`naive`] is the O(N) scan that the accelerated
//! structures in [`crate::accel`] are checked against.

pub mod filter;
pub mod geometry;
pub mod hit;
pub mod instance;
pub mod naive;
pub mod overlap;
pub mod shapecast;

pub use filter::QueryFilter;
pub use geometry::QueryGeometry;
pub use hit::{Hit, LeafHit};
pub use instance::InstancedGeometry;
pub use overlap::Overlap;
pub use shapecast::ShapeCast;
