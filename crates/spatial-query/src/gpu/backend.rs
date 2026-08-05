//! The wgpu version this build targets, behind a single alias.
//!
//! The GPU backend names `backend::*` instead of `wgpu::*` so that selecting a
//! wgpu major version is this one module's concern, mirroring viewport-lib's own
//! seam. The `wgpu27` / `wgpu29` cargo features pick the version by re-exporting
//! the matching dependency; exactly one must be enabled, and only that crate is
//! compiled. A new leg (e.g. wgpu 30) grows one more branch here.

#[cfg(all(feature = "wgpu27", feature = "wgpu29"))]
compile_error!("the `wgpu27` and `wgpu29` features are mutually exclusive: enable exactly one");

// The 29 crate keeps its real name `wgpu` (the default `gpu` pick); only 27 is
// aliased.
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub use wgpu::*;
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub use wgpu27::*;

/// Construct a wgpu `Instance` with default options, papering over the
/// `InstanceDescriptor` construction that differs across legs: 27 derives
/// `Default` and takes the descriptor by reference, while 29 gained a
/// display-handle field and constructs by value through
/// `new_without_display_handle`.
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub fn default_instance() -> Instance {
    Instance::new(InstanceDescriptor::new_without_display_handle())
}
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub fn default_instance() -> Instance {
    Instance::new(&InstanceDescriptor::default())
}
