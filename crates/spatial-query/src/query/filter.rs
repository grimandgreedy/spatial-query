//! Query filtering.
//!
//! The core does not interpret a [`QueryFilter`]; it hands it to the provider's
//! [`QueryGeometry::accepts`](crate::QueryGeometry::accepts), which decides what
//! a filter means for its own data (collision layers, sensors, exclusion, ...).
//! The fields here are a neutral starting vocabulary; providers may ignore them.

/// A filter passed through every query to the provider's `accepts` hook.
///
/// Start from [`QueryFilter::default`] and set fields as needed. The struct is
/// `#[non_exhaustive]` so the filter vocabulary can grow without breaking callers.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct QueryFilter {
    /// A layer bitmask. Its meaning is the provider's; the core only carries it.
    pub layer_mask: u64,
    /// A single leaf id the caller wants excluded (e.g. the querying object
    /// itself). Interpreted by the provider.
    pub exclude: Option<u64>,
}

impl Default for QueryFilter {
    fn default() -> Self {
        QueryFilter {
            layer_mask: u64::MAX,
            exclude: None,
        }
    }
}
