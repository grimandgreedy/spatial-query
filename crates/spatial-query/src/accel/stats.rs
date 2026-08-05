//! Structural statistics about a tree, and opt-in per-query counters.
//!
//! [`TreeStats`] describes a tree's current shape and is read on demand, with no
//! query and no internal ticker. [`QueryStats`] counts traversal work for one
//! query and is opt-in: pass it to a query's `*_stats` variant. Both are
//! deterministic, so a replay produces the same numbers and they can be asserted
//! in tests. Neither carries wall-clock time; timing is the consumer's concern,
//! measured around the call.

use crate::maths::Scalar;

/// A read-out of a tree's current shape and size. Cheap to compute and
/// deterministic; read it any time from [`Bvh::stats`](crate::Bvh::stats) or
/// [`Tlas::stats`](crate::Tlas::stats) without running a query.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct TreeStats {
    /// Total nodes, interior plus leaf.
    pub node_count: usize,
    /// Leaf nodes.
    pub leaf_count: usize,
    /// Primitives (provider leaves or instances) held across all leaf nodes.
    pub prim_count: usize,
    /// Longest root-to-leaf path, counted in nodes. Zero for an empty tree.
    pub max_depth: usize,
    /// The interior-surface quality metric (the same one
    /// [`needs_rebuild`](crate::Tlas::needs_rebuild) uses); a larger value means
    /// a looser tree.
    pub quality: Scalar,
    /// Approximate heap bytes held by the node and index arrays.
    pub bytes: usize,
}

/// Per-query traversal counters. Opt in by passing `&mut QueryStats` to a
/// query's `*_stats` variant; the plain variants never touch it.
///
/// Counts are deterministic (a replay produces the same numbers), so they can be
/// asserted in tests. They carry no wall-clock time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryStats {
    /// Nodes popped from the traversal stack and inspected.
    pub nodes_visited: u64,
    /// Ray or shape versus node-AABB tests performed.
    pub aabb_tests: u64,
    /// Provider narrow tests invoked (`test_ray`, `test_ray_crossings`, ...).
    pub narrow_tests: u64,
    /// Hits produced by narrow tests, before nearest selection or sorting.
    pub hits: u64,
}

impl QueryStats {
    /// A fresh zeroed counter.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Traversal event sink. [`NoStats`] is zero-sized and its methods are empty, so
/// the plain query path compiles down to no counting at all; `&mut QueryStats`
/// counts.
pub(crate) trait Observe {
    fn node(&mut self);
    fn aabb(&mut self);
    fn narrow(&mut self);
    fn hit(&mut self);
}

/// The zero-cost observer the plain (non-`_stats`) query methods use.
pub(crate) struct NoStats;

impl Observe for NoStats {
    #[inline(always)]
    fn node(&mut self) {}
    #[inline(always)]
    fn aabb(&mut self) {}
    #[inline(always)]
    fn narrow(&mut self) {}
    #[inline(always)]
    fn hit(&mut self) {}
}

impl Observe for &mut QueryStats {
    #[inline]
    fn node(&mut self) {
        self.nodes_visited += 1;
    }
    #[inline]
    fn aabb(&mut self) {
        self.aabb_tests += 1;
    }
    #[inline]
    fn narrow(&mut self) {
        self.narrow_tests += 1;
    }
    #[inline]
    fn hit(&mut self) {
        self.hits += 1;
    }
}
