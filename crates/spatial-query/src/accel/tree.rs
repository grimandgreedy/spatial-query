//! Shared BVH node storage, the top-down median-split builder, and the bottom-up
//! refit, used by both the single-level [`Bvh`](super::bvh::Bvh) and the
//! two-level [`Tlas`](super::tlas::Tlas).

// The median split sorts by centroid on a chosen axis and walks per-axis
// extents; indexed loops read clearest there.
#![allow(clippy::needless_range_loop)]

use crate::maths::{Aabb, Point, Scalar};

/// Maximum primitives in a node before it must split.
pub(crate) const LEAF_SIZE: usize = 4;

/// A flat BVH node.
///
/// A leaf has `prim_count > 0` and `child_a` indexes the first of `prim_count`
/// contiguous entries in the tree's index array. An interior node has
/// `prim_count == 0` and children at node indices `child_a` and `child_b`, which
/// are stored explicitly because the recursive build appends whole subtrees.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Node<const D: usize> {
    pub bounds: Aabb<D>,
    pub child_a: u32,
    pub child_b: u32,
    pub prim_count: u32,
}

impl<const D: usize> Node<D> {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.prim_count > 0
    }
}

/// Build a median-split BVH over `bounds`. Returns the node array (root at index
/// 0) and the primitive index order that leaves slice into. The build order
/// guarantees every child has a higher index than its parent, which is what lets
/// [`refit`] work in a single reverse pass.
pub(crate) fn build<const D: usize>(bounds: &[Aabb<D>]) -> (Vec<Node<D>>, Vec<u32>) {
    let n = bounds.len();
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let mut nodes = Vec::new();
    if n == 0 {
        return (nodes, indices);
    }
    let centroids: Vec<Point<D>> = bounds.iter().map(|b| b.center()).collect();
    build_recursive(&mut nodes, &mut indices, bounds, &centroids, 0, n as u32);
    (nodes, indices)
}

/// Refit node bounds bottom-up from fresh per-primitive bounds, keeping the tree
/// topology and index order. `prim_bounds` is indexed by primitive id (the
/// values stored in `indices`).
pub(crate) fn refit<const D: usize>(
    nodes: &mut [Node<D>],
    indices: &[u32],
    prim_bounds: &[Aabb<D>],
) {
    for ni in (0..nodes.len()).rev() {
        let node = nodes[ni];
        if node.is_leaf() {
            let start = node.child_a as usize;
            let end = start + node.prim_count as usize;
            let mut b = Aabb::<D>::EMPTY;
            for &pi in &indices[start..end] {
                b = b.union(prim_bounds[pi as usize]);
            }
            nodes[ni].bounds = b;
        } else {
            let a = nodes[node.child_a as usize].bounds;
            let c = nodes[node.child_b as usize].bounds;
            nodes[ni].bounds = a.union(c);
        }
    }
}

/// Structural counts over a node array: `(leaf_count, prim_count, max_depth)`.
/// Depth is the longest root-to-leaf path in nodes. The build guarantees every
/// child has a higher index than its parent, so a single reverse pass computes
/// each node's depth after its children.
pub(crate) fn structural<const D: usize>(nodes: &[Node<D>]) -> (usize, usize, usize) {
    if nodes.is_empty() {
        return (0, 0, 0);
    }
    let mut depth = vec![0usize; nodes.len()];
    let mut leaf_count = 0;
    let mut prim_count = 0;
    for ni in (0..nodes.len()).rev() {
        let node = nodes[ni];
        if node.is_leaf() {
            leaf_count += 1;
            prim_count += node.prim_count as usize;
            depth[ni] = 1;
        } else {
            depth[ni] = 1 + depth[node.child_a as usize].max(depth[node.child_b as usize]);
        }
    }
    (leaf_count, prim_count, depth[0])
}

/// The sum of interior-node surface measures. Used as a proxy for tree quality:
/// after a refit spreads the boxes out, this grows, and a large enough increase
/// over the value at the last rebuild signals that a rebuild would pay off.
pub(crate) fn interior_surface_sum<const D: usize>(nodes: &[Node<D>]) -> Scalar {
    nodes
        .iter()
        .filter(|n| !n.is_leaf())
        .map(|n| n.bounds.half_surface())
        .sum()
}

fn build_recursive<const D: usize>(
    nodes: &mut Vec<Node<D>>,
    indices: &mut [u32],
    bounds: &[Aabb<D>],
    centroids: &[Point<D>],
    start: u32,
    end: u32,
) -> u32 {
    let mut node_bounds = Aabb::<D>::EMPTY;
    for &pi in &indices[start as usize..end as usize] {
        node_bounds = node_bounds.union(bounds[pi as usize]);
    }

    let node_index = nodes.len() as u32;
    let count = end - start;

    // Reserve this node's slot up front so children append after it.
    nodes.push(Node {
        bounds: node_bounds,
        child_a: start,
        child_b: 0,
        prim_count: count,
    });

    if count as usize <= LEAF_SIZE {
        return node_index;
    }

    let mut centroid_bounds = Aabb::<D>::EMPTY;
    for &pi in &indices[start as usize..end as usize] {
        centroid_bounds = centroid_bounds.union_point(centroids[pi as usize]);
    }
    let axis = centroid_bounds.longest_axis();
    if centroid_bounds.extents()[axis] <= Scalar::EPSILON {
        // All centroids coincide: keep it a leaf rather than loop forever.
        return node_index;
    }

    // TODO: this is a median split. A surface-area-heuristic split would build a
    // higher-quality tree for uneven distributions, at some build-time cost.
    indices[start as usize..end as usize]
        .sort_by(|&a, &b| centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis]));
    let mid = start + count / 2;

    let child_a = build_recursive(nodes, indices, bounds, centroids, start, mid);
    let child_b = build_recursive(nodes, indices, bounds, centroids, mid, end);
    let node = &mut nodes[node_index as usize];
    node.child_a = child_a;
    node.child_b = child_b;
    node.prim_count = 0;
    node_index
}
