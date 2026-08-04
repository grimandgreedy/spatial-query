//! Single-level bounding-volume hierarchy and CPU ray traversal.
//!
//! Built once over a provider's leaf AABBs by top-down median split on the
//! longest centroid axis. Traversal is an explicit-stack walk that prunes whole
//! subtrees by the ray-AABB slab test and, for nearest queries, by the current
//! best `time_of_impact`.
//!
//! The BVH stores only bounds and leaf indices; exact geometry lives in the
//! provider, tested through [`QueryGeometry::test_ray`]. This tree is rebuilt
//! wholesale rather than refit.

use crate::maths::{Aabb, Point, Ray, Scalar};
use crate::query::filter::QueryFilter;
use crate::query::geometry::QueryGeometry;
use crate::query::hit::{hit_ordering, Hit};
use crate::query::naive::assemble_hit;

/// Max leaves in a BVH node before it must split.
const LEAF_SIZE: usize = 4;

/// A flat BVH node.
///
/// When `prim_count > 0` the node is a leaf and `child_a` indexes into
/// [`Bvh::prim_indices`] at the first of `prim_count` contiguous leaves.
/// Otherwise it is interior with children at node indices `child_a` and
/// `child_b` (stored explicitly rather than assumed contiguous, since the
/// recursive build appends whole subtrees).
#[derive(Clone, Copy, Debug)]
struct Node<const D: usize> {
    bounds: Aabb<D>,
    child_a: u32,
    child_b: u32,
    prim_count: u32,
}

/// A single-level BVH over a provider's leaves.
pub struct Bvh<const D: usize> {
    nodes: Vec<Node<D>>,
    /// Leaf indices reordered so each node's primitives are contiguous.
    prim_indices: Vec<u32>,
}

impl<const D: usize> Bvh<D> {
    /// Build a BVH over every leaf of `g`.
    pub fn build<G: QueryGeometry<D>>(g: &G) -> Self {
        let n = g.leaf_count();
        let mut prim_indices: Vec<u32> = (0..n as u32).collect();
        let nodes: Vec<Node<D>> = Vec::new();
        if n == 0 {
            return Bvh {
                nodes,
                prim_indices,
            };
        }
        // Cache bounds and centroids so the build never re-queries the provider.
        let bounds: Vec<Aabb<D>> = (0..n).map(|i| g.world_aabb(i)).collect();
        let centroids: Vec<Point<D>> = bounds.iter().map(|b| b.center()).collect();

        let mut nodes = nodes;
        build_recursive(
            &mut nodes,
            &mut prim_indices,
            &bounds,
            &centroids,
            0,
            n as u32,
        );
        Bvh {
            nodes,
            prim_indices,
        }
    }

    /// Nearest hit along `ray` within `max_toi`.
    pub fn raycast_nearest<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Option<Hit<G::Id, D>> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<Hit<G::Id, D>> = None;
        let mut limit = max_toi;
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            // Prune: miss, or the whole box is beyond the current best hit.
            match node.bounds.ray_intersect(ray, limit) {
                Some((t_enter, _)) if t_enter <= limit => {}
                _ => continue,
            }
            if node.prim_count > 0 {
                let start = node.child_a as usize;
                let end = start + node.prim_count as usize;
                for &pi in &self.prim_indices[start..end] {
                    let leaf = pi as usize;
                    if !g.accepts(leaf, filter) {
                        continue;
                    }
                    if let Some(lh) = g.test_ray(leaf, ray, limit) {
                        let hit = assemble_hit(g, ray, leaf, lh);
                        match &best {
                            Some(prev) if hit_ordering(prev, &hit).is_le() => {}
                            _ => {
                                limit = hit.time_of_impact;
                                best = Some(hit);
                            }
                        }
                    }
                }
            } else {
                stack.push(node.child_a);
                stack.push(node.child_b);
            }
        }
        best
    }

    /// All hits along `ray` within `max_toi`, sorted nearest-first (then leaf).
    pub fn raycast_all<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Hit<G::Id, D>> {
        let mut hits = Vec::new();
        if self.nodes.is_empty() {
            return hits;
        }
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if node.bounds.ray_intersect(ray, max_toi).is_none() {
                continue;
            }
            if node.prim_count > 0 {
                let start = node.child_a as usize;
                let end = start + node.prim_count as usize;
                for &pi in &self.prim_indices[start..end] {
                    let leaf = pi as usize;
                    if !g.accepts(leaf, filter) {
                        continue;
                    }
                    if let Some(lh) = g.test_ray(leaf, ray, max_toi) {
                        hits.push(assemble_hit(g, ray, leaf, lh));
                    }
                }
            } else {
                stack.push(node.child_a);
                stack.push(node.child_b);
            }
        }
        hits.sort_by(hit_ordering);
        hits
    }

    /// The number of nodes in the tree (diagnostics / tests).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Recursively build a subtree over `prim_indices[start..end]`, appending nodes
/// to `nodes`, and return the node index of the subtree root.
fn build_recursive<const D: usize>(
    nodes: &mut Vec<Node<D>>,
    prim_indices: &mut [u32],
    bounds: &[Aabb<D>],
    centroids: &[Point<D>],
    start: u32,
    end: u32,
) -> u32 {
    let mut node_bounds = Aabb::<D>::EMPTY;
    for &pi in &prim_indices[start as usize..end as usize] {
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

    // Leaf: small enough to stop.
    if count as usize <= LEAF_SIZE {
        return node_index;
    }

    // Split on the longest axis of the centroid bounds, at the median.
    let mut centroid_bounds = Aabb::<D>::EMPTY;
    for &pi in &prim_indices[start as usize..end as usize] {
        centroid_bounds = centroid_bounds.union_point(centroids[pi as usize]);
    }
    let axis = centroid_bounds.longest_axis();
    if centroid_bounds.extents()[axis] <= Scalar::EPSILON {
        // All centroids coincide: keep it as a leaf rather than loop forever.
        return node_index;
    }

    prim_indices[start as usize..end as usize]
        .sort_by(|&a, &b| centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis]));
    let mid = start + count / 2;

    let child_a = build_recursive(nodes, prim_indices, bounds, centroids, start, mid);
    let child_b = build_recursive(nodes, prim_indices, bounds, centroids, mid, end);
    let node = &mut nodes[node_index as usize];
    node.child_a = child_a;
    node.child_b = child_b;
    node.prim_count = 0;
    node_index
}
