//! Single-level bounding-volume hierarchy and CPU ray traversal.
//!
//! Built over a provider's leaf AABBs. Traversal is an explicit-stack walk that
//! prunes whole subtrees by the ray-AABB slab test and, for nearest queries, by
//! the current best `time_of_impact`.
//!
//! The BVH stores only bounds and leaf indices; exact geometry lives in the
//! provider, tested through [`QueryGeometry::test_ray`]. When leaves move but
//! stay the same in number, [`Bvh::refit`] updates the bounds in place instead
//! of rebuilding the tree.

use super::tree::{self, Node};
use crate::maths::{Aabb, Ray, Scalar};
use crate::query::filter::QueryFilter;
use crate::query::geometry::QueryGeometry;
use crate::query::hit::{hit_ordering, Hit};
use crate::query::naive::{assemble_hit, assemble_shape_hit};
use crate::query::overlap::Overlap;
use crate::query::shapecast::ShapeCast;

/// A single-level BVH over a provider's leaves.
pub struct Bvh<const D: usize> {
    nodes: Vec<Node<D>>,
    /// Leaf indices reordered so each node's primitives are contiguous.
    prim_indices: Vec<u32>,
}

impl<const D: usize> Bvh<D> {
    /// Build a BVH over every leaf of `g`.
    pub fn build<G: QueryGeometry<D>>(g: &G) -> Self {
        let bounds: Vec<_> = (0..g.leaf_count()).map(|i| g.world_aabb(i)).collect();
        let (nodes, prim_indices) = tree::build(&bounds);
        Bvh {
            nodes,
            prim_indices,
        }
    }

    /// Update the tree bounds from the current leaf AABBs, keeping the topology.
    ///
    /// Much cheaper than [`build`](Self::build) for a scene whose leaves moved
    /// but did not change in number or identity. The tree quality can drift after
    /// many refits; rebuild when that matters.
    pub fn refit<G: QueryGeometry<D>>(&mut self, g: &G) {
        let bounds: Vec<_> = (0..g.leaf_count()).map(|i| g.world_aabb(i)).collect();
        tree::refit(&mut self.nodes, &self.prim_indices, &bounds);
    }

    /// Nearest hit along `ray` within `max_toi`.
    pub fn raycast_nearest<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Option<Hit<G::Id, D, G::SubObject>> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<Hit<G::Id, D, G::SubObject>> = None;
        let mut limit = max_toi;
        // TODO: allocates a traversal stack per query. A reusable scratch buffer
        // would avoid the allocation in hot query loops.
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            match node.bounds.ray_intersect(ray, limit) {
                Some((t_enter, _)) if t_enter <= limit => {}
                _ => continue,
            }
            if node.is_leaf() {
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
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
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
            if node.is_leaf() {
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

    /// Nearest shape-cast contact within `max_toi`.
    pub fn shapecast_nearest<G: QueryGeometry<D>>(
        &self,
        g: &G,
        cast: &ShapeCast<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Option<Hit<G::Id, D, G::SubObject>> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<Hit<G::Id, D, G::SubObject>> = None;
        let mut limit = max_toi;
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            match cast.broadphase(node.bounds, limit) {
                Some((t_enter, _)) if t_enter <= limit => {}
                _ => continue,
            }
            if node.is_leaf() {
                let start = node.child_a as usize;
                let end = start + node.prim_count as usize;
                for &pi in &self.prim_indices[start..end] {
                    let leaf = pi as usize;
                    if !g.accepts(leaf, filter) {
                        continue;
                    }
                    if let Some(lh) = g.test_shape_cast(leaf, cast, limit) {
                        let hit = assemble_shape_hit(g, cast, leaf, lh);
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

    /// All shape-cast contacts within `max_toi`, sorted nearest-first (then leaf).
    pub fn shapecast_all<G: QueryGeometry<D>>(
        &self,
        g: &G,
        cast: &ShapeCast<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
        let mut hits = Vec::new();
        if self.nodes.is_empty() {
            return hits;
        }
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if cast.broadphase(node.bounds, max_toi).is_none() {
                continue;
            }
            if node.is_leaf() {
                let start = node.child_a as usize;
                let end = start + node.prim_count as usize;
                for &pi in &self.prim_indices[start..end] {
                    let leaf = pi as usize;
                    if !g.accepts(leaf, filter) {
                        continue;
                    }
                    if let Some(lh) = g.test_shape_cast(leaf, cast, max_toi) {
                        hits.push(assemble_shape_hit(g, cast, leaf, lh));
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

    /// Every leaf whose exact geometry intersects `region`, in ascending leaf
    /// order.
    pub fn overlap<G: QueryGeometry<D>>(
        &self,
        g: &G,
        region: &Aabb<D>,
        filter: &QueryFilter,
    ) -> Vec<Overlap<G::Id>> {
        let mut out = Vec::new();
        if self.nodes.is_empty() {
            return out;
        }
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if !node.bounds.intersects(*region) {
                continue;
            }
            if node.is_leaf() {
                let start = node.child_a as usize;
                let end = start + node.prim_count as usize;
                for &pi in &self.prim_indices[start..end] {
                    let leaf = pi as usize;
                    if !g.accepts(leaf, filter) {
                        continue;
                    }
                    if g.test_overlap(leaf, region) {
                        out.push(Overlap {
                            id: g.id(leaf),
                            leaf,
                        });
                    }
                }
            } else {
                stack.push(node.child_a);
                stack.push(node.child_b);
            }
        }
        out.sort_by_key(|o| o.leaf);
        out
    }

    /// The number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
