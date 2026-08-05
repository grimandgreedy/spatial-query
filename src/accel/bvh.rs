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

use super::stats::{NoStats, Observe, QueryStats, TreeStats};
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
        self.raycast_nearest_impl(g, ray, max_toi, filter, NoStats)
    }

    /// Nearest hit along `ray`, also accumulating traversal counts into `stats`.
    pub fn raycast_nearest_stats<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
        stats: &mut QueryStats,
    ) -> Option<Hit<G::Id, D, G::SubObject>> {
        self.raycast_nearest_impl(g, ray, max_toi, filter, stats)
    }

    fn raycast_nearest_impl<G: QueryGeometry<D>, O: Observe>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
        mut obs: O,
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
            obs.node();
            obs.aabb();
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
                    obs.narrow();
                    if let Some(lh) = g.test_ray(leaf, ray, limit) {
                        obs.hit();
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

    /// The nearest hit for each ray in `rays`, one entry per input ray in the
    /// same order.
    ///
    /// Each ray is answered by an independent [`raycast_nearest`](Self::raycast_nearest),
    /// so the result is the deterministic per-ray outcome laid out in ray order.
    /// This is the sequential reference that
    /// [`raycast_nearest_batch_parallel`](Self::raycast_nearest_batch_parallel)
    /// reproduces bit for bit.
    pub fn raycast_nearest_batch<G: QueryGeometry<D>>(
        &self,
        g: &G,
        rays: &[Ray<D>],
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Option<Hit<G::Id, D, G::SubObject>>> {
        rays.iter()
            .map(|ray| self.raycast_nearest(g, ray, max_toi, filter))
            .collect()
    }

    /// The nearest hit for each ray, computed across up to `threads` worker
    /// threads, returning the exact same values in the exact same order as
    /// [`raycast_nearest_batch`](Self::raycast_nearest_batch).
    ///
    /// The batch is split into contiguous index ranges, one per worker, and each
    /// worker writes only into its own slice of the output. No result depends on
    /// another ray, and nothing is reduced across rays, so the output is a
    /// function of the ray index alone: the thread count and scheduling cannot
    /// change it. This is the fixed-order parallelism the deterministic contract
    /// allows -- identical to the sequential path, never an unordered reduction.
    ///
    /// `threads` is clamped to at least one and at most the number of rays; a
    /// value of one (or an empty or single-ray batch) runs sequentially without
    /// spawning.
    pub fn raycast_nearest_batch_parallel<G>(
        &self,
        g: &G,
        rays: &[Ray<D>],
        max_toi: Scalar,
        filter: &QueryFilter,
        threads: usize,
    ) -> Vec<Option<Hit<G::Id, D, G::SubObject>>>
    where
        G: QueryGeometry<D> + Sync,
        G::Id: Send,
        G::SubObject: Send,
    {
        let workers = threads.clamp(1, rays.len().max(1));
        if workers == 1 {
            return self.raycast_nearest_batch(g, rays, max_toi, filter);
        }

        let mut out: Vec<Option<Hit<G::Id, D, G::SubObject>>> =
            (0..rays.len()).map(|_| None).collect();
        // Ceiling division so `workers` chunks cover every ray; the last chunk may
        // be shorter. Chunk boundaries are fixed by index, not by timing.
        let chunk = rays.len().div_ceil(workers);
        std::thread::scope(|scope| {
            for (ray_chunk, out_chunk) in rays.chunks(chunk).zip(out.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (ray, slot) in ray_chunk.iter().zip(out_chunk.iter_mut()) {
                        *slot = self.raycast_nearest(g, ray, max_toi, filter);
                    }
                });
            }
        });
        out
    }

    /// All hits along `ray` within `max_toi`, sorted nearest-first (then leaf).
    pub fn raycast_all<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
        self.raycast_all_impl(g, ray, max_toi, filter, NoStats)
    }

    /// All hits along `ray`, also accumulating traversal counts into `stats`.
    pub fn raycast_all_stats<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
        stats: &mut QueryStats,
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
        self.raycast_all_impl(g, ray, max_toi, filter, stats)
    }

    fn raycast_all_impl<G: QueryGeometry<D>, O: Observe>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
        mut obs: O,
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
        let mut hits = Vec::new();
        if self.nodes.is_empty() {
            return hits;
        }
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            obs.node();
            obs.aabb();
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
                    obs.narrow();
                    if let Some(lh) = g.test_ray(leaf, ray, max_toi) {
                        obs.hit();
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

    /// Every crossing of `ray` with the leaves within `max_toi`, sorted
    /// nearest-first (then leaf). A concave or self-intersecting leaf can appear
    /// more than once; there is no best-toi pruning, so the whole ray is walked.
    pub fn raycast_crossings<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
        self.raycast_crossings_impl(g, ray, max_toi, filter, NoStats)
    }

    /// Every crossing of `ray`, also accumulating traversal counts into `stats`.
    pub fn raycast_crossings_stats<G: QueryGeometry<D>>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
        stats: &mut QueryStats,
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
        self.raycast_crossings_impl(g, ray, max_toi, filter, stats)
    }

    fn raycast_crossings_impl<G: QueryGeometry<D>, O: Observe>(
        &self,
        g: &G,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
        mut obs: O,
    ) -> Vec<Hit<G::Id, D, G::SubObject>> {
        let mut hits = Vec::new();
        if self.nodes.is_empty() {
            return hits;
        }
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            obs.node();
            obs.aabb();
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
                    obs.narrow();
                    let mut push = |lh| {
                        obs.hit();
                        hits.push(assemble_hit(g, ray, leaf, lh));
                    };
                    g.test_ray_crossings(leaf, ray, max_toi, &mut push);
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

    /// The flat node array (root at index 0). Used by the GPU backend to pack the
    /// tree into a device buffer; the CPU tree stays canonical.
    #[cfg(any(feature = "wgpu27", feature = "wgpu29"))]
    pub(crate) fn nodes(&self) -> &[Node<D>] {
        &self.nodes
    }

    /// The primitive index order leaves slice into. Uploaded once as device
    /// topology; a refit re-uploads bounds, not this.
    #[cfg(any(feature = "wgpu27", feature = "wgpu29"))]
    pub(crate) fn prim_indices(&self) -> &[u32] {
        &self.prim_indices
    }

    /// The tree's current structural statistics, computed on demand.
    pub fn stats(&self) -> TreeStats {
        let (leaf_count, prim_count, max_depth) = tree::structural(&self.nodes);
        TreeStats {
            node_count: self.nodes.len(),
            leaf_count,
            prim_count,
            max_depth,
            quality: tree::interior_surface_sum(&self.nodes),
            bytes: self.nodes.len() * core::mem::size_of::<Node<D>>()
                + self.prim_indices.len() * core::mem::size_of::<u32>(),
        }
    }
}
