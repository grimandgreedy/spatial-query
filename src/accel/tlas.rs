//! Two-level tree over transformed instances.
//!
//! The top level is a BVH over instance world AABBs. Each instance carries a
//! rigid transform and its own local geometry (its lower level). A ray query
//! walks the top level, transforms the ray into each candidate instance's local
//! space, and tests it there.
//!
//! When instances move, [`Tlas::refit`] recomputes their world AABBs and updates
//! the tree bounds in place, in time linear in the instance count, without
//! rebuilding. After enough motion the tree quality drifts;
//! [`Tlas::needs_rebuild`] reports when a rebuild would pay off.

use super::tree::{self, Node};
use crate::maths::{Aabb, Ray, Scalar};
use crate::query::filter::QueryFilter;
use crate::query::hit::{hit_ordering, Hit};
use crate::query::instance::InstancedGeometry;
use crate::query::overlap::Overlap;
use crate::query::shapecast::ShapeCast;

/// A top-level acceleration structure over a set of transformed instances.
pub struct Tlas<const D: usize> {
    nodes: Vec<Node<D>>,
    /// Instance indices reordered so each node's instances are contiguous.
    instance_indices: Vec<u32>,
    /// World AABB per instance, indexed by instance id. Recomputed on refit.
    world_aabbs: Vec<Aabb<D>>,
    /// Interior surface sum right after the last build, for the rebuild metric.
    baseline: Scalar,
}

impl<const D: usize> Tlas<D> {
    /// Build the tree over every instance of `scene`.
    pub fn build<S: InstancedGeometry<D>>(scene: &S) -> Self {
        let world_aabbs = instance_world_aabbs(scene);
        let (nodes, instance_indices) = tree::build(&world_aabbs);
        let baseline = tree::interior_surface_sum(&nodes);
        Tlas {
            nodes,
            instance_indices,
            world_aabbs,
            baseline,
        }
    }

    /// Recompute instance world AABBs from their current transforms and update
    /// the tree bounds in place, keeping the topology.
    pub fn refit<S: InstancedGeometry<D>>(&mut self, scene: &S) {
        for (i, slot) in self.world_aabbs.iter_mut().enumerate() {
            *slot = scene.transform(i).transform_aabb(scene.local_aabb(i));
        }
        tree::refit(&mut self.nodes, &self.instance_indices, &self.world_aabbs);
    }

    /// Whether the tree has drifted past `factor` times its quality at the last
    /// build and should be rebuilt. A `factor` around 1.5 to 2 is typical.
    pub fn needs_rebuild(&self, factor: Scalar) -> bool {
        self.baseline > 0.0 && tree::interior_surface_sum(&self.nodes) > factor * self.baseline
    }

    /// The number of nodes in the top-level tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Nearest hit along the world-space `ray` within `max_toi`.
    pub fn raycast_nearest<S: InstancedGeometry<D>>(
        &self,
        scene: &S,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Option<Hit<S::Id, D, S::SubObject>> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<Hit<S::Id, D, S::SubObject>> = None;
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
                for &ii in &self.instance_indices[start..end] {
                    let i = ii as usize;
                    if !scene.accepts(i, filter) {
                        continue;
                    }
                    if let Some(hit) = test_instance(scene, i, ray, limit) {
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

    /// All hits along the world-space `ray` within `max_toi`, sorted
    /// nearest-first (then by instance index).
    pub fn raycast_all<S: InstancedGeometry<D>>(
        &self,
        scene: &S,
        ray: &Ray<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Hit<S::Id, D, S::SubObject>> {
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
                for &ii in &self.instance_indices[start..end] {
                    let i = ii as usize;
                    if !scene.accepts(i, filter) {
                        continue;
                    }
                    if let Some(hit) = test_instance(scene, i, ray, max_toi) {
                        hits.push(hit);
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

    /// Nearest shape-cast contact along the world-space sweep within `max_toi`.
    pub fn shapecast_nearest<S: InstancedGeometry<D>>(
        &self,
        scene: &S,
        cast: &ShapeCast<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Option<Hit<S::Id, D, S::SubObject>> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<Hit<S::Id, D, S::SubObject>> = None;
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
                for &ii in &self.instance_indices[start..end] {
                    let i = ii as usize;
                    if !scene.accepts(i, filter) {
                        continue;
                    }
                    if let Some(hit) = shapecast_instance(scene, i, cast, limit) {
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

    /// All shape-cast contacts within `max_toi`, sorted nearest-first (then by
    /// instance index).
    pub fn shapecast_all<S: InstancedGeometry<D>>(
        &self,
        scene: &S,
        cast: &ShapeCast<D>,
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Hit<S::Id, D, S::SubObject>> {
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
                for &ii in &self.instance_indices[start..end] {
                    let i = ii as usize;
                    if !scene.accepts(i, filter) {
                        continue;
                    }
                    if let Some(hit) = shapecast_instance(scene, i, cast, max_toi) {
                        hits.push(hit);
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

    /// Every instance whose exact geometry intersects the world-space `region`,
    /// in ascending instance order.
    pub fn overlap<S: InstancedGeometry<D>>(
        &self,
        scene: &S,
        region: &Aabb<D>,
        filter: &QueryFilter,
    ) -> Vec<Overlap<S::Id>> {
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
                for &ii in &self.instance_indices[start..end] {
                    let i = ii as usize;
                    if !scene.accepts(i, filter) {
                        continue;
                    }
                    if scene.test_overlap(i, region) {
                        out.push(Overlap {
                            id: scene.id(i),
                            leaf: i,
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
}

/// Broad-phase against instance `i`'s world AABB, then the provider's exact
/// world-space swept test, assembled into a [`Hit`].
#[inline]
fn shapecast_instance<const D: usize, S: InstancedGeometry<D>>(
    scene: &S,
    i: usize,
    cast: &ShapeCast<D>,
    max_toi: Scalar,
) -> Option<Hit<S::Id, D, S::SubObject>> {
    let lh = scene.test_shape_cast(i, cast, max_toi)?;
    Some(Hit {
        id: scene.id(i),
        leaf: i,
        time_of_impact: lh.toi,
        point: cast.at(lh.toi),
        normal: lh.normal,
        sub_object: lh.sub_object,
    })
}

/// Transform the world ray into instance `i`'s local space, test it, and lift the
/// result back to world space.
// TODO: each instance is tested as a whole. Instances with many primitives could
// carry their own lower-level tree so the narrow-phase test is accelerated too.
#[inline]
fn test_instance<const D: usize, S: InstancedGeometry<D>>(
    scene: &S,
    i: usize,
    ray: &Ray<D>,
    max_toi: Scalar,
) -> Option<Hit<S::Id, D, S::SubObject>> {
    let iso = scene.transform(i);
    // The rotation is orthonormal, so the local direction stays unit length and
    // the local time-of-impact is the world time-of-impact.
    let local = Ray::new_unnormalized(
        iso.inverse_transform_point(ray.origin),
        iso.inverse_transform_vector(ray.dir),
    );
    let lh = scene.test_ray_local(i, &local, max_toi)?;
    Some(Hit {
        id: scene.id(i),
        leaf: i,
        time_of_impact: lh.toi,
        point: ray.at(lh.toi),
        normal: iso.transform_vector(lh.normal),
        sub_object: lh.sub_object,
    })
}

fn instance_world_aabbs<const D: usize, S: InstancedGeometry<D>>(scene: &S) -> Vec<Aabb<D>> {
    (0..scene.instance_count())
        .map(|i| scene.transform(i).transform_aabb(scene.local_aabb(i)))
        .collect()
}
