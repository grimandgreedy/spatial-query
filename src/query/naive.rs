//! Naive O(N) query baseline.
//!
//! These scan every leaf. They are the reference the accelerated BVH is checked
//! against, and a reasonable choice for tiny leaf counts. They use the same
//! tie-breaking as the BVH (`hit_ordering`), so results match exactly.

use super::filter::QueryFilter;
use super::geometry::QueryGeometry;
use super::hit::{hit_ordering, Hit, LeafHit};
use super::overlap::Overlap;
use super::shapecast::ShapeCast;
use crate::maths::{Aabb, Ray, Scalar};

/// Assemble a [`Hit`] from a provider leaf and its [`LeafHit`], computing the
/// world point from the ray.
#[inline]
pub(crate) fn assemble_hit<const D: usize, G: QueryGeometry<D>>(
    g: &G,
    ray: &Ray<D>,
    leaf: usize,
    lh: LeafHit<D, G::SubObject>,
) -> Hit<G::Id, D, G::SubObject> {
    Hit {
        id: g.id(leaf),
        leaf,
        time_of_impact: lh.toi,
        point: ray.at(lh.toi),
        normal: lh.normal,
        sub_object: lh.sub_object,
    }
}

/// Nearest hit along `ray` within `max_toi`, scanning all leaves.
pub fn raycast_nearest<const D: usize, G: QueryGeometry<D>>(
    g: &G,
    ray: &Ray<D>,
    max_toi: Scalar,
    filter: &QueryFilter,
) -> Option<Hit<G::Id, D, G::SubObject>> {
    let mut best: Option<Hit<G::Id, D, G::SubObject>> = None;
    let mut limit = max_toi;
    for leaf in 0..g.leaf_count() {
        if !g.accepts(leaf, filter) {
            continue;
        }
        // Cheap AABB reject before the exact test.
        if Aabb::ray_intersect(g.world_aabb(leaf), ray, limit).is_none() {
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
    best
}

/// All hits along `ray` within `max_toi`, sorted nearest-first (then by leaf).
pub fn raycast_all<const D: usize, G: QueryGeometry<D>>(
    g: &G,
    ray: &Ray<D>,
    max_toi: Scalar,
    filter: &QueryFilter,
) -> Vec<Hit<G::Id, D, G::SubObject>> {
    let mut hits = Vec::new();
    for leaf in 0..g.leaf_count() {
        if !g.accepts(leaf, filter) {
            continue;
        }
        if Aabb::ray_intersect(g.world_aabb(leaf), ray, max_toi).is_none() {
            continue;
        }
        if let Some(lh) = g.test_ray(leaf, ray, max_toi) {
            hits.push(assemble_hit(g, ray, leaf, lh));
        }
    }
    hits.sort_by(hit_ordering);
    hits
}

/// Assemble a [`Hit`] from a shape-cast [`LeafHit`], placing the world point at
/// the probe's reference position at contact.
#[inline]
pub(crate) fn assemble_shape_hit<const D: usize, G: QueryGeometry<D>>(
    g: &G,
    cast: &ShapeCast<D>,
    leaf: usize,
    lh: LeafHit<D, G::SubObject>,
) -> Hit<G::Id, D, G::SubObject> {
    Hit {
        id: g.id(leaf),
        leaf,
        time_of_impact: lh.toi,
        point: cast.at(lh.toi),
        normal: lh.normal,
        sub_object: lh.sub_object,
    }
}

/// Nearest shape-cast contact within `max_toi`, scanning all leaves.
pub fn shapecast_nearest<const D: usize, G: QueryGeometry<D>>(
    g: &G,
    cast: &ShapeCast<D>,
    max_toi: Scalar,
    filter: &QueryFilter,
) -> Option<Hit<G::Id, D, G::SubObject>> {
    let mut best: Option<Hit<G::Id, D, G::SubObject>> = None;
    let mut limit = max_toi;
    for leaf in 0..g.leaf_count() {
        if !g.accepts(leaf, filter) {
            continue;
        }
        if cast.broadphase(g.world_aabb(leaf), limit).is_none() {
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
    best
}

/// All shape-cast contacts within `max_toi`, sorted nearest-first (then leaf).
pub fn shapecast_all<const D: usize, G: QueryGeometry<D>>(
    g: &G,
    cast: &ShapeCast<D>,
    max_toi: Scalar,
    filter: &QueryFilter,
) -> Vec<Hit<G::Id, D, G::SubObject>> {
    let mut hits = Vec::new();
    for leaf in 0..g.leaf_count() {
        if !g.accepts(leaf, filter) {
            continue;
        }
        if cast.broadphase(g.world_aabb(leaf), max_toi).is_none() {
            continue;
        }
        if let Some(lh) = g.test_shape_cast(leaf, cast, max_toi) {
            hits.push(assemble_shape_hit(g, cast, leaf, lh));
        }
    }
    hits.sort_by(hit_ordering);
    hits
}

/// Every leaf whose exact geometry intersects `region`, in ascending leaf order.
pub fn overlap<const D: usize, G: QueryGeometry<D>>(
    g: &G,
    region: &Aabb<D>,
    filter: &QueryFilter,
) -> Vec<Overlap<G::Id>> {
    let mut out = Vec::new();
    for leaf in 0..g.leaf_count() {
        if !g.accepts(leaf, filter) {
            continue;
        }
        if !g.world_aabb(leaf).intersects(*region) {
            continue;
        }
        if g.test_overlap(leaf, region) {
            out.push(Overlap {
                id: g.id(leaf),
                leaf,
            });
        }
    }
    out
}
