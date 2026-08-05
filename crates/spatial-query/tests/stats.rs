//! Phase 5.3: structural tree stats read on demand, and opt-in per-query
//! counters with exact, deterministic values.

use spatial_query::{
    Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, QueryStats, Ray, Scalar,
};

/// Unit-ish balls strung along +x so a ray from the origin pierces them in
/// order.
struct Balls {
    centers: Vec<Point<3>>,
}

impl Balls {
    fn line(n: usize) -> Self {
        Balls {
            centers: (0..n)
                .map(|i| Point([2.0 * (i as f32 + 1.0), 0.0, 0.0]))
                .collect(),
        }
    }
}

impl QueryGeometry<3> for Balls {
    type Id = usize;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        self.centers.len()
    }
    fn id(&self, leaf: usize) -> usize {
        leaf
    }
    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        let c = self.centers[leaf];
        Aabb::new(c - Point::splat(0.5), c + Point::splat(0.5))
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        let c = self.centers[leaf];
        let oc = ray.origin - c;
        let b = oc.dot(ray.dir);
        let disc = b * b - (oc.length_squared() - 0.25);
        if disc < 0.0 {
            return None;
        }
        let t = -b - disc.sqrt();
        if t < 0.0 || t > max_toi {
            return None;
        }
        Some(LeafHit::new(t, (ray.at(t) - c).normalize_or_zero()))
    }
}

fn axis_ray() -> Ray<3> {
    Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]))
}

// ---------------------------------------------------------------------------
// Structural stats.
// ---------------------------------------------------------------------------

#[test]
fn structural_stats_of_a_single_leaf_tree() {
    // Three balls fit under LEAF_SIZE, so the tree is one leaf node.
    let balls = Balls::line(3);
    let bvh = Bvh::build(&balls);
    let s = bvh.stats();
    assert_eq!(s.node_count, 1, "single node");
    assert_eq!(s.node_count, bvh.node_count(), "matches node_count");
    assert_eq!(s.leaf_count, 1, "single leaf");
    assert_eq!(s.prim_count, 3, "holds all three prims");
    assert_eq!(s.max_depth, 1, "depth one");
    assert!(s.bytes > 0, "non-empty");
}

#[test]
fn structural_stats_invariants_on_a_larger_tree() {
    let balls = Balls::line(50);
    let bvh = Bvh::build(&balls);
    let s = bvh.stats();
    assert_eq!(s.node_count, bvh.node_count());
    assert_eq!(s.prim_count, 50, "every leaf indexed exactly once");
    assert!(
        s.leaf_count >= 1 && s.leaf_count < s.node_count,
        "interior + leaf"
    );
    assert!(s.max_depth >= 1 && s.max_depth <= s.node_count);
    assert!(s.quality.is_finite() && s.quality >= 0.0);
}

#[test]
fn empty_tree_stats_are_zero() {
    let balls = Balls::line(0);
    let bvh = Bvh::build(&balls);
    let s = bvh.stats();
    assert_eq!(s.node_count, 0);
    assert_eq!(s.leaf_count, 0);
    assert_eq!(s.prim_count, 0);
    assert_eq!(s.max_depth, 0);
}

// ---------------------------------------------------------------------------
// Per-query counters: exact values on a one-leaf scene.
// ---------------------------------------------------------------------------

#[test]
fn all_hits_counters_are_exact() {
    let balls = Balls::line(3); // one leaf node holding three balls
    let bvh = Bvh::build(&balls);
    let mut stats = QueryStats::new();
    let hits = bvh.raycast_all_stats(
        &balls,
        &axis_ray(),
        100.0,
        &QueryFilter::default(),
        &mut stats,
    );
    assert_eq!(hits.len(), 3, "ray pierces all three");
    assert_eq!(stats.nodes_visited, 1);
    assert_eq!(stats.aabb_tests, 1);
    assert_eq!(stats.narrow_tests, 3, "one narrow test per ball");
    assert_eq!(stats.hits, 3);
}

#[test]
fn nearest_counters_reflect_best_toi_pruning() {
    // Same one-leaf scene: nearest still narrow-tests all three (they share the
    // leaf), but only the first produces a surviving hit once the limit shrinks.
    let balls = Balls::line(3);
    let bvh = Bvh::build(&balls);
    let mut stats = QueryStats::new();
    let hit = bvh.raycast_nearest_stats(
        &balls,
        &axis_ray(),
        100.0,
        &QueryFilter::default(),
        &mut stats,
    );
    assert_eq!(hit.unwrap().leaf, 0, "nearest is the first ball");
    assert_eq!(stats.nodes_visited, 1);
    assert_eq!(stats.aabb_tests, 1);
    assert_eq!(stats.narrow_tests, 3);
    assert_eq!(
        stats.hits, 1,
        "only the nearest survives the shrinking limit"
    );
}

#[test]
fn miss_does_no_narrow_tests() {
    let balls = Balls::line(3);
    let bvh = Bvh::build(&balls);
    // A ray pointing away from the balls misses the node bounds outright.
    let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([0.0, 0.0, 1.0]));
    let mut stats = QueryStats::new();
    let hits = bvh.raycast_all_stats(&balls, &ray, 100.0, &QueryFilter::default(), &mut stats);
    assert!(hits.is_empty());
    assert_eq!(stats.nodes_visited, 1);
    assert_eq!(stats.aabb_tests, 1);
    assert_eq!(stats.narrow_tests, 0, "node-AABB miss skips the leaf");
    assert_eq!(stats.hits, 0);
}

#[test]
fn stats_path_matches_plain_path_and_repeats() {
    let balls = Balls::line(40);
    let bvh = Bvh::build(&balls);
    let filter = QueryFilter::default();

    let plain = bvh.raycast_all(&balls, &axis_ray(), 100.0, &filter);
    let mut s1 = QueryStats::new();
    let a = bvh.raycast_all_stats(&balls, &axis_ray(), 100.0, &filter, &mut s1);
    let mut s2 = QueryStats::new();
    let b = bvh.raycast_all_stats(&balls, &axis_ray(), 100.0, &filter, &mut s2);

    // Same results whether or not stats are collected, and across stats runs.
    let key = |h: &spatial_query::Hit<usize, 3>| (h.leaf, h.time_of_impact);
    let leaves = |v: &[spatial_query::Hit<usize, 3>]| v.iter().map(key).collect::<Vec<_>>();
    assert_eq!(leaves(&plain), leaves(&a));
    assert_eq!(leaves(&a), leaves(&b));
    // And the counters are deterministic across runs.
    assert_eq!(s1, s2);
    assert!(s1.nodes_visited > 1, "a 40-ball tree has interior nodes");
    assert!(s1.nodes_visited <= s1.narrow_tests + bvh.node_count() as u64);
}
