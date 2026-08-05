//! Phase 6: the cost model picks CPU for small / headless / deterministic
//! batches and an alternate backend for the regimes those win, and a registered
//! stub backend both gets chosen and agrees with CPU on hit identity.

use spatial_query::{
    Aabb, BackendKind, Bvh, Dispatcher, LeafHit, Point, QueryBackend, QueryContext, QueryFilter,
    QueryGeometry, Ray, Scalar,
};

// A small deterministic RNG.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: Scalar, hi: Scalar) -> Scalar {
        let u = (self.next_u64() >> 11) as Scalar / (1u64 << 53) as Scalar;
        lo + u * (hi - lo)
    }
    fn point(&mut self, lo: Scalar, hi: Scalar) -> Point<3> {
        Point([self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)])
    }
}

struct Balls {
    centers: Vec<Point<3>>,
}

impl spatial_query::QueryGeometry<3> for Balls {
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
        Aabb::new(c - Point::splat(1.0), c + Point::splat(1.0))
    }
    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: Scalar) -> Option<LeafHit<3>> {
        let oc = ray.origin - self.centers[leaf];
        let b = oc.dot(ray.dir);
        let disc = b * b - (oc.length_squared() - 1.0);
        if disc < 0.0 {
            return None;
        }
        let t = -b - disc.sqrt();
        if t < 0.0 || t > max_toi {
            return None;
        }
        Some(LeafHit::new(
            t,
            (ray.at(t) - self.centers[leaf]).normalize_or_zero(),
        ))
    }
}

fn scene(n: usize) -> (Balls, Bvh<3>) {
    let mut rng = Rng::new(0xD15_0A7C);
    let balls = Balls {
        centers: (0..n).map(|_| rng.point(-40.0, 40.0)).collect(),
    };
    let bvh = Bvh::build(&balls);
    (balls, bvh)
}

// A batch of rays aimed roughly through the origin.
fn rays(n: usize) -> Vec<Ray<3>> {
    let mut rng = Rng::new(0x11A5);
    (0..n)
        .map(|_| {
            let o = rng.point(-50.0, 50.0);
            Ray::new(o, Point::ZERO - o)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stub alternate backends. Each stands in for a real backend (a draw-pass
// picker, a GPU-compute traversal); both delegate to the CPU tree so they agree
// with CPU on results, and each declares its own cost shape and availability.
// ---------------------------------------------------------------------------

struct StubDrawPass<'a> {
    bvh: &'a Bvh<3>,
}
impl QueryBackend<3, Balls> for StubDrawPass<'_> {
    fn kind(&self) -> BackendKind {
        BackendKind::DrawPass
    }
    fn can_serve(&self, ctx: &QueryContext) -> bool {
        ctx.device_available
    }
    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // The first view piggybacks the frame's draw (nearly free); each extra
        // distinct direction costs a full scene pass.
        (ctx.distinct_directions.saturating_sub(1)) as f64 * ctx.scene_leaves as f64 + 1.0
    }
    fn raycast_nearest_batch(
        &self,
        g: &Balls,
        rays: &[Ray<3>],
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Option<spatial_query::Hit<usize, 3>>> {
        rays.iter()
            .map(|r| self.bvh.raycast_nearest(g, r, max_toi, filter))
            .collect()
    }
}

struct StubGpu<'a> {
    bvh: &'a Bvh<3>,
}
impl QueryBackend<3, Balls> for StubGpu<'_> {
    fn kind(&self) -> BackendKind {
        BackendKind::Gpu
    }
    fn can_serve(&self, ctx: &QueryContext) -> bool {
        ctx.device_available
    }
    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // One buffer upload plus one dispatch over the batch, direction-agnostic.
        ctx.scene_leaves as f64 + ctx.batch_size as f64
    }
    fn raycast_nearest_batch(
        &self,
        g: &Balls,
        rays: &[Ray<3>],
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> Vec<Option<spatial_query::Hit<usize, 3>>> {
        rays.iter()
            .map(|r| self.bvh.raycast_nearest(g, r, max_toi, filter))
            .collect()
    }
}

fn full_dispatcher<'a>(bvh: &'a Bvh<3>) -> Dispatcher<'a, 3, Balls> {
    let mut d = Dispatcher::new(bvh);
    d.register(Box::new(StubDrawPass { bvh }));
    d.register(Box::new(StubGpu { bvh }));
    d
}

// ---------------------------------------------------------------------------
// Selection matrix.
// ---------------------------------------------------------------------------

#[test]
fn deterministic_request_pins_cpu() {
    let (_g, bvh) = scene(1000);
    let d = full_dispatcher(&bvh);
    // A scattered batch that would otherwise pick GPU, but determinism pins CPU.
    let ctx = QueryContext::new(100_000, 100_000, 100_000)
        .with_device(true)
        .deterministic(true);
    assert_eq!(d.select(&ctx), BackendKind::Cpu);
}

#[test]
fn no_device_falls_back_to_cpu() {
    let (_g, bvh) = scene(1000);
    let d = full_dispatcher(&bvh);
    // Single view, huge scene: draw-pass would win, but no device is present.
    let ctx = QueryContext::new(1, 1, 100_000).with_device(false);
    assert_eq!(d.select(&ctx), BackendKind::Cpu);
}

#[test]
fn single_view_large_scene_picks_draw_pass() {
    let (_g, bvh) = scene(1000);
    let d = full_dispatcher(&bvh);
    // One cursor click on a huge scene.
    let click = QueryContext::new(1, 1, 100_000).with_device(true);
    assert_eq!(d.select(&click), BackendKind::DrawPass);
    // A rectangle select: many rays, still one view direction.
    let rect = QueryContext::new(5_000, 1, 100_000).with_device(true);
    assert_eq!(d.select(&rect), BackendKind::DrawPass);
}

#[test]
fn scattered_large_batch_picks_gpu() {
    let (_g, bvh) = scene(1000);
    let d = full_dispatcher(&bvh);
    // Thousands of rays from thousands of directions: many draw passes would be
    // needed, and the CPU walk is linear in the batch, so GPU compute wins.
    let ctx = QueryContext::new(100_000, 100_000, 100_000).with_device(true);
    assert_eq!(d.select(&ctx), BackendKind::Gpu);
}

#[test]
fn small_batch_stays_cpu_without_alternates_winning() {
    let (_g, bvh) = scene(1000);
    // A dispatcher with only CPU registered always selects CPU.
    let d: Dispatcher<3, Balls> = Dispatcher::new(&bvh);
    let ctx = QueryContext::new(1, 1, 100_000).with_device(true);
    assert_eq!(d.select(&ctx), BackendKind::Cpu);
}

#[test]
fn force_override_wins_over_everything() {
    let (_g, bvh) = scene(1000);
    let mut d = full_dispatcher(&bvh);
    d.force(Some(BackendKind::Gpu));
    // Even a deterministic request now reports the forced backend.
    let ctx = QueryContext::new(1, 1, 10).deterministic(true);
    assert_eq!(d.select(&ctx), BackendKind::Gpu);
    d.force(None);
    assert_eq!(d.select(&ctx), BackendKind::Cpu);
}

// ---------------------------------------------------------------------------
// Chosen backend runs and agrees with CPU on identity.
// ---------------------------------------------------------------------------

#[test]
fn chosen_backend_agrees_with_cpu_on_identity() {
    let (g, bvh) = scene(2000);
    let d = full_dispatcher(&bvh);
    let batch = rays(500);
    let filter = QueryFilter::default();

    // Single-view large scene selects the draw-pass stub.
    let ctx = QueryContext::new(batch.len(), 1, g.leaf_count()).with_device(true);
    let (kind, hits) = d.raycast_nearest_batch(&g, &batch, 500.0, &filter, &ctx);
    assert_eq!(kind, BackendKind::DrawPass, "single view picks draw-pass");

    // The dispatched results match a direct CPU traversal ray-for-ray.
    for (r, got) in batch.iter().zip(hits.iter()) {
        let cpu = bvh.raycast_nearest(&g, r, 500.0, &filter);
        assert_eq!(
            got.as_ref().map(|h| (h.leaf, h.time_of_impact)),
            cpu.as_ref().map(|h| (h.leaf, h.time_of_impact)),
        );
    }
}
