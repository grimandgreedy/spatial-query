//! Watch the cost model choose a backend as the batch shape changes.
//!
//! A scene of spheres, rendered by viewport-lib, with a sweeping ray marking the
//! nearest hit so the window is live. An overlay box in the corner is coloured by
//! the backend the cost model picks and labelled with the choice, and the example
//! cycles through three query shapes, printing on each change what every backend
//! would cost and which one the `Dispatcher` auto-selects:
//!
//! - a single cursor click (one ray, one view direction),
//! - a rectangle select (many rays, still one view direction),
//! - a scattered batch (many rays, many directions, e.g. ambient occlusion).
//!
//! The draw-pass and GPU backends here are stubs that delegate to the CPU tree,
//! so their results agree with CPU but their timing is not meaningful yet (the
//! real backends land with the renderer adapter and the compute feature). What
//! is meaningful is the selection and the cost estimates that drive it, plus the
//! measured CPU time for the one backend that really runs.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{
    BackendKind, BatchHits, Bvh, Dispatcher, Point, QueryBackend, QueryContext, QueryFilter, Ray,
};
use spatial_query_viewport_examples::{sweep_ray, to_point, to_vec3, SphereScene};
use viewport_lib::{
    primitives, AppConfig, LabelAnchor, LabelAnchorY, LabelItem, Material, NodeId, OverlayFill,
    OverlayShape, OverlayShapeItem, ViewportApp,
};

const SPHERES: usize = 120;
const MAX_TOI: f32 = 100.0;
const RAY_DOTS: usize = 30;
const RAY_LENGTH: f32 = 24.0;
const HIDDEN: Vec3 = Vec3::new(0.0, 0.0, -10000.0);

fn main() {
    let scene = Rc::new(SphereScene::scatter(SPHERES, 0x0DDB_A110));
    let bvh = Rc::new(Bvh::build(&*scene));
    let state: Rc<RefCell<State>> = Rc::new(RefCell::new(State::default()));

    let setup_scene = scene.clone();
    let setup_state = state.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : backend_dispatch")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let ball = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 16, 10))
            .unwrap();
        let dot = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 8, 6))
            .unwrap();

        let mut st = setup_state.borrow_mut();
        let sc = session.scene_mut();
        for i in 0..setup_scene.len() {
            let c = to_vec3(setup_scene.centers[i]);
            let r = setup_scene.radii[i];
            sc.add(
                Some(ball),
                Mat4::from_scale_rotation_translation(Vec3::splat(r), Quat::IDENTITY, c),
                Material::from_colour([0.72, 0.74, 0.78]),
            );
        }
        for _ in 0..RAY_DOTS {
            st.dots.push(sc.add(
                Some(dot),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.06), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.95, 0.85, 0.15]),
            ));
        }
        st.marker = Some(sc.add(
            Some(dot),
            Mat4::from_scale_rotation_translation(Vec3::splat(0.35), Quat::IDENTITY, HIDDEN),
            Material::from_colour([0.9, 0.15, 0.15]),
        ));
        st.last_scenario = usize::MAX;

        session.camera_mut().distance = 16.0;
    })
    .run(move |ctx| {
        // A dispatcher over the scene, rebuilt each frame (cheap): CPU plus stub
        // draw-pass and GPU backends. The stubs borrow the same tree.
        let bvh_ref: &Bvh<3> = &bvh;
        let mut dispatcher = Dispatcher::new(bvh_ref);
        dispatcher.register(Box::new(StubDrawPass { bvh: bvh_ref }));
        dispatcher.register(Box::new(StubGpu { bvh: bvh_ref }));

        // Cycle a scenario every couple of seconds and report on each change.
        let scenario = ((ctx.time() / 2.0) as usize) % SCENARIOS.len();
        let mut st = state.borrow_mut();
        if scenario != st.last_scenario {
            st.last_scenario = scenario;
            report(&dispatcher, &scene, &bvh, scenario);
        }
        drop(st);

        // On-screen indicator: a box coloured by the backend the cost model picks
        // for the current scenario, with the choice labelled inside it.
        let (label, batch, dirs) = SCENARIOS[scenario];
        let picked =
            dispatcher.select(&QueryContext::new(batch, dirs, scene.len()).with_device(true));
        {
            let ov = ctx.overlays_mut();
            ov.shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect {
                        corner_radius: 10.0,
                    },
                    [20.0, 20.0],
                    [300.0, 64.0],
                )
                .with_fill(OverlayFill::Solid(backend_colour(picked)))
                .with_border([1.0, 1.0, 1.0, 0.9], 1.5),
            );
            ov.labels.push(
                LabelItem::new(format!("auto: {picked:?}"))
                    .with_screen_anchor([170.0, 44.0])
                    .with_align_x(LabelAnchor::Middle)
                    .with_align_y(LabelAnchorY::Middle)
                    .with_colour([1.0, 1.0, 1.0, 1.0])
                    .with_font_size(22.0),
            );
            ov.labels.push(
                LabelItem::new(format!("{label}  ({batch} rays, {dirs} dirs)"))
                    .with_screen_anchor([170.0, 68.0])
                    .with_align_x(LabelAnchor::Middle)
                    .with_align_y(LabelAnchorY::Middle)
                    .with_colour([0.95, 0.95, 0.95, 0.9])
                    .with_font_size(13.0),
            );
        }

        // The live visual: sweep a ray, mark the nearest hit.
        let (origin, dir) = sweep_ray(ctx.time());
        let ray = Ray::new(to_point(origin), to_point(dir));
        let hit = bvh.raycast_nearest(&*scene, &ray, MAX_TOI, &QueryFilter::default());

        let st = state.borrow();
        let sc = ctx.scene_mut();
        let seg = hit.as_ref().map(|h| h.time_of_impact).unwrap_or(RAY_LENGTH);
        let spacing = seg / RAY_DOTS as f32;
        for (i, &id) in st.dots.iter().enumerate() {
            let pos = origin + dir * ((i as f32 + 1.0) * spacing);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.06), Quat::IDENTITY, pos),
            );
        }
        if let Some(m) = st.marker {
            let pos = hit.as_ref().map(|h| to_vec3(h.point)).unwrap_or(HIDDEN);
            sc.set_local_transform(
                m,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.35), Quat::IDENTITY, pos),
            );
        }
    });
}

#[derive(Default)]
struct State {
    dots: Vec<NodeId>,
    marker: Option<NodeId>,
    last_scenario: usize,
}

/// The overlay-box colour for each backend.
fn backend_colour(kind: BackendKind) -> [f32; 4] {
    match kind {
        BackendKind::Cpu => [0.28, 0.52, 0.92, 0.85],
        BackendKind::DrawPass => [0.24, 0.68, 0.40, 0.85],
        BackendKind::Gpu => [0.95, 0.55, 0.15, 0.85],
    }
}

/// The three query shapes the example cycles through: `(label, batch_size,
/// distinct_directions)`.
const SCENARIOS: [(&str, usize, usize); 3] = [
    ("cursor click", 1, 1),
    ("rectangle select", 4000, 1),
    ("scattered AO batch", 4000, 4000),
];

/// Print the per-backend cost estimates, the auto-selected backend, and the
/// measured CPU time for one scenario.
fn report(
    dispatcher: &Dispatcher<3, SphereScene>,
    scene: &SphereScene,
    bvh: &Bvh<3>,
    scenario: usize,
) {
    let (label, batch, directions) = SCENARIOS[scenario];
    let ctx = QueryContext::new(batch, directions, scene.len()).with_device(true);

    // Cost the three backends the same way the dispatcher does.
    let cpu = StubCpuCost.estimate(&ctx);
    let draw = StubDrawPass { bvh }.estimate_cost(&ctx);
    let gpu = StubGpu { bvh }.estimate_cost(&ctx);
    let picked = dispatcher.select(&ctx);

    // Measure the one backend that really runs, over a batch of that size.
    let rays: Vec<Ray<3>> = (0..batch)
        .map(|k| {
            let a = k as f32 * 0.013;
            let origin = Point([12.0 * a.cos(), 12.0 * a.sin(), 3.0 * (a * 0.5).sin()]);
            Ray::new(origin, Point::ZERO - origin)
        })
        .collect();
    let t = Instant::now();
    for r in &rays {
        let _ = bvh.raycast_nearest(scene, r, MAX_TOI, &QueryFilter::default());
    }
    let cpu_us = t.elapsed().as_micros();

    println!(
        "{label:<19} | batch {batch:>5}, dirs {directions:>5} | cost cpu {cpu:>10.0} draw {draw:>12.0} gpu {gpu:>10.0} | auto: {picked:?} | cpu ran in {cpu_us} us",
    );
}

// ---------------------------------------------------------------------------
// Stub backends: they delegate to the CPU tree for correctness, and each
// declares its own cost shape and availability. They stand in for the real
// draw-pass picker (a renderer adapter) and GPU-compute backend.
// ---------------------------------------------------------------------------

/// Mirrors `CpuBackend`'s cost so the report can show it next to the others.
struct StubCpuCost;
impl StubCpuCost {
    fn estimate(&self, ctx: &QueryContext) -> f64 {
        let per_ray = ((ctx.scene_leaves + 1) as f64).log2().max(1.0);
        ctx.batch_size as f64 * per_ray
    }
}

struct StubDrawPass<'a> {
    bvh: &'a Bvh<3>,
}
impl QueryBackend<3, SphereScene> for StubDrawPass<'_> {
    fn kind(&self) -> BackendKind {
        BackendKind::DrawPass
    }
    fn can_serve(&self, ctx: &QueryContext) -> bool {
        ctx.device_available
    }
    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        (ctx.distinct_directions.saturating_sub(1)) as f64 * ctx.scene_leaves as f64 + 1.0
    }
    fn raycast_nearest_batch(
        &self,
        g: &SphereScene,
        rays: &[Ray<3>],
        max_toi: f32,
        filter: &QueryFilter,
    ) -> BatchHits<usize, 3, ()> {
        rays.iter()
            .map(|r| self.bvh.raycast_nearest(g, r, max_toi, filter))
            .collect()
    }
}

struct StubGpu<'a> {
    bvh: &'a Bvh<3>,
}
impl QueryBackend<3, SphereScene> for StubGpu<'_> {
    fn kind(&self) -> BackendKind {
        BackendKind::Gpu
    }
    fn can_serve(&self, ctx: &QueryContext) -> bool {
        ctx.device_available
    }
    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        ctx.scene_leaves as f64 + ctx.batch_size as f64
    }
    fn raycast_nearest_batch(
        &self,
        g: &SphereScene,
        rays: &[Ray<3>],
        max_toi: f32,
        filter: &QueryFilter,
    ) -> BatchHits<usize, 3, ()> {
        rays.iter()
            .map(|r| self.bvh.raycast_nearest(g, r, max_toi, filter))
            .collect()
    }
}
