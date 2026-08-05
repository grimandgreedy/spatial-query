//! The case that motivated the crate: tens of thousands of arbitrary rays from
//! many directions at once, where the GPU-compute backend wins and per-ray
//! render passes would not.
//!
//! A grid of points on the floor computes a soft shadow cast by a sphere cloud.
//! Each floor point fires a fan of rays toward points spread across an area
//! light; the fraction of those rays the cloud blocks is how deep in shadow that
//! point is. Blocked points turn pink, so the cloud's soft-edged silhouette is
//! painted onto the floor. The light orbits the scene, so the whole shadow
//! sweeps around and the entire ray batch is rebuilt and re-answered every frame.
//!
//! The batch runs every frame on both the CPU (the fixed-order parallel batch)
//! and the GPU backend; the overlay reports each batch's time, and toggling the
//! display between them shows the results are the same.
//!
//! Run with the GPU backend enabled:
//!
//! ```text
//! cargo run -p spatial-query-viewport-examples --example many_rays --features gpu
//! ```
//!
//! If no GPU adapter is present the example still runs, CPU-only.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use spatial_query::gpu::GpuRaycaster;
use spatial_query::{Bvh, QueryFilter, Ray};
use spatial_query_viewport_examples::{to_point, to_vec3, SphereScene};
use viewport_lib::{
    primitives, AppConfig, ItemSettings, LabelAnchor, LabelItem, Material, NodeId, OverlayFill,
    OverlayShape, OverlayShapeItem, ViewportApp,
};

const SPHERES: usize = 160; // the occluder cloud
const GRID: usize = 96; // GRID x GRID floor points
const SPAN: f32 = 21.0; // floor half-extent (wide, to catch the sweeping shadow)
const PLANE_Z: f32 = -4.0; // floor height (z-up world)
const DOT_SIZE: f32 = 0.18; // sized to the grid spacing
const LIGHT_ORBIT: f32 = 16.0; // radius of the sun's circular path
const LIGHT_HEIGHT: f32 = 9.0; // sun height above the origin
const LIGHT_SPEED: f32 = 0.5; // radians per second
const LIGHT_RADIUS: f32 = 2.2; // area light: bigger radius, softer penumbra
const LIGHT_SAMPLES: usize = 20; // rays per floor point (toward the light's area)
const MAX_TOI: f32 = 80.0;
const TOGGLE_SECS: f32 = 2.5;

fn main() {
    let scene = Rc::new(SphereScene::scatter(SPHERES, 0x0DDB_A110));
    let bvh = Rc::new(Bvh::build(&*scene));
    let samples = floor_grid();

    let gpu = GpuRaycaster::headless(&bvh, &*scene);
    println!(
        "many_rays: {} floor points x {} light rays = {} rays/batch over {} spheres",
        samples.len(),
        LIGHT_SAMPLES,
        samples.len() * LIGHT_SAMPLES,
        SPHERES,
    );
    if gpu.is_none() {
        println!("no GPU adapter: running CPU-only");
    }

    // One-shot check at the light's starting position: time both backends, report
    // the shadow spread and how many points the two backends shade identically.
    let filter = QueryFilter::default();
    let rays0 = build_rays(&samples, &light_samples(light_center(0.0)));
    let cpu_lit = {
        let t = Instant::now();
        let hits = bvh.raycast_nearest_batch(&*scene, &rays0, MAX_TOI, &filter);
        let us = t.elapsed().as_micros();
        let lit = lit_fraction(&hits.iter().map(Option::is_some).collect::<Vec<_>>());
        let min = lit.iter().copied().fold(f32::INFINITY, f32::min);
        let max = lit.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("cpu batch: {us} us  |  lit fraction min {min:.2} max {max:.2}");
        lit
    };
    if let Some(g) = &gpu {
        let t = Instant::now();
        let hits = g.raycast_nearest_batch(&rays0, MAX_TOI);
        let us = t.elapsed().as_micros();
        let gpu_lit = lit_fraction(&hits.iter().map(Option::is_some).collect::<Vec<_>>());
        let agree = cpu_lit
            .iter()
            .zip(&gpu_lit)
            .filter(|(a, b)| (*a - *b).abs() < 1.0e-6)
            .count();
        println!(
            "gpu batch: {us} us  |  shadings agree on {agree}/{} points",
            cpu_lit.len()
        );
    }

    let state: Rc<RefCell<State>> = Rc::new(RefCell::new(State {
        dots: Vec::new(),
        light: None,
    }));

    let setup_scene = scene.clone();
    let setup_state = state.clone();
    let run_scene = scene.clone();
    let run_bvh = bvh.clone();
    let run_state = state.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : many_rays")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let ball = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 16, 10))
            .unwrap();
        let dot = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 6, 4))
            .unwrap();

        let sc = session.scene_mut();
        // The occluder: an opaque sphere cloud.
        for i in 0..setup_scene.len() {
            let c = to_vec3(setup_scene.centers[i]);
            let r = setup_scene.radii[i];
            sc.add(
                Some(ball),
                Mat4::from_scale_rotation_translation(Vec3::splat(r), Quat::IDENTITY, c),
                Material::from_colour([0.62, 0.64, 0.70]),
            );
        }

        let mut st = setup_state.borrow_mut();
        // The orbiting light, an unlit marker so its direction is obvious.
        let light = sc.add(
            Some(ball),
            Mat4::from_scale_rotation_translation(
                Vec3::splat(LIGHT_RADIUS),
                Quat::IDENTITY,
                light_center(0.0),
            ),
            Material::from_colour([1.0, 0.93, 0.5]),
        );
        set_unlit(sc, light);
        st.light = Some(light);

        // The floor field. Unlit, so each dot's colour is exactly its shadow
        // value, not shaded by scene lighting.
        for &p in &floor_grid() {
            let id = sc.add(
                Some(dot),
                Mat4::from_scale_rotation_translation(Vec3::splat(DOT_SIZE), Quat::IDENTITY, p),
                Material::from_colour([0.5, 0.5, 0.5]),
            );
            set_unlit(sc, id);
            st.dots.push(id);
        }

        session.camera_mut().distance = 52.0;
    })
    .run(move |ctx| {
        let filter = QueryFilter::default();

        // The light orbits, so rebuild the whole ray batch aimed at where it is
        // now, then answer it fresh.
        let center = light_center(ctx.time());
        let rays = build_rays(&samples, &light_samples(center));

        // CPU (fixed-order parallel), timed.
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let t = Instant::now();
        let cpu_hits =
            run_bvh.raycast_nearest_batch_parallel(&*run_scene, &rays, MAX_TOI, &filter, threads);
        let cpu_us = t.elapsed().as_micros();

        // GPU, if present, timed.
        let mut gpu_us = 0;
        let gpu_hits = gpu.as_ref().map(|g| {
            let t = Instant::now();
            let h = g.raycast_nearest_batch(&rays, MAX_TOI);
            gpu_us = t.elapsed().as_micros();
            h
        });

        // Which backend's result is shown; toggles every few seconds.
        let source = if gpu.is_some() && ((ctx.time() / TOGGLE_SECS) as u64) % 2 == 1 {
            Source::Gpu
        } else {
            Source::Cpu
        };
        let mask: Vec<bool> = match source {
            Source::Gpu => gpu_hits
                .as_ref()
                .unwrap()
                .iter()
                .map(Option::is_some)
                .collect(),
            Source::Cpu => cpu_hits.iter().map(Option::is_some).collect(),
        };
        let lit = lit_fraction(&mask);

        // The shadow moves every frame, so recolour the whole field and move the
        // light marker.
        let st = run_state.borrow();
        let dots = st.dots.clone();
        let light = st.light;
        drop(st);
        let sc = ctx.scene_mut();
        for (&id, &l) in dots.iter().zip(&lit) {
            sc.set_material(id, Material::from_colour(shade(l)));
        }
        if let Some(light) = light {
            sc.set_local_transform(
                light,
                Mat4::from_scale_rotation_translation(
                    Vec3::splat(LIGHT_RADIUS),
                    Quat::IDENTITY,
                    center,
                ),
            );
        }

        // Overlay: which backend is shown, and both batch times.
        let colour = match source {
            Source::Cpu => [0.28, 0.52, 0.92, 0.85],
            Source::Gpu => [0.95, 0.55, 0.15, 0.85],
        };
        let ov = ctx.overlays_mut();
        ov.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect {
                    corner_radius: 10.0,
                },
                [20.0, 20.0],
                [380.0, 78.0],
            )
            .with_fill(OverlayFill::Solid(colour))
            .with_border([1.0, 1.0, 1.0, 0.9], 1.5),
        );
        ov.labels.push(
            LabelItem::new(format!("shading from: {}", source.name()))
                .with_screen_anchor([210.0, 42.0])
                .with_anchor_align(LabelAnchor::Center)
                .with_colour([1.0, 1.0, 1.0, 1.0])
                .with_font_size(22.0),
        );
        // Both backends run every frame; the one currently driving the picture is
        // bracketed. The bracket jumps cpu <-> gpu every few seconds while the
        // shadow itself stays identical -- that is the point.
        let cpu_txt = format!("cpu {:.2} ms", cpu_us as f32 / 1000.0);
        let gpu_txt = if gpu_us == 0 {
            "gpu n/a".to_string()
        } else {
            format!("gpu {:.2} ms", gpu_us as f32 / 1000.0)
        };
        let (cpu_txt, gpu_txt) = match source {
            Source::Cpu => (format!("[ {cpu_txt} ]"), gpu_txt),
            Source::Gpu => (cpu_txt, format!("[ {gpu_txt} ]")),
        };
        ov.labels.push(
            LabelItem::new(format!("{} rays   {cpu_txt}   {gpu_txt}", rays.len()))
                .with_screen_anchor([210.0, 70.0])
                .with_anchor_align(LabelAnchor::Center)
                .with_colour([0.97, 0.97, 0.97, 0.95])
                .with_font_size(13.0),
        );
    });
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Cpu,
    Gpu,
}

impl Source {
    fn name(self) -> &'static str {
        match self {
            Source::Cpu => "CPU (parallel batch)",
            Source::Gpu => "GPU (compute batch)",
        }
    }
}

struct State {
    dots: Vec<NodeId>,
    light: Option<NodeId>,
}

/// Mark a node unlit, so its material colour renders exactly as set.
fn set_unlit(sc: &mut viewport_lib::Scene, id: NodeId) {
    let mut look = ItemSettings::default();
    look.unlit = true;
    sc.set_appearance(id, look);
}

/// The centre of the area light at time `t`: a circular orbit at fixed height.
fn light_center(t: f32) -> Vec3 {
    let a = t * LIGHT_SPEED;
    Vec3::new(LIGHT_ORBIT * a.cos(), LIGHT_ORBIT * a.sin(), LIGHT_HEIGHT)
}

/// The GRID x GRID floor points at `PLANE_Z`.
fn floor_grid() -> Vec<Vec3> {
    let mut out = Vec::with_capacity(GRID * GRID);
    for iy in 0..GRID {
        for ix in 0..GRID {
            let fx = ix as f32 / (GRID - 1) as f32;
            let fy = iy as f32 / (GRID - 1) as f32;
            out.push(Vec3::new(
                (fx - 0.5) * 2.0 * SPAN,
                (fy - 0.5) * 2.0 * SPAN,
                PLANE_Z,
            ));
        }
    }
    out
}

/// `LIGHT_SAMPLES` points spread over the area light (a small sphere of radius
/// `LIGHT_RADIUS` around `center`), via a Fibonacci sphere. The spread is what
/// softens the shadow edge.
fn light_samples(center: Vec3) -> Vec<Vec3> {
    const GOLDEN: f32 = 2.399_963_2; // pi * (3 - sqrt 5)
    (0..LIGHT_SAMPLES)
        .map(|k| {
            let z = 1.0 - 2.0 * (k as f32 + 0.5) / LIGHT_SAMPLES as f32;
            let r = (1.0 - z * z).sqrt();
            let phi = k as f32 * GOLDEN;
            center + Vec3::new(r * phi.cos(), r * phi.sin(), z) * LIGHT_RADIUS
        })
        .collect()
}

/// One flat batch: floor point p, light sample k -> ray index p * K + k.
fn build_rays(samples: &[Vec3], lights: &[Vec3]) -> Vec<Ray<3>> {
    samples
        .iter()
        .flat_map(|&p| {
            lights
                .iter()
                .map(move |&l| Ray::new(to_point(p), to_point(l - p)))
        })
        .collect()
}

/// Per floor point, the fraction of its light rays that reach the light: 1 fully
/// lit (nothing blocks), 0 fully shadowed (every ray blocked).
fn lit_fraction(mask: &[bool]) -> Vec<f32> {
    (0..mask.len() / LIGHT_SAMPLES)
        .map(|s| {
            let blocked = mask[s * LIGHT_SAMPLES..(s + 1) * LIGHT_SAMPLES]
                .iter()
                .filter(|&&h| h)
                .count();
            1.0 - blocked as f32 / LIGHT_SAMPLES as f32
        })
        .collect()
}

/// Warm where lit, vivid pink where shadowed, blending through the penumbra.
fn shade(lit: f32) -> [f32; 3] {
    let shadow = (1.0 - lit).clamp(0.0, 1.0);
    let litc = [0.90, 0.86, 0.55]; // reaches the light
    let pink = [1.0, 0.12, 0.62]; // blocked from the light
    [
        litc[0] + (pink[0] - litc[0]) * shadow,
        litc[1] + (pink[1] - litc[1]) * shadow,
        litc[2] + (pink[2] - litc[2]) * shadow,
    ]
}
