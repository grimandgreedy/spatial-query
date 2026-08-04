//! Every sphere a ray pierces, in order.
//!
//! A sweeping ray, but instead of the nearest hit this queries all hits along
//! the ray. The ray is drawn as a trail of yellow dots, and a green marker is
//! parked at each pierced sphere's entry point. The console prints the hit order
//! (by leaf id and distance) whenever the set of pierced spheres changes. This
//! uses `Bvh::raycast_all`, which returns hits sorted nearest-first with a fixed
//! tie-break.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Bvh, QueryFilter, Ray};
use spatial_query_viewport_examples::{sweep_ray, to_point, to_vec3, SphereScene};
use viewport_lib::{primitives, AppConfig, Material, NodeId, ViewportApp};

const RAY_DOTS: usize = 40;
const RAY_LENGTH: f32 = 24.0;
const MAX_MARKERS: usize = 24;
const MAX_TOI: f32 = 100.0;
const HIDDEN: Vec3 = Vec3::new(0.0, 0.0, -1000.0);

fn main() {
    let scene = Rc::new(SphereScene::scatter(60, 0x00C0_FFEE));
    let bvh = Rc::new(Bvh::build(&*scene));
    let dots: Rc<RefCell<Vec<NodeId>>> = Rc::new(RefCell::new(Vec::new()));
    let markers: Rc<RefCell<Vec<NodeId>>> = Rc::new(RefCell::new(Vec::new()));
    let last_count: Rc<Cell<usize>> = Rc::new(Cell::new(usize::MAX));

    let setup_scene = scene.clone();
    let setup_dots = dots.clone();
    let setup_markers = markers.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : ray_all_hits")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let ball_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 20, 12))
            .unwrap();
        let dot_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 8, 6))
            .unwrap();

        let sc = session.scene_mut();
        for i in 0..setup_scene.len() {
            let c = to_vec3(setup_scene.centers[i]);
            let r = setup_scene.radii[i];
            sc.add(
                Some(ball_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(r), Quat::IDENTITY, c),
                Material::from_colour([0.72, 0.74, 0.78]),
            );
        }
        for _ in 0..RAY_DOTS {
            let id = sc.add(
                Some(dot_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.08), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.95, 0.85, 0.15]),
            );
            setup_dots.borrow_mut().push(id);
        }
        for _ in 0..MAX_MARKERS {
            let id = sc.add(
                Some(ball_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.25), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.2, 0.85, 0.35]),
            );
            setup_markers.borrow_mut().push(id);
        }

        session.camera_mut().distance = 18.0;
    })
    .run(move |ctx| {
        let (origin, dir) = sweep_ray(ctx.time());
        let ray = Ray::new(to_point(origin), to_point(dir));

        let hits = bvh.raycast_all(&*scene, &ray, MAX_TOI, &QueryFilter::default());

        if hits.len() != last_count.get() {
            last_count.set(hits.len());
            let order: Vec<String> = hits
                .iter()
                .map(|h| format!("#{} @ {:.2}", h.leaf, h.time_of_impact))
                .collect();
            println!("{} hits: {}", hits.len(), order.join(", "));
        }

        let sc = ctx.scene_mut();

        // The ray, drawn full length so it is always visible sweeping through.
        let spacing = RAY_LENGTH / RAY_DOTS as f32;
        for (i, &id) in dots.borrow().iter().enumerate() {
            let pos = origin + dir * ((i as f32 + 1.0) * spacing);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.08), Quat::IDENTITY, pos),
            );
        }

        // A green marker at each pierced sphere; unused markers park off screen.
        for (i, &id) in markers.borrow().iter().enumerate() {
            let pos = hits.get(i).map(|h| to_vec3(h.point)).unwrap_or(HIDDEN);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.25), Quat::IDENTITY, pos),
            );
        }
    });
}
