//! Find the nearest sphere a ray hits.
//!
//! A scene of spheres, rendered by viewport-lib. Each frame a ray sweeps from a
//! fixed origin, `spatial-query` finds the nearest sphere it hits, and a red
//! marker jumps to the hit point. The ray is drawn as a trail of small yellow
//! dots. The same sphere centres feed both the render scene and the
//! `SphereScene` query provider, so what you see is what is queried.
//!
//! This casts a world-space ray rather than a ray through the cursor, so it
//! needs no camera unprojection.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Bvh, QueryFilter, Ray};
use spatial_query_viewport_examples::{sweep_ray, to_point, to_vec3, SphereScene};
use viewport_lib::{primitives, AppConfig, Material, NodeId, ViewportApp};

const RAY_DOTS: usize = 30;
const RAY_LENGTH: f32 = 24.0;
const MAX_TOI: f32 = 100.0;

fn main() {
    let scene = Rc::new(SphereScene::scatter(60, 0x00C0_FFEE));
    let bvh = Rc::new(Bvh::build(&*scene));
    let dots: Rc<RefCell<Vec<NodeId>>> = Rc::new(RefCell::new(Vec::new()));
    let marker: Rc<Cell<Option<NodeId>>> = Rc::new(Cell::new(None));

    let setup_scene = scene.clone();
    let setup_dots = dots.clone();
    let setup_marker = marker.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : ray_pick")
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
        // The queried spheres, unit mesh scaled per radius.
        for i in 0..setup_scene.len() {
            let c = to_vec3(setup_scene.centers[i]);
            let r = setup_scene.radii[i];
            sc.add(
                Some(ball_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(r), Quat::IDENTITY, c),
                Material::from_colour([0.72, 0.74, 0.78]),
            );
        }
        // Ray trail dots (yellow), positioned per frame.
        for _ in 0..RAY_DOTS {
            let id = sc.add(
                Some(dot_mesh),
                Mat4::from_scale_rotation_translation(
                    Vec3::splat(0.05),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 0.0, -1000.0),
                ),
                Material::from_colour([0.95, 0.85, 0.15]),
            );
            setup_dots.borrow_mut().push(id);
        }
        // Hit marker (red).
        let m = sc.add(
            Some(ball_mesh),
            Mat4::from_scale_rotation_translation(
                Vec3::splat(0.35),
                Quat::IDENTITY,
                Vec3::new(0.0, 0.0, -1000.0),
            ),
            Material::from_colour([0.9, 0.15, 0.15]),
        );
        setup_marker.set(Some(m));

        session.camera_mut().distance = 18.0;
    })
    .run(move |ctx| {
        let (origin, dir) = sweep_ray(ctx.time());
        let ray = Ray::new(to_point(origin), to_point(dir));

        let hit = bvh.raycast_nearest(&*scene, &ray, MAX_TOI, &QueryFilter::default());

        let sc = ctx.scene_mut();

        // Lay the ray trail evenly from the origin to the hit. With no hit, draw
        // the ray at a fixed length so it stays visible passing through.
        let seg = hit.as_ref().map(|h| h.time_of_impact).unwrap_or(RAY_LENGTH);
        let spacing = seg / RAY_DOTS as f32;
        let dots = dots.borrow();
        for (i, &id) in dots.iter().enumerate() {
            let pos = origin + dir * ((i as f32 + 1.0) * spacing);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.08), Quat::IDENTITY, pos),
            );
        }

        if let Some(m) = marker.get() {
            let pos = hit
                .as_ref()
                .map(|h| to_vec3(h.point))
                .unwrap_or(Vec3::new(0.0, 0.0, -1000.0));
            sc.set_local_transform(
                m,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.4), Quat::IDENTITY, pos),
            );
        }
    });
}
