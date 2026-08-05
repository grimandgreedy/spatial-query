//! Sweep a sphere and stop it at the first surface it touches.
//!
//! A scene of spheres, rendered by viewport-lib. Each frame a probe sphere is
//! swept along a sweeping direction; `spatial-query` finds the first scene
//! sphere it touches, and a translucent ghost sphere parks at that contact
//! placement. The contact normal is drawn as a short trail of dots leaving the
//! surface. The sweep path is drawn as yellow dots from the start to the ghost.
//!
//! This casts a world-space sweep rather than one aimed through the cursor, so
//! it needs no camera unprojection.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Aabb, Bvh, Point, QueryFilter, ShapeCast};
use spatial_query_viewport_examples::{sweep_ray, to_point, to_vec3, SphereScene};
use viewport_lib::{primitives, AlphaMode, AppConfig, ItemSettings, Material, NodeId, ViewportApp};

const SWEEP_DOTS: usize = 30;
const SWEEP_LENGTH: f32 = 24.0;
const NORMAL_DOTS: usize = 6;
const PROBE_RADIUS: f32 = 0.9;
const MAX_TOI: f32 = 100.0;
const HIDDEN: Vec3 = Vec3::new(0.0, 0.0, -10000.0);

fn main() {
    let scene = Rc::new(SphereScene::scatter(60, 0x00C0_FFEE));
    let bvh = Rc::new(Bvh::build(&*scene));
    let nodes: Rc<RefCell<Nodes>> = Rc::new(RefCell::new(Nodes::default()));

    let setup_scene = scene.clone();
    let setup_nodes = nodes.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : shape_cast")
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

        let mut n = setup_nodes.borrow_mut();
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
        // Sweep-path trail dots.
        for _ in 0..SWEEP_DOTS {
            n.sweep.push(sc.add(
                Some(dot_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.06), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.95, 0.85, 0.15]),
            ));
        }
        // Contact-normal trail dots.
        for _ in 0..NORMAL_DOTS {
            n.normal.push(sc.add(
                Some(dot_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.05), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.2, 0.9, 0.35]),
            ));
        }
        // The swept probe, drawn as a translucent ghost that parks at contact.
        let mut ghost_mat = Material::from_colour([0.95, 0.35, 0.2]);
        ghost_mat.alpha_mode = AlphaMode::Blend;
        let ghost = sc.add(
            Some(ball_mesh),
            Mat4::from_scale_rotation_translation(
                Vec3::splat(PROBE_RADIUS),
                Quat::IDENTITY,
                HIDDEN,
            ),
            ghost_mat,
        );
        let mut ghost_look = ItemSettings::default();
        ghost_look.opacity = 0.35;
        sc.set_appearance(ghost, ghost_look);
        n.ghost = Some(ghost);

        session.camera_mut().distance = 18.0;
    })
    .run(move |ctx| {
        let (origin, dir) = sweep_ray(ctx.time());
        let probe = Aabb::new(Point::splat(-PROBE_RADIUS), Point::splat(PROBE_RADIUS));
        let cast = ShapeCast::new(to_point(origin), to_point(dir), probe);

        let hit = bvh.shapecast_nearest(&*scene, &cast, MAX_TOI, &QueryFilter::default());

        let n = nodes.borrow();
        let sc = ctx.scene_mut();

        // Lay the sweep trail from the start to the contact placement. With no
        // contact, draw a fixed length so the sweep stays visible.
        let seg = hit
            .as_ref()
            .map(|h| h.time_of_impact)
            .unwrap_or(SWEEP_LENGTH);
        let spacing = seg / SWEEP_DOTS as f32;
        for (i, &id) in n.sweep.iter().enumerate() {
            let pos = origin + dir * ((i as f32 + 1.0) * spacing);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.06), Quat::IDENTITY, pos),
            );
        }

        // The ghost sphere sits where the probe centre is at contact (the hit
        // point), and the normal trail leaves the touched surface.
        let ghost_pos = hit.as_ref().map(|h| to_vec3(h.point)).unwrap_or(HIDDEN);
        if let Some(g) = n.ghost {
            sc.set_local_transform(
                g,
                Mat4::from_scale_rotation_translation(
                    Vec3::splat(PROBE_RADIUS),
                    Quat::IDENTITY,
                    ghost_pos,
                ),
            );
        }
        for (i, &id) in n.normal.iter().enumerate() {
            let pos = match hit.as_ref() {
                // The surface contact point is the probe centre pulled back along
                // the outward normal by the probe radius; step out from there.
                Some(h) => {
                    let normal = to_vec3(h.normal);
                    let surface = to_vec3(h.point) - normal * PROBE_RADIUS;
                    surface + normal * (i as f32 * 0.25)
                }
                None => HIDDEN,
            };
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.05), Quat::IDENTITY, pos),
            );
        }
    });
}

#[derive(Default)]
struct Nodes {
    sweep: Vec<NodeId>,
    normal: Vec<NodeId>,
    ghost: Option<NodeId>,
}
