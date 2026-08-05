//! Highlight every object a moving box overlaps.
//!
//! A scene of spheres, rendered by viewport-lib. A translucent query box drifts
//! through the cloud, and each frame `spatial-query` reports every sphere whose
//! geometry it overlaps; those spheres turn orange while the rest stay grey. The
//! box motion is scripted (no cursor dragging) so the demo needs no camera
//! unprojection.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Aabb, Bvh, QueryFilter};
use spatial_query_viewport_examples::{to_point, to_vec3, SphereScene};
use viewport_lib::{primitives, AppConfig, ItemSettings, Material, NodeId, ViewportApp};

const BOX_HALF: Vec3 = Vec3::new(2.2, 1.6, 1.8);
const BASE_COLOUR: [f32; 3] = [0.62, 0.64, 0.68];
const HIT_COLOUR: [f32; 3] = [0.95, 0.55, 0.15];

fn main() {
    let scene = Rc::new(SphereScene::scatter(60, 0x00C0_FFEE));
    let bvh = Rc::new(Bvh::build(&*scene));
    let state: Rc<RefCell<State>> = Rc::new(RefCell::new(State::default()));

    let setup_scene = scene.clone();
    let setup_state = state.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : shape_cast_overlap")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let ball_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 20, 12))
            .unwrap();
        let cube_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::cube(1.0))
            .unwrap();

        let mut st = setup_state.borrow_mut();
        let sc = session.scene_mut();

        for i in 0..setup_scene.len() {
            let c = to_vec3(setup_scene.centers[i]);
            let r = setup_scene.radii[i];
            st.spheres.push(sc.add(
                Some(ball_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(r), Quat::IDENTITY, c),
                Material::from_colour(BASE_COLOUR),
            ));
        }

        // The query box, drawn as a wireframe so the spheres inside stay visible.
        // The unit cube spans -0.5 to 0.5, so the scale is the full box size.
        let query_box = sc.add(
            Some(cube_mesh),
            Mat4::from_scale_rotation_translation(BOX_HALF * 2.0, Quat::IDENTITY, Vec3::ZERO),
            Material::from_colour([0.35, 0.7, 0.95]),
        );
        let mut box_look = ItemSettings::default();
        box_look.wireframe = true;
        sc.set_appearance(query_box, box_look);
        st.query_box = Some(query_box);

        session.camera_mut().distance = 18.0;
    })
    .run(move |ctx| {
        let time = ctx.time();
        // Drift the box centre on a slow Lissajous path through the cloud.
        let center = Vec3::new(
            3.2 * (time * 0.6).sin(),
            2.0 * (time * 0.9).cos(),
            2.4 * (time * 0.45).sin(),
        );
        let region = Aabb::new(to_point(center - BOX_HALF), to_point(center + BOX_HALF));

        let hits = bvh.overlap(&*scene, &region, &QueryFilter::default());
        let mut overlapped = vec![false; scene.len()];
        for h in &hits {
            overlapped[h.leaf] = true;
        }

        let st = state.borrow();
        let sc = ctx.scene_mut();
        if let Some(b) = st.query_box {
            sc.set_local_transform(
                b,
                Mat4::from_scale_rotation_translation(BOX_HALF * 2.0, Quat::IDENTITY, center),
            );
        }
        for (leaf, &id) in st.spheres.iter().enumerate() {
            let colour = if overlapped[leaf] {
                HIT_COLOUR
            } else {
                BASE_COLOUR
            };
            sc.set_material(id, Material::from_colour(colour));
        }
    });
}

#[derive(Default)]
struct State {
    spheres: Vec<NodeId>,
    query_box: Option<NodeId>,
}
