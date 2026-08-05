//! One `QueryGeometry` provider spanning two kinds of thing at once: visible
//! render meshes and invisible trigger volumes, behind a single combining
//! provider with a sum-type `Id`. A swept ray selects whichever it hits first -
//! a mesh or a trigger - and the overlay names the selection.
//!
//! The point (ADR-001): combining render geometry with physics-style trigger
//! volumes in one query needs no core support. The core sees leaves; the consumer
//! decides what a leaf is and what its id means. Visible meshes are queried
//! through the `viewport-lib-query` adapter; triggers are plain AABBs; both live
//! behind `EditorScene`.
//!
//! ```text
//! cargo run -p spatial-query-viewport-examples --example editor_combining_provider
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray};
use spatial_query_viewport_examples::{sweep_ray, to_point, to_vec3};
use viewport_lib::{
    primitives, AppConfig, ItemSettings, LabelAnchor, LabelItem, Material, NodeId, OverlayFill,
    OverlayShape, OverlayShapeItem, ViewportApp,
};
use viewport_lib_query::SceneQuery;

const MESH_KEY: u64 = 1;
const CUBE: f32 = 1.6;
const MAX_TOI: f32 = 100.0;

/// The sum-type id: a query result is either a visible mesh or a trigger volume.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Selection {
    Mesh(u64),
    Trigger(u32),
}

/// A provider combining the adapter's mesh instances with trigger AABBs. Leaves
/// `0..meshes.leaf_count()` are meshes; the rest are triggers.
struct EditorScene {
    meshes: SceneQuery,
    triggers: Vec<Aabb<3>>,
}

impl EditorScene {
    fn split(&self) -> usize {
        self.meshes.leaf_count()
    }
}

impl QueryGeometry<3> for EditorScene {
    type Id = Selection;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        self.meshes.leaf_count() + self.triggers.len()
    }

    fn id(&self, leaf: usize) -> Selection {
        let split = self.split();
        if leaf < split {
            Selection::Mesh(self.meshes.id(leaf).0)
        } else {
            Selection::Trigger((leaf - split) as u32)
        }
    }

    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        let split = self.split();
        if leaf < split {
            self.meshes.world_aabb(leaf)
        } else {
            self.triggers[leaf - split]
        }
    }

    fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
        let split = self.split();
        if leaf < split {
            // Delegate to the adapter's triangle test; drop the face sub-object
            // (this provider's identity is the sum type, not a face).
            self.meshes
                .test_ray(leaf, ray, max_toi)
                .map(|lh| LeafHit::new(lh.toi, lh.normal))
        } else {
            ray_aabb(&self.triggers[leaf - split], ray, max_toi)
        }
    }
}

/// Slab ray-AABB test returning the entry hit as a `LeafHit`.
fn ray_aabb(b: &Aabb<3>, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
    let mut t_enter = 0.0f32;
    let mut t_exit = max_toi;
    let mut axis = 0usize;
    let mut sign = -1.0f32;
    for k in 0..3 {
        let inv = 1.0 / ray.dir[k];
        let mut t0 = (b.min[k] - ray.origin[k]) * inv;
        let mut t1 = (b.max[k] - ray.origin[k]) * inv;
        let mut s = -1.0;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
            s = 1.0;
        }
        if t0 > t_enter {
            t_enter = t0;
            axis = k;
            sign = s;
        }
        if t1 < t_exit {
            t_exit = t1;
        }
        if t_enter > t_exit {
            return None;
        }
    }
    if t_enter < 0.0 || t_enter > max_toi {
        return None;
    }
    let mut n = [0.0; 3];
    n[axis] = sign;
    Some(LeafHit::new(t_enter, Point(n)))
}

struct State {
    mesh_nodes: Vec<NodeId>,
    trigger_nodes: Vec<NodeId>,
    marker: Option<NodeId>,
    last: Option<Selection>,
}

fn main() {
    // A few visible cubes and a few invisible trigger boxes, all near the origin
    // so the swept ray crosses them.
    let mesh_xf = [
        Mat4::from_translation(Vec3::new(-3.0, 0.0, 1.5)),
        Mat4::from_translation(Vec3::new(3.0, 0.0, -1.0)),
        Mat4::from_scale_rotation_translation(
            Vec3::splat(1.3),
            Quat::from_rotation_z(0.6),
            Vec3::new(0.0, 3.0, 0.0),
        ),
    ];
    let triggers = vec![
        aabb_at(Vec3::new(0.0, -3.0, 0.0), 1.4),
        aabb_at(Vec3::new(-3.5, 0.0, -3.0), 1.2),
    ];

    let cube = primitives::cube(CUBE);
    let mut meshes = SceneQuery::new();
    meshes.register_mesh_data(MESH_KEY, &cube);
    for (i, xf) in mesh_xf.iter().enumerate() {
        meshes.push_instance((i + 1) as u64, MESH_KEY, *xf); // ids 1..=N
    }
    let scene = Rc::new(EditorScene { meshes, triggers });
    let bvh = Rc::new(Bvh::build(&*scene));

    // Startup check (headless, before the window): a ray into a known mesh
    // selects a Mesh; a ray into a known trigger selects a Trigger.
    {
        let filter = QueryFilter::default();
        let into_mesh = Ray::new(Point([-3.0, 0.0, 20.0]), Point([0.0, 0.0, -1.0]));
        let into_trigger = Ray::new(Point([0.0, -3.0, 20.0]), Point([0.0, 0.0, -1.0]));
        let m = bvh.raycast_nearest(&*scene, &into_mesh, MAX_TOI, &filter);
        let t = bvh.raycast_nearest(&*scene, &into_trigger, MAX_TOI, &filter);
        assert!(
            matches!(m.map(|h| h.id), Some(Selection::Mesh(_))),
            "ray hit a mesh"
        );
        assert!(
            matches!(t.map(|h| h.id), Some(Selection::Trigger(_))),
            "ray hit a trigger"
        );
        println!("editor_combining_provider: one provider selects both meshes and triggers.");
        println!("  ray into mesh column   -> {:?}", m.map(|h| h.id));
        println!("  ray into trigger volume -> {:?}", t.map(|h| h.id));
    }

    let state: Rc<RefCell<State>> = Rc::new(RefCell::new(State {
        mesh_nodes: Vec::new(),
        trigger_nodes: Vec::new(),
        marker: None,
        last: None,
    }));

    let setup_state = state.clone();
    let run_scene = scene.clone();
    let run_bvh = bvh.clone();
    let run_state = state.clone();
    let mesh_xf_setup = mesh_xf;
    let trigger_setup: Vec<Aabb<3>> = scene.triggers.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : editor_combining_provider")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let ball = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 12, 8))
            .unwrap();
        let cube_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::cube(CUBE))
            .unwrap();

        let sc = session.scene_mut();
        let mut st = setup_state.borrow_mut();
        // Visible meshes: solid grey cubes.
        for xf in &mesh_xf_setup {
            st.mesh_nodes.push(sc.add(
                Some(cube_mesh),
                *xf,
                Material::from_colour([0.62, 0.64, 0.70]),
            ));
        }
        // Trigger volumes: wireframe boxes, faint (they are "invisible" gameplay
        // volumes, shown here only so you can see what the query hits).
        for b in &trigger_setup {
            let c = to_vec3((b.min + b.max) * 0.5);
            let ext = to_vec3(b.max - b.min);
            let id = sc.add(
                Some(cube_mesh),
                Mat4::from_scale_rotation_translation(ext / CUBE, Quat::IDENTITY, c),
                Material::from_colour([0.30, 0.55, 0.95]),
            );
            let mut look = ItemSettings::default();
            look.wireframe = true;
            look.unlit = true;
            sc.set_appearance(id, look);
            st.trigger_nodes.push(id);
        }
        // Hit marker.
        st.marker = Some(sc.add(
            Some(ball),
            Mat4::from_scale_rotation_translation(
                Vec3::splat(0.25),
                Quat::IDENTITY,
                Vec3::new(0.0, 0.0, -10000.0),
            ),
            Material::from_colour([0.95, 0.2, 0.2]),
        ));
        session.camera_mut().distance = 16.0;
    })
    .run(move |ctx| {
        let filter = QueryFilter::default();
        let (origin, dir) = sweep_ray(ctx.time());
        let ray = Ray::new(to_point(origin), to_point(dir));
        let hit = run_bvh.raycast_nearest(&*run_scene, &ray, MAX_TOI, &filter);
        let selection = hit.as_ref().map(|h| h.id);

        let st = run_state.borrow();
        let mesh_nodes = st.mesh_nodes.clone();
        let trigger_nodes = st.trigger_nodes.clone();
        let marker = st.marker;
        let changed = st.last != selection;
        drop(st);

        let sc = ctx.scene_mut();
        // Recolour to show the current selection (only when it changes).
        if changed {
            for &id in &mesh_nodes {
                sc.set_material(id, Material::from_colour([0.62, 0.64, 0.70]));
            }
            for &id in &trigger_nodes {
                sc.set_material(id, Material::from_colour([0.30, 0.55, 0.95]));
            }
            match selection {
                Some(Selection::Mesh(oid)) => {
                    if let Some(&id) = mesh_nodes.get((oid as usize).wrapping_sub(1)) {
                        sc.set_material(id, Material::from_colour([0.98, 0.78, 0.20]));
                    }
                }
                Some(Selection::Trigger(ti)) => {
                    if let Some(&id) = trigger_nodes.get(ti as usize) {
                        sc.set_material(id, Material::from_colour([0.35, 0.95, 0.55]));
                    }
                }
                None => {}
            }
            run_state.borrow_mut().last = selection;
        }
        // Move the hit marker.
        if let Some(m) = marker {
            let pos = hit
                .as_ref()
                .map(|h| to_vec3(h.point))
                .unwrap_or(Vec3::new(0.0, 0.0, -10000.0));
            sc.set_local_transform(
                m,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.25), Quat::IDENTITY, pos),
            );
        }

        // Overlay: name the selection.
        let (label, colour) = match selection {
            Some(Selection::Mesh(id)) => (format!("selected: Mesh({id})"), [0.98, 0.78, 0.20, 0.9]),
            Some(Selection::Trigger(i)) => {
                (format!("selected: Trigger({i})"), [0.35, 0.95, 0.55, 0.9])
            }
            None => ("selected: nothing".to_string(), [0.4, 0.4, 0.45, 0.85]),
        };
        let ov = ctx.overlays_mut();
        ov.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect {
                    corner_radius: 10.0,
                },
                [20.0, 20.0],
                [340.0, 56.0],
            )
            .with_fill(OverlayFill::Solid(colour))
            .with_border([1.0, 1.0, 1.0, 0.9], 1.5),
        );
        ov.labels.push(
            LabelItem::new(label)
                .with_screen_anchor([190.0, 48.0])
                .with_anchor_align(LabelAnchor::Center)
                .with_colour([0.1, 0.1, 0.1, 1.0])
                .with_font_size(20.0),
        );
    });
}

/// An axis-aligned box of half-size `h` centred at `c`.
fn aabb_at(c: Vec3, h: f32) -> Aabb<3> {
    Aabb::new(
        Point([c.x - h, c.y - h, c.z - h]),
        Point([c.x + h, c.y + h, c.z + h]),
    )
}
