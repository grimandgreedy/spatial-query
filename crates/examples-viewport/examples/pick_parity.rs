//! Parity: the `viewport-lib-query` adapter resolves the same object as
//! viewport-lib's own CPU picker on the same rays.
//!
//! This is a parity check, not a claim of new capability: viewport-lib already
//! picks from arbitrary directions (`pick_scene_cpu`, parry3d-backed). The point
//! is that the spatial-query adapter agrees with it, object for object, so a
//! consumer can move a scene onto the shared engine without changing pick
//! results.
//!
//! Headless by design: no window and no GPU device. It scripts a batch of world
//! rays (the "scripted cursor positions" of the plan), asks both pickers, and
//! asserts they agree.
//!
//! ```text
//! cargo run -p spatial-query-viewport-examples --example pick_parity
//! ```

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Point, Ray};
use viewport_lib::interaction::query::picking::pick_scene_cpu;
use viewport_lib::{primitives, MeshData, ViewportObject};
use viewport_lib_query::{SceneIndex, SceneQuery};

const OBJECTS: usize = 24;
const RAYS: usize = 400;
const MESH_KEY: u64 = 1;
const MAX_TOI: f32 = 1000.0;

/// SplitMix64, so the scene and rays are reproducible without an RNG crate.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        lo + u * (hi - lo)
    }
}

/// A viewport-lib object: what its CPU picker needs. The adapter is fed the same
/// world transform, so the two see identical geometry.
struct Obj {
    id: u64,
    pos: Vec3,
    rot: Quat,
    scale: Vec3,
}

impl Obj {
    fn model(&self) -> Mat4 {
        // Must match how `pick_scene_cpu` composes it: translate * rotate * scale.
        Mat4::from_scale_rotation_translation(self.scale, self.rot, self.pos)
    }
}

impl ViewportObject for Obj {
    fn id(&self) -> u64 {
        self.id
    }
    fn mesh_id(&self) -> Option<u64> {
        Some(MESH_KEY)
    }
    fn model_matrix(&self) -> Mat4 {
        self.model()
    }
    fn position(&self) -> Vec3 {
        self.pos
    }
    fn rotation(&self) -> Quat {
        self.rot
    }
    fn scale(&self) -> Vec3 {
        self.scale
    }
    fn is_visible(&self) -> bool {
        true
    }
    fn colour(&self) -> Vec3 {
        Vec3::splat(0.7)
    }
}

fn main() {
    let cube: MeshData = primitives::cube(1.6);
    let mut rng = Rng(0x5CA1_AB1E_D0D0_1234);

    // A scatter of rotated, scaled cubes.
    let objects: Vec<Obj> = (0..OBJECTS)
        .map(|i| Obj {
            id: (i + 1) as u64, // nonzero pick ids
            pos: Vec3::new(
                rng.range(-14.0, 14.0),
                rng.range(-14.0, 14.0),
                rng.range(-14.0, 14.0),
            ),
            rot: Quat::from_euler(
                glam::EulerRot::XYZ,
                rng.range(0.0, std::f32::consts::TAU),
                rng.range(0.0, std::f32::consts::TAU),
                rng.range(0.0, std::f32::consts::TAU),
            ),
            scale: Vec3::splat(rng.range(0.7, 1.6)),
        })
        .collect();

    // viewport-lib picker inputs.
    let mut mesh_lookup: HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)> = HashMap::new();
    mesh_lookup.insert(MESH_KEY, (cube.positions.clone(), cube.indices.clone()));
    let obj_refs: Vec<&dyn ViewportObject> =
        objects.iter().map(|o| o as &dyn ViewportObject).collect();

    // Adapter: register the mesh once, push an instance per object.
    let mut query = SceneQuery::new();
    query.register_mesh_data(MESH_KEY, &cube);
    for o in &objects {
        query.push_instance(o.id, MESH_KEY, o.model());
    }
    let index = SceneIndex::build(query);

    // Scripted rays: each aimed at a random object's centre from a random
    // direction, so most land squarely on some object (occlusion decides which).
    let mut agree = 0usize;
    let mut both_hit = 0usize;
    let mut both_miss = 0usize;
    let mut disagree = 0usize;
    for _ in 0..RAYS {
        let target = objects[(rng.next_u64() as usize) % objects.len()].pos;
        let dir = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        )
        .normalize_or_zero();
        let origin = target - dir * 40.0;

        let vp = pick_scene_cpu(origin, dir, &obj_refs, &mesh_lookup);
        let ray = Ray::new(
            Point([origin.x, origin.y, origin.z]),
            Point([dir.x, dir.y, dir.z]),
        );
        let ad = index.raycast(&ray, MAX_TOI);

        match (vp, ad) {
            (None, None) => {
                agree += 1;
                both_miss += 1;
            }
            (Some(v), Some(a)) if v.id == a.id.0 => {
                agree += 1;
                both_hit += 1;
            }
            _ => disagree += 1,
        }
    }

    println!("pick_parity: {OBJECTS} objects, {RAYS} scripted rays");
    println!("  agree      {agree}/{RAYS}  (both-hit {both_hit}, both-miss {both_miss})");
    println!("  disagree   {disagree}");
    assert_eq!(
        disagree, 0,
        "adapter and viewport-lib's CPU picker must resolve the same object on every ray"
    );
    println!("PASS: the adapter matches viewport-lib's own picker on every ray.");
}
