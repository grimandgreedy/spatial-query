//! Make a viewport-lib [`Scene`] queryable through `spatial-query`.
//!
//! This is the reference render-side adapter (Phase 9, thin). viewport-lib already
//! has its own arbitrary-direction CPU picker, its own BVH, and a CPU/GPU pick
//! switch, so this crate does not reimplement querying; it exposes a
//! [`QueryGeometry`] provider over the scene so the *same* `spatial-query` engine a
//! headless consumer uses can also answer against a render scene, and so a scene
//! can be combined with non-render geometry behind one provider (see the
//! `editor_combining_provider` example).
//!
//! # The mesh cache
//!
//! viewport-lib retains CPU triangle data after upload but does not vend it back
//! from a `MeshId` (the fields are crate-private). So the adapter keeps its own
//! `MeshId -> (local AABB, triangles)` cache: register each mesh's [`MeshData`]
//! when you upload it, then [`sync`](SceneQuery::sync) the instance list from the
//! live scene each frame. This mirrors how viewport-lib's own picking examples
//! keep a caller-side mesh map, and needs no viewport-lib change.
//!
//! # What the provider reports
//!
//! `Id` is [`PickId`] (`= PickId(node.id())`, the same identity the GPU pick
//! buffer uses), and `test_ray` returns [`SubObjectRef::Face`] for the triangle
//! hit, so a spatial-query result lines up with viewport-lib's own picker.
//!
//! # Limitations
//!
//! Skinned / GPU-deformed meshes are seen at their bind pose: viewport-lib keeps
//! only the bind pose on the CPU and computes deformed positions on the GPU. For
//! deformed geometry, feed deformed positions in yourself (re-register the mesh
//! with updated `MeshData`), mirroring viewport-lib's per-frame-refresh strategy.
//! Layer visibility is not consulted; per-node `is_visible` and `hidden` are.

use std::collections::HashMap;

use glam::{Mat3, Mat4, Vec3};
use spatial_query::{Aabb, Bvh, Hit, LeafHit, Point, QueryFilter, QueryGeometry, Ray, Scalar};
use viewport_lib::{MeshData, MeshId, PickId, Scene, SubObjectRef};

/// Core point -> glam vector.
#[inline]
fn to_vec3(p: Point<3>) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// glam vector -> core point.
#[inline]
fn to_point(v: Vec3) -> Point<3> {
    Point([v.x, v.y, v.z])
}

/// A cached mesh: local-space triangles and the local bounding box.
struct MeshGeom {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    local_aabb: Aabb<3>,
}

/// One queryable instance: a scene node carrying a registered mesh.
struct Instance {
    /// The node id, which is also its pick id (`PickId(id)`).
    id: u64,
    /// Key into the mesh cache (`MeshId::index()`).
    mesh: u64,
    world: Mat4,
    inv_world: Mat4,
    /// Inverse-transpose of the world rotation/scale, for transforming normals.
    normal_mat: Mat3,
    world_aabb: Aabb<3>,
}

/// A [`QueryGeometry`] provider over a viewport-lib scene.
///
/// Register meshes as you upload them, then [`sync`](Self::sync) the instances
/// from the scene and build a [`Bvh`] over it (or use [`SceneIndex`], which does
/// both).
#[derive(Default)]
pub struct SceneQuery {
    meshes: HashMap<u64, MeshGeom>,
    instances: Vec<Instance>,
}

impl SceneQuery {
    /// An empty provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cache a mesh's CPU geometry under an explicit key. Use this in headless
    /// contexts where there is no `MeshId`; prefer [`register_mesh`](Self::register_mesh)
    /// with a real handle.
    pub fn register_mesh_data(&mut self, key: u64, data: &MeshData) {
        let local_aabb = aabb_of(&data.positions);
        self.meshes.insert(
            key,
            MeshGeom {
                positions: data.positions.clone(),
                indices: data.indices.clone(),
                local_aabb,
            },
        );
    }

    /// Cache a mesh's CPU geometry under its `MeshId`. Call this at upload time,
    /// since viewport-lib does not return triangles from a handle.
    pub fn register_mesh(&mut self, id: MeshId, data: &MeshData) {
        self.register_mesh_data(id.index() as u64, data);
    }

    /// Drop all instances (keeps the mesh cache).
    pub fn clear_instances(&mut self) {
        self.instances.clear();
    }

    /// Add one instance of a registered mesh at `world`. A node whose mesh was
    /// never registered is skipped (nothing to test against).
    pub fn push_instance(&mut self, id: u64, mesh_key: u64, world: Mat4) {
        let Some(geom) = self.meshes.get(&mesh_key) else {
            return;
        };
        let inv_world = world.inverse();
        self.instances.push(Instance {
            id,
            mesh: mesh_key,
            world,
            inv_world,
            normal_mat: Mat3::from_mat4(inv_world).transpose(),
            world_aabb: world_aabb_of(&world, &geom.local_aabb),
        });
    }

    /// Rebuild the instance list from the visible, mesh-bearing nodes of `scene`.
    ///
    /// Reads each node's world transform, so the scene's transforms must be
    /// current (a `ViewportApp`/runtime updates them each frame before the run
    /// callback). Nodes that are hidden, invisible, mesh-less, or whose mesh is
    /// unregistered are skipped.
    pub fn sync(&mut self, scene: &Scene) {
        self.instances.clear();
        for node in scene.nodes() {
            if !node.is_visible() || node.appearance().hidden {
                continue;
            }
            let Some(mesh) = node.mesh_id() else {
                continue;
            };
            let key = mesh.index() as u64;
            if !self.meshes.contains_key(&key) {
                continue;
            }
            self.push_instance(node.id(), key, node.world_transform());
        }
    }

    /// Number of queryable instances.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether there are no queryable instances.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

impl QueryGeometry<3> for SceneQuery {
    type Id = PickId;
    type SubObject = SubObjectRef;

    fn leaf_count(&self) -> usize {
        self.instances.len()
    }

    fn id(&self, leaf: usize) -> PickId {
        PickId(self.instances[leaf].id)
    }

    fn world_aabb(&self, leaf: usize) -> Aabb<3> {
        self.instances[leaf].world_aabb
    }

    fn test_ray(
        &self,
        leaf: usize,
        ray: &Ray<3>,
        max_toi: Scalar,
    ) -> Option<LeafHit<3, SubObjectRef>> {
        let inst = &self.instances[leaf];
        let geom = self.meshes.get(&inst.mesh)?;

        // Test in the mesh's local space, then map the hit back to world so the
        // reported time of impact is correct even under non-uniform scale.
        let o = to_vec3(ray.origin);
        let d = to_vec3(ray.dir);
        let lo = inst.inv_world.transform_point3(o);
        let ld = inst.inv_world.transform_vector3(d);

        let tri_count = geom.indices.len() / 3;
        let mut best_t = Scalar::INFINITY;
        let mut best_face = 0u32;
        for f in 0..tri_count {
            let (v0, v1, v2) = tri(geom, f);
            if let Some(t) = ray_triangle(lo, ld, v0, v1, v2) {
                if t < best_t {
                    best_t = t;
                    best_face = f as u32;
                }
            }
        }
        if !best_t.is_finite() {
            return None;
        }

        let hit_world = inst.world.transform_point3(lo + ld * best_t);
        let world_toi = (hit_world - o).dot(d); // d is unit, so this is world distance
        if world_toi < 0.0 || world_toi > max_toi {
            return None;
        }

        let (v0, v1, v2) = tri(geom, best_face as usize);
        let n_local = (v1 - v0).cross(v2 - v0);
        let normal = (inst.normal_mat * n_local).normalize_or_zero();
        Some(
            LeafHit::new(world_toi, to_point(normal))
                .with_sub_object(SubObjectRef::Face(best_face)),
        )
    }
}

/// A [`SceneQuery`] bundled with a built [`Bvh`], the common case: sync then
/// query. The mesh cache lives in `query`; register meshes there before building.
pub struct SceneIndex {
    /// The provider. Register meshes on it before [`rebuild`](Self::rebuild).
    pub query: SceneQuery,
    bvh: Bvh<3>,
}

impl SceneIndex {
    /// Build an index over `query`'s current instances.
    pub fn build(query: SceneQuery) -> Self {
        let bvh = Bvh::build(&query);
        SceneIndex { query, bvh }
    }

    /// Re-sync instances from `scene` and rebuild the tree. The instance count can
    /// change frame to frame (nodes added/removed/hidden), so this rebuilds rather
    /// than refits.
    pub fn sync(&mut self, scene: &Scene) {
        self.query.sync(scene);
        self.bvh = Bvh::build(&self.query);
    }

    /// Rebuild the tree from the current instances without touching the scene.
    pub fn rebuild(&mut self) {
        self.bvh = Bvh::build(&self.query);
    }

    /// Nearest hit along `ray` within `max_toi`: which object (`PickId`) and which
    /// face.
    pub fn raycast(&self, ray: &Ray<3>, max_toi: Scalar) -> Option<Hit<PickId, 3, SubObjectRef>> {
        self.bvh
            .raycast_nearest(&self.query, ray, max_toi, &QueryFilter::default())
    }
}

/// The three local-space vertices of triangle `f`.
#[inline]
fn tri(geom: &MeshGeom, f: usize) -> (Vec3, Vec3, Vec3) {
    let i0 = geom.indices[3 * f] as usize;
    let i1 = geom.indices[3 * f + 1] as usize;
    let i2 = geom.indices[3 * f + 2] as usize;
    (
        Vec3::from(geom.positions[i0]),
        Vec3::from(geom.positions[i1]),
        Vec3::from(geom.positions[i2]),
    )
}

/// Moeller-Trumbore ray-triangle intersection (double-sided). Returns the ray
/// parameter `t` (in the units of `d`) of the front intersection, or `None`.
fn ray_triangle(o: Vec3, d: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = o - v0;
    let u = tvec.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tvec.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t > 1e-6 {
        Some(t)
    } else {
        None
    }
}

/// The AABB of a set of local positions.
fn aabb_of(positions: &[[f32; 3]]) -> Aabb<3> {
    let mut a = Aabb::<3>::EMPTY;
    for p in positions {
        a = a.union_point(Point(*p));
    }
    a
}

/// The world AABB of a local box under a transform (all eight corners).
fn world_aabb_of(world: &Mat4, local: &Aabb<3>) -> Aabb<3> {
    let mn = local.min.0;
    let mx = local.max.0;
    let mut out = Aabb::<3>::EMPTY;
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { mn[0] } else { mx[0] },
            if i & 2 == 0 { mn[1] } else { mx[1] },
            if i & 4 == 0 { mn[2] } else { mx[2] },
        );
        out = out.union_point(to_point(world.transform_point3(corner)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use viewport_lib::primitives;

    #[test]
    fn ray_hits_registered_cube_with_face_and_id() {
        // A unit-ish cube, translated out along +x.
        let cube = primitives::cube(2.0);
        let local = aabb_of(&cube.positions);
        let front = 5.0 + local.min[0]; // world x of the near (-x) face

        let mut q = SceneQuery::new();
        q.register_mesh_data(7, &cube);
        q.push_instance(42, 7, Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        let idx = SceneIndex::build(q);

        let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
        let hit = idx.raycast(&ray, 100.0).expect("ray should hit the cube");
        assert_eq!(hit.id, PickId(42), "reports the node's pick id");
        assert!(
            (hit.time_of_impact - front).abs() < 1e-3,
            "toi {} at near face {front}",
            hit.time_of_impact
        );
        assert!(hit.normal.0[0].abs() > 0.9, "front-face normal along x");
        assert!(
            matches!(hit.sub_object, Some(SubObjectRef::Face(_))),
            "a face"
        );
    }

    #[test]
    fn nearest_instance_wins() {
        let cube = primitives::cube(2.0);
        let mut q = SceneQuery::new();
        q.register_mesh_data(1, &cube);
        q.push_instance(100, 1, Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)));
        q.push_instance(200, 1, Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        let idx = SceneIndex::build(q);

        let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
        let hit = idx.raycast(&ray, 100.0).expect("hit");
        assert_eq!(hit.id, PickId(200), "the nearer cube is picked");
    }

    #[test]
    fn ray_pointing_away_misses() {
        let cube = primitives::cube(2.0);
        let mut q = SceneQuery::new();
        q.register_mesh_data(1, &cube);
        q.push_instance(1, 1, Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        let idx = SceneIndex::build(q);

        let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([-1.0, 0.0, 0.0]));
        assert!(idx.raycast(&ray, 100.0).is_none());
    }

    #[test]
    fn scaled_instance_reports_world_distance() {
        // A cube scaled 3x: the local-space toi would be wrong, the world toi right.
        let cube = primitives::cube(2.0);
        let local = aabb_of(&cube.positions);
        let mut q = SceneQuery::new();
        q.register_mesh_data(1, &cube);
        q.push_instance(
            1,
            1,
            Mat4::from_scale_rotation_translation(
                Vec3::splat(3.0),
                glam::Quat::IDENTITY,
                Vec3::new(20.0, 0.0, 0.0),
            ),
        );
        let idx = SceneIndex::build(q);

        let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
        let hit = idx.raycast(&ray, 100.0).expect("hit");
        let front = 20.0 + 3.0 * local.min[0]; // scaled near (-x) face
        assert!(
            (hit.time_of_impact - front).abs() < 1e-2,
            "world toi {} at scaled face {front}",
            hit.time_of_impact
        );
    }
}
