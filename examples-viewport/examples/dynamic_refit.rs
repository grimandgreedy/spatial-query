//! Thousands of moving boxes, all queryable in real time.
//!
//! Every box drifts, tumbles, and some pulse in size. Each frame the two-level
//! tree is refit (not rebuilt) from the new transforms, and a sweeping ray finds
//! the nearest box it hits, marked in red. Once a second the console prints the
//! refit time next to the cost of a full rebuild and a batch of queries, which
//! is the whole point: refit stays cheap while the scene keeps moving.

// Several loops index parallel per-instance arrays by the same index.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use glam::{Mat3, Mat4, Quat, Vec3};
use spatial_query::{Aabb, InstancedGeometry, Isometry, LeafHit, Point, QueryFilter, Ray, Tlas};
use spatial_query_viewport_examples::{sweep_ray, to_point, to_vec3};
use viewport_lib::{primitives, AppConfig, Material, NodeId, ViewportApp};

const INSTANCES: usize = 1800;
const BOUND: f32 = 8.5;
const RAY_DOTS: usize = 30;
const RAY_LENGTH: f32 = 26.0;
const MAX_TOI: f32 = 100.0;
const HIDDEN: Vec3 = Vec3::new(0.0, 0.0, -10000.0);

fn main() {
    let boxes = Boxes::new(INSTANCES, 0x0D19_A3F7);
    let tlas = Tlas::build(&boxes);
    let state = Rc::new(RefCell::new(State {
        boxes,
        tlas,
        box_nodes: Vec::new(),
        dot_nodes: Vec::new(),
        marker: None,
        frame: 0,
    }));

    let setup_state = state.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : dynamic_refit")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let cube = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::cube(1.0))
            .unwrap();
        let dot = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 8, 6))
            .unwrap();

        let mut st = setup_state.borrow_mut();
        let sc = session.scene_mut();

        for i in 0..st.boxes.len() {
            let id = sc.add(
                Some(cube),
                st.boxes.render_transform(i),
                Material::from_colour([0.45, 0.55, 0.72]),
            );
            st.box_nodes.push(id);
        }
        for _ in 0..RAY_DOTS {
            let id = sc.add(
                Some(dot),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.1), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.95, 0.85, 0.15]),
            );
            st.dot_nodes.push(id);
        }
        st.marker = Some(sc.add(
            Some(dot),
            Mat4::from_scale_rotation_translation(Vec3::splat(0.5), Quat::IDENTITY, HIDDEN),
            Material::from_colour([0.9, 0.15, 0.15]),
        ));

        session.camera_mut().distance = 30.0;
    })
    .run(move |ctx| {
        let dt = ctx.dt().min(0.05);
        let time = ctx.time();
        let mut st = state.borrow_mut();
        st.frame += 1;

        // Advance the scene, then refit the tree from the new transforms.
        st.boxes.advance(dt, time, BOUND);
        let refit_us = {
            let State { boxes, tlas, .. } = &mut *st;
            let t = Instant::now();
            tlas.refit(boxes);
            if tlas.needs_rebuild(2.0) {
                *tlas = Tlas::build(boxes);
            }
            t.elapsed().as_micros()
        };

        // Query: nearest box the sweeping ray hits.
        let (origin, dir) = sweep_ray(time);
        let ray = Ray::new(to_point(origin), to_point(dir));
        let hit = st
            .tlas
            .raycast_nearest(&st.boxes, &ray, MAX_TOI, &QueryFilter::default());

        // Once a second, compare refit against a full rebuild and a query batch.
        if st.frame.is_multiple_of(60) {
            let State { boxes, tlas, .. } = &*st;
            let t = Instant::now();
            let _rebuilt = Tlas::build(boxes);
            let rebuild_us = t.elapsed().as_micros();
            let t = Instant::now();
            for _ in 0..1000 {
                let _ = tlas.raycast_nearest(boxes, &ray, MAX_TOI, &QueryFilter::default());
            }
            let query_us = t.elapsed().as_micros();
            println!(
                "{} boxes | refit {refit_us} us | rebuild {rebuild_us} us | 1000 nearest queries {query_us} us",
                boxes.len(),
            );
        }

        // Render: box transforms, the ray trail, and the hit marker.
        let sc = ctx.scene_mut();
        for i in 0..st.box_nodes.len() {
            sc.set_local_transform(st.box_nodes[i], st.boxes.render_transform(i));
        }

        let seg = hit.as_ref().map(|h| h.time_of_impact).unwrap_or(RAY_LENGTH);
        let spacing = seg / RAY_DOTS as f32;
        for (i, &id) in st.dot_nodes.iter().enumerate() {
            let pos = origin + dir * ((i as f32 + 1.0) * spacing);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.1), Quat::IDENTITY, pos),
            );
        }
        if let Some(m) = st.marker {
            let pos = hit.as_ref().map(|h| to_vec3(h.point)).unwrap_or(HIDDEN);
            sc.set_local_transform(
                m,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.5), Quat::IDENTITY, pos),
            );
        }
    });
}

struct State {
    boxes: Boxes,
    tlas: Tlas<3>,
    box_nodes: Vec<NodeId>,
    dot_nodes: Vec<NodeId>,
    marker: Option<NodeId>,
    frame: u64,
}

/// A cloud of oriented boxes that drift, tumble, and pulse.
struct Boxes {
    pos: Vec<Vec3>,
    quat: Vec<Quat>,
    half: Vec<Vec3>,
    base_half: Vec<Vec3>,
    lin_vel: Vec<Vec3>,
    ang_vel: Vec<Vec3>,
}

impl Boxes {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut unit = move || (next() >> 11) as f32 / (1u64 << 53) as f32;
        let mut span = |lo: f32, hi: f32| lo + unit() * (hi - lo);
        let mut vec = |lo, hi| Vec3::new(span(lo, hi), span(lo, hi), span(lo, hi));

        let mut b = Boxes {
            pos: Vec::with_capacity(n),
            quat: Vec::with_capacity(n),
            half: Vec::with_capacity(n),
            base_half: Vec::with_capacity(n),
            lin_vel: Vec::with_capacity(n),
            ang_vel: Vec::with_capacity(n),
        };
        for _ in 0..n {
            b.pos.push(vec(-BOUND * 0.85, BOUND * 0.85));
            b.quat.push(Quat::from_scaled_axis(vec(-3.0, 3.0)));
            let h = vec(0.25, 0.6).abs();
            b.half.push(h);
            b.base_half.push(h);
            b.lin_vel.push(vec(-2.5, 2.5));
            b.ang_vel.push(vec(-2.0, 2.0));
        }
        b
    }

    fn len(&self) -> usize {
        self.pos.len()
    }

    /// Integrate motion, bounce off the bounding cube, and pulse a fraction of
    /// the boxes in size (the deforming instances).
    fn advance(&mut self, dt: f32, time: f32, bound: f32) {
        for i in 0..self.len() {
            let p = self.pos[i];
            let v = &mut self.lin_vel[i];
            if (p.x > bound && v.x > 0.0) || (p.x < -bound && v.x < 0.0) {
                v.x = -v.x;
            }
            if (p.y > bound && v.y > 0.0) || (p.y < -bound && v.y < 0.0) {
                v.y = -v.y;
            }
            if (p.z > bound && v.z > 0.0) || (p.z < -bound && v.z < 0.0) {
                v.z = -v.z;
            }
            self.pos[i] = p + *v * dt;
            self.quat[i] =
                (Quat::from_scaled_axis(self.ang_vel[i] * dt) * self.quat[i]).normalize();
            if i.is_multiple_of(9) {
                let scale = 1.0 + 0.4 * (time * 2.5 + i as f32).sin();
                self.half[i] = self.base_half[i] * scale;
            }
        }
    }

    /// The full affine (non-uniform scale, rotation, translation) that places
    /// this box in the render scene. The unit cube spans -0.5 to 0.5, so the
    /// scale is the full box size.
    fn render_transform(&self, i: usize) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.half[i] * 2.0, self.quat[i], self.pos[i])
    }
}

impl InstancedGeometry<3> for Boxes {
    type Id = usize;

    fn instance_count(&self) -> usize {
        self.len()
    }
    fn id(&self, i: usize) -> usize {
        i
    }
    fn transform(&self, i: usize) -> Isometry<3> {
        Isometry::new(mat3_rows(self.quat[i]), to_point(self.pos[i]))
    }
    fn local_aabb(&self, i: usize) -> Aabb<3> {
        let h = to_point(self.half[i]);
        Aabb::new(-h, h)
    }
    fn test_ray_local(&self, i: usize, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
        ray_local_box(ray, to_point(self.half[i]), max_toi)
    }
}

/// A glam quaternion as the row-major rotation matrix `Isometry` expects.
fn mat3_rows(q: Quat) -> [[f32; 3]; 3] {
    let m = Mat3::from_quat(q);
    [
        [m.x_axis.x, m.y_axis.x, m.z_axis.x],
        [m.x_axis.y, m.y_axis.y, m.z_axis.y],
        [m.x_axis.z, m.y_axis.z, m.z_axis.z],
    ]
}

/// Ray against a local axis-aligned box `[-half, half]`. Returns the entry
/// point-of-impact and the outward face normal.
fn ray_local_box(ray: &Ray<3>, half: Point<3>, max_toi: f32) -> Option<LeafHit<3>> {
    let mut t_min = 0.0f32;
    let mut t_max = max_toi;
    let mut axis: Option<usize> = None;
    for i in 0..3 {
        let o = ray.origin[i];
        let d = ray.dir[i];
        if d.abs() < 1.0e-9 {
            if o < -half[i] || o > half[i] {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut t1 = (-half[i] - o) * inv;
            let mut t2 = (half[i] - o) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            if t1 > t_min {
                t_min = t1;
                axis = Some(i);
            }
            if t2 < t_max {
                t_max = t2;
            }
            if t_min > t_max {
                return None;
            }
        }
    }
    let axis = match axis {
        Some(a) if t_min > 1.0e-6 && t_min <= max_toi => a,
        _ => return None,
    };
    let sign = if ray.dir[axis] > 0.0 { -1.0 } else { 1.0 };
    let mut n = [0.0; 3];
    n[axis] = sign;
    Some(LeafHit {
        toi: t_min,
        normal: Point(n),
        sub_object: None,
    })
}
