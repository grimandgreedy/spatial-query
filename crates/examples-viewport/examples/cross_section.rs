//! Reconstruct a cross-section from ray crossings.
//!
//! A torus lies flat in the xy-plane (axis +z, matching viewport-lib's z-up
//! world) and is rendered translucent. A horizontal cutting plane sweeps up and
//! down along z, and each frame a fan of coplanar rays is cast across the plane.
//! `spatial-query`'s `raycast_crossings` reports every point where each ray
//! pierces the torus surface (up to four per ray, since a line through the plane
//! crosses both walls of the tube), and a dot is parked at every crossing. The
//! dots trace the section: two concentric rings that widen and narrow as the
//! plane rises and falls through the tube.
//!
//! The core produces the crossing samples; this example does the reconstruction
//! (here, a point cloud). A consumer wanting a filled contour would stitch the
//! samples into polylines.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray};
use spatial_query_viewport_examples::to_vec3;
use viewport_lib::{primitives, AppConfig, ItemSettings, Material, NodeId, ViewportApp};

const MAJOR: f32 = 3.0;
const MINOR: f32 = 1.0;
const SCAN_LINES: usize = 72;
const DOTS: usize = 320;
const MAX_TOI: f32 = 2.0 * (MAJOR + MINOR) + 2.0;
const HIDDEN: Vec3 = Vec3::new(0.0, 0.0, -10000.0);

fn main() {
    let torus = Rc::new(Torus {
        major: MAJOR,
        minor: MINOR,
    });
    let bvh = Rc::new(Bvh::build(&*torus));
    let nodes: Rc<RefCell<Nodes>> = Rc::new(RefCell::new(Nodes::default()));

    let setup_nodes = nodes.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : cross_section")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let torus_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::torus(MAJOR, MINOR, 48, 24))
            .unwrap();
        let plane_mesh = session
            .resources_mut()
            .upload_mesh_data(
                device,
                &primitives::plane(2.0 * (MAJOR + MINOR) + 1.0, 2.0 * (MAJOR + MINOR) + 1.0),
            )
            .unwrap();
        let dot_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 8, 6))
            .unwrap();

        let mut n = setup_nodes.borrow_mut();
        let sc = session.scene_mut();

        // The torus being cut, drawn translucent so the section shows through.
        let mut torus_mat = Material::from_colour([0.55, 0.58, 0.66]);
        torus_mat.alpha_mode = viewport_lib::AlphaMode::Blend;
        let torus_node = sc.add(Some(torus_mesh), Mat4::IDENTITY, torus_mat);
        let mut torus_look = ItemSettings::default();
        torus_look.opacity = 0.25;
        sc.set_appearance(torus_node, torus_look);

        // The cutting plane, a faint translucent quad that slides on y.
        let mut plane_mat = Material::from_colour([0.35, 0.7, 0.95]);
        plane_mat.alpha_mode = viewport_lib::AlphaMode::Blend;
        n.plane = Some(sc.add(Some(plane_mesh), Mat4::IDENTITY, plane_mat));
        let mut plane_look = ItemSettings::default();
        plane_look.opacity = 0.12;
        sc.set_appearance(n.plane.unwrap(), plane_look);

        // The section point cloud.
        for _ in 0..DOTS {
            n.dots.push(sc.add(
                Some(dot_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.06), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.95, 0.75, 0.15]),
            ));
        }

        session.camera_mut().distance = 16.0;
    })
    .run(move |ctx| {
        // Sweep the cut height along +z (up) through the tube. Keep it inside
        // (-minor, minor) so both rings of the section stay real.
        let h = 0.92 * MINOR * (ctx.time() * 0.6).sin();

        // Cast a fan of +x rays lying in the horizontal plane z = h, one per
        // scanline in y, and collect every crossing of every ray.
        let span = MAJOR + MINOR + 0.5;
        let mut points: Vec<Vec3> = Vec::new();
        for k in 0..SCAN_LINES {
            let y = -span + 2.0 * span * (k as f32 / (SCAN_LINES - 1) as f32);
            let ray = Ray::new(Point([-span - 0.1, y, h]), Point([1.0, 0.0, 0.0]));
            for hit in bvh.raycast_crossings(&*torus, &ray, MAX_TOI, &QueryFilter::default()) {
                points.push(to_vec3(hit.point));
            }
        }

        let n = nodes.borrow();
        let sc = ctx.scene_mut();

        if let Some(p) = n.plane {
            sc.set_local_transform(
                p,
                Mat4::from_scale_rotation_translation(
                    Vec3::ONE,
                    Quat::IDENTITY,
                    Vec3::new(0.0, 0.0, h),
                ),
            );
        }

        for (i, &id) in n.dots.iter().enumerate() {
            let pos = points.get(i).copied().unwrap_or(HIDDEN);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.06), Quat::IDENTITY, pos),
            );
        }
    });
}

#[derive(Default)]
struct Nodes {
    dots: Vec<NodeId>,
    plane: Option<NodeId>,
}

/// A torus as an implicit surface, so a ray reports every crossing by marching
/// the field and bracketing each sign change. The axis is +z and the ring lies
/// in the xy-plane, matching `primitives::torus` and viewport-lib's z-up world.
struct Torus {
    major: f32,
    minor: f32,
}

impl Torus {
    /// The implicit function: zero on the surface, negative inside the tube.
    fn field(&self, p: Point<3>) -> f32 {
        let q = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let a = q - self.major;
        a * a + p[2] * p[2] - self.minor * self.minor
    }

    /// The outward gradient of the field, used as the surface normal.
    fn gradient(&self, p: Point<3>) -> Point<3> {
        let q = (p[0] * p[0] + p[1] * p[1]).sqrt().max(1e-6);
        let a = q - self.major;
        Point([2.0 * a * p[0] / q, 2.0 * a * p[1] / q, 2.0 * p[2]])
    }
}

impl QueryGeometry<3> for Torus {
    type Id = usize;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        1
    }
    fn id(&self, _leaf: usize) -> usize {
        0
    }
    fn world_aabb(&self, _leaf: usize) -> Aabb<3> {
        let o = self.major + self.minor;
        Aabb::new(Point([-o, -o, -self.minor]), Point([o, o, self.minor]))
    }
    fn test_ray(&self, _leaf: usize, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
        // The first crossing the march emits is the nearest.
        let mut nearest = None;
        self.test_ray_crossings(0, ray, max_toi, &mut |lh| {
            if nearest.is_none() {
                nearest = Some(lh);
            }
        });
        nearest
    }
    fn test_ray_crossings(
        &self,
        _leaf: usize,
        ray: &Ray<3>,
        max_toi: f32,
        out: &mut dyn FnMut(LeafHit<3>),
    ) {
        const STEP: f32 = 0.02;
        let mut t = 0.0;
        let mut f_prev = self.field(ray.at(t));
        while t < max_toi {
            let t_next = (t + STEP).min(max_toi);
            let f_next = self.field(ray.at(t_next));
            if f_prev.signum() != f_next.signum() && f_prev != 0.0 {
                // Bracket [t, t_next]: bisect to the crossing.
                let (mut a, mut b) = (t, t_next);
                let mut fa = f_prev;
                for _ in 0..24 {
                    let m = 0.5 * (a + b);
                    let fm = self.field(ray.at(m));
                    if (fa < 0.0) != (fm < 0.0) {
                        b = m;
                    } else {
                        a = m;
                        fa = fm;
                    }
                }
                let root = 0.5 * (a + b);
                let normal = self.gradient(ray.at(root)).normalize_or_zero();
                out(LeafHit::new(root, normal));
            }
            if t_next >= max_toi {
                break;
            }
            t = t_next;
            f_prev = f_next;
        }
    }
}
