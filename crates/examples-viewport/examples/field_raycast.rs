//! Ray-march an implicit iso-surface and show the hits and gradient normals.
//!
//! The queried leaf is not a mesh or an analytic primitive: it is a density
//! field, the sum of a handful of moving Gaussian blobs (a metaball field, the
//! shape a fluid density iso-surface takes). Its surface is where the density
//! equals a threshold, and there is no closed-form ray intersection, so the
//! provider's `test_ray` marches the field along the ray, brackets the crossing,
//! bisects to it, and reports the field gradient as the surface normal. The core
//! traverses and queries this leaf exactly as it would a triangle mesh.
//!
//! A grid of rays fires straight down at the field. Each ray's nearest hit parks
//! a dot on the surface, and a short spike is drawn along the gradient normal
//! there, so the sampled iso-surface and its normals show as the blobs drift and
//! merge. The core produces the hit points and normals; the drawing is the
//! example's.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use spatial_query::{Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray};
use spatial_query_viewport_examples::to_vec3;
use viewport_lib::{primitives, AppConfig, Material, NodeId, ViewportApp};

const GRID: usize = 22; // rays per side of the sampling grid
const SPAN: f32 = 6.0; // half-width of the sampled region in x and y
const TOP: f32 = 8.0; // rays start here on +z and march down
const MAX_TOI: f32 = 2.0 * TOP;
const ISO: f32 = 0.6; // density threshold defining the surface
const FALLOFF: f32 = 0.6; // Gaussian tightness
const NORMAL_LEN: f32 = 0.6;
const HIDDEN: Vec3 = Vec3::new(0.0, 0.0, -10000.0);

fn main() {
    let field = Rc::new(RefCell::new(MetaballField::new()));
    // The field's bounding box is fixed (the blobs stay within it), so the
    // one-leaf tree is built once and stays valid as the blobs move.
    let bvh = Rc::new(Bvh::build(&*field.borrow()));
    let markers: Rc<RefCell<Markers>> = Rc::new(RefCell::new(Markers::default()));

    let setup_markers = markers.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("spatial-query : field_raycast")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        let dot_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 8, 6))
            .unwrap();
        let spike_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::cube(1.0))
            .unwrap();
        // A faint blob mesh per metaball, so the field itself is visible as a
        // cluster of translucent spheres under the sampled surface.
        let blob_mesh = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(1.0, 20, 12))
            .unwrap();

        let mut m = setup_markers.borrow_mut();
        let sc = session.scene_mut();

        for _ in 0..MetaballField::BLOBS {
            let mut look = viewport_lib::ItemSettings::default();
            look.opacity = 0.18;
            let id = sc.add(
                Some(blob_mesh),
                Mat4::from_translation(HIDDEN),
                Material::from_colour([0.4, 0.7, 0.95]),
            );
            sc.set_appearance(id, look);
            m.blobs.push(id);
        }
        for _ in 0..GRID * GRID {
            m.dots.push(sc.add(
                Some(dot_mesh),
                Mat4::from_scale_rotation_translation(Vec3::splat(0.08), Quat::IDENTITY, HIDDEN),
                Material::from_colour([0.95, 0.55, 0.15]),
            ));
            m.spikes.push(sc.add(
                Some(spike_mesh),
                Mat4::from_translation(HIDDEN),
                Material::from_colour([0.95, 0.85, 0.25]),
            ));
        }

        session.camera_mut().distance = 22.0;
    })
    .run(move |ctx| {
        // Drift the blobs, then sample the surface with the ray grid.
        field.borrow_mut().animate(ctx.time());
        let f = field.borrow();

        let filter = QueryFilter::default();
        let mut hits: Vec<LeafHit<3>> = Vec::with_capacity(GRID * GRID);
        let mut hit_points: Vec<Vec3> = Vec::with_capacity(GRID * GRID);
        for iy in 0..GRID {
            for ix in 0..GRID {
                let x = lerp(-SPAN, SPAN, ix, GRID);
                let y = lerp(-SPAN, SPAN, iy, GRID);
                let ray = Ray::new(Point([x, y, TOP]), Point([0.0, 0.0, -1.0]));
                if let Some(h) = bvh.raycast_nearest(&*f, &ray, MAX_TOI, &filter) {
                    hit_points.push(to_vec3(h.point));
                    hits.push(h);
                }
            }
        }

        let m = markers.borrow();
        let sc = ctx.scene_mut();

        for (i, &id) in m.blobs.iter().enumerate() {
            let (c, r) = f.blob(i);
            sc.set_local_transform(
                id,
                Mat4::from_scale_rotation_translation(Vec3::splat(r), Quat::IDENTITY, to_vec3(c)),
            );
        }

        for (slot, &dot) in m.dots.iter().enumerate() {
            let (pos, spike_tf) = match hit_points.get(slot) {
                Some(&p) => {
                    let n = to_vec3(hits[slot].normal).normalize_or_zero();
                    let rot = if n.length_squared() > 1e-6 {
                        Quat::from_rotation_arc(Vec3::Z, n)
                    } else {
                        Quat::IDENTITY
                    };
                    let tf = Mat4::from_scale_rotation_translation(
                        Vec3::new(0.03, 0.03, NORMAL_LEN),
                        rot,
                        p + n * NORMAL_LEN * 0.5,
                    );
                    (p, tf)
                }
                None => (HIDDEN, Mat4::from_translation(HIDDEN)),
            };
            sc.set_local_transform(
                dot,
                Mat4::from_scale_rotation_translation(Vec3::splat(0.08), Quat::IDENTITY, pos),
            );
            sc.set_local_transform(m.spikes[slot], spike_tf);
        }
    });
}

fn lerp(lo: f32, hi: f32, i: usize, n: usize) -> f32 {
    lo + (hi - lo) * (i as f32 / (n - 1) as f32)
}

#[derive(Default)]
struct Markers {
    blobs: Vec<NodeId>,
    dots: Vec<NodeId>,
    spikes: Vec<NodeId>,
}

/// A metaball density field: the sum of Gaussian blobs. The surface is the
/// `ISO` level set. One `QueryGeometry` leaf whose `test_ray` marches the field.
struct MetaballField {
    centers: Vec<Point<3>>,
    radii: Vec<f32>,
    phases: Vec<f32>,
}

impl MetaballField {
    const BLOBS: usize = 5;

    fn new() -> Self {
        let mut centers = Vec::new();
        let mut radii = Vec::new();
        let mut phases = Vec::new();
        for i in 0..Self::BLOBS {
            centers.push(Point([0.0, 0.0, 0.0]));
            radii.push(1.1 + 0.25 * i as f32);
            phases.push(i as f32 * 1.7);
        }
        let mut f = MetaballField {
            centers,
            radii,
            phases,
        };
        f.animate(0.0);
        f
    }

    /// Move the blobs along slow Lissajous paths so the surface morphs.
    fn animate(&mut self, time: f32) {
        for i in 0..Self::BLOBS {
            let p = self.phases[i];
            self.centers[i] = Point([
                3.2 * (0.4 * time + p).cos(),
                3.2 * (0.31 * time + p * 1.3).sin(),
                1.6 * (0.5 * time + p).sin(),
            ]);
        }
    }

    /// Blob `i`'s current centre and display radius (for the faint blob meshes).
    fn blob(&self, i: usize) -> (Point<3>, f32) {
        (self.centers[i], self.radii[i])
    }

    /// The density at `p`: the sum of the blob Gaussians. Zero far away, large
    /// where blobs overlap.
    fn density(&self, p: Point<3>) -> f32 {
        let mut s = 0.0;
        for (c, r) in self.centers.iter().zip(&self.radii) {
            let d = p - *c;
            s += r * r * (-FALLOFF * d.dot(d)).exp();
        }
        s
    }

    /// The field whose zero set is the surface: positive outside, negative inside.
    fn field(&self, p: Point<3>) -> f32 {
        ISO - self.density(p)
    }

    /// The outward gradient of the field (points from dense to sparse), used as
    /// the surface normal.
    fn gradient(&self, p: Point<3>) -> Point<3> {
        // d/dp (ISO - sum r^2 exp(-k |p-c|^2)) = sum r^2 exp(...) * 2k (p-c).
        let mut g = [0.0f32; 3];
        for (c, r) in self.centers.iter().zip(&self.radii) {
            let d = p - *c;
            let w = r * r * (-FALLOFF * d.dot(d)).exp() * 2.0 * FALLOFF;
            for k in 0..3 {
                g[k] += w * d[k];
            }
        }
        Point(g)
    }
}

impl QueryGeometry<3> for MetaballField {
    type Id = usize;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        1
    }
    fn id(&self, _leaf: usize) -> usize {
        0
    }
    fn world_aabb(&self, _leaf: usize) -> Aabb<3> {
        // A fixed box that always contains the blobs and their iso-surface, so the
        // prebuilt tree stays valid as the field animates.
        let e = SPAN + 2.0;
        Aabb::new(Point([-e, -e, -e]), Point([e, e, e]))
    }
    fn test_ray(&self, _leaf: usize, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
        // March the ray, take the first sign change of the field, bisect to it.
        const STEP: f32 = 0.06;
        let mut t = 0.0;
        let mut f_prev = self.field(ray.at(t));
        while t < max_toi {
            let t_next = (t + STEP).min(max_toi);
            let f_next = self.field(ray.at(t_next));
            if f_prev != 0.0 && (f_prev < 0.0) != (f_next < 0.0) {
                let (mut a, mut b) = (t, t_next);
                let mut fa = f_prev;
                for _ in 0..30 {
                    let mid = 0.5 * (a + b);
                    let fm = self.field(ray.at(mid));
                    if (fa < 0.0) != (fm < 0.0) {
                        b = mid;
                    } else {
                        a = mid;
                        fa = fm;
                    }
                }
                let root = 0.5 * (a + b);
                let normal = self.gradient(ray.at(root)).normalize_or_zero();
                return Some(LeafHit::new(root, normal));
            }
            if t_next >= max_toi {
                break;
            }
            t = t_next;
            f_prev = f_next;
        }
        None
    }
}
