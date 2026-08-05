//! A spatial acceleration and query engine that works in any dimension.
//!
//! This crate answers "what geometry does this ray touch?" over a set of
//! objects, without knowing what those objects are. A consumer implements the
//! [`QueryGeometry`] trait to describe its own data: render-scene items, physics
//! bodies, or anything with a bounding box and a ray test. The crate owns the
//! acceleration structure and traversal.
//!
//! The core pulls in no renderer, no physics engine, and no `wgpu`, so it can be
//! embedded anywhere. Everything is generic over the dimension `D`: the point
//! type is [`Point<D>`](maths::Point) over `[f32; D]` rather than a fixed
//! three-component vector, which is why the crate has no linear-algebra
//! dependency. Most callers instantiate at `D = 3`.
//!
//! Nearest-hit selection and all-hits ordering are fully determined (by
//! `time_of_impact`, then leaf index), so the [`Bvh`] returns the same results
//! as the [`naive`] scan on the same input.
//!
//! # Example
//!
//! ```
//! use spatial_query::{Aabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray};
//!
//! // A tiny provider: unit-radius balls at given centres, in 3D.
//! struct Balls(Vec<Point<3>>);
//!
//! impl QueryGeometry<3> for Balls {
//!     type Id = usize;
//!     fn leaf_count(&self) -> usize { self.0.len() }
//!     fn id(&self, leaf: usize) -> usize { leaf }
//!     fn world_aabb(&self, leaf: usize) -> Aabb<3> {
//!         let c = self.0[leaf];
//!         Aabb::new(c - Point::splat(1.0), c + Point::splat(1.0))
//!     }
//!     fn test_ray(&self, leaf: usize, ray: &Ray<3>, max_toi: f32) -> Option<LeafHit<3>> {
//!         let oc = ray.origin - self.0[leaf];
//!         let b = oc.dot(ray.dir);
//!         let c = oc.length_squared() - 1.0;
//!         let disc = b * b - c;
//!         if disc < 0.0 { return None; }
//!         let t = -b - disc.sqrt();
//!         if t < 0.0 || t > max_toi { return None; }
//!         let normal = (ray.at(t) - self.0[leaf]).normalize_or_zero();
//!         Some(LeafHit { toi: t, normal, sub_object: None })
//!     }
//! }
//!
//! let balls = Balls(vec![Point([5.0, 0.0, 0.0]), Point([10.0, 0.0, 0.0])]);
//! let bvh = Bvh::build(&balls);
//! let ray = Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]));
//! let hit = bvh.raycast_nearest(&balls, &ray, 100.0, &QueryFilter::default()).unwrap();
//! assert_eq!(hit.id, 0);
//! assert!((hit.time_of_impact - 4.0).abs() < 1e-4);
//! ```

#![forbid(unsafe_code)]

pub mod accel;
pub mod maths;
pub mod query;

// The public surface stays flat: consumers use `spatial_query::Bvh`,
// `spatial_query::QueryGeometry`, etc., regardless of the internal module tree.
pub use accel::{Bvh, Tlas};
pub use maths::{plane_rotation, Aabb, Isometry, Point, Ray, Scalar};
pub use query::{
    naive, Hit, InstancedGeometry, LeafHit, Overlap, QueryFilter, QueryGeometry, ShapeCast,
};
