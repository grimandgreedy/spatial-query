//! Device-side batched-ray traversal, behind the optional GPU legs.
//!
//! [`GpuRaycaster`] uploads a built [`Bvh`] and a consumer's device geometry to
//! GPU buffers and answers a whole batch of rays in one compute dispatch, then
//! reads the nearest hit per ray back. It is the backend for the case that
//! motivated the crate: thousands of arbitrary rays from many directions, where
//! per-ray render passes do not help and the CPU walk is linear in the batch.
//!
//! # Three dimensions only
//!
//! The compute kernel works in `vec3<f32>`, so this backend is 3D. The rest of
//! the crate stays dimension-generic; this is the one place `D = 3` is baked in.
//!
//! # The geometry seam
//!
//! The kernel cannot call a consumer's CPU `test_ray`, so the consumer describes
//! its geometry for the device instead, through [`GpuGeometry`]: a flat block of
//! per-leaf primitive parameters plus a WGSL `narrow` function that tests one
//! leaf. The core owns the traversal (the flattened BVH walk and nearest-hit
//! bookkeeping) and stitches the consumer's `narrow` into it. The CPU tree stays
//! canonical; the device layout is derived from it.
//!
//! # Determinism
//!
//! The GPU path is explicitly *not* bitwise-reproducible: floating-point
//! narrow-phase results differ from the CPU by rounding, and exact
//! `time_of_impact` ties between two leaves can resolve to either. Results match
//! the CPU backend within a tolerance, not to the bit. Replay-sensitive
//! consumers stay on the deterministic CPU path.
//!
//! ```no_run
//! use spatial_query::{Bvh, Point, Ray};
//! use spatial_query::gpu::{GpuGeometry, GpuRaycaster};
//!
//! # struct Spheres;
//! # impl spatial_query::QueryGeometry<3> for Spheres {
//! #   type Id = usize; type SubObject = ();
//! #   fn leaf_count(&self) -> usize { 0 }
//! #   fn id(&self, _: usize) -> usize { 0 }
//! #   fn world_aabb(&self, _: usize) -> spatial_query::Aabb<3> { spatial_query::Aabb::new(Point::ZERO, Point::ZERO) }
//! #   fn test_ray(&self, _: usize, _: &Ray<3>, _: f32) -> Option<spatial_query::LeafHit<3>> { None }
//! # }
//! # impl GpuGeometry for Spheres {
//! #   fn prim_stride(&self) -> u32 { 4 }
//! #   fn prim_data(&self) -> Vec<f32> { Vec::new() }
//! #   fn narrow_wgsl(&self) -> String { String::new() }
//! # }
//! # let spheres = Spheres;
//! let bvh = Bvh::build(&spheres);
//! if let Some(gpu) = GpuRaycaster::headless(&bvh, &spheres) {
//!     let rays = [Ray::new(Point([0.0, 0.0, 0.0]), Point([1.0, 0.0, 0.0]))];
//!     let hits = gpu.raycast_nearest_batch(&rays, 100.0);
//!     assert_eq!(hits.len(), 1);
//! }
//! ```

mod backend;

use crate::accel::Bvh;
use crate::maths::{Point, Ray};
use backend::util::DeviceExt;

/// A consumer's geometry described for the device.
///
/// The GPU kernel does the narrow-phase entirely on-device, so the consumer
/// hands over two things: the per-leaf primitive parameters as a flat `f32`
/// block, and a WGSL snippet that tests one leaf against a ray.
///
/// The WGSL from [`narrow_wgsl`](Self::narrow_wgsl) must define exactly:
///
/// ```wgsl
/// fn narrow(leaf: u32, ro: vec3<f32>, rd: vec3<f32>, max_toi: f32) -> Hit { ... }
/// ```
///
/// where `ro` is the ray origin, `rd` the (unit) ray direction, and `max_toi`
/// the current search limit. It returns the kernel-provided `Hit`:
///
/// ```wgsl
/// struct Hit { hit: u32, toi: f32, normal: vec3<f32> }
/// ```
///
/// with `hit = 1u` on a real intersection at `0.0 <= toi <= max_toi` and
/// `hit = 0u` otherwise. Per-leaf parameters are readable as
/// `prims[leaf * params.prim_stride + k]`; the `prims` binding, the `params`
/// uniform (with `prim_stride`), and math intrinsics are already in scope.
pub trait GpuGeometry {
    /// The number of `f32`s of parameters per leaf. The kernel reads leaf `l`'s
    /// block at `prims[l * stride ..][.. stride]`.
    fn prim_stride(&self) -> u32;

    /// The per-leaf parameters, tightly packed and indexed by leaf id: leaf `l`
    /// occupies `[l * prim_stride, (l + 1) * prim_stride)`. Length must be
    /// `leaf_count * prim_stride`.
    fn prim_data(&self) -> Vec<f32>;

    /// WGSL source defining `narrow` (and any helpers it needs), stitched into
    /// the traversal kernel. See the trait docs for the required signature.
    fn narrow_wgsl(&self) -> String;
}

/// One ray's nearest hit from the GPU batch. The core fills the world hit point
/// from the ray; `leaf` is the same leaf index the CPU backend reports and is
/// the identity to cross-check against it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuHit {
    /// The leaf index within the provider (`0..leaf_count`).
    pub leaf: u32,
    /// Distance along the ray direction to the hit.
    pub time_of_impact: f32,
    /// World-space hit position (`ray.at(time_of_impact)`).
    pub point: Point<3>,
    /// Outward surface normal reported by the consumer's `narrow`.
    pub normal: Point<3>,
}

/// The fixed traversal kernel. The consumer's `narrow` is spliced in at the
/// marker; everything else (the flattened BVH walk, the nearest-hit reduction
/// with a leaf tie-break, the output packing) is owned here.
const KERNEL_TEMPLATE: &str = r#"
struct Params {
    max_toi: f32,
    ray_count: u32,
    prim_stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Three vec4s per node: [min.xyz | child_a], [max.xyz | child_b], [prim_count | _ | _ | _].
// A leaf has prim_count > 0 and child_a as the start index into prim_indices;
// an interior node has prim_count == 0 and child_a / child_b as node indices.
@group(0) @binding(1) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> prim_indices: array<u32>;
@group(0) @binding(3) var<storage, read> prims: array<f32>;
// Two vec4s per ray: [origin.xyz | _], [dir.xyz | _]. Two per hit likewise:
// [toi | leaf-bits | found-bits | _], [normal.xyz | _].
@group(0) @binding(4) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> hits: array<vec4<f32>>;

struct Hit {
    hit: u32,
    toi: f32,
    normal: vec3<f32>,
};

//__NARROW__

// Slab test: does the ray reach the box within `limit`? Conservative; used only
// to prune, never to decide a hit.
fn slab(ro: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, limit: f32) -> bool {
    let t0 = (lo - ro) * inv;
    let t1 = (hi - ro) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let enter = max(max(tmin.x, tmin.y), tmin.z);
    let exit = min(min(tmax.x, tmax.y), tmax.z);
    return exit >= max(enter, 0.0) && enter <= limit;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ri = gid.x;
    if (ri >= params.ray_count) {
        return;
    }
    let ro = rays[ri * 2u].xyz;
    let rd = rays[ri * 2u + 1u].xyz;
    let inv = vec3<f32>(1.0) / rd;

    var limit = params.max_toi;
    var found = 0u;
    var best_toi = params.max_toi;
    var best_leaf = 0u;
    var best_n = vec3<f32>(0.0, 0.0, 0.0);

    // Iterative DFS. The stack holds node indices; its depth is bounded by the
    // tree height, so 64 covers any tree this crate builds.
    var stack: array<u32, 64>;
    var sp = 0u;
    stack[0] = 0u;
    sp = 1u;
    loop {
        if (sp == 0u) {
            break;
        }
        sp = sp - 1u;
        let ni = stack[sp];
        let base = ni * 3u;
        let lo = nodes[base].xyz;
        let hi = nodes[base + 1u].xyz;
        if (!slab(ro, inv, lo, hi, limit)) {
            continue;
        }
        let prim_count = bitcast<u32>(nodes[base + 2u].x);
        let ca = bitcast<u32>(nodes[base].w);
        let cb = bitcast<u32>(nodes[base + 1u].w);
        if (prim_count > 0u) {
            var i = 0u;
            loop {
                if (i >= prim_count) {
                    break;
                }
                let leaf = prim_indices[ca + i];
                let h = narrow(leaf, ro, rd, limit);
                // Nearest by (toi, then leaf), matching the CPU total order.
                if (h.hit == 1u && h.toi <= limit) {
                    if (found == 0u || h.toi < best_toi || (h.toi == best_toi && leaf < best_leaf)) {
                        found = 1u;
                        best_toi = h.toi;
                        best_leaf = leaf;
                        best_n = h.normal;
                        limit = h.toi;
                    }
                }
                i = i + 1u;
            }
        } else {
            stack[sp] = ca;
            sp = sp + 1u;
            stack[sp] = cb;
            sp = sp + 1u;
        }
    }

    hits[ri * 2u] = vec4<f32>(best_toi, bitcast<f32>(best_leaf), bitcast<f32>(found), 0.0);
    hits[ri * 2u + 1u] = vec4<f32>(best_n, 0.0);
}
"#;

/// A device-side batched raycaster over a built tree and a device geometry.
///
/// Build once with [`new`](Self::new) (sharing a consumer's device) or
/// [`headless`](Self::headless) (creating its own), then call
/// [`raycast_nearest_batch`](Self::raycast_nearest_batch) per batch. When the
/// scene moves but keeps its topology, [`refit`](Self::refit) re-uploads node
/// bounds and primitive parameters in place, without rebuilding the tree or the
/// pipeline.
pub struct GpuRaycaster {
    device: backend::Device,
    queue: backend::Queue,
    pipeline: backend::ComputePipeline,
    node_buf: backend::Buffer,
    prim_index_buf: backend::Buffer,
    prim_buf: backend::Buffer,
    node_count: u32,
    prim_stride: u32,
}

impl GpuRaycaster {
    /// Build a raycaster on a device the consumer already owns. The device and
    /// queue are cloned (both are cheap handles), so the caller keeps using
    /// theirs. Returns the pipeline and the uploaded scene.
    pub fn new<G: GpuGeometry>(
        device: &backend::Device,
        queue: &backend::Queue,
        bvh: &Bvh<3>,
        geom: &G,
    ) -> Self {
        let node_count = bvh.node_count() as u32;
        let prim_stride = geom.prim_stride();

        let source = KERNEL_TEMPLATE.replace("//__NARROW__", &geom.narrow_wgsl());
        let module = device.create_shader_module(backend::ShaderModuleDescriptor {
            label: Some("spatial-query gpu traversal"),
            source: backend::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&backend::ComputePipelineDescriptor {
            label: Some("spatial-query gpu traversal"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Buffers never shrink to zero: an empty scene still gets one padding
        // element so buffer creation and binding stay valid; raycasts short out.
        let node_buf = device.create_buffer_init(&backend::util::BufferInitDescriptor {
            label: Some("spatial-query nodes"),
            contents: bytemuck::cast_slice(&pack_nodes(bvh)),
            usage: backend::BufferUsages::STORAGE | backend::BufferUsages::COPY_DST,
        });
        let prim_index_buf = device.create_buffer_init(&backend::util::BufferInitDescriptor {
            label: Some("spatial-query prim indices"),
            contents: bytemuck::cast_slice(&pad_u32(bvh.prim_indices())),
            usage: backend::BufferUsages::STORAGE,
        });
        let prim_buf = device.create_buffer_init(&backend::util::BufferInitDescriptor {
            label: Some("spatial-query prims"),
            contents: bytemuck::cast_slice(&pad_f32(&geom.prim_data())),
            usage: backend::BufferUsages::STORAGE | backend::BufferUsages::COPY_DST,
        });

        GpuRaycaster {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            node_buf,
            prim_index_buf,
            prim_buf,
            node_count,
            prim_stride,
        }
    }

    /// Build a raycaster on its own headless device, or `None` if no GPU adapter
    /// is available (headless CI, no driver). Convenient for tests and batch
    /// tools that do not already hold a device.
    pub fn headless<G: GpuGeometry>(bvh: &Bvh<3>, geom: &G) -> Option<Self> {
        let instance = backend::default_instance();
        let adapter =
            pollster::block_on(instance.request_adapter(&backend::RequestAdapterOptions {
                power_preference: backend::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&backend::DeviceDescriptor {
                label: Some("spatial-query headless"),
                ..Default::default()
            }))
            .ok()?;
        Some(Self::new(&device, &queue, bvh, geom))
    }

    /// Re-upload node bounds and primitive parameters from a refit tree, keeping
    /// the topology and the pipeline. `bvh` must be the same tree (same node and
    /// leaf count) after a [`Bvh::refit`]; `geom` supplies the moved parameters.
    ///
    /// This writes into the existing buffers, so a moving scene re-queries
    /// without rebuilding the tree, re-creating the pipeline, or re-uploading the
    /// primitive index topology.
    pub fn refit<G: GpuGeometry>(&mut self, bvh: &Bvh<3>, geom: &G) {
        self.queue
            .write_buffer(&self.node_buf, 0, bytemuck::cast_slice(&pack_nodes(bvh)));
        self.queue.write_buffer(
            &self.prim_buf,
            0,
            bytemuck::cast_slice(&pad_f32(&geom.prim_data())),
        );
    }

    /// The nearest hit for each ray, one entry per input ray in the same order.
    /// A ray that hits nothing within `max_toi` yields `None`.
    ///
    /// The whole batch runs in one compute dispatch. Directions are arbitrary;
    /// there is no per-direction cost as there would be for a render-pass picker.
    pub fn raycast_nearest_batch(&self, rays: &[Ray<3>], max_toi: f32) -> Vec<Option<GpuHit>> {
        if rays.is_empty() {
            return Vec::new();
        }
        if self.node_count == 0 {
            return vec![None; rays.len()];
        }

        let ray_count = rays.len() as u32;
        let ray_data = pack_rays(rays);
        let ray_buf = self
            .device
            .create_buffer_init(&backend::util::BufferInitDescriptor {
                label: Some("spatial-query rays"),
                contents: bytemuck::cast_slice(&ray_data),
                usage: backend::BufferUsages::STORAGE,
            });

        // Output: two vec4 per ray. Read back through a MAP_READ staging buffer.
        let hit_bytes = (rays.len() * 2 * 16) as u64;
        let hit_buf = self.device.create_buffer(&backend::BufferDescriptor {
            label: Some("spatial-query hits"),
            size: hit_bytes,
            usage: backend::BufferUsages::STORAGE | backend::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&backend::BufferDescriptor {
            label: Some("spatial-query hits readback"),
            size: hit_bytes,
            usage: backend::BufferUsages::MAP_READ | backend::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params: [u32; 4] = [max_toi.to_bits(), ray_count, self.prim_stride, 0];
        let param_buf = self
            .device
            .create_buffer_init(&backend::util::BufferInitDescriptor {
                label: Some("spatial-query params"),
                contents: bytemuck::cast_slice(&params),
                usage: backend::BufferUsages::UNIFORM,
            });

        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self
            .device
            .create_bind_group(&backend::BindGroupDescriptor {
                label: Some("spatial-query bindings"),
                layout: &layout,
                entries: &[
                    backend::BindGroupEntry {
                        binding: 0,
                        resource: param_buf.as_entire_binding(),
                    },
                    backend::BindGroupEntry {
                        binding: 1,
                        resource: self.node_buf.as_entire_binding(),
                    },
                    backend::BindGroupEntry {
                        binding: 2,
                        resource: self.prim_index_buf.as_entire_binding(),
                    },
                    backend::BindGroupEntry {
                        binding: 3,
                        resource: self.prim_buf.as_entire_binding(),
                    },
                    backend::BindGroupEntry {
                        binding: 4,
                        resource: ray_buf.as_entire_binding(),
                    },
                    backend::BindGroupEntry {
                        binding: 5,
                        resource: hit_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&backend::CommandEncoderDescriptor {
                label: Some("spatial-query dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&backend::ComputePassDescriptor {
                label: Some("spatial-query traversal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 64 rays per workgroup, matching the kernel's workgroup_size.
            pass.dispatch_workgroups(ray_count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&hit_buf, 0, &staging, 0, hit_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(backend::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(backend::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        rx.recv()
            .expect("map_async callback fires")
            .expect("hits map for read");

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        let hits = (0..rays.len())
            .map(|i| decode_hit(words, i, rays[i]))
            .collect();
        drop(data);
        staging.unmap();
        hits
    }
}

/// One GPU hit record: `[toi, leaf_bits, found_bits, _] , [nx, ny, nz, _]`.
fn decode_hit(words: &[u32], i: usize, ray: Ray<3>) -> Option<GpuHit> {
    let base = i * 8;
    let found = words[base + 2];
    if found == 0 {
        return None;
    }
    let toi = f32::from_bits(words[base]);
    let leaf = words[base + 1];
    let normal = Point([
        f32::from_bits(words[base + 4]),
        f32::from_bits(words[base + 5]),
        f32::from_bits(words[base + 6]),
    ]);
    Some(GpuHit {
        leaf,
        time_of_impact: toi,
        point: ray.at(toi),
        normal,
    })
}

/// Pack the tree into three `vec4<f32>` per node (see the kernel comment). Index
/// fields are bit-cast into the `f32` slots. Always at least one node so the
/// buffer is never empty.
fn pack_nodes(bvh: &Bvh<3>) -> Vec<f32> {
    let nodes = bvh.nodes();
    if nodes.is_empty() {
        return vec![0.0; 12];
    }
    let mut out = Vec::with_capacity(nodes.len() * 12);
    for n in nodes {
        let mn = n.bounds.min.0;
        let mx = n.bounds.max.0;
        out.extend_from_slice(&[mn[0], mn[1], mn[2], f32::from_bits(n.child_a)]);
        out.extend_from_slice(&[mx[0], mx[1], mx[2], f32::from_bits(n.child_b)]);
        out.extend_from_slice(&[f32::from_bits(n.prim_count), 0.0, 0.0, 0.0]);
    }
    out
}

/// Two `vec4<f32>` per ray: origin then (unit) direction.
fn pack_rays(rays: &[Ray<3>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rays.len() * 8);
    for r in rays {
        let o = r.origin.0;
        let d = r.dir.0;
        out.extend_from_slice(&[o[0], o[1], o[2], 0.0]);
        out.extend_from_slice(&[d[0], d[1], d[2], 0.0]);
    }
    out
}

/// A never-empty `u32` copy (buffer creation rejects zero-size).
fn pad_u32(v: &[u32]) -> Vec<u32> {
    if v.is_empty() {
        vec![0]
    } else {
        v.to_vec()
    }
}

/// A never-empty `f32` copy.
fn pad_f32(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        vec![0.0]
    } else {
        v.to_vec()
    }
}
