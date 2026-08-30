# spatial-query

A spatial query engine over an arbitrary set of objects in n-space.

The motivation came from picking in viewport-lib. Its GPU pick is a pixel pick
from the camera direction, so a single render pass gives you the object under the
cursor. That does not stretch to general queries: 100 raycasts from 100
directions would mean rendering the scene once per direction, and it says nothing
about "which objects overlap this box" or "what does this swept shape hit".

So this crate does the query side on its own. You describe your objects once and
it builds a BVH over them; then you ask ray, shape-cast, and overlap queries. It
knows nothing about rendering or any particular host: the point type is a
const-generic `[f32; D]`, not a `glam::Vec3`, so the core carries no glam, no
wgpu, and no renderer.

It is fully general, but in practice it is mostly an internal tool for viewport-lib
and its siblings, so the API may still change.

## Using it

Implement `QueryGeometry` for whatever you want to query over, then build a `Bvh`
and cast against it. The core does the broad phase (the BVH walk); your `test_ray`
does the exact narrow phase and can report a sub-object. There is a two-level
`InstancedGeometry` / `Tlas` variant for scenes of rigidly transformed instances.

```rust
let bvh = Bvh::build(&scene);
let ray = Ray::new(Point([0.0, 0.0, 5.0]), Point([0.0, 0.0, -1.0]));
let hit = bvh.raycast_nearest(&scene, &ray, 100.0, &QueryFilter::default());
```

## Features

The default build is the pure CPU core with no optional dependencies.

- `gpu` (`wgpu29` / `wgpu27` legs): device-side batched-ray traversal for large
  batches, with a consumer-supplied WGSL narrow phase.
- `f64`: double-precision query maths.
