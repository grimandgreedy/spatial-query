//! Backend abstraction and cost-model dispatch for batched queries.
//!
//! For a batch of queries the [`Dispatcher`] estimates the cost of each
//! registered [`QueryBackend`] and runs the cheapest one that can serve the
//! request. CPU traversal is always available and always deterministic; a
//! consumer can register alternates (a draw-pass picker, a GPU-compute backend)
//! that win for their own regimes.
//!
//! The cost estimates are advisory and uncalibrated: they encode the *shape* of
//! each backend's cost, not measured constants. Do not trust the numbers as
//! absolute times until a benchmark study calibrates them. Two guardrails make a
//! wrong model safe: a deterministic or headless request always pins to CPU, and
//! the caller can force a backend outright.

use crate::accel::Bvh;
use crate::maths::{Ray, Scalar};
use crate::query::filter::QueryFilter;
use crate::query::geometry::QueryGeometry;
use crate::query::hit::Hit;

/// The per-ray nearest-hit results of a batch: one `Option<Hit>` per input ray,
/// in the same order.
pub type BatchHits<Id, const D: usize, S> = Vec<Option<Hit<Id, D, S>>>;

/// The three backend classes the cost model reasons about.
///
/// This crate implements only [`Cpu`](BackendKind::Cpu) (via [`CpuBackend`]).
/// The other two are named and cost-modelled here but implemented elsewhere and
/// registered through [`QueryBackend`]: `DrawPass` by a rendering consumer's
/// adapter, `Gpu` by the optional compute backend. Naming them lets the model
/// weigh their distinctive cost shapes without depending on either; a consumer
/// that never registers one simply never has it selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendKind {
    /// Deterministic CPU traversal. The only backend this crate implements, and
    /// the always-available fallback that deterministic and headless requests
    /// pin to.
    Cpu,
    /// A consumer-registered draw-pass picker; the canonical example is a
    /// renderer's GPU ID-buffer pick (render object ids to an offscreen target,
    /// read back the pixel under the cursor). It resolves one view direction per
    /// render pass, so it is nearly free for a single view because it rides the
    /// frame's existing draw, and costs another pass for each extra direction.
    /// That makes it win for a single cursor click or a rectangle select on a
    /// large scene, and lose as the number of distinct directions grows. Not
    /// implemented here; a rendering consumer supplies and registers it.
    DrawPass,
    /// A GPU-compute batched traversal: upload the tree and geometry once, then
    /// dispatch over the whole ray batch, direction-agnostic. It wins for large
    /// batches of arbitrary rays where a draw-pass would need many passes and the
    /// CPU walk is linear in the batch. Not implemented here; it is the optional
    /// device-side backend behind the `gpu` feature.
    Gpu,
}

/// The shape and environment of a batch request, read by the cost model.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct QueryContext {
    /// Number of queries in the batch.
    pub batch_size: usize,
    /// Distinct ray directions in the batch: about one for a single-view cursor
    /// pick or a rectangle select, up to `batch_size` for scattered rays.
    pub distinct_directions: usize,
    /// Number of leaves in the scene.
    pub scene_leaves: usize,
    /// Whether a GPU device is available for device-side backends.
    pub device_available: bool,
    /// Whether the caller requires the deterministic path (e.g. replay). When
    /// set, selection pins to CPU regardless of cost.
    pub require_deterministic: bool,
}

impl QueryContext {
    /// A context for `batch_size` rays spanning `distinct_directions` directions
    /// over a scene of `scene_leaves` leaves, with no device and no determinism
    /// requirement.
    pub fn new(batch_size: usize, distinct_directions: usize, scene_leaves: usize) -> Self {
        QueryContext {
            batch_size,
            distinct_directions,
            scene_leaves,
            device_available: false,
            require_deterministic: false,
        }
    }

    /// Mark whether a GPU device is available.
    pub fn with_device(mut self, available: bool) -> Self {
        self.device_available = available;
        self
    }

    /// Mark whether the deterministic path is required.
    pub fn deterministic(mut self, required: bool) -> Self {
        self.require_deterministic = required;
        self
    }
}

/// A backend that can answer a batch of raycasts. The CPU backend
/// ([`CpuBackend`]) is the always-available default; consumers register others.
pub trait QueryBackend<const D: usize, G: QueryGeometry<D>> {
    /// Which backend this is.
    fn kind(&self) -> BackendKind;

    /// Whether this backend can serve `ctx` at all (a draw-pass or GPU backend
    /// needs a device, for instance). The CPU backend serves everything.
    fn can_serve(&self, ctx: &QueryContext) -> bool;

    /// An advisory relative cost for `ctx`; lower is cheaper. Only the ordering
    /// between backends matters, and only until calibration.
    fn estimate_cost(&self, ctx: &QueryContext) -> f64;

    /// Answer the batch: the nearest hit for each ray, in the same order.
    fn raycast_nearest_batch(
        &self,
        g: &G,
        rays: &[Ray<D>],
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> BatchHits<G::Id, D, G::SubObject>;
}

/// The always-available CPU traversal backend, over a single-level [`Bvh`].
pub struct CpuBackend<'a, const D: usize> {
    bvh: &'a Bvh<D>,
}

impl<'a, const D: usize> CpuBackend<'a, D> {
    /// Wrap a built tree as the CPU backend.
    pub fn new(bvh: &'a Bvh<D>) -> Self {
        CpuBackend { bvh }
    }
}

impl<const D: usize, G: QueryGeometry<D>> QueryBackend<D, G> for CpuBackend<'_, D> {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn can_serve(&self, _ctx: &QueryContext) -> bool {
        true
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // One tree walk per ray, roughly the tree depth in node tests.
        let per_ray = ((ctx.scene_leaves + 1) as f64).log2().max(1.0);
        ctx.batch_size as f64 * per_ray
    }

    fn raycast_nearest_batch(
        &self,
        g: &G,
        rays: &[Ray<D>],
        max_toi: Scalar,
        filter: &QueryFilter,
    ) -> BatchHits<G::Id, D, G::SubObject> {
        rays.iter()
            .map(|ray| self.bvh.raycast_nearest(g, ray, max_toi, filter))
            .collect()
    }
}

/// Chooses among registered backends per batch and runs the choice.
///
/// Built with a mandatory CPU backend so there is always a deterministic
/// fallback; alternates are registered on top.
pub struct Dispatcher<'a, const D: usize, G: QueryGeometry<D>> {
    backends: Vec<Box<dyn QueryBackend<D, G> + 'a>>,
    forced: Option<BackendKind>,
}

impl<'a, const D: usize, G: QueryGeometry<D>> Dispatcher<'a, D, G> {
    /// A dispatcher whose default backend is CPU traversal over `bvh`.
    pub fn new(bvh: &'a Bvh<D>) -> Self {
        Dispatcher {
            backends: vec![Box::new(CpuBackend::new(bvh))],
            forced: None,
        }
    }

    /// Register an alternate backend (a draw-pass picker, a GPU backend).
    pub fn register(&mut self, backend: Box<dyn QueryBackend<D, G> + 'a>) {
        self.backends.push(backend);
    }

    /// Force every selection to `kind`, or clear the override with `None`. An
    /// override is absolute: it bypasses the determinism pin, so use it only when
    /// you know the backend is acceptable.
    pub fn force(&mut self, kind: Option<BackendKind>) {
        self.forced = kind;
    }

    /// The backend the cost model would pick for `ctx`, without running it.
    pub fn select(&self, ctx: &QueryContext) -> BackendKind {
        if let Some(kind) = self.forced {
            return kind;
        }
        if ctx.require_deterministic {
            return BackendKind::Cpu;
        }
        self.backends
            .iter()
            .filter(|b| b.can_serve(ctx))
            .min_by(|a, b| a.estimate_cost(ctx).total_cmp(&b.estimate_cost(ctx)))
            .map(|b| b.kind())
            .unwrap_or(BackendKind::Cpu)
    }

    /// Select a backend for the batch and run it. Returns which backend ran and
    /// the nearest hit per ray. `ctx`'s `batch_size` is taken from `rays`.
    pub fn raycast_nearest_batch(
        &self,
        g: &G,
        rays: &[Ray<D>],
        max_toi: Scalar,
        filter: &QueryFilter,
        ctx: &QueryContext,
    ) -> (BackendKind, BatchHits<G::Id, D, G::SubObject>) {
        let mut ctx = *ctx;
        ctx.batch_size = rays.len();
        let kind = self.select(&ctx);
        let backend = self
            .backends
            .iter()
            .find(|b| b.kind() == kind)
            .expect("selected backend is registered");
        (
            kind,
            backend.raycast_nearest_batch(g, rays, max_toi, filter),
        )
    }
}
