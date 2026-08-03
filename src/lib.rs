//! `spatial-query` — a host-agnostic spatial acceleration and query engine.
//!
//! This crate answers "what geometry does this ray / shape / region touch?"
//! over a moving set of objects, without knowing what those objects *are*. A
//! consumer implements a small provider trait (`QueryGeometry`, forthcoming)
//! that projects its own data — render-scene items, physics bodies, anything
//! with a bounding box and a primitive test — down into a neutral vocabulary
//! of AABBs, transforms, and a per-primitive ray test. The crate owns the
//! acceleration structure (a two-level BLAS/TLAS BVH with refit-not-rebuild),
//! the traversal, and the CPU/GPU backend dispatch.
//!
//! # Design commitments
//!
//! - **Host-agnostic core.** The default build pulls in no renderer, no
//!   physics engine, and no `wgpu`. Both `viewport-lib` and `hamilton_engine`
//!   depend on this crate; neither owns it, and it must never carry a
//!   consumer's name into a consumer's dependency-free build.
//! - **Refit, not rebuild.** Dynamic scenes update bounds in place; full
//!   rebuilds are rare and quality-driven.
//! - **Deterministic CPU path.** The CPU traversal is bitwise-repeatable so
//!   replay-sensitive consumers (e.g. Hamilton's determinism mode) can rely on
//!   it. The GPU backend is opt-in and not held to that guarantee.
//!
//! The public API is under active design; see `docs/plans/` for the milestone
//! plan and phase breakdown.

#![forbid(unsafe_code)]

// Intentionally empty for now: the milestone plan (docs/plans) drives the
// build-out phase by phase. This file exists so the crate compiles from the
// first commit and CI has something to run.
