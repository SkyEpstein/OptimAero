# Stage B — CFD-trained 3D surrogate (spec)

*SDD: specify. Governed by `memory/constitution.md`. Follows the 3D-enclosure/airframe work.*
Date: 2026-07-06

## Problem

The airframe designer currently evaluates aero with AeroSandbox's physics methods — not Sky's
ML. Stage B trains a **3D surrogate + confidence model on OpenFOAM CFD data** and drops it into
the designer, so the design loop is driven by Sky's own ML. Because the surrogate is fast
(~ms/eval), the optimizer can then run **thousands** of candidate airframes (CFD cannot) —
which is the entire reason to use ML instead of CFD, and Sky's stated vision.

## Approach (mirrors the proven 2D pipeline, now in 3D)

1. **F2 gate (IN PROGRESS):** verify OpenFOAM steady-RANS runs in Docker on this Mac end-to-end
   (STL → snappyHexMesh → simpleFoam → forceCoeffs) and measure per-case cost. *Everything below
   is sized from this result — no pipeline is built until F2 reports.*
2. **Geometry parameterization + sampling:** a parametric family (bodies/wings/airframes) with
   a modest number of design variables; sample it (space-filling) at feasible scale.
   `[DECISION post-F2: bodies-only (easy auto-mesh) vs wing+body airframes (harder mesh).]`
3. **CFD generation:** for each sample, auto-generate geometry → mesh → solve → extract
   Cl/Cd(/Cm). Parallelized across cores; per-airfoil-style sharding + resume. Scale set by F2
   cost (likely hundreds–low-thousands over days, NOT the 213k of 2D).
4. **Bake-off:** predictor + confidence on `(geometry params, Re/V) → (Cl, Cd)`, group-aware
   splits, honest R²+RMSE, selective prediction — the same nested bake-off as Phase 2.
5. **Wire in + verify-in-the-loop:** swap the surrogate into the airframe designer; the
   optimizer's optimum is verified against OpenFOAM (the CFD-fallback). Then crank the candidate
   count to thousands (fast, because the surrogate is fast).

## Requirements

- **R1.** Robust auto-meshing: every sampled geometry meshes + solves without manual work
  (the pipeline can't need hand-holding per case). Failures are logged + skipped, not faked.
- **R2.** Honesty discipline (constitution): no leakage (group-aware splits), R²+RMSE together,
  verify-against-CFD, honest data card (counts, convergence yield, cost).
- **R3.** The surrogate is fast enough (~ms) that the designer can evaluate thousands of
  candidates per design.

## Feasibility gates

- **F2 (running):** OpenFOAM RANS in Docker works end-to-end + per-case cost is acceptable at
  the target dataset scale. If a case is too expensive (e.g. >~15 min) or meshing is not
  robust, re-scope (coarser mesh, bodies-only, or fewer/simpler samples) — reported honestly.
