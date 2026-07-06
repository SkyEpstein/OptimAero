# Phase 1 — Data Foundation (spec)

*SDD: specify → clarify. Governed by `memory/constitution.md`.*
Status: **DRAFT** — resolving the `[NEEDS CLARIFICATION]` markers via MCQ.
Date: 2026-07-05

## Problem

The Phase 2 nested bake-off cannot start without a clean, leakage-controlled dataset of
airfoil **section aerodynamics**. Phase 1 produces that dataset and — the part that actually
protects every downstream number — the **leakage map and evaluation contract**.

## Scope

- **In:** curate public airfoil data; generate our own controlled polars; implement the two
  evaluation splits (new-geometry headline, new-condition secondary); version data with
  per-row provenance and fidelity labels.
- **Out:** 3D/vehicle data (Phase 3+); any model training (Phase 2).

## Requirements

- **R1.** A unified table `(geometry, α, Re, M) → (C_l, C_d, C_m)` with per-row provenance
  and a **fidelity label** (e.g. `xfoil`, `rans-airfrans`, `cfd`, `windtunnel`).
- **R2.** Geometry stored in a canonical form: normalized coordinates **+** a parametric
  fit. `[NEEDS CLARIFICATION: parametric form — CST vs PARSEC vs B-spline. The predictor's
  input representation is a Phase 2 bake-off dimension, but a canonical storage form is
  chosen here.]`
- **R3.** **New-geometry split (headline):** an "airfoil family" grouping such that no shape
  or its scaled/rotated/near-duplicate variants straddle train/test. **Family = hybrid
  (decided 2026-07-05):** group by catalogued provenance/lineage (NACA 4-/5-digit, Selig,
  Eppler, etc.) **and** merge families that are near-duplicates in CST/coordinate space
  (scaled/rotated/thickened variants across sources). An automated near-duplicate check (CST
  or coordinate-distance threshold) merges geometric twins that lineage alone misses; the
  threshold is recorded and justified in the data card.
- **R4.** **New-condition split (secondary):** a known shape may recur across splits at new
  `α/Re/M`; define and implement its construction.
- **R5.** **Data sources & mix (decided 2026-07-05): XFOIL backbone + higher-fidelity
  anchors.** Backbone = XFOIL over the UIUC set (~1,600 airfoils) across the R6 grid — fast,
  consistent, fully controlled. Anchors = AirfRANS (RANS) now; real CFD (SU2/OpenFOAM) and
  wind-tunnel later — each row fidelity-labeled (R1), used for validation and a future
  fidelity-correction model, never silently blended into the backbone.
- **R6.** Generation grid — proposed default (adjustable): `α ∈ [−8°, +18°]` step 1°;
  `Re ∈ {5e4, 1e5, 2e5, 5e5, 1e6}`; `M ∈ {0.0–0.3}` (low-speed first).
- **R7.** Data-generation scripts checked in; seeds fixed; a documented QA pass (e.g. XFOIL
  convergence filtering, post-stall reliability flags — XFOIL is approximate near/after stall
  and at very low Re, so those rows are labeled, not silently trusted).

## Acceptance criteria

- **AC1.** Dataset builds reproducibly from scripts; row count + provenance + fidelity mix
  reported honestly (no inflated counts).
- **AC2.** Both splits implemented and **verified to have zero family leakage** (new-geometry)
  and correct construction (new-condition), with an automated check.
- **AC3.** A **data card** documents sources, conditions, counts, fidelity mix, and known gaps.

## Open clarifications

- **C1** — RESOLVED (2026-07-05): hybrid family = lineage + shape-space near-duplicate
  merge. See R3.
- **C2** — RESOLVED (2026-07-05): XFOIL backbone + higher-fidelity anchors. See R5.
- **C3** — OPEN: canonical parametric storage form (R2). Deferred to the Phase 2
  representation bake-off unless it affects storage. Default storage = normalized
  coordinates **+** a CST fit.
