# ML-Pluggable Architecture (spec)

*SDD: specify → clarify. Governed by `memory/constitution.md`.*
Date: 2026-07-06

## Problem

Build the rest of OptimAero so the trained ML (surrogate + confidence) is a **drop-in
component**, not the whole app. Everything downstream is written against one interface —
`optimaero.surrogate.Surrogate` — and validated NOW using the `NeuralFoilSurrogate`
placeholder, then the Phase-2 bake-off produces the real model behind the same interface.

## The plug-in contract (DONE)

`Surrogate.predict(coords, alpha, Re, mach) -> AeroPrediction(Cl, Cd, Cm, *_err, trusted, ood)`.
The confidence fields (`*_err`, `trusted`, `ood`) are the trust-gate → CFD-fallback mechanism.

## Component map (all consume `Surrogate`)

| Module | Role | Blocking on |
|---|---|---|
| `surrogate.py` | interface + placeholder | ✅ done |
| `bakeoff/` | Phase 2: train predictor + confidence → a trained `Surrogate` | backbone data (finishing) |
| `physics/bemt.py`, `physics/lifting_line.py` | section → propeller/wing/UAV performance | nothing (physics) |
| `requirements.py` | user requirements (top speed, lift, drag, envelope) → objective + constraints | nothing |
| `optimize/inverse_design.py` | envelope-constrained shape optimization over CST space | `[C-OPT]` optimizer choice |
| `cad/io.py` | STEP/STL/IGES import (envelope) + export (optimized shape) | `[C-CAD]` CAD library |

## Requirements

- **R1.** Downstream modules depend ONLY on `Surrogate` (swap real model in with zero changes).
- **R2.** End-to-end demo runnable now with the placeholder: envelope + requirements → optimized
  airfoil/section → CAD export, with per-evaluation confidence surfaced.
- **R3.** The optimizer respects (a) the packaging envelope (hard constraint), (b) the
  requirement targets, and (c) the confidence gate — it should prefer shapes the surrogate is
  *confident* about, and flag when the optimum sits in low-confidence territory.
- **R4.** CAD I/O round-trips through neutral formats without lock-in.

## Clarifications (resolved 2026-07-06)

- **[C-DATA]** → **Confidence-driven active-learning augmentation.** Build surrogate +
  confidence first; then generate synthetic CST-perturbed airfoils (via our XFOIL)
  *targeted where the confidence model is uncertain*. The confidence model directs its own
  data collection. (AirfRANS anchor already ingested as the higher-fidelity reference.)
- **[C-OPT]** → **Black-box optimizer** (CMA-ES / genetic; scipy `differential_evolution`
  for v1). Works with any bake-off winner (LightGBM isn't differentiable); decoupled from
  the model class.
- **[C-CAD]** → **CadQuery / OpenCASCADE** — true STEP (parametric) + STL/IGES import/export.

## AirfRANS anchor — status & caveats (recorded for the data card)

- 1,000 RANS sims ingested (`airfrans_anchor.parquet`), `fidelity=rans-airfrans`.
- **Re = 2–6M** — does NOT overlap the XFOIL backbone (≤1M): extends coverage, does not
  directly validate XFOIL. A matched-condition fidelity study needs future data.
- **No per-airfoil geometry** in the mirror (mesh-only, index IDs, Cm=NaN) → not yet
  featurizable for the surrogate / shape-space splits. Recover geometry from the PLAID mesh
  before training on it.
