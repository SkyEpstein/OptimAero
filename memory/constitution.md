# OptimAero — Project Constitution

*SDD phase 0. This document governs every later phase. The `analyze` step of every
change checks compliance with it. Amendments require an explicit note in `CHANGELOG.md`.*

Status: **DRAFT — awaiting Sky's confirmation.** All sections drafted; §4 (confidence
model) resolved from the verified `f-elements-2` formulation. No open `[NEEDS CLARIFICATION]`.

Last updated: 2026-07-05

---

## 1. Mission

Build **OptimAero**: a tool where the user **imports a CAD file** of the volume their
components must occupy and selects a purpose, and OptimAero grows an **optimized aerodynamic
enclosure** around that volume and **exports it as CAD** (STEP/STL). Workflow: **import CAD →
aerodynamic optimization → export CAD**. Under the hood is an *uncertainty-aware
machine-learning surrogate* that replaces/augments CFD, wrapped in an inverse-design optimizer
with **verify-against-truth** so the reported performance is never the surrogate's unverified
guess.

The scientific thesis (from Sky's research paper): **a well-calibrated ML surrogate,
which knows when it is uncertain and defers to real CFD, can stand in for CFD across most
of the design loop** — delivering millisecond evaluations without sacrificing trust.

## 2. Scope

**Core target (revised 2026-07-06 — the true product vision).** The user supplies a **3D
volume** their internal components must fit inside and selects a **purpose**; OptimAero grows
an **optimized aerodynamic enclosure** around that volume (min drag / purpose objective,
subject to fully containing the volume) and exports it as CAD. This is *aerodynamic-enclosure
optimization in 3D* — the literal "a shape that must fit in the final product." Delivered in
two stages: **Stage A** a working pipeline on a fast 3D method (AeroSandbox panel/component
buildup); **Stage B** a learned surrogate + confidence trained on feasible-scale 3D CFD
(OpenFOAM), with verify-against-CFD. Data strategy mirrors the proven 2D pattern (fast
backbone + higher-fidelity anchor).

**Validated foundation (the 2D work, complete).** 2D airfoils/wings — section aero
`C_l, C_d, C_m(α, Re, M)`, low-speed, `Re ≈ 1e4–1e6`. This proved the *methodology* that now
transfers to 3D: leakage-controlled bake-off, the confidence model, inverse design with
verify-against-truth, CAD I/O, the GUI. It is a reusable base, not the final deliverable.

**Application (v1):** drones / UAVs / propellers — which are physically *composed of
airfoil sections*, so the section surrogate is the reusable core.

**3D bridge (staged):** physics-coupled first (BEMT for propellers, lifting-line/panel for
wings), then a *learned 3D residual-correction* model once 3D CFD data exists.

**In scope:** curated + self-generated training data; the forward surrogate; the
confidence/UQ model; physics coupling to vehicle-level metrics; the inverse-design
optimizer; neutral-format CAD I/O.

**Out of scope (v1):** transonic/supersonic and strong compressibility; automotive/ground-
vehicle aero; aeroacoustics; structural/multidisciplinary coupling; a bespoke commercial-
CAD plugin. Revisit in later phases.

## 3. Build order (roadmap)

- **Phase 0 — Constitution** *(this document).*
- **Phase 1 — Data foundation.** Curate UIUC airfoil DB + AirfRANS + NASA TMR; build a
  controlled XFOIL generation pipeline for `C_l/C_d/C_m(α, Re, M)`; author the **leakage
  map** and the **evaluation contract** before any model is trained.
- **Phase 2 — Forward section surrogate (nested bake-off).** (1) *Predictor bake-off* over a
  wide pool of models **and** ensembles, ranked on held-out **new-geometry RMSE** → keep the
  **top-K** (default: top 3 + best ensemble). (2) *Confidence bake-off per surviving
  predictor* — error-model/ensemble candidates trained on that predictor's **out-of-fold
  residuals**. (3) *Winner = the (predictor, confidence) pair* that minimizes RMSE on the
  **retained** points at target coverage ("deployed trust-gated accuracy"), gated by
  calibration (empirical≈nominal coverage) and Spearman. All under group-aware CV; R² and
  RMSE reported together; top-X% R² is never the selection metric.
- **Phase 3 — Physics-coupled vehicle prediction.** BEMT / lifting-line assembling section
  predictions into propeller/UAV/wing performance, with uncertainty propagated section→vehicle.
- **Phase 4 — Inverse design.** Envelope-constrained optimizer over shape space + neutral-
  format CAD I/O (STEP/STL/IGES).
- **Phase 5 — 3D residual correction.** Learned correction over the physics coupling,
  validated against higher-fidelity 3D CFD.

## 4. The confidence model

Formulation carried over (verified against the source code) from Sky's
`Machine-learning-to-separate-f-elements-2` project. It is a **learned error model** used
as a trust gate — *not* an ensemble/dropout heuristic:

1. **Error model.** A *separate* learned regressor (LightGBM in the f-elements work) is fit
   on the primary surrogate's **absolute residual** `|ŷ − y|`, one per output
   (`C_l, C_d, C_m`). It is its own model fit on residuals — its algorithm may match the
   primary's or differ (e.g. the deployed f-elements Track A is LightGBM-on-LightGBM). The
   *predicted error* is floored to a small positive value (1e-6/0.05 in the source) so the
   conformal ratio below stays well-defined.
2. **Out-of-fold training (the honesty guard).** The primary surrogate's predictions are
   generated out-of-fold under group-aware cross-validation; residuals are computed from
   those OOF predictions; the error model is then fit on training-fold residuals and applied
   to held-out folds. **No in-sample residuals — ever.** (This is the single line that keeps
   every confidence number honest.)
3. **Error-model features (bake-off).** Condition descriptors (`α, Re, M`) + the primary
   prediction ("plain"), optionally + ensemble spread and geometry-novelty (e.g. PCA
   reconstruction error) ("lean"/"rich"). Recipe chosen by bake-off on
   **RMSE@top-k + Spearman(predicted error, actual error)**. Top-X% R² is reported only as a
   secondary view — the retained subset's shrinking variance confounds it (a lesson the
   f-elements project learned explicitly), so it is never the selection metric.
4. **Gate — two modes.**
   (a) *Selective prediction:* rank predictions by predicted error; report honest metrics
   at coverage operating points (top 100 / 50 / 25 / 10 %). Low predicted-error points are
   "let through"; the rest defer to higher-fidelity CFD.
   (b) *Split-conformal intervals:* normalized score `s = resid / err`; the quantile is
   calibrated on one disjoint group-set and coverage measured on another, so intervals are
   calibrated (target-vs-empirical coverage reported). *(In f-elements this delivered 90%→
   90.2% / 80%→80.2% empirical coverage on the ΔG per-pair model specifically — evidence the
   method calibrates, not a transferred guarantee.)*
5. **Fallback trigger.** Predicted error (or conformal interval width, or high novelty/OOD
   score) above the operating threshold → defer that evaluation to real CFD.

**Grouping is matched to the evaluation regime — it is not a single fixed variable.** The
f-elements project uses *molecule*-grouping to test generalization to unseen molecules, but
*condition-key* grouping (a molecule may recur across splits at new conditions) to test
interpolation of new conditions for a known molecule. OptimAero adopts **both regimes**
(decided 2026-07-05):
- **Headline test — new-geometry generalization.** Group by **airfoil family**: no shape or
  its scaled/rotated/near-duplicate variants straddle the split. This is the number that
  matters most, because the inverse-design optimizer proposes shapes never seen in training.
- **Secondary test — new-condition interpolation.** A known shape may recur across splits at
  new `α/Re/M`; measures how well the surrogate fills the operating envelope for a known
  geometry.

The two are reported separately and never conflated. The precise definition of "airfoil
family" (parent-shape lineage vs clustering the coordinate/CST space; scaled/rotated/
near-duplicate handling) remains the Phase 1 leakage-map detail.

## 5. Non-negotiable principles

1. **No leakage, ever.** Train/val/test are separated by *geometry family and condition*.
   Airfoils from the same family, scaled variants, or near-duplicates never straddle
   splits. The leakage map is written and reviewed before training.
2. **Honest metrics, stated regime.** Report **R² and RMSE together**, per output
   (`C_l, C_d, C_m`), against the ground-truth solver; report confidence-model
   **calibration** (empirical vs nominal coverage). State the evaluation regime for every
   number. No cherry-picking, no overclaiming.
3. **Trust before speed.** Every prediction carries calibrated uncertainty; out-of-
   distribution inputs trigger fallback to higher-fidelity CFD. A fast but overconfident
   model is a *failure*, not a win.
4. **Physics as a guardrail.** Prefer physically-grounded couplings (BEMT/lifting-line) and
   physical sanity checks over unbounded black-box extrapolation.
5. **Reproducibility.** Every dataset is versioned with full provenance (source, solver,
   conditions, mesh/settings); data-generation scripts are checked in; seeds fixed.
6. **Bake-offs, not hunches.** Model class, shape representation, and hyperparameters are
   chosen by recorded, fair comparison on held-out data — never arbitrarily.
7. **SDD governs every change.** specify → clarify → plan → tasks → analyze → implement →
   verify → record, scaled to the size of the change.

## 6. Tech stack (initial; individual choices confirmed by bake-off where noted)

- **Language:** Python.
- **ML / UQ:** PyTorch; scikit-learn baselines; GPyTorch or an ensemble for uncertainty.
- **Data generation:** XFOIL (fast viscous-panel polars) for volume; SU2 and/or OpenFOAM
  (RANS) for higher-fidelity spot-checks and validation.
- **Curated datasets:** UIUC airfoil coordinate database, AirfRANS, NASA Turbulence
  Modeling Resource; (Phase 5) DrivAerNet++-style 3D sources.
- **Physics coupling:** BEMT (propellers) + lifting-line/panel (wings); XROTOR/QPROP/QBlade
  lineage as references.
- **Geometry / CAD:** neutral STEP/STL/IGES via OpenCASCADE (pythonOCC / FreeCAD) and
  trimesh/gmsh; airfoil parameterization via CST / PARSEC / B-spline (bake-off).
- **Optimization:** gradient-based over a differentiable surrogate where possible, plus
  Bayesian optimization / genetic algorithms for the constrained inverse design.

## 7. Governance

- **Account: `SkyEpstein`** (per-project exception, not the `skyepstein1` default) — the
  MIT-facing research identity that also hosts the `Machine-learning-to-separate-f-elements`
  repos. OptimAero is a showcase research project and belongs there. Confirm the active
  `gh` account is `SkyEpstein` before any push.
- **Published** at https://github.com/SkyEpstein/OptimAero (public, 2026-07-06) once the
  first real milestone was in hand (full working pipeline + verified results + writeup).
- Commit messages confirmed with Sky before pushing.
- `CHANGELOG.md` updated as part of every commit, with honest numbers.
- Constitution amendments recorded in `CHANGELOG.md`.
