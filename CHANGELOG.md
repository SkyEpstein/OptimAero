# Changelog

All notable decisions and milestones for **OptimAero**. Honest numbers only.

## [Unreleased]

### 2026-07-06 — MIT writeup + figures
- `docs/METHODS_AND_RESULTS.md`: full honest methods-and-results (motivation, data, leakage
  control, nested bake-off, confidence model, physics + inverse design, the trust-verification
  finding, limitations, reproducibility). Figures in `docs/figures/`: predictor bake-off R²,
  selective-prediction curve, and the surrogate-vs-XFOIL verification (441→129).
- Preparing first commit to **`SkyEpstein/OptimAero`** (past the constitution's milestone bar).

### 2026-07-06 — End-to-end on the trained model + off-manifold finding
- **`TrainedSurrogate`** (`optimaero/bakeoff/deploy.py`): winning MLP + LightGBM confidence
  trained on the full backbone, calibrated trust/OOD gate, implements the `Surrogate` socket.
  Placeholder retired. Section aero excellent in-distribution (NACA 4412 Cl(0)=0.459±0.015).
- **Finale demo** (`scripts/finale_demo.py`): section aero → inverse design → BEMT → CAD STEP,
  all on the trained model. Section/CAD/BEMT + confidence propagation work.
- **Important finding — surrogate-exploitation off-manifold.** The optimizer initially returned
  garbage (L/D=418k, Cd≈0, envelope violated): the MLP extrapolates to fake near-zero drag far
  off the training manifold, AND the learned error-model is itself unreliable there so it did
  NOT flag `ood`. Confidence caught `trusted=False` but soft penalties were dwarfed by the
  exploited metric.
- **Guards tried (partial):** (1) geometry-**novelty OOD** (per-dimension range → then JOINT
  KDTree nearest-neighbour distance to the training manifold); (2) **hard** optimizer
  constraints (reject infeasible-envelope / OOD / sub-physical-Cd). Each helped but none fully
  closed it — the optimizer still found in-manifold shapes where the MLP under-predicts Cd at
  high-α near stall.
- **Key insight (XFOIL-verified):** for the optimizer's shape the surrogate claimed L/D=441
  but **XFOIL truth = 129** (surrogate over-promised 3.4× — Cl was accurate at 1.44, Cd was
  under-predicted 0.0033 vs 0.0112). No static guard fully prevents an optimizer from exploiting
  a surrogate's blind spots. BUT the optimizer **did** find a real airfoil — XFOIL-confirmed
  **L/D=129, ~+29% over baseline within the envelope**. Inverse design works *when verified*.
- **Resolution (Sky: "both, staged"):** Stage 1 — **verification-in-the-loop** (`optimize_verified`:
  surrogate searches from N seeds, real XFOIL confirms each optimum, best-verified returned;
  the surrogate's number is never returned unverified). Stage 2 — **active-learning loop**
  (feed XFOIL-verified misses back into training, retrain, re-optimize) as the surrogate-
  refinement layer. This is the constitution's CFD-fallback + [C-DATA] realized.
- **Stage 1 demonstrated end-to-end:** `finale_demo` now reports the XFOIL-VERIFIED optimum
  (L/D=129, +29% over baseline, within envelope) instead of the surrogate's inflated 441.
  Full pipeline (section aero → inverse design → BEMT → CAD STEP/STL) runs trustworthy on the
  trained model. Stage 2 (active-learning) is the next build.

### 2026-07-06 — Phase 2 bake-off COMPLETE (headline research result)
- Full nested bake-off on 213,406 rows / 2,103 families, 5-fold GroupKFold-by-family
  (new-geometry, zero leakage), 14.2 min. Results in `results/phase2_bakeoff.json`.
- **Winner: MLP predictor + LightGBM confidence.** Honest new-geometry generalization:
  **Cl R²=0.985, Cd R²=0.964, Cm R²=0.902** (RMSE 0.078 / 0.0084 / 0.0155). Ranking:
  mlp > avg-top3 > lightgbm ≈ extratrees ≈ hist_gbr > knn > ridge.
- **Confidence model works:** selective prediction improves retained-point RMSE — Cl
  0.078→0.028 @50%, Cd 0.0084→0.0018 (4.7×), Cm 0.0155→0.0049. Spearman(pred_err, err)
  ≈ 0.55 / 0.62 / 0.47.
- **Variance-confound observed as predicted:** Cd R² *drops* under tighter coverage
  (0.964→0.838) while RMSE *improves* — live confirmation that RMSE, not top-X% R², is the
  honest selection metric (constitution §4).
- **Adversarially verified (harness-verifier):** grouped CV genuinely leakage-free, OOF
  correct, no feature leakage, metrics match sklearn, MLP seed-stable. Honesty correction:
  2,044/2,103 families are singletons → "unseen airfoils, duplicates grouped," NOT
  "clustered-family generalization" (data-model §4a updated).
- **Headline independently reproduced** (full 213k, 3 seeds): **Cl R²=0.984±0.001,
  Cd 0.963±0.001, Cm 0.927±0.018** (`results/mlp_reproduction.json`). Cl/Cd essentially
  deterministic; Cm carries a real ±0.018 spread (reported as a range, not cherry-picked).

### 2026-07-06 — BEMT propeller model (Phase 3 physics coupling, first deliverable)
- Built `optimaero/physics/bemt.py` (+ `physics/__init__.py`): standard low-speed
  Blade-Element Momentum Theory. Discretizes the blade hub→tip, iterates axial `a` and
  swirl `a'` inductions to convergence with a Prandtl tip+hub loss `F`, queries the
  `Surrogate` (section Cl/Cd) per element, and integrates T/Q/P. `Propeller` dataclass +
  `solve(...) -> PropResult`. rho=1.225, nu=1.5e-5.
- **Confidence propagated section→vehicle** (constitution §3/§4): `PropResult.any_ood` /
  `frac_trusted` aggregated from per-element surrogate predictions.
- **VALIDATED (physics confirmed, not asserted).** Test prop R=0.15 m, 2 blades, naca4412,
  twist 28°→10°, 5000 rpm, J∈[0.1,0.9] via NeuralFoil placeholder: thrust 6.44 N (J=0.1) →
  −2.95 N (J=0.9), monotonic decreasing ✓; efficiency a smooth interior hump peaking at
  **η≈0.62 near J≈0.45** ✓; η∈(0,1) everywhere it is a producing point ✓. Two independent
  η formulas (T·V/P and CT/CP·J) agree to 4 decimals → nondimensionalization self-consistent.
- **Fix during validation:** η was reported as a large negative number at the first
  post-thrust-reversal point (T<0, P≈0). Corrected to define propulsive efficiency only where
  the prop is actually producing (T>0 ∧ P>0), else nan — a reporting fix, not a physics change.
- Placeholder-surrogate OOD note: only the highest-J point (J=0.9, deep windmill, section α
  driven negative) trips `any_ood`; frac_trusted stays 1.0 through the useful envelope.

### 2026-07-05 — Project inception (SDD phase 0)
- Scope locked via clarifying MCQs:
  - **Domain:** 2D airfoils/wings (foundation) → drones/UAVs/propellers (application).
  - **Data:** curate public benchmarks (UIUC, AirfRANS, NASA TMR) **+** generate our own
    (XFOIL for volume; SU2/OpenFOAM for higher-fidelity spot-checks).
  - **Build order:** forward surrogate + confidence model first, then envelope-constrained
    inverse design.
  - **3D bridge:** physics-coupled (BEMT/lifting-line) now → learned 3D residual correction later.
  - **CAD I/O:** neutral formats (STEP/STL/IGES).
- Drafted `memory/constitution.md` (awaiting Sky's confirmation).
- **Confidence model resolved** (§4): verified formulation carried over from
  `SkyEpstein/Machine-learning-to-separate-f-elements-2` — a LightGBM error-model
  regressing the surrogate's absolute out-of-fold residual, gated by selective-prediction
  percentiles + split-conformal intervals, grouped by airfoil family to prevent leakage.
  Verified against `dg_coverage.py:39-53` (out-of-fold residuals, disjoint conformal
  calibration). No open `[NEEDS CLARIFICATION]` remain in the constitution.
- **Hosting decision:** OptimAero will be hosted under the **`SkyEpstein`** GitHub account
  (MIT-facing research identity, alongside the f-elements repos) — a per-project exception
  to the `skyepstein1` default. Repo creation deliberately held off until the first
  milestone; building locally until then.
- **§4 corrected after adversarial verification** (harness-verifier vs the f-elements-2
  source). Four overclaims fixed: (1) the error-model is a *separate model on residuals*,
  not necessarily a different algorithm (deployed Track A is LightGBM-on-LightGBM);
  (2) grouping is *matched to the evaluation regime* (molecule vs condition-key), not a
  single universal "molecule" split; (3) bake-off selection metric is **RMSE@top-k +
  Spearman**, with top-X% R² demoted as variance-confounded; (4) the clip floors the
  *predicted error*, not the target. The 90.2%/80.2% conformal coverage is scoped to the
  ΔG per-pair model as supporting evidence, not a transferred guarantee.
- **Evaluation regime decided:** OptimAero reports **both** an honest new-geometry test
  (airfoil-family-grouped — the headline, since inverse design proposes unseen shapes) and a
  secondary new-condition interpolation test (a shape may recur at new α/Re/M), never
  conflated. Resolves the regime half of the Phase 1 leakage-map decision; the precise
  "airfoil family" definition remains a Phase 1 detail.
- **Constitution APPROVED** by Sky (phase 0 closed).
- **Phase 2 bake-off protocol locked:** nested design — predictor bake-off (models +
  ensembles) ranked on new-geometry RMSE → top-K → per-candidate confidence bake-off on OOF
  residuals → winner = the (predictor, confidence) pair minimizing RMSE on retained points at
  target coverage ("deployed trust-gated accuracy"), gated by calibration + Spearman.
- **Phase 1 spec opened:** `specs/2026-07-05-data-foundation/spec.md` (SDD specify step;
  open `[NEEDS CLARIFICATION]` markers being resolved via MCQ).
- **Phase 1 clarifications resolved:** (C1) airfoil-family = **hybrid** (catalogued lineage
  + automated shape-space near-duplicate merge); (C2) data = **XFOIL backbone over UIUC +
  higher-fidelity anchors** (AirfRANS now; CFD/wind-tunnel later), fidelity-labeled per row.
  (C3 parametric storage form deferred to the Phase 2 representation bake-off.)

### 2026-07-06 (cont.) — Phase 2 launched + downstream filled in
- **Phase-2 bake-off built + RUNNING** (`optimaero/bakeoff/`): featurization (CST + conditions),
  predictor pool (LightGBM, HistGBR, ExtraTrees, MLP, KNN, Ridge + avg ensemble), GroupKFold-by-
  family CV, LightGBM confidence model on OOF residuals, selective-prediction + Spearman, winner
  by deployed trust-gated accuracy. Smoke test clean; full run on 213k in progress. Early:
  LightGBM new-geometry R² Cl 0.968 / Cd 0.924 / Cm 0.877; smoke selective-prediction Cl RMSE
  0.134→0.034 as coverage tightens (confidence model works).
- **CAD I/O** (`optimaero/cad/io.py`, CadQuery): STEP + STL export of an optimized section;
  envelope import round-trips (recovered t/c 0.122 for NACA 2412). Neutral-format, no lock-in.
- **Data card** (`docs/DATA_CARD.md`): honest inventory (213k rows, τ sensitivity, AirfRANS caveats).
- **BEMT propeller coupling** being built + validated (section→vehicle, confidence propagates).

### 2026-07-06 — Backbone complete + ML-pluggable downstream stood up
- **XFOIL backbone DONE:** `xfoil_backbone.parquet` — **213,406 rows, 2,169 airfoils**, 5 Re,
  72.7% alpha-yield (141k ok / 43k low_re / 29k post_stall). Ran ~3.5h unattended.
- **Leakage gate holds on the full dataset:** `test_leakage.py` 3/3 pass on all 213k rows
  (no family straddle, no shared rows, non-vacuous guard).
- **AirfRANS anchor ingested:** 1,000 RANS sims (`airfrans_anchor.parquet`). Caveats recorded:
  Re 2-6M (no overlap with XFOIL ≤1M — extends, doesn't validate); mirror lacks per-airfoil
  geometry (not yet training-integrable) and Cm.
- **ML-pluggable architecture** (`specs/2026-07-06-ml-pluggable-architecture/`): `Surrogate`
  interface + `NeuralFoilSurrogate` placeholder; `requirements.py` (Envelope + DesignRequirement);
  `optimize/inverse_design.py` (black-box CST-space optimizer). Decisions locked: confidence-
  driven active-learning augmentation, black-box optimizer, CadQuery.
- **End-to-end inverse-design demo works** — and surfaced a real lesson: a soft OOD penalty is
  dwarfed by an exploited metric, so **confidence must be a hard gate** (fixed). With the
  placeholder's weak confidence the optimizer still returns implausible L/D; this motivates the
  real Phase-2 confidence model rather than being papered over. Honest framing kept.

### 2026-07-05 — Phase 1 build begins
- Project env: venv + Phase-1 deps (aerosandbox 4.2.10, neuralfoil 0.3.2, scikit-learn 1.9.0,
  lightgbm, pandas, scipy, matplotlib); `requirements.txt` pinned (25 pkgs). Torch deferred to
  Phase 2. SDD artifacts written: `plan.md`, `data-model.md` (leakage map + eval contract),
  `tasks.md`.
- **Backbone geometry needs no web scraping:** AeroSandbox bundles the UIUC database (2,174
  airfoils, version-pinned). `optimaero/datasets/uiuc.py` (catalog + loader) built and
  verified — all 2,174 loadable; 25 exact/rename-duplicate groups already detected.
- **Generation pipeline built + full sweep launched.** `optimaero/generate.py` (parallel,
  per-airfoil sharded, resumable) + `optimaero/splits.py` (new-geometry & new-condition
  splits + L1/L2 leakage checks, self-checked clean on pilot). Pilot validated end-to-end
  (268 rows, NACA 4412 Cl(0)=0.481). **Full backbone sweep running**: 2,174 airfoils × 5 Re,
  15 workers, ~6-8h, checkpointed. Measured cost ~7h (corrected from an optimistic 1-2h
  estimate; low-Re sweeps dominate). Sky chose full-quality scope.
- **XFOIL built + validated** — XFOIL 6.99 compiled from source for arm64 (headless, no X11)
  at `tools/xfoil/xfoil`, reproducible via `tools/xfoil/install_xfoil.sh`, with 3 documented
  I/O-only Fortran patches for the gfortran/AeroSandbox path. Independently validated beyond
  the smoke test: NACA 0012 lift slope 0.108/deg (2π/rad✓); NACA 4412 zero-lift angle ≈ −4°
  and Cl(0)≈0.48 (textbook) → confirms the patches did not alter the solver. Data generation
  unblocked.
- `optimaero/geometry.py` (CST fit) + `optimaero/families.py` (hybrid family merge) built.
  Empirical: CST **12 weights/side** is the sweet spot (higher destabilizes); family
  **τ = 0.003** locked (largest merged family = 4 airfoils).
- **Leakage-critical modules adversarially verified** (harness-verifier): no leakage bug in
  the shipped config (34 diff-name geometric twins, zero straddle at defensible τ). Two
  findings fixed: (a) `_residual` measured CST error on a lossy resample and reported 0 for a
  5%-chord defect → now computed on raw coordinates (spike now reads 0.0499); (b) added a
  signed-area **orientation guard** to `geometric_signature` to prevent a future
  loop-reversed source from straddling a split.
