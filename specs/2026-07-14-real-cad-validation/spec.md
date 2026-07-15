# Spec — Validate the surrogate + optimizer on a real downloaded CAD model

Date: 2026-07-14
Status: implementing

## Problem
Every validation so far used shapes OptimAero generated (parametric families, canonical benchmark
geometries). The strongest real-world test — now that the wing/plane ranking ceiling is fixed
(commit 3d61ce2, wing held-out 0.61→0.78) — is a real third-party CAD model downloaded from online:
does the universal surrogate predict its drag well from geometry alone (no CFD), does CFD confirm the
prediction, and does the optimizer improve it?

## Scope
- IN: download ONE public-domain / CC0 CAD model (aircraft/glider/UAV preferred — exercises the fixed
  wing/fuselage path — or a drone). Import → repair to watertight → predict drag with `predict_drag`
  (no CFD) → CFD-verify the prediction at a matched speed → run the appropriate optimizer
  (`optimize_universal` for a general body, or the drone fairing path for a multirotor) → CFD-verify the
  optimized result is actually lower drag and (for drones) additive-only.
- OUT: retraining; downloading multiple models; CAD cleanup beyond the existing watertight repair.

## Approvals / safety
- Downloading a file needs Sky's explicit OK on the SPECIFIC file (name, source, size) — ask via MCQ
  before downloading. Public-domain/CC0 only.

## Conditions
- Match the training/validation regime for the CFD check: RHO=1.225, refine 4 + layers 2, alpha=0,
  a representative V (scale the model to the ~0.1 m training size regime OR pick V so Reynolds is
  comparable; record which). Cd is frontal-area based (same convention as training + `predict_drag`).

## Acceptance criteria
- AC1: the model imports and repairs to a watertight, positive-volume mesh (or honestly report if the
  CAD is too broken/thin-shell to repair — that is itself a finding).
- AC2: report `predict_drag` (drag N, Cd, confidence) from geometry alone, then the CFD drag at the same
  V; report the % error honestly (this is a true out-of-distribution real-world shape, not a benchmark).
- AC3: run the optimizer; CFD-verify the optimized shape is lower drag; for a drone confirm additive-only
  (contains the original). Report before→after drag and the % reduction, never-worse respected.
- AC4: no overclaiming — state the regime, that the model is a real OOD shape, the confidence value, and
  any caveats (scale/Reynolds choice, mesh repair quality, single sample).

## Deliverables
- The imported+repaired mesh + a short eval script (scratchpad) → printed predict-vs-CFD + optimize result.
- CHANGELOG entry with honest numbers; note the model source + license.

## Results (2026-07-14) — NASA SOFIA (Boeing 747SP), public domain, genuinely OOD
Model was frozen ~9 h BEFORE this mesh existed (verifier-confirmed temporal no-leakage). 6 external STL
parts boolean-unioned in-memory (watertight; **STL export drops connectivity → not watertight on reload**,
a real gotcha the deform optimizer hit), scaled to 0.15 m, flow +x.
- **Prediction (geometry only, no CFD):** Cd **0.411** vs coarse-RANS CFD Cd **0.387** → **+6.2%**
  (reproducible via `optimaero/universal/validate_sofia.py`). The model's own confidence implied ~25%
  typical Cd error, so ~6% on a real unseen aircraft is well inside — a strong single-sample result, NOT
  proof of general accuracy (n=1). A differently-tessellated instance gave 0.392 vs 0.395 (−0.8%); the
  prediction varies ~5% with mesh tessellation (512-pt surface sampling) — honest robustness caveat.
- **Optimization (universal streamlining, CFD-verified, never-worse):** reshaped the airframe
  (elongate +29%, boat-tail aft) → CFD drag **3.130 → 2.224 N (−28.9%)**. Same coarse-RANS setup
  before/after, so the relative reduction is solid; a pure aero-shape result (longer aircraft, not free).
- **Honesty:** CFD is this project's coarse steady RANS (±30–50% stated); frontal-area RANS Cd≈0.4 at
  0.15 m / 134 m/s is NOT a real 747 cruise Cd (~0.03, wing-area, compressible). "Matched what our RANS
  says for this shape," not real-world drag. AC1–AC4 met (AC1: watertight in-memory only — reported).
