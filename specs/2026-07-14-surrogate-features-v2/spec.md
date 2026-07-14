# Spec — Richer geometry features to separate low- from mid-drag within a type

Date: 2026-07-14
Status: implementing

## Problem
The benchmark validation (spec 2026-07-13) proved the universal drag surrogate **regresses canonical
shapes toward each type's training-median Cd**: benchmark NACA wings (CFD Cd ≈ 0.36, at the training-wing
minimum) are predicted ≈ 0.47–0.50 — exactly the training-wing median (0.463). It is NOT a data floor
(training has Cd down to 0.15; the model predicts down to 0.15 on training) and NOT extrapolation (the
shapes look in-distribution feature-wise, so a novelty detector would not fire). The current 21 features
do not **separate low- from mid-drag within a type** — held-out within-type rank is only wing 0.62,
plane 0.63. The hypothesis: the features miss the physical drivers of pressure drag.

## Approach (chosen by MCQ 2026-07-14: "Richer features first — cheap probe, no CFD")
Add a small, physically-motivated set of features to `features._core` (shared by train + serve, so no
skew). Chosen from first principles — the classic pressure-drag drivers — NOT tuned to the 9 benchmark
shapes (that would leak the test):
- **aft_taper** — area shed from the mid-station to the base (boat-tail extent / base-drag proxy).
- **max_slope** — sharpest area change along the body (adverse-pressure-gradient / separation risk).
- **tc_ratio** — thickness-to-chord = min transverse extent / streamwise length (isolates thin wings).
- **transverse_aspect** — width/thickness (distinguishes flat wings from bodies of revolution).
- **fore_aft** — streamwise area centroid (teardrop / front- vs aft-loaded volume).

Recompute features from the ALREADY-SAVED point clouds (data/processed/xtype_*/*.json carry points +
normals), retrain the GBR + confidence model. **No new CFD.**

## Evaluation contract (honest, leakage-aware)
- Primary metric: held-out (KFold-5) **within-type rank** on the 720 shapes (larger n, the robust signal).
  Report per-type before → after so any regression (e.g. drones 0.91) is visible.
- Secondary OOD check: re-run the 9 saved-CFD benchmark shapes through the retrained model (rebuild mesh,
  recompute features, predict; compare to the saved CFD Cd). No CFD except optionally recovering naca4412.
- Leakage guard: features are fixed from physics BEFORE looking at benchmark results; the feature set is
  NOT iterated against the 9-shape benchmark (single principled expansion, evaluated once).
- No overclaiming: the benchmark is n=8 (naca4412 CFD-failed) with a wide CI; lean on the 720 held-out
  within-type ranks as the headline, the benchmark as directional corroboration.

## Acceptance criteria
- AC1: features._core emits the 5 new features at both train and serve with no train/serve skew
  (features_from_saved and universal_features agree on a spot-checked shape).
- AC2: retrained model reported with held-out overall + per-type rank, before vs after.
- AC3: benchmark re-run reported: cross-shape rank + median |%err| before vs after.
- AC4: honest verdict — if within-type wing/plane rank and benchmark rank both improve, representation was
  (part of) the gap; if not, it is a data-coverage gap → go to the diverse-data spec. State which.

## Deliverables
- `optimaero/universal/features.py` — 5 new features in `_core`, extended `FEATURE_NAMES`.
- Retrained `results/universal_drag_surrogate.joblib` + `_report.json` (only if it improves; else keep old).
- Re-run benchmark numbers; CHANGELOG entry with before/after; verdict feeds the next step.

## Verdict (2026-07-14, after retrain — no CFD)
Held-out (720, KFold-5): overall rank 0.970 → **0.975**; per-type after/before —
fuselage 0.820/0.757 (+0.063), plane 0.670/0.620 (+0.050), bluff 0.921/0.884 (+0.038),
drones 0.929/0.922 (+0.007), **wing 0.622/0.626 (−0.004, UNMOVED)**, nacelle 0.782/0.790,
bodies 0.893/0.903. Benchmark (n=8): median |%err| 34.1% → **27.5%** (streamlined body 30→11%),
cross-shape rank 0.57 → 0.52 (within n=8 noise). **Conclusion: richer features gave a modest
CALIBRATION win and lifted several types, but the wing/plane RANKING ceiling did not move → the
"can't separate low- from mid-drag wings" problem is a DATA-COVERAGE gap, not representation.**
Decision (MCQ 2026-07-14): KEEP v2 (net positive on operational metrics), proceed to diverse-data.
