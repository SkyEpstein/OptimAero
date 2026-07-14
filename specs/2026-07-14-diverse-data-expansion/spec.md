# Spec — Diverse-data expansion to fix the wing/streamlined ranking ceiling

Date: 2026-07-14
Status: implementing

## Problem
The features probe (spec 2026-07-14-surrogate-features-v2) proved the wing/plane ranking ceiling
(held-out rank ~0.62) is a **data-coverage gap**, not representation: the surrogate regresses canonical
low-drag wings toward the training-wing median because it has too few, too-similar wing/streamlined
examples to learn what separates a low-drag section from an average one. Fix: enlarge and DIVERSIFY the
training set, especially the weak types (wing 0.62, plane 0.62, fuselage 0.76), with canonical shape
families, then retrain and re-validate.

## Approach (MCQ 2026-07-14: "Keep v2, generate diverse data")
Reuse the `optimaero/universal/benchmarks.py` canonical generators (naca_wing, sphere, ellipsoid,
cylinder, box, ahmed_body, streamlined_body, onera-like) with PARAMETER SWEEPS to generate ~120–150 new
shapes, CFD-label them with the exact training pipeline (refine 4, layers 2, V=134.11, RHO=1.225,
alpha=0), and append to the existing `data/processed/xtype_{wing,bluff,bodies}` buckets. Retrain the
26-feature (v2) GBR + confidence model.

## Leakage map (CRITICAL — the honest test depends on it)
- The 9 benchmark shapes are the held-out test. The generated training shapes MUST exclude the exact
  benchmark parameter tuples: sphere r=0.05; cylinder r=0.03,L=0.12; cube side=0.08; ahmed scale=1.0,
  slant=25; streamlined_body r=0.018,L=0.11; naca_wing 0012/2412/4412 at chord=0.08,span=0.12; onera_m6
  defaults. Enforced by an explicit exclusion check in the generator.
- HONESTY RE-FRAME: after this, the benchmark is a **leave-these-9-instances-out** generalization test
  (train on many sphere/wing/… instances, predict the specific held-out ones), NOT the original
  "never-saw-this-family" OOD test. Report it as such. It still measures real generalization across each
  family's parameter space, but the framing changes and must be stated.
- Primary metric stays the **held-out KFold-5 within-type rank** on the enlarged dataset (leave-shapes-out
  cross-validation the pipeline already does) — the honest, larger-n measure of "can it rank within a type."

## Acceptance criteria
- AC1: ~120+ new shapes CFD-labeled, watertight, Cd in (0,6); none coincide with a benchmark tuple.
- AC2: retrained model reported with held-out overall + per-type rank, before (720) vs after (enlarged);
  the target is wing/plane rank climbing meaningfully above 0.62.
- AC3: benchmark re-run (now leave-instances-out) reported: cross-shape rank + median |%err|, before/after.
- AC4: honest verdict — did more wing/streamlined data move the wing/plane held-out rank? State the
  regime (steady RANS refine-4) and the re-framed benchmark meaning. No overclaiming.
- AC5: never regress the strong types badly (drones 0.92, bodies 0.90) — report any regression.

## Deliverables
- Generation+labeling script (scratchpad; reusable) → new JSONs in data/processed/xtype_*.
- Retrained `results/universal_drag_surrogate.joblib` + `_report.json` (keep only if net-better).
- Re-run benchmark numbers; CHANGELOG entry with honest before/after; commit under SkyEpstein.
