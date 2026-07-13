# General drone surrogate — optimize ANY multirotor, not one (spec + plan)

**Date:** 2026-07-09 · **Status:** in progress · Sky: "the tool doesn't optimize drones, it optimizes MY
drone. This must be fixed." Chosen path: **full general surrogate.**

## Problem
Audit confirmed: the optimizer, segmentation, and fairing-builder are already general (everything scales to
the imported drone's measured geometry; blind CFD optimizes any drone). The ONLY specialization is the ML
surrogate — trained on 300 forms of ONE drone (`n_base_drones: 1`, features = the 3 knobs only), so its
fast rankings are valid only for that drone. To make the fast path general, the surrogate must be trained
across MANY drones and conditioned on drone-shape descriptors.

## Approach
1. **Parametric multirotor generator** (`drone/generator.py`): synthesize diverse watertight multirotors —
   randomize arm count {3,4,6,8}, body size/fineness, arm length/thickness, pod size, so training spans the
   multirotor design space. Each must segment (segment_multirotor finds the pods) and build treatments.
2. **Multi-drone × treatment CFD dataset** (`drone/dataset.py::generate_multi`): for N generated drones ×
   M treatments (tail/chord/thick), build (airfoil_arms + add_tail), CFD-label. Record per row: **drone
   descriptors** + treatment knobs + bare cda + treated cda. Grouped by drone id. Resumable/sharded.
3. **General surrogate** (`drone/surrogate.py`): features = drone descriptors {n_rotors, rmax, r_core,
   arm_thickness, body_fineness, frontal_area, r_core/rmax, pod ratio, ...} + knobs {tail,chord,thick};
   **target = normalized reduction ratio `treated_cda / bare_cda`** (dimensionless → transfers across drone
   sizes; ranking by predicted ratio = drag ranking for a given drone, so NO bare-CFD needed at serve time).
   Bake-off recipe + confidence model. **GroupKFold BY DRONE** — train on some drones, test on HELD-OUT
   drones — so the reported R² is honest cross-drone generalization, not within-drone interpolation.
4. **Wire in** (`optimize_drone_surrogate` + GUI): compute drone descriptors from `seg`+mesh at serve time,
   feed them to the surrogate, re-enable surrogate mode for ANY imported drone.

## Evaluation contract (honesty / leakage)
- **GroupKFold by drone id** is mandatory — random KFold would leak (same drone in train & test) and
  overstate generality. Report held-out-drone R²/RMSE + a confidence curve.
- Also report a "leave-Sky's-drone-out" number: train on generated drones, test on his real drone.
- The final optimum is always CFD-verified at the real V/AoA (surrogate only chooses which forms to CFD),
  so surrogate error cannot inflate the returned result.

## Acceptance criteria
1. Generator produces diverse watertight multirotors that segment + build + CFD cleanly.
2. Multi-drone dataset: ≥ ~12–20 drones × treatments, grouped, resumable.
3. General surrogate: honest **held-out-drone** R² reported (not within-drone); confidence curve.
4. Surrogate-driven optimizer works on a drone NOT in training (e.g. Sky's) and its CFD-verified winner
   is ≤ the blind search's best, using far fewer CFD calls. Re-enabled for any imported drone.

## Open
- Scope = multirotors (body + radial arms + pods). Fixed-wing/other frames are out of scope for v1.
- If cross-drone R² is weak, fall back: keep blind CFD as the general default, surrogate as a pre-screen only.
