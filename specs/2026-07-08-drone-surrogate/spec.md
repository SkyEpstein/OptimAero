# Surrogate-driven autonomous drone optimizer (spec + plan)

**Date:** 2026-07-08 · **Status:** in progress · Sky's directive + chosen path recorded below.

## Problem / vision (Sky, verbatim intent)
The autonomous drone optimizer should **use an ML surrogate**, not CFD, inside the search loop: "continually
generate different [forms], quickly run them through the ML tests until an ideal form is made — more tuning
than doing that with CFD would allow." CFD then only verifies the final top few. Surrogate eval is ~ms;
CFD is ~75 s. So the surrogate buys thousands of trial forms where CFD affords ~12.

## Verified blocker (why we can't reuse the envelope surrogate)
The trained Cd/Cl surrogate is **envelope-only**: 4 of its 20 features (`grow, nose_frac, tail_frac,
round_exp`) are envelope-generator knobs undefined for drones, and 100% of its training rows are smooth
enclosing envelopes — a drone with protruding airfoil arms + boat-tail is out of distribution. Scoring
drones with it would be a confident guess, not knowledge (honesty rule). → **Build a drone-form surrogate.**

## Chosen approach (Sky picked "Full sweep first")
1. **Drone-form CFD dataset** (`optimaero/drone/dataset.py`): sample the treatment space
   `[tail_len, chord×rmax, thick]`, build each form (`optimize._build` = airfoil_arms + add_tail), CFD-label
   (refine 4 + boundary layers), record **drone features** (geometric + area-rule, flow along +x, same
   recipe as envelopes MINUS envelope-only params, PLUS the 3 treatment knobs and base-drone descriptors
   rmax/r_core/n_rotors). Resumable, sharded. Target: several hundred forms of the benchmark drone.
2. **Drone surrogate** (`optimaero/drone/surrogate.py`): the proven bake-off recipe (ExtraTrees/LGBM/HGB
   pool + OOF-residual LGBM **confidence model** + trust gate) on drone features → predict drag/Cd.
3. **Rewire `optimize_drone`**: surrogate-search thousands of forms (e.g. differential evolution over the
   knobs, or dense sampling) → rank by surrogate (confidence-aware) → **CFD-verify the top-K** →
   additive-only gate (`additive_ok`) → return the best CFD-confirmed. Never worse than bare.

## Design decision — speed-invariant target (from the adversarial review)
The review caught a **train/serve condition skew**: training is at a single condition (V=134 m/s, 0° AoA),
so V/Mach/AoA don't vary, and feeding them as features let the surrogate silently ignore a different
serve-time speed (the GUI default is 25 m/s). Fix: **target the drag area `cda = drag/q = Cd·A_front`**
(speed-invariant in the turbulent regime) with **only the 3 varying knobs** as features. Ranking forms by
predicted `cda` gives the same ordering as ranking by drag at ANY speed, and the returned optimum is always
CFD-verified at the user's actual V/AoA — so the surrogate is honestly usable across speeds. Constants
(rmax/Mach/AoA) are NOT features (they'd overstate what the 3-knob model learns).

## Review fixes (8 confirmed, all resolved)
HIGH train/serve skew + MEDIUM constant-feature overclaim → cda target + 3-knob features (above).
LOW: graceful blind-CFD fallback if the surrogate can't load; NaN-safe ranking (all-NaN → blind);
`_clean` drops additive-invalid forms from training; R² honestly labelled "interpolation within one drone."
Accepted LOWs: failed-row NaN cells (mitigated by `dropna` in `_clean`); resume progress-count cosmetic.

## Evaluation contract (honesty / leakage)
- The v1 surrogate is a **per-drone response surface** over THIS drone's treatment space — it interpolates
  the knobs, it is NOT a universal drone model. Report it as such.
- Features are deterministic functions of (params, base drone) — no target leakage.
- Metric: drag/Cd **R² + RMSE** on held-out forms; a random split AND a param-edge (extrapolation) split.
- The final reported optimum is always **CFD-verified**, so the surrogate's error can't inflate the claim —
  it only decides which forms are worth a CFD run.

## Acceptance criteria — ALL MET (2026-07-08)
1. ✅ Drone-form dataset: 300 forms, all converged, all additive-valid, drag 31.4–137.7 N.
2. ✅ Surrogate trained; **knobs-only R²=0.971** (cda target), confidence RMSE 0.00031→0.00019@25%,
   extrapolation RMSE≈interpolation RMSE; R² honestly labelled interpolation-within-one-drone.
3. ✅ `optimize_drone_surrogate` searches 8000 forms, CFD-verifies diverse top-K, additive-only + never-worse
   (shared helpers; unit + fallback tests pass).
4. ✅ Live: **137.7 → 31.4 N (−77%) in 200 s, 7 CFD runs** (vs blind 57%/12 CFD). Surrogate predicted the
   winner to 0.0 N; 31.4 N = the best of all 300 swept forms (found the global optimum with 7 CFD calls).

## Open
- v1 knob space is 3-D (tail/chord/thick); widen (nacelle fairings, arm count, tail shape) once the loop
  is proven — each new knob just needs the builder + more samples.
- Cl for drones is secondary (drag is the objective); Cl-feature work tracked separately.
