# Spec — Validate the universal drag surrogate on canonical benchmark geometries

Date: 2026-07-13
Status: implementing

## Problem
The universal drag surrogate reports a strong headline number (overall rank 0.97 on held-out KFold),
but that number is **cross-type** — it is easy to rank a bluff body above a wing. The honest,
never-yet-run test is: does the surrogate hold on **canonical textbook geometries it never trained on**
(the specific proportions of a sphere, an Ahmed body, real NACA sections, the ONERA M6 planform),
rather than on our own random parametric shapes? These shapes have known aerodynamic behavior and, in
several cases, published drag coefficients — the strongest available ground truth short of a wind tunnel.

## Scope
- IN: generate a fixed set of canonical benchmark meshes (bluff → streamlined), oriented flow=+x and
  scaled to the training size regime; CFD-label each with the SAME pipeline the surrogate trained on
  (refine 4, layers 2, V=134.11 m/s, RHO=1.225); run the surrogate's `predict_drag` on each; compare.
- IN: an honest report — cross-shape rank correlation, per-shape % error (calibration), whether the
  surrogate's own confidence flags its worst predictions, and (where published) CFD-vs-literature as a
  check on the ground truth itself.
- OUT (this spec): downloading arbitrary CAD; retraining the surrogate; the optimizer end-to-end run.
  Improvements found here feed a follow-up spec.

## Conditions (must match training so Reynolds is comparable)
- V = 134.11 m/s, alpha = 0, RHO = 1.225, refine = 4, layers = 2.
- Characteristic size ~0.1 m (training bodies had bbox ~0.06–0.12 m).
- Cd is frontal-area based: Cd = drag / (½ρV²·A_front); the surrogate and `cfd_label` use the SAME
  A_front (projected area along +x), so predicted vs CFD Cd are apples-to-apples.

## Benchmark set (out-of-distribution canonical shapes)
Bluff → streamlined, chosen to span the widest Cd range:
1. sphere               — Cd_lit ≈ 0.47 (subcritical); RANS bluff caveat noted
2. cylinder (crossflow) — Cd_lit ≈ 1.0–1.2; bluff extreme
3. cube                 — bluff reference (already seen ~high Cd once end-to-end)
4. Ahmed body (25° slant, no legs) — Cd_lit ≈ 0.29 (frontal); canonical automotive bluff-tapered
5. capsule / streamlined body — low-Cd rounded body
6. NACA 0012 wing (symmetric)
7. NACA 2412 wing (cambered)
8. NACA 4412 wing (more cambered)
9. ONERA M6 wing (swept, tapered; symmetric section approx)

## Acceptance criteria
- AC1: ≥7 benchmark shapes build watertight and CFD-converge (drag > 0, |Cd| < 10).
- AC2: report Spearman rank corr of predicted vs CFD Cd across the set, AND per-shape signed % error.
- AC3: report the surrogate confidence (predicted |log-resid|) per shape and whether it is higher on the
  worst-predicted shapes than the best (confidence validity, directional).
- AC4: for sphere/cylinder/Ahmed, report CFD Cd next to the published value so the reader can judge
  whether steady RANS is a trustworthy ground truth for that shape (bluff-separated caveat).
- AC5: no overclaiming — state the regime (steady RANS, refine 4), the OOD framing (canonical instances
  of families the surrogate saw parametric versions of; the sphere/Ahmed proportions are novel), and
  name the shapes where it breaks.

## Deliverables
- `optimaero/universal/benchmarks.py` — the canonical geometry generators (reusable, watertight, flow=+x).
- `scripts`/scratchpad eval → `results/benchmark_validation.json` + a printed table.
- CHANGELOG entry with honest numbers. Findings → follow-up improvement spec.
