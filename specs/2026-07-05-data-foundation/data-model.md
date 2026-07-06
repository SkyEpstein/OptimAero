# Phase 1 — Data Model, Leakage Map & Evaluation Contract

*SDD: data-model (the ML-specific artifact required by the working agreement).*
*This document is the single source of truth for how data is shaped and how it is split so
that no reported number leaks.*
Date: 2026-07-05

## 1. Row schema (unified table)

One row = one airfoil evaluated at one condition.

| Column | Type | Notes |
|---|---|---|
| `airfoil_id` | str | canonical name/id of the shape |
| `family_id` | int | assigned by the hybrid grouping (§3) — the leakage unit |
| `source` | str | provenance: `uiuc`, `airfrans`, … |
| `fidelity` | enum | `xfoil`, `rans-airfrans`, `cfd`, `windtunnel` |
| `alpha_deg` | float | angle of attack |
| `Re` | float | Reynolds number |
| `Mach` | float | Mach number |
| `Cl`, `Cd`, `Cm` | float | targets |
| `converged` | bool | XFOIL convergence flag |
| `regime_flag` | enum | `ok`, `post_stall`, `low_re` (reliability caveats) |
| `cst_params` | list[float] | CST fit of the geometry (canonical parametric form) |
| `coord_ref` | str | pointer to normalized coordinate file |

**Honesty rule:** rows with `converged=False` are dropped from training but counted in the
data card; `regime_flag != ok` rows are kept but labeled, never silently trusted as
ground truth.

## 2. Geometry canonicalization

- Coordinates normalized to unit chord, leading edge at origin, consistent point ordering.
- **CST fit** (Kulfan) stored as `cst_params`; fit residual recorded (a bad fit is a data-
  quality flag, not silently accepted).

## 3. Airfoil-family grouping (hybrid) — the leakage unit

`family_id` (implemented in `optimaero/families.py`) is assigned so that no shape or its
**duplicates** can straddle a split. Two-part construction:

1. **Must-link floor (exact/rename dedup).** Airfoils sharing a `normalized_name` (e.g.
   `n0012`→`naca0012`, `naca0012-il`→`naca0012`) are force-merged, even if their raw
   geometry differs slightly (open vs sealed TE, coarse sampling). *Design-series tags
   (NACA-4, Eppler, …) are descriptive only and are NOT used to group — `naca0012` and
   `naca4412` are genuinely different airfoils and may split.*
2. **Shape-space merge.** Each airfoil gets an 80-dim geometric signature (upper & lower `y`
   at 40 cosine-spaced `x`, chord-normalized, **orientation-guarded** by signed area so
   loop-reversed twins from future sources match). A KD-tree union-find merges any pair
   within Euclidean distance **`τ`**.

**Threshold `τ = 0.003`** (`families.FAMILY_TAU`), chosen empirically and **verified**:
- The max distance between any two same-shape/different-name geometric twins in the UIUC DB
  is **0.00195** (34 such twins found, e.g. `rae2822`/`rae69ck`, `goe523`/`goe525`), so
  `τ = 0.003` captures every known twin with ~50 % margin.
- At `τ = 0.003` the **largest merged family is 4 airfoils** (78 total merged, 2174→2096
  families) — it merges near-identical duplicates but **not** genuine thickness/camber
  variants, which remain distinct: testing across a nearby-but-different airfoil is
  legitimate generalization, *not* leakage.
- Adversarially verified: **zero** known twins straddle a split at `τ ≥ p05`. Sensitivity
  across `τ` is reported in the data card.

## 4. The two evaluation regimes (both reported, never conflated)

### 4a. New-geometry generalization — **headline**
- **Split:** `GroupKFold`/grouped hold-out on `family_id`. No `family_id` appears in both
  train and test.
- **Question answered:** does the surrogate generalize to shapes it has never seen? (What the
  inverse-design optimizer actually requires.)
- **Honest note (verified 2026-07-06):** 2,044/2,103 families are singletons, so in practice
  this is **airfoil-level hold-out with near-duplicates merged** (the multi-airfoil families
  are exactly the duplicate groups the τ-merge caught). It is a genuine unseen-shape test —
  describe it as *"unseen airfoils, duplicates grouped,"* NOT "clustered-family generalization."

### 4b. New-condition interpolation — **secondary**
- **Split:** a known shape *may* recur across train/test, but at **held-out conditions**
  (disjoint `α/Re/M` cells for that shape). Grouping is on `(family_id, condition-cell)` such
  that no exact `(shape, condition)` pair is shared.
- **Question answered:** does the surrogate fill the `α/Re/M` envelope for a known geometry?

## 5. Evaluation contract (applies to every model number, Phase 2+)

- **Metrics per output** (`Cl, Cd, Cm`): **R² and RMSE reported together**, under the
  group-aware CV above. Never one without the other.
- **Selective-prediction curve:** RMSE (and R²) at coverage operating points
  **top 100 / 50 / 25 / 10 %** ranked by the confidence model's predicted error.
- **Confidence quality:** **Spearman(predicted error, actual error)** and **calibration**
  (empirical vs nominal coverage from split-conformal). Top-X% R² is reported only as a
  secondary view — it is variance-confounded and is **never** a selection metric.
- **Winner rule (Phase 2 bake-off):** the (predictor, confidence) pair minimizing RMSE on the
  **retained** points at target coverage, gated by calibration + Spearman ("deployed
  trust-gated accuracy").
- **Both regimes (§4a, §4b) reported side by side**; the new-geometry number is the headline.

## 6. Automated leakage checks (build fails if violated)

- **L1 — no family straddle (new-geometry):** assert `set(train.family_id) ∩ set(test.family_id) == ∅`.
- **L2 — no exact (shape, condition) shared (new-condition):** assert disjoint `(airfoil_id, α, Re, M)` keys.
- **L3 — near-duplicate audit:** assert no cross-split pair of airfoils is within `τ` in
  descriptor space.
- **L4 — fidelity honesty:** anchors (`rans-airfrans`, `cfd`, `windtunnel`) never silently
  merged into the XFOIL backbone; any fidelity mixing is explicit and flagged.
