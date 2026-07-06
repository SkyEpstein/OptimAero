# OptimAero

**An uncertainty-aware ML surrogate for aerodynamics — and an envelope-constrained
inverse-design optimizer built on top of it.**

Replace slow CFD with a fast ML surrogate that *knows when it is uncertain* and defers to
real CFD when it is. Then wrap it in an optimizer: give it a packaging envelope a shape must
fit inside plus requirements (top speed, lift, drag), and it returns an optimized geometry —
importable/exportable through neutral CAD formats (STEP/STL/IGES).

Foundation: 2D airfoils/wings. Application: drones / UAVs / propellers (built from airfoil
sections). Physics-coupled to vehicle level (BEMT / lifting-line), with a learned 3D
correction added later.

## Status

Phase 1 (data foundation) — in progress. See `CHANGELOG.md` for the honest running log.

- **`memory/constitution.md`** — the governing document (mission, scope, principles).
- **`specs/2026-07-05-data-foundation/`** — the current spec: `spec.md`, `plan.md`,
  `data-model.md` (the leakage map & evaluation contract), `tasks.md`.

## The confidence model (why this is trustworthy)

A **learned error model** (carried over from the sibling f-elements research) predicts the
surrogate's error on each input; low-predicted-error points are trusted, the rest defer to
CFD. Trained only on **out-of-fold residuals** (no leakage), gated by selective-prediction
percentiles + split-conformal intervals, evaluated under **group-aware splits by airfoil
family** so no shape or its near-duplicates straddle train/test.

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# XFOIL (ground-truth generator): scripts/install_xfoil.sh   (built during Phase 1)
```

Non-negotiables (from the constitution): no leakage ever · report R² *and* RMSE together ·
trust before speed · bake-offs not hunches · SDD governs every change.
