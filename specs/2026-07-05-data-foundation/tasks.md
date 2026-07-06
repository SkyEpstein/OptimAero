# Phase 1 — Data Foundation (tasks)

*SDD: tasks. Dependency-ordered; each task ties to an acceptance criterion in `spec.md`.*
Date: 2026-07-05

Legend: ☐ todo · ◐ in progress · ☑ done

## Environment (foundation)
- ☑ **T0.1** Project venv + Phase-1 deps installed; `requirements.txt` pinned (25 pkgs).
- ☑ **T0.2** Working **XFOIL** 6.99 binary (`tools/xfoil/xfoil`, arm64, headless) +
  reproducible `tools/xfoil/install_xfoil.sh`. Built from source with 3 documented I/O-only
  patches. **Independently validated**: NACA 0012 slope 0.108/deg; NACA 4412 zero-lift at
  ≈ −4° (textbook) → solver intact after patching. → AC1
- ☐ **T0.3** Package skeleton `optimaero/` + thin `scripts/` entry points.

## Geometry & curation (XFOIL-independent — buildable now)
- ☑ **T1.1** `datasets/uiuc.py`: airfoil catalog + loader. **No scraping needed** — read the
  UIUC DB bundled with AeroSandbox (2,174 airfoils, all loadable, version-pinned). Self-check
  found 25 exact/rename-duplicate groups (50 airfoils) → confirms the near-dup merge matters. → AC1
- ☑ **T1.2** `geometry.py`: normalize + **CST (Kulfan) fit** with recorded round-trip
  residual + `fit_ok` flag. Empirical: **12 weights/side** is the sweet spot (191/200 fit
  <1% chord); higher orders destabilize. Raw coords stay source of truth. → AC1, R2
- ☑ **T1.3** `families.py`: hybrid grouping — must-link exact/rename dupes **+** shape-space
  near-duplicate merge (80-dim orientation-guarded signature, KD-tree union-find).
  **Adversarially verified** (no leakage bug; 34 diff-name twins, zero straddle). **τ = 0.003
  locked** (largest family = 4). → AC2, R3

## Data generation
- ☑ **T2.1** `generate.py`: XFOIL sweep driver (AeroSandbox), parallel across cores +
  per-airfoil sharding (crash-safe resume). `regime_flag` for post-stall/low-Re; honest
  alpha-yield reporting; 45s per-sweep timeout. → AC1
- ☑ **T2.2** **Full backbone sweep DONE** → `xfoil_backbone.parquet`: **213,406 rows,
  2,169 airfoils**, 5 Re; 72.7% alpha-yield; regime 141k ok / 43k low_re / 29k post_stall.
  Ran ~3.5h, resumable, zero manual intervention. → AC1

## Anchors & unification
- ☐ **T3.1** `datasets/airfrans.py`: ingest an AirfRANS subset → schema, `fidelity=rans-airfrans`.
  Record exactly what was pulled. → AC1
- ☐ **T3.2** Unify all sources into one parquet table (schema per `data-model.md §1`). → AC1

## Splits & leakage (the honesty gate)
- ☑ **T4.1** `splits.py`: new-geometry (group by `family_id`) + new-condition (per-shape
  disjoint rows) splits + L1/L2 check helpers. Self-checked on pilot: L1 no-family-straddle
  ✓, L2 no-shared-row ✓, every shape in both sides ✓. → AC2
- ☑ **T4.2** `tests/test_leakage.py`: pytest for L1 (family straddle) + L2 (shared rows) +
  a **non-vacuous** test proving the guard catches a deliberate straddle. 3 passed. Auto-uses
  the backbone once it exists, else the pilot. *(L4 fidelity-honesty test added with T3.)* → AC2

## Documentation
- ☑ **T5.1** `datacard.py` → `docs/DATA_CARD.md`: honest counts (213,406 rows), regime mix,
  τ sensitivity table, AirfRANS caveats, known gaps. → AC3
- ☐ **T5.2** Update `CHANGELOG.md` with honest row counts and the fidelity breakdown. → AC3

## Verify (adversarial reflection — SDD phase 7)
- ☐ **V.1** Skeptical pass / verifier: confirm zero leakage, honest counts, no XFOIL
  post-stall rows silently used as ground truth, CST fits sane. Then Phase 1 is done.
