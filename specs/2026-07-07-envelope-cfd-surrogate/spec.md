# Spec — Envelope CFD surrogate + confidence model

## Decision (Sky, 2026-07-07, MCQs)
- **Scope:** ML surrogate over the **envelope parameter space** — inputs = envelope params
  (grow, nose_frac, tail_frac, round_exp) + the base shape's silhouette features (fineness,
  frontal area, planform area, wetted area, bbox aspect) + conditions (V, alpha) → **drag & lift**.
- **Truth:** **capped OpenFOAM CFD from the start** (real forces, not the physics estimate).
- Reuse the proven bake-off (`optimaero/bakeoff/`) + confidence model (LightGBM on OOF residuals,
  `AeroPrediction` interface with trusted/ood gates) — mirror the working 2D airfoil pipeline in 3D.

## Problem
The 3D envelope/shape optimizer scores candidates with a fast physics estimate. Replace it with an
ML surrogate trained on real CFD, gated by the confidence model, so the optimizer searches at CFD
fidelity in milliseconds and flags out-of-distribution / low-confidence points for a CFD check.

## Requirements
1. **CFD labeler** (this phase): envelope params → `build_envelope` mesh (clean, watertight) →
   OpenFOAM external-flow case (background blockMesh + snappyHexMesh around the STL, simpleFoam
   k-omega SST, forceCoeffs) → drag [N], lift [N], Cd, Cl. MUST run resource-capped
   (`--memory=12g --memory-swap=12g --cpus=6`) with a host memory watchdog — never crash the Mac.
2. **Dataset generation:** Latin-hypercube sample the param space × a set of base silhouettes ×
   conditions; label each with the CFD labeler; store parquet (features + CFD drag/lift + a
   `converged` flag + a `family_id` for group-aware CV). Size/time confirmed with Sky before launch.
3. **Bake-off + confidence:** feed the dataset through the existing bake-off (targets drag, lift);
   GroupKFold by family; select by RMSE @ 50% coverage; train LightGBM confidence on OOF residuals.
   **Per-speed-regime models (Sky, 2026-07-07):** a single model strains across the full Re/Mach
   range (300 mph vs 30 mph is different physics), so the bake-off also trains one model per speed
   band (low / mid / high, recorded as `speed_regime` in the dataset) and compares the regime-gated
   ensemble against the single global model — the winner is whichever generalizes better on held-out
   new-geometry folds. `SPEEDS = [15,25,40,70,100,134] m/s` span the regimes.
4. **Integration:** wrap the winner behind a `Surrogate`-style API returning drag/lift + predicted
   error + trusted/ood; swap it into `optimize_envelope`'s objective, falling back to the physics
   estimate (or a CFD check) when not trusted / OOD.

## Acceptance criteria
- **CFD labeler:** on 2–3 envelopes returns positive, physically-ordered drag (bluffer → more drag),
  runs with peak container memory < cap and host free RAM never dangerously low; deterministic
  enough to reuse (report residual noise). A streamlined envelope has lower Cd than a sphere.
- **Dataset:** ≥ N labeled rows (N set with Sky), converged fraction reported honestly.
- **Surrogate:** honest group-CV R²/RMSE for drag & lift reported with both metrics; confidence
  model improves RMSE at reduced coverage (selective-prediction curve), same regime as the 2D win.
- **Integration:** optimizer using the surrogate matches CFD-verified drag within the confidence
  band on held-out shapes; OOD envelopes correctly flagged.

## Risks / mitigations
- **Auto-mesh robustness** (Stage-B's flagged risk): mitigated because envelopes are smooth closed
  watertight bodies (not arbitrary imports). Still: validate mesh + fall back / mark non-converged.
- **Machine safety:** hard Docker caps + watchdog; single-case test before any sweep; sweep batched
  and resumable; Sky confirms size/time before launch.
- **CFD cost:** each case is minutes; a few hundred rows = hours. Batch, checkpoint, resume.

## Plan (phased)
1. Build + prove the capped CFD labeler on one envelope (this turn).
2. Silhouette featurizer + LHS sampler over params × base shapes × conditions.
3. Confirm dataset size/time with Sky → run the sweep (background, checkpointed).
4. Bake-off + confidence on the CFD dataset.
5. Integrate the surrogate into `optimize_envelope` with confidence gating; verify vs CFD.
