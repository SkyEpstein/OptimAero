"""Phase-2 nested bake-off (constitution §4, roadmap Phase 2).

(1) Predictor bake-off over a pool of models + ensembles, ranked on held-out NEW-GEOMETRY
    RMSE (GroupKFold by family_id — no leakage).
(2) Confidence bake-off on the top predictors: a LightGBM error-model on OUT-OF-FOLD
    residuals (the honesty guard), gated by selective-prediction percentiles + Spearman.
(3) Winner = the (predictor, confidence) pair with the best DEPLOYED trust-gated accuracy
    (RMSE on retained points at the target coverage).

Honest reporting: R2 AND RMSE per output, at coverage operating points. Top-X% R2 is shown
only as a secondary view (variance-confounded), never the selection metric.

Env: OPTIMAERO_BAKEOFF_SMOKE=N  -> subsample to N rows + 3 folds for a fast end-to-end check.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from optimaero.bakeoff import dataset as D

OUTPUTS = ["Cl", "Cd", "Cm"]
_REPO = D._REPO
COVERAGES = (100, 50, 25, 10)
TARGET_COVERAGE = 50  # deployed operating point for the winner metric


def predictor_pool():
    """name -> factory. Single-output regressors (fit once per target)."""
    return {
        "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                              num_leaves=63, n_jobs=-1, verbosity=-1),
        "hist_gbr": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                          max_leaf_nodes=63),
        "extratrees": lambda: ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=0),
        "mlp": lambda: make_pipeline(StandardScaler(),
                                     MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=150,
                                                  early_stopping=True, random_state=0)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=12)),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def _metrics(y, p):
    rmse = np.sqrt(np.mean((y - p) ** 2, axis=0))
    denom = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - np.sum((y - p) ** 2, axis=0) / np.where(denom == 0, 1, denom)
    return rmse, r2


def oof_predict(factory, X, Y, folds):
    oof = np.zeros_like(Y)
    for tr, va in folds:
        for j in range(Y.shape[1]):
            m = factory()
            m.fit(X[tr], Y[tr, j])
            oof[va, j] = m.predict(X[va])
    return oof


def confidence_eval(X, Y, oof, folds):
    """Learned error-model (LightGBM) on OUT-OF-FOLD residuals; selective-prediction curve."""
    resid = np.abs(Y - oof)
    err = np.zeros_like(resid)
    for tr, va in folds:
        Ftr = np.c_[X[tr], oof[tr]]   # "plain" recipe: features + the prediction
        Fva = np.c_[X[va], oof[va]]
        for j in range(Y.shape[1]):
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  n_jobs=-1, verbosity=-1)
            m.fit(Ftr, resid[tr, j])
            err[va, j] = m.predict(Fva)
    err = np.clip(err, 1e-9, None)
    out = {}
    for j, name in enumerate(OUTPUTS):
        order = np.argsort(err[:, j])           # most-confident first
        curve = {}
        for q in COVERAGES:
            idx = order[: max(1, int(len(order) * q / 100))]
            rmse, r2 = _metrics(Y[idx, j:j + 1], oof[idx, j:j + 1])
            curve[q] = {"rmse": round(float(rmse[0]), 4), "r2": round(float(r2[0]), 3)}
        sp = spearmanr(err[:, j], resid[:, j]).correlation
        out[name] = {"coverage": curve, "spearman": round(float(sp), 3)}
    return out


def main():
    t0 = time.time()
    X, Y, groups, cols, d = D.load_dataset()
    smoke = int(os.environ.get("OPTIMAERO_BAKEOFF_SMOKE", "0"))
    if smoke:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(X), size=min(smoke, len(X)), replace=False)
        X, Y, groups = X[keep], Y[keep], groups[keep]
    n_splits = 3 if smoke else 5
    folds = list(GroupKFold(n_splits).split(X, Y[:, 0], groups))
    print(f"bake-off: rows={len(X):,} features={len(cols)} families={len(np.unique(groups))} "
          f"folds={n_splits}", flush=True)

    # (1) predictor bake-off
    pred_results, oofs = {}, {}
    for name, fac in predictor_pool().items():
        t = time.time()
        try:
            oof = oof_predict(fac, X, Y, folds)
        except Exception as e:
            print(f"  {name}: FAILED ({e})", flush=True)
            continue
        rmse, r2 = _metrics(Y, oof)
        oofs[name] = oof
        pred_results[name] = {"rmse": rmse.round(4).tolist(), "r2": r2.round(3).tolist(),
                              "sec": round(time.time() - t, 1)}
        print(f"  {name:11s} Cl/Cd/Cm RMSE={rmse.round(4).tolist()} "
              f"R2={r2.round(3).tolist()} ({pred_results[name]['sec']}s)", flush=True)

    # ensemble: average of top-3 by Cl RMSE
    ranked = sorted(pred_results, key=lambda n: pred_results[n]["rmse"][0])
    topk = ranked[:3]
    avg = np.mean([oofs[n] for n in topk], axis=0)
    rmse, r2 = _metrics(Y, avg)
    oofs["avg_top3"] = avg
    pred_results["avg_top3"] = {"rmse": rmse.round(4).tolist(), "r2": r2.round(3).tolist(),
                                "members": topk}
    print(f"  avg_top3    Cl/Cd/Cm RMSE={rmse.round(4).tolist()} R2={r2.round(3).tolist()}",
          flush=True)

    # (2) confidence bake-off on the top-K predictors (+ the ensemble)
    ranked_all = sorted(pred_results, key=lambda n: pred_results[n]["rmse"][0])
    conf_candidates = ranked_all[:3]
    conf_results = {}
    for name in conf_candidates:
        print(f"  confidence on {name} ...", flush=True)
        conf_results[name] = confidence_eval(X, Y, oofs[name], folds)

    # (3) winner = candidate with best deployed trust-gated Cl RMSE @ TARGET_COVERAGE
    def gated(name):
        return conf_results[name]["Cl"]["coverage"][TARGET_COVERAGE]["rmse"]
    winner = min(conf_candidates, key=gated)

    report = {
        "rows": int(len(X)), "features": cols, "families": int(len(np.unique(groups))),
        "regime": "new-geometry (GroupKFold by family_id)",
        "predictors": pred_results,
        "confidence": conf_results,
        "winner": winner,
        "winner_gated_Cl_rmse@%d%%" % TARGET_COVERAGE: gated(winner),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    out = os.path.join(_REPO, "results",
                       "phase2_bakeoff_smoke.json" if smoke else "phase2_bakeoff.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWINNER: {winner}  (deployed Cl RMSE @ {TARGET_COVERAGE}% = {gated(winner)})")
    print(f"wrote {out}  |  {report['runtime_min']} min")


if __name__ == "__main__":
    main()
