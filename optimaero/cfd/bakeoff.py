"""Bake-off + confidence model for the envelope CFD surrogate.

Predicts Cd/Cl (V-normalized → more learnable; drag reconstructs as Cd·½ρV²·A_front) from envelope
silhouette + condition features. Runs the proven predictor pool, compares a single GLOBAL model
against PER-SPEED-REGIME models (Sky's ask), and trains a LightGBM confidence model on out-of-fold
residuals (selective prediction). Honest new-geometry-ish CV; reports R² and RMSE together.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

FEATURES = ["fineness", "A_front", "A_plan", "A_wet", "Dmax", "vol", "wet_front", "plan_front",
            "grow", "nose_frac", "tail_frac", "round_exp", "Re", "Mach", "alpha_deg",
            "prismatic", "x_maxarea", "area_smooth", "base_area", "nose_area"]  # +area-rule (2026-07-07)
TARGETS = ["Cd", "Cl"]


def pool():
    p = {
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06),
        "extratrees": lambda: ExtraTreesRegressor(n_estimators=250, n_jobs=-1, random_state=0),
        "mlp": lambda: make_pipeline(StandardScaler(),
                                     MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=1000,
                                                  random_state=0)),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge()),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
    }
    if HAVE_LGB:
        p["lgbm"] = lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                              num_leaves=31, verbose=-1)
    return p


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _r2(a, b):
    st = float(np.sum((b - b.mean()) ** 2))
    return float(1 - np.sum((a - b) ** 2) / st) if st > 0 else 0.0


def _oof(factory, X, y, folds):
    pred = np.zeros(len(y))
    for tr, te in folds:
        m = factory(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def _oof_per_regime(factory, X, y, reg, folds):
    """Route each test row to a model trained only on its own speed regime (within the fold)."""
    pred = np.zeros(len(y))
    for tr, te in folds:
        for g in np.unique(reg):
            trg = tr[reg[tr] == g]
            teg = te[reg[te] == g]
            if len(teg) == 0:
                continue
            if len(trg) < 8:                       # too few in-regime → fall back to full-fold model
                m = factory(); m.fit(X[tr], y[tr])
            else:
                m = factory(); m.fit(X[trg], y[trg])
            pred[teg] = m.predict(X[teg])
    return pred


def run(path: str, out: str = "results/envelope_cfd_bakeoff.json"):
    df = pd.read_parquet(path)
    df = df[df["converged"].astype(bool)]
    # drop unphysical CFD outliers (diverged cases that still wrote a coefficient)
    df = df[np.isfinite(df["Cd"]) & np.isfinite(df["Cl"]) &
            (df["Cd"].abs() < 10) & (df["Cl"].abs() < 10)]
    if "fineness" in df.columns:                            # drop under-resolved extreme needles
        df = df[df["fineness"] < 12]
    df = df.reset_index(drop=True)
    n = len(df)
    if "camber" in df.columns:                              # old rows predate camber → treat as 0
        df["camber"] = df["camber"].fillna(0.0)
    feats = [f for f in FEATURES if f in df.columns]
    if "camber" in df.columns and int(df["camber"].nunique()) > 1:   # include only when it varies (Cl)
        feats.append("camber")
    X = df[feats].values.astype(float)
    reg = df["speed_regime"].values
    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    report = {"rows": int(n), "features": feats, "regime_counts":
              {k: int((reg == k).sum()) for k in np.unique(reg)}, "targets": {}}

    for tgt in TARGETS:
        y = df[tgt].values.astype(float)
        glob = {}
        best_oof = None
        best = (None, 1e18)
        for name, f in pool().items():
            p = _oof(f, X, y, folds)
            glob[name] = {"rmse": _rmse(p, y), "r2": _r2(p, y)}
            if glob[name]["rmse"] < best[1]:
                best = (name, glob[name]["rmse"]); best_oof = p
        # per-regime version of the best global model
        pr = _oof_per_regime(pool()[best[0]], X, y, reg, folds)
        per_regime = {"rmse": _rmse(pr, y), "r2": _r2(pr, y)}
        # confidence model on |residual| of the best OOF predictions
        conf = {}
        if HAVE_LGB and best_oof is not None:
            resid = np.abs(best_oof - y)
            cp = _oof(lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1),
                      X, resid, folds)
            order = np.argsort(cp)                 # low predicted error first
            for cov in (1.0, 0.5, 0.25):
                k = max(3, int(cov * n))
                sel = order[:k]
                conf[f"rmse@{int(cov*100)}%"] = _rmse(best_oof[sel], y[sel])
        report["targets"][tgt] = {
            "global": glob, "best_global": {"model": best[0], **glob[best[0]]},
            "per_regime_best": per_regime,
            "winner": ("per_regime" if per_regime["rmse"] < best[1] else "global"),
            "confidence_selective_rmse": conf}

    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    import sys
    r = run(sys.argv[1] if len(sys.argv) > 1 else "data/processed/envelope_cfd.parquet")
    for tgt, d in r["targets"].items():
        bg = d["best_global"]; pr = d["per_regime_best"]
        print(f"{tgt}: best global = {bg['model']} (R²={bg['r2']:.3f} RMSE={bg['rmse']:.4f}) | "
              f"per-regime R²={pr['r2']:.3f} RMSE={pr['rmse']:.4f} | winner={d['winner']}")
        if d["confidence_selective_rmse"]:
            print("     confidence:", {k: round(v, 4) for k, v in d["confidence_selective_rmse"].items()})
