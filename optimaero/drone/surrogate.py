"""Drone-form surrogate: rank drone forms by a speed-invariant form-quality metric so the optimizer can
search thousands of forms in milliseconds and CFD-verify only the top few.

Target = **drag area** cda = drag / q  (q = ½ρV²), i.e. Cd·A_front. This is speed-invariant in the
fully-turbulent regime, so ranking forms by predicted cda gives the SAME ordering as ranking by drag at
ANY single speed — the surrogate is usable at any airspeed even though it was trained at one, and the final
answer is always CFD-verified at the user's actual V/AoA (so surrogate error only chooses which forms to CFD).

Features = the 3 varying treatment knobs [tail_len, chord, thick]. For a FIXED base drone every geometric
feature is a deterministic function of these, so a 3-knob model loses no information; we still bake off
knobs-only vs knobs+geometry and run an extrapolation split, and we honestly label the R² as
INTERPOLATION WITHIN ONE DRONE (not a universal drone model). Trained at a single condition
(V, AoA) — recorded in meta; ranking transfers across speed, AoA assumed ~0° cruise.

Run:  python -m optimaero.drone.surrogate [drone_form.parquet]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimaero.cfd.bakeoff import pool, _oof, _rmse, _r2, HAVE_LGB
from optimaero.cfd.foam import RHO
if HAVE_LGB:
    import lightgbm as lgb
from sklearn.model_selection import KFold

# the varying search knobs — the honest feature set (constants like rmax/Mach/alpha are NOT features here)
PARAM_FEATS = ["tail_len", "chord", "thick"]
GEOM_FEATS = ["fineness", "A_front", "A_plan", "A_wet", "Dmax", "vol", "wet_front", "plan_front",
              "prismatic", "x_maxarea", "area_smooth", "base_area", "nose_area"]
FULL_FEATS = PARAM_FEATS + GEOM_FEATS
TARGET = "cda"                       # drag area = drag / q = Cd·A_front (speed-invariant)
ARTIFACT = "results/drone_surrogate.joblib"
REPORT = "results/drone_surrogate_bakeoff.json"


@dataclass
class DroneSurrogate:
    model: object                 # predicts cda (drag area), fit on `feats`
    conf_model: object | None     # predicts |residual| (lower = more confident); may be None
    feats: list                   # feature order the model expects
    needs_mesh: bool              # True if `feats` include geometry (must build the mesh to score)
    meta: dict

    def predict(self, rows) -> tuple:
        """rows: dict or DataFrame with the feature columns. Returns (cda_pred, pred_abs_err)."""
        df = pd.DataFrame([rows]) if isinstance(rows, dict) else pd.DataFrame(rows)
        missing = [c for c in self.feats if c not in df.columns]
        if missing:
            raise KeyError(f"surrogate.predict missing feature columns: {missing}")
        X = df[self.feats].values.astype(float)     # df[list] preserves the trained feature order
        cda = np.asarray(self.model.predict(X), float)
        err = (np.asarray(self.conf_model.predict(X), float) if self.conf_model is not None
               else np.zeros(len(X)))
        return cda, err


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["converged"] == True].copy()                          # noqa: E712
    if "additive_ok" in df.columns:                                 # don't train on §5.8-violating forms
        df = df[df["additive_ok"] == True]                          # noqa: E712
    if "drag" in df.columns and "V" in df.columns:
        df["cda"] = df["drag"] / (0.5 * RHO * df["V"] ** 2)         # drag area (speed-invariant target)
    df = df[np.isfinite(df[TARGET]) & (df[TARGET] > 0)]
    for c in FULL_FEATS:
        if c not in df.columns:
            df[c] = 0.0
    return df.dropna(subset=FULL_FEATS)


def _bakeoff(df: pd.DataFrame, feats: list, folds) -> dict:
    X = df[feats].values.astype(float)
    y = df[TARGET].values.astype(float)
    glob, best, best_oof = {}, (None, 1e18), None
    for name, f in pool().items():
        p = _oof(f, X, y, folds)
        glob[name] = {"rmse": _rmse(p, y), "r2": _r2(p, y)}
        if glob[name]["rmse"] < best[1]:
            best = (name, glob[name]["rmse"]); best_oof = p
    conf = {}
    if HAVE_LGB and best_oof is not None:
        resid = np.abs(best_oof - y)
        cp = _oof(lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1),
                  X, resid, folds)
        order = np.argsort(cp)
        for cov in (1.0, 0.5, 0.25):
            k = max(3, int(cov * len(y))); sel = order[:k]
            conf[f"rmse@{int(cov*100)}%"] = _rmse(best_oof[sel], y[sel])
    return {"best_model": best[0], "global": glob, "best": glob[best[0]], "confidence": conf}


def _extrapolation_split(df: pd.DataFrame, feats: list, knob: str = "tail_len") -> dict:
    """Train on the lower 80% of a knob, test on the top 20% — does it generalize past what it saw?"""
    thr = df[knob].quantile(0.8)
    tr, te = df[df[knob] <= thr], df[df[knob] > thr]
    if len(te) < 4 or len(tr) < 20:
        return {"note": "too few rows for an extrapolation split"}
    m = pool()["extratrees"]()
    m.fit(tr[feats].values.astype(float), tr[TARGET].values.astype(float))
    p = m.predict(te[feats].values.astype(float)); y = te[TARGET].values.astype(float)
    return {"knob": knob, "thr": float(thr), "n_test": int(len(te)), "r2": _r2(p, y), "rmse": _rmse(p, y)}


def train_and_save(dataset_path: str = "data/processed/drone_form.parquet") -> DroneSurrogate:
    import joblib
    raw = pd.read_parquet(dataset_path)
    df = _clean(raw)
    n = len(df)
    folds = list(KFold(5, shuffle=True, random_state=0).split(df))
    cond = {"V": float(raw["V"].iloc[0]) if "V" in raw and len(raw) else None,
            "alpha_deg": float(raw["alpha_deg"].iloc[0]) if "alpha_deg" in raw and len(raw) else None,
            "n_base_drones": 1}
    report = {"rows": int(n), "target": TARGET, "condition": cond,
              "eval_regime": "OOF R2 = INTERPOLATION within ONE drone's treatment space (not universal)",
              "knobs_only": _bakeoff(df, PARAM_FEATS, folds),
              "knobs_plus_geometry": _bakeoff(df, FULL_FEATS, folds),
              "extrapolation_tail": _extrapolation_split(df, PARAM_FEATS)}

    feats = PARAM_FEATS
    best = report["knobs_only"]["best_model"]
    X = df[feats].values.astype(float); y = df[TARGET].values.astype(float)
    model = pool()[best](); model.fit(X, y)
    conf_model = None
    if HAVE_LGB:
        oof = _oof(pool()[best], X, y, folds)
        conf_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)
        conf_model.fit(X, np.abs(oof - y))
    sur = DroneSurrogate(model=model, conf_model=conf_model, feats=feats, needs_mesh=False,
                         meta={"n": int(n), "target": TARGET, "best_model": best, "condition": cond,
                               "eval_regime": report["eval_regime"],
                               "knobs_only_r2": report["knobs_only"]["best"]["r2"],
                               "knobs_geom_r2": report["knobs_plus_geometry"]["best"]["r2"]})
    os.makedirs("results", exist_ok=True)
    joblib.dump(sur, ARTIFACT)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    return sur


def surrogate_available(path: str = ARTIFACT) -> bool:
    return os.path.exists(path)


def load_surrogate(path: str = ARTIFACT) -> "DroneSurrogate":
    import joblib
    return joblib.load(path)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/drone_form.parquet"
    s = train_and_save(path)
    r = json.load(open(REPORT))
    print(f"rows={r['rows']} target={r['target']} (drag area = Cd·A_front, speed-invariant) "
          f"condition={r['condition']}")
    print(f"  NOTE: {r['eval_regime']}")
    print(f"  knobs-only      : {r['knobs_only']['best_model']:10s} "
          f"R2={r['knobs_only']['best']['r2']:.3f} RMSE={r['knobs_only']['best']['rmse']:.5f} "
          f"conf@25%={r['knobs_only']['confidence'].get('rmse@25%', float('nan')):.5f}")
    print(f"  knobs+geometry  : {r['knobs_plus_geometry']['best_model']:10s} "
          f"R2={r['knobs_plus_geometry']['best']['r2']:.3f}")
    print(f"  extrapolation   : {r['extrapolation_tail']}")
