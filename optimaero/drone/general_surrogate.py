"""GENERAL drone surrogate — predicts the drag-reduction of a treatment for ANY multirotor.

Trained on the multi-drone dataset (`dataset.generate_multi`): many synthesized drones × treatments.
Features = drone-shape DESCRIPTORS (constant per drone, geometric) + the 3 treatment KNOBS. Target = the
**normalized reduction ratio** `treated_cda / bare_cda` — dimensionless, so it transfers across drone
sizes, and for a fixed drone, ranking by predicted ratio = ranking by treated drag. So at serve time we
compute the imported drone's descriptors (no CFD), predict the ratio for thousands of treatments, rank,
and CFD-verify the top few — for ANY drone, not one.

Honesty: evaluated with **GroupKFold BY DRONE** (train on some drones, test on HELD-OUT drones), so the
reported R² is genuine cross-drone generalization, not within-drone interpolation.

Run:  python -m optimaero.drone.general_surrogate [multidrone.parquet]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimaero.cfd.bakeoff import pool, _oof, _rmse, _r2, HAVE_LGB
from optimaero.drone.generator import DESCRIPTOR_FEATS
if HAVE_LGB:
    import lightgbm as lgb
from sklearn.model_selection import GroupKFold

KNOBS_3 = ["tail_len", "chord", "thick"]
KNOBS_HD = ["tail_len", "tail_base", "arm_chord", "arm_thick", "nose_len", "nose_base"]
BARE_FEATS = ["bare_cd", "bare_cda"]     # the drone's MEASURED bluffness (from its 1 bare CFD)
GEN_FEATS = DESCRIPTOR_FEATS + KNOBS_3   # default (3-knob); train_and_save picks the actual set per dataset
TARGET = "ratio"                     # treated_cda / bare_cda (lower = more drag reduction)


def _feature_cols(df) -> list:
    """Pick the feature columns present in the dataset: shape descriptors + whichever treatment knobs vary
    (3-knob or 6-knob) + the bare-drone bluffness features. Keeps train and serve on the same set."""
    cand = list(dict.fromkeys(KNOBS_HD + KNOBS_3))                 # dedup, keep order
    knobs = [c for c in cand if c in df.columns and df[c].nunique() > 1]
    feats = list(DESCRIPTOR_FEATS) + knobs + [c for c in BARE_FEATS if c in df.columns]
    return [c for c in feats if c in df.columns]
ARTIFACT = "results/general_drone_surrogate.joblib"
REPORT = "results/general_drone_surrogate_bakeoff.json"


@dataclass
class GeneralDroneSurrogate:
    model: object
    conf_model: object | None
    feats: list
    meta: dict

    def predict(self, rows) -> tuple:
        """rows: dict or DataFrame with descriptor + knob columns. Returns (ratio_pred, pred_abs_err)."""
        df = pd.DataFrame([rows]) if isinstance(rows, dict) else pd.DataFrame(rows)
        missing = [c for c in self.feats if c not in df.columns]
        if missing:
            raise KeyError(f"general surrogate missing feature columns: {missing}")
        X = df[self.feats].values.astype(float)
        ratio = np.asarray(self.model.predict(X), float)
        err = (np.asarray(self.conf_model.predict(X), float) if self.conf_model is not None
               else np.zeros(len(X)))
        return ratio, err


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Per treatment row, attach its drone's bare drag-area + Cd (bluffness features) and the reduction
    ratio target; keep only additive-valid, converged rows."""
    df = df[df["converged"] == True].copy()                                  # noqa: E712
    barecda = df[df["is_bare"] == True].groupby("drone_id")["cda"].first()   # noqa: E712
    barecd = df[df["is_bare"] == True].groupby("drone_id")["Cd"].first()     # noqa: E712
    trt = df[df["is_bare"] == False].copy()                                  # noqa: E712
    if "additive_ok" in trt.columns:
        trt = trt[trt["additive_ok"] == True]                               # noqa: E712
    trt["bare_cda"] = trt["drone_id"].map(barecda)
    trt["bare_cd"] = trt["drone_id"].map(barecd)
    trt = trt[trt["bare_cda"].notna() & trt["cda"].notna() & (trt["bare_cda"] > 0) & (trt["cda"] > 0)]
    trt["ratio"] = trt["cda"] / trt["bare_cda"]
    return trt


def train_and_save(dataset_path: str = "data/processed/multidrone.parquet") -> "GeneralDroneSurrogate":
    import joblib
    df = _prepare(pd.read_parquet(dataset_path))
    feats = _feature_cols(df)
    df = df.dropna(subset=feats + [TARGET])
    n = len(df); n_drones = int(df["drone_id"].nunique())
    X = df[feats].values.astype(float); y = df[TARGET].values.astype(float)
    groups = df["drone_id"].values

    # HELD-OUT-DRONE evaluation: GroupKFold so no drone is in both train and test
    n_splits = int(min(5, max(2, n_drones)))
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
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
    # does ranking survive per-drone? (Spearman of predicted vs actual ratio within each held-out drone)
    rank_ok = _within_drone_rank(df, best_oof)

    report = {"rows": int(n), "n_drones": n_drones, "target": TARGET,
              "eval_regime": ("GroupKFold BY DRONE (held-out drones, no leakage). CAVEAT: the headline R2 "
                              "is the BEST of the model pool on ONE split — selection-optimistic, no error "
                              "bar; see per-model spread in 'global'. The optimizer relies on the "
                              "within-drone RANK corr + CFD-verify, not this R2."),
              "held_out_drone": {"best_model": best[0], "global": glob, "best": glob[best[0]],
                                 "confidence": conf, "within_drone_rank_corr": rank_ok,
                                 "model_r2_spread": sorted([round(v["r2"], 3) for v in glob.values()])}}

    model = pool()[best[0]](); model.fit(X, y)
    conf_model = None
    if HAVE_LGB:
        oof = _oof(pool()[best[0]], X, y, folds)
        conf_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)
        conf_model.fit(X, np.abs(oof - y))
    report["n_features"] = len(feats); report["features"] = feats
    sur = GeneralDroneSurrogate(model=model, conf_model=conf_model, feats=feats,
                                meta={"n": int(n), "n_drones": n_drones, "target": TARGET,
                                      "best_model": best[0], "feats": feats,
                                      "held_out_drone_r2": glob[best[0]]["r2"],
                                      "within_drone_rank_corr": rank_ok,
                                      "eval_regime": report["eval_regime"]})
    os.makedirs("results", exist_ok=True)
    joblib.dump(sur, ARTIFACT)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    return sur


def _within_drone_rank(df: pd.DataFrame, oof_pred) -> float:
    """Mean per-drone rank correlation (Spearman) of predicted vs actual ratio — does the surrogate rank
    a HELD-OUT drone's treatments in the right order? (That is what the optimizer actually needs.)"""
    from scipy.stats import spearmanr
    d = df.copy(); d["_pred"] = oof_pred
    cors = []
    for _, g in d.groupby("drone_id"):
        if len(g) >= 4:
            c = spearmanr(g["_pred"], g[TARGET]).correlation
            if np.isfinite(c):
                cors.append(c)
    return float(np.mean(cors)) if cors else float("nan")


def general_available(path: str = ARTIFACT) -> bool:
    return os.path.exists(path)


def load_general(path: str = ARTIFACT) -> "GeneralDroneSurrogate":
    import joblib
    return joblib.load(path)


if __name__ == "__main__":
    import sys
    # Import the trainer under the fully-qualified module so the pickled GeneralDroneSurrogate carries its
    # real module path (not __main__), otherwise the saved artifact can't be unpickled in other processes.
    from optimaero.drone.general_surrogate import train_and_save as _train
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/multidrone.parquet"
    s = _train(path)
    r = json.load(open(REPORT))
    h = r["held_out_drone"]
    print(f"rows={r['rows']} across {r['n_drones']} drones | target=reduction ratio (treated/bare cda)")
    print(f"  NOTE: {r['eval_regime']}")
    print(f"  HELD-OUT-DRONE : {h['best_model']} R2={h['best']['r2']:.3f} RMSE={h['best']['rmse']:.4f}")
    print(f"  within-drone rank corr (Spearman): {h['within_drone_rank_corr']:.3f}  <-- what the optimizer needs")
    print(f"  confidence RMSE: 100%={h['confidence'].get('rmse@100%', float('nan')):.4f} "
          f"25%={h['confidence'].get('rmse@25%', float('nan')):.4f}")
