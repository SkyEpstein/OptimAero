"""Universal drag surrogate: one model that predicts a shape's drag coefficient across shape types.

Trained on the diverse multi-type CFD dataset (data/processed/xtype_*/), each shape a JSON with its CFD Cd
and a surface point cloud. Features come from `features.features_from_saved` (training) / `universal_features`
(serving) — the SAME computation, no skew. Held out per shape (KFold): reports overall + per-type rank and a
confidence model (predict the |log-Cd residual|) so the caller knows when to trust it.

Run:  python -m optimaero.universal.surrogate
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import trimesh

from optimaero.universal.features import features_from_saved, universal_features, FEATURE_NAMES
from optimaero.cfd.foam import RHO

ARTIFACT = "results/universal_drag_surrogate.joblib"
REPORT = "results/universal_drag_surrogate_report.json"


def assemble_dataset(root: str = "data/processed"):
    """Load every xtype_* shape into (features, Cd, type). Skips non-converged / featureless rows."""
    X, y, types = [], [], []
    for dirn in sorted(glob.glob(os.path.join(root, "xtype_*"))):
        t = os.path.basename(dirn).replace("xtype_", "")
        for f in glob.glob(os.path.join(dirn, "*.json")):
            try:
                d = json.load(open(f))
                if not (np.isfinite(d["Cd"]) and 0 < d["Cd"] < 5) or "points" not in d:
                    continue
                X.append(features_from_saved(d)); y.append(float(d["Cd"])); types.append(t)
            except Exception:
                pass
    return np.asarray(X, float), np.asarray(y, float), np.asarray(types)


@dataclass
class UniversalDragSurrogate:
    model: object              # predicts log(Cd)
    conf_model: object         # predicts |log-Cd residual| (lower = more confident)
    feat_names: list
    meta: dict

    def predict_drag(self, mesh: trimesh.Trimesh, V: float, flow_axis: str = "x", rho: float = RHO):
        """Return (drag[N], Cd, confidence_error) for a mesh. drag = Cd·½ρV²·A_front."""
        m = mesh
        if flow_axis != "x":
            m = mesh.copy()
            m.apply_transform(trimesh.geometry.align_vectors(np.eye(3)[{"x": 0, "y": 1, "z": 2}[flow_axis]],
                                                             [1.0, 0.0, 0.0]))
        f = universal_features(mesh, flow_axis).reshape(1, -1)
        cd = float(np.exp(self.model.predict(f))[0])
        err = float(self.conf_model.predict(f)[0])
        try:
            A_front = float(m.projected([1, 0, 0]).area)
        except Exception:
            A_front = float(m.extents[1] * m.extents[2])
        drag = cd * 0.5 * rho * V ** 2 * A_front
        return drag, cd, err


def _oof(mk, X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        m = mk(); m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return p


def train_and_save(root: str = "data/processed") -> "UniversalDragSurrogate":
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr

    X, cd, types = assemble_dataset(root)
    if len(X) < 10:
        raise ValueError(f"universal surrogate needs >=10 labeled shapes, found {len(X)} in {root}/xtype_*")
    y = np.log(cd)
    folds = list(KFold(min(5, len(X)), shuffle=True, random_state=0).split(X))
    mk = lambda: GradientBoostingRegressor(n_estimators=500, max_depth=3, subsample=0.8, random_state=0)
    oof = _oof(mk, X, y, folds)
    resid = np.abs(oof - y)
    cmk = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    coof = _oof(cmk, X, resid, folds)

    def rc(i):
        return float(spearmanr(oof[i], y[i]).correlation) if len(i) >= 4 else float("nan")

    order = np.argsort(coof); n = len(y)
    conf = {f"rank@{int(c*100)}%": rc(order[:max(4, int(c * n))]) for c in (1.0, 0.5, 0.25)}
    per_type = {t: {"n": int((types == t).sum()), "rank": rc(np.where(types == t)[0])}
                for t in sorted(set(types))}
    report = {"n": int(n), "n_types": len(set(types)), "features": FEATURE_NAMES,
              "overall_rank": rc(np.arange(n)), "confidence_gated": conf, "per_type": per_type,
              "eval": "held-out (KFold-5) rank of predicted vs CFD Cd; confidence = GBR on |log-resid|"}

    model = mk(); model.fit(X, y)
    conf_model = cmk(); conf_model.fit(X, resid)
    sur = UniversalDragSurrogate(model=model, conf_model=conf_model, feat_names=FEATURE_NAMES,
                                 meta={"n": int(n), "n_types": len(set(types)),
                                       "overall_rank": report["overall_rank"]})
    os.makedirs("results", exist_ok=True)
    joblib.dump(sur, ARTIFACT)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    return sur


def available(path: str = ARTIFACT) -> bool:
    return os.path.exists(path)


def load(path: str = ARTIFACT) -> "UniversalDragSurrogate":
    import joblib
    return joblib.load(path)


if __name__ == "__main__":
    from optimaero.universal.surrogate import train_and_save as _t   # real module path for the pickle
    _t()
    r = json.load(open(REPORT))
    print(f"universal drag surrogate — {r['n']} shapes / {r['n_types']} types | overall rank {r['overall_rank']:.3f}")
    print(f"  confidence-gated: {r['confidence_gated']}")
    print("  per-type rank:")
    for t, v in r["per_type"].items():
        print(f"    {t:<10} n={v['n']:<4} rank={v['rank']:.3f}")
