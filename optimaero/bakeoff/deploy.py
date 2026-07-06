"""Deployable trained surrogate — the bake-off winner behind the `Surrogate` socket.

`train_and_save()` fits the winning MLP predictor + the LightGBM confidence (error) model on
the full backbone and calibrates the trust/OOD gate from out-of-fold predicted errors.
`TrainedSurrogate` loads it and implements `Surrogate.predict`, so the inverse-design
optimizer, BEMT, and CAD pipeline run on OUR model with OUR confidence — placeholder gone.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from optimaero.surrogate import Surrogate, AeroPrediction
from optimaero.bakeoff import dataset as D
from optimaero import geometry as G

OUTPUTS = ["Cl", "Cd", "Cm"]
MODEL_PATH = os.path.join(D._REPO, "results", "trained_surrogate.joblib")


def _mlp():
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=150,
                                      early_stopping=True, random_state=0))


def train_and_save(path: str = MODEL_PATH) -> dict:
    X, Y, groups, feat_cols, d = D.load_dataset()
    folds = list(GroupKFold(5).split(X, Y[:, 0], groups))

    # OOF residuals (for the confidence model + gate calibration) — honest, no leakage
    oof = np.zeros_like(Y)
    for tr, va in folds:
        for j in range(3):
            m = _mlp(); m.fit(X[tr], Y[tr, j]); oof[va, j] = m.predict(X[va])
    resid = np.abs(Y - oof)

    # Final predictor (all data) + final confidence error-models (features + prediction)
    predictors, err_models = [], []
    for j in range(3):
        p = _mlp(); p.fit(X, Y[:, j]); predictors.append(p)
        e = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              n_jobs=-1, verbosity=-1)
        e.fit(np.c_[X, oof[:, j]], resid[:, j]); err_models.append(e)

    # Gate thresholds from OOF predicted errors (calibrated per output)
    err_oof = np.zeros_like(resid)
    for tr, va in folds:
        for j in range(3):
            e = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  n_jobs=-1, verbosity=-1)
            e.fit(np.c_[X[tr], oof[tr, j]], resid[tr, j])
            err_oof[va, j] = e.predict(np.c_[X[va], oof[va, j]])
    trust_thr = np.quantile(err_oof, 0.50, axis=0).tolist()   # median → 50% deployed coverage
    ood_thr = np.quantile(err_oof, 0.95, axis=0).tolist()

    bundle = {
        "predictors": predictors, "err_models": err_models, "feat_cols": feat_cols,
        "n_cst": D.N_CST, "trust_thr": trust_thr, "ood_thr": ood_thr,
        "cst_mean": X[:, :len(feat_cols) - 3].mean(0).tolist(),  # for reference/novelty
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    return {"path": path, "trust_thr": trust_thr, "ood_thr": ood_thr, "rows": int(len(X))}


class TrainedSurrogate(Surrogate):
    name = "optimaero-mlp+lgbm-confidence"

    def __init__(self, path: str = MODEL_PATH):
        self.b = joblib.load(path)
        # Geometry-novelty reference: the learned error-model is unreliable FAR off the
        # training manifold, so we add a hard geometry-based OOD check. A shape whose CST
        # features fall outside the training range (0.5–99.5 pct, with margin) is flagged ood.
        import pandas as pd
        from scipy.spatial import cKDTree
        feats = pd.read_parquet(D.FEATS_CACHE)
        self._geom_cols = self.b["feat_cols"][:-3]  # all but alpha_deg, logRe, Mach
        Gm = feats[self._geom_cols].to_numpy(float)
        # JOINT novelty: standardized nearest-neighbour distance to the training-airfoil
        # manifold. Per-dimension ranges miss shapes that are in-range on each axis but
        # jointly off-manifold — exactly what the optimizer exploits. Threshold = 1.5× the
        # 99th-pct of real airfoils' leave-one-out NN distance (real airfoils stay in-dist).
        self._g_mean = np.nanmean(Gm, axis=0)
        self._g_std = np.nanstd(Gm, axis=0) + 1e-9
        Gs = np.nan_to_num((Gm - self._g_mean) / self._g_std)
        self._tree = cKDTree(Gs)
        d, _ = self._tree.query(Gs, k=2)
        self._novel_thr = float(np.quantile(d[:, 1], 0.99) * 1.5)

    def _novel(self, geom_vec) -> bool:
        gs = np.nan_to_num((np.asarray(geom_vec, float) - self._g_mean) / self._g_std)
        d, _ = self._tree.query(gs, k=1)
        return bool(d > self._novel_thr)

    def _features(self, coords, alpha_deg, Re, mach):
        fit = G.cst_fit(np.asarray(coords, float), n_weights_per_side=self.b["n_cst"])
        f = {"cst_fit_ok": float(fit["fit_ok"]), "cst_te": float(fit["TE_thickness"]),
             "alpha_deg": float(alpha_deg), "logRe": float(np.log10(Re)), "Mach": float(mach)}
        for i, w in enumerate(fit["upper_weights"]):
            f[f"cst_u{i}"] = float(w)
        for i, w in enumerate(fit["lower_weights"]):
            f[f"cst_l{i}"] = float(w)
        return np.array([f[c] for c in self.b["feat_cols"]], dtype=float).reshape(1, -1)

    def predict(self, coords, alpha_deg, Re, mach: float = 0.0) -> AeroPrediction:
        x = self._features(coords, alpha_deg, Re, mach)
        vals = [float(self.b["predictors"][j].predict(x)[0]) for j in range(3)]
        errs = [float(self.b["err_models"][j].predict(np.c_[x, vals[j]])[0]) for j in range(3)]
        errs = [max(e, 1e-9) for e in errs]
        trusted = all(errs[j] <= self.b["trust_thr"][j] for j in range(3))
        ood = (any(errs[j] > self.b["ood_thr"][j] for j in range(3))
               or self._novel(x[0, :len(self._geom_cols)]))
        return AeroPrediction(Cl=vals[0], Cd=vals[1], Cm=vals[2],
                              Cl_err=errs[0], Cd_err=errs[1], Cm_err=errs[2],
                              trusted=trusted, ood=ood)

    def predict_batch(self, coords, alphas, Re, mach: float = 0.0):
        # Featurize the geometry ONCE (constant across an alpha sweep); vary only alpha.
        alphas = np.atleast_1d(alphas).astype(float)
        fit = G.cst_fit(np.asarray(coords, float), n_weights_per_side=self.b["n_cst"])
        base = {"cst_fit_ok": float(fit["fit_ok"]), "cst_te": float(fit["TE_thickness"]),
                "logRe": float(np.log10(Re)), "Mach": float(mach)}
        for i, w in enumerate(fit["upper_weights"]):
            base[f"cst_u{i}"] = float(w)
        for i, w in enumerate(fit["lower_weights"]):
            base[f"cst_l{i}"] = float(w)
        Xb = np.array([[({**base, "alpha_deg": float(a)})[c] for c in self.b["feat_cols"]]
                       for a in alphas], dtype=float)
        vals = [self.b["predictors"][j].predict(Xb) for j in range(3)]
        errs = [np.clip(self.b["err_models"][j].predict(np.c_[Xb, vals[j]]), 1e-9, None)
                for j in range(3)]
        novel = self._novel(Xb[0, :len(self._geom_cols)])  # geometry constant across alphas
        out = []
        for k in range(len(alphas)):
            e = [float(errs[j][k]) for j in range(3)]
            out.append(AeroPrediction(
                Cl=float(vals[0][k]), Cd=float(vals[1][k]), Cm=float(vals[2][k]),
                Cl_err=e[0], Cd_err=e[1], Cm_err=e[2],
                trusted=all(e[j] <= self.b["trust_thr"][j] for j in range(3)),
                ood=(any(e[j] > self.b["ood_thr"][j] for j in range(3)) or novel)))
        return out


if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        print(train_and_save())
    else:
        from optimaero.datasets import uiuc
        s = TrainedSurrogate()
        for a in (-4, 0, 4, 8):
            p = s.predict(uiuc.load_coordinates("naca4412"), a, 1e6)
            print(f"a={a:+d} Cl={p.Cl:+.3f}(±{p.Cl_err:.3f}) Cd={p.Cd:.4f}(±{p.Cd_err:.4f}) "
                  f"trusted={p.trusted} ood={p.ood}")
