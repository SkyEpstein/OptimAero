"""Deployable envelope CFD surrogate — the SAVED artifact (not just a bake-off metric).

Trains the winning predictor per target (Cd, Cl) + a LightGBM confidence model on out-of-fold
residuals + a 50%-coverage trust gate, and serializes everything to results/envelope_surrogate.joblib.
The optimizer calls predict(features) → {Cd, Cl, Cd_err, Cl_err, Cd_trusted, Cl_trusted}; untrusted /
out-of-distribution points fall back to CFD. Mirrors the 2D TrainedSurrogate for the 3D envelope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from optimaero.cfd.bakeoff import FEATURES, pool, _oof, _r2, _rmse

ARTIFACT = "results/envelope_surrogate.joblib"


def _conf_factory():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)


class EnvelopeSurrogate:
    """Predictor + confidence + trust gate per target, trained on the CFD dataset."""

    def __init__(self, feats, pred, conf, gate, metrics):
        self.feats = feats           # feature-name order the models expect
        self.pred = pred             # {'Cd': model, 'Cl': model}
        self.conf = conf             # {'Cd': error-model, 'Cl': error-model}
        self.gate = gate             # {'Cd': thr, 'Cl': thr}  (predicted-error at 50% coverage)
        self.metrics = metrics       # honest OOF R²/RMSE + selective RMSE per target

    @classmethod
    def train(cls, dataset_path: str, best: dict | None = None) -> "EnvelopeSurrogate":
        best = best or {"Cd": "extratrees", "Cl": "lgbm"}
        df = pd.read_parquet(dataset_path)
        df = df[df["converged"].astype(bool)]
        df = df[np.isfinite(df["Cd"]) & np.isfinite(df["Cl"]) &
                (df["Cd"].abs() < 10) & (df["Cl"].abs() < 10)]
        if "fineness" in df.columns:
            df = df[df["fineness"] < 12]
        df = df.reset_index(drop=True)
        if "camber" in df.columns:
            df["camber"] = df["camber"].fillna(0.0)
        feats = [f for f in FEATURES if f in df.columns]
        if "camber" in df.columns and int(df["camber"].nunique()) > 1:
            feats.append("camber")
        X = df[feats].values.astype(float)
        folds = list(KFold(5, shuffle=True, random_state=0).split(X))

        pred, conf, gate, metrics = {}, {}, {}, {}
        for tgt in ("Cd", "Cl"):
            y = df[tgt].values.astype(float)
            oof = _oof(pool()[best[tgt]], X, y, folds)              # honest OOF predictions
            resid = np.abs(oof - y)
            coof = _oof(_conf_factory, X, resid, folds)            # OOF predicted error
            thr = float(np.median(coof))                           # 50%-coverage gate
            order = np.argsort(coof); k = max(1, len(y) // 2)
            metrics[tgt] = {"model": best[tgt], "n": int(len(y)),
                            "r2": _r2(oof, y), "rmse": _rmse(oof, y),
                            "rmse@50%": _rmse(oof[order[:k]], y[order[:k]])}
            gate[tgt] = thr
            m = pool()[best[tgt]](); m.fit(X, y); pred[tgt] = m    # final predictor on all data
            cm = _conf_factory(); cm.fit(X, resid); conf[tgt] = cm  # final confidence on all data
        return cls(feats, pred, conf, gate, metrics)

    def save(self, path: str = ARTIFACT):
        import os
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str = ARTIFACT) -> "EnvelopeSurrogate":
        import joblib
        return joblib.load(path)

    def predict(self, feat: dict) -> dict:
        """feat: dict with at least self.feats keys (e.g. from dataset.envelope_features)."""
        x = np.array([[float(feat[f]) for f in self.feats]], dtype=float)
        out = {}
        for tgt in ("Cd", "Cl"):
            v = float(self.pred[tgt].predict(x)[0])
            e = float(self.conf[tgt].predict(x)[0])
            out[tgt] = v
            out[tgt + "_err"] = e
            out[tgt + "_trusted"] = bool(e <= self.gate[tgt])
        return out


def train_and_save(dataset_path: str = "data/processed/envelope_cfd_v2.parquet",
                   out: str = ARTIFACT) -> "EnvelopeSurrogate":
    s = EnvelopeSurrogate.train(dataset_path)
    s.save(out)
    return s


if __name__ == "__main__":
    import sys
    s = train_and_save(sys.argv[1] if len(sys.argv) > 1 else "data/processed/envelope_cfd_v2.parquet")
    print(f"saved {ARTIFACT} | features={len(s.feats)}")
    for t, m in s.metrics.items():
        print(f"  {t}: {m['model']} OOF R²={m['r2']:.3f} RMSE={m['rmse']:.4f} "
              f"| gated RMSE@50%={m['rmse@50%']:.4f} | gate={s.gate[t]:.4f} | n={m['n']}")
