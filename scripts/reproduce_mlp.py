"""Independently reproduce the winning MLP's new-geometry generalization on the FULL 213k
backbone across seeds, to confirm the headline number for the writeup.

Reuses the exact featurization, grouped folds, and metric code from the bake-off.
"""
import json
import os

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optimaero.bakeoff import dataset as D
from optimaero.bakeoff.run import _metrics, oof_predict, OUTPUTS

if __name__ == "__main__":
    X, Y, groups, cols, d = D.load_dataset()
    folds = list(GroupKFold(5).split(X, Y[:, 0], groups))
    print(f"reproducing MLP on {len(X):,} rows, {len(np.unique(groups))} families, 5 grouped folds",
          flush=True)

    runs = {}
    for seed in (0, 1, 42):
        fac = (lambda s=seed: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=150,
                         early_stopping=True, random_state=s)))
        oof = oof_predict(fac, X, Y, folds)
        rmse, r2 = _metrics(Y, oof)
        runs[seed] = {"r2": r2.round(4).tolist(), "rmse": rmse.round(4).tolist()}
        print(f"  seed={seed:2d}: R2={dict(zip(OUTPUTS, r2.round(4)))}  "
              f"RMSE={dict(zip(OUTPUTS, rmse.round(4)))}", flush=True)

    r2s = np.array([runs[s]["r2"] for s in runs])
    summary = {"seeds": list(runs), "per_seed": runs,
               "mean_r2": dict(zip(OUTPUTS, r2s.mean(0).round(4).tolist())),
               "std_r2": dict(zip(OUTPUTS, r2s.std(0).round(4).tolist()))}
    print(f"\nREPRODUCED (mean±std over seeds):")
    for i, o in enumerate(OUTPUTS):
        print(f"  {o}: R2 = {r2s.mean(0)[i]:.4f} ± {r2s.std(0)[i]:.4f}")
    out = os.path.join(D._REPO, "results", "mlp_reproduction.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
