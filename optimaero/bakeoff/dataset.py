"""Featurize the XFOIL backbone for the Phase-2 bake-off.

Features = geometry (CST weights) + flow conditions; targets = Cl, Cd, Cm; groups = family_id
for leakage-controlled GroupKFold. Per-airfoil geometry + family are cached (they're the slow
part) so the bake-off can re-run cheaply.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from optimaero.datasets import uiuc
from optimaero import families as F, geometry as G

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKBONE = os.path.join(_REPO, "data", "processed", "xfoil_backbone.parquet")
FEATS_CACHE = os.path.join(_REPO, "data", "processed", "airfoil_features.parquet")
N_CST = 8  # weights per side used as geometry features (compact, captures shape)


def build_airfoil_features(names) -> pd.DataFrame:
    """Per-airfoil geometry features (CST weights) + family_id."""
    rows = []
    for nm in names:
        c = uiuc.load_coordinates(nm)
        if c is None:
            continue
        fit = G.cst_fit(c, n_weights_per_side=N_CST)
        rec = {"airfoil_id": nm, "cst_fit_ok": fit["fit_ok"]}
        for i, w in enumerate(fit["upper_weights"]):
            rec[f"cst_u{i}"] = float(w)
        for i, w in enumerate(fit["lower_weights"]):
            rec[f"cst_l{i}"] = float(w)
        rec["cst_te"] = fit["TE_thickness"]
        rows.append(rec)
    feats = pd.DataFrame(rows)
    cat = F.assign_families(pd.DataFrame({"airfoil_id": feats.airfoil_id}), uiuc.load_coordinates)
    return feats.merge(cat[["airfoil_id", "family_id"]], on="airfoil_id", how="left")


def load_dataset(regimes=("ok", "low_re", "post_stall"), rebuild=False):
    """Return X, Y (Cl,Cd,Cm), groups (family_id), feature names, and the merged frame."""
    df = pd.read_parquet(BACKBONE)
    df = df[df.regime_flag.isin(regimes)].copy()
    if rebuild or not os.path.exists(FEATS_CACHE):
        feats = build_airfoil_features(sorted(df.airfoil_id.unique()))
        feats.to_parquet(FEATS_CACHE, index=False)
    else:
        feats = pd.read_parquet(FEATS_CACHE)

    d = df.merge(feats, on="airfoil_id", how="inner")
    d["logRe"] = np.log10(d["Re"])
    geom_cols = [c for c in feats.columns if c.startswith("cst_")]
    cond_cols = ["alpha_deg", "logRe", "Mach"]
    feat_cols = geom_cols + cond_cols
    d = d.dropna(subset=feat_cols + ["Cl", "Cd", "Cm", "family_id"])
    d = d[d.family_id >= 0]
    X = d[feat_cols].to_numpy(dtype=float)
    Y = d[["Cl", "Cd", "Cm"]].to_numpy(dtype=float)
    groups = d["family_id"].to_numpy()
    return X, Y, groups, feat_cols, d


if __name__ == "__main__":
    X, Y, g, cols, d = load_dataset()
    print(f"rows={len(X):,}  features={len(cols)}  families={len(np.unique(g))}")
    print("feature cols:", cols)
    print("target ranges: Cl", Y[:, 0].min().round(2), Y[:, 0].max().round(2),
          "| Cd", Y[:, 1].min().round(4), Y[:, 1].max().round(3),
          "| Cm", Y[:, 2].min().round(2), Y[:, 2].max().round(2))
