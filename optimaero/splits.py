"""Train/test splits for the two evaluation regimes + leakage checks (data-model §4, §6).

- `new_geometry_split`  — grouped by `family_id`; no airfoil family straddles (HEADLINE).
- `new_condition_split` — a shape may recur, but only at held-out rows/conditions (SECONDARY).
- `check_*` — the automated leakage assertions (L1, L2) the build gate relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLS = ("airfoil_id", "alpha_deg", "Re", "Mach")


def attach_family_id(df: pd.DataFrame, family_catalog: pd.DataFrame,
                     on: str = "airfoil_id") -> pd.DataFrame:
    """Merge `family_id` from a family-assigned catalog onto a data table."""
    fam = family_catalog[[on, "family_id"]].drop_duplicates(on)
    return df.merge(fam, on=on, how="left")


def new_geometry_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Grouped split by `family_id` — no family appears in both train and test."""
    if "family_id" not in df.columns:
        raise ValueError("df needs family_id — call attach_family_id first")
    fams = df["family_id"].to_numpy()
    uniq = np.unique(fams[fams >= 0])
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_fams = set(uniq[:n_test].tolist())
    test = np.array([f in test_fams for f in fams]) & (fams >= 0)
    train = ~test & (fams >= 0)
    return train, test


def new_condition_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0,
                        group_col: str = "airfoil_id"):
    """Per-shape row split: each multi-row shape appears in BOTH sides, at disjoint rows —
    tests interpolation of the α/Re/M envelope for a known geometry."""
    rng = np.random.default_rng(seed)
    df2 = df.reset_index(drop=True)
    test = np.zeros(len(df2), dtype=bool)
    for _, g in df2.groupby(group_col):
        gi = g.index.to_numpy()
        if len(gi) < 2:
            continue  # cannot be in both sides; leave in train
        k = max(1, int(round(len(gi) * test_frac)))
        test[rng.choice(gi, size=k, replace=False)] = True
    return ~test, test


def check_new_geometry(df: pd.DataFrame, train, test):
    """L1: no `family_id` straddles train/test. Returns (ok, offending_families)."""
    fams = df["family_id"].to_numpy()
    inter = (set(fams[train].tolist()) & set(fams[test].tolist())) - {-1}
    return len(inter) == 0, inter


def check_no_shared_rows(df: pd.DataFrame, train, test, keys=KEY_COLS):
    """L2: no exact (shape, condition) row appears in both sides."""
    cols = [df[c].to_numpy() for c in keys]
    tr = {tuple(v[i] for v in cols) for i in np.flatnonzero(train)}
    te = {tuple(v[i] for v in cols) for i in np.flatnonzero(test)}
    shared = tr & te
    return len(shared) == 0, len(shared)


if __name__ == "__main__":  # self-check on the pilot data
    import os
    from optimaero.datasets import uiuc
    from optimaero import families as F

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_parquet(os.path.join(repo, "data", "processed", "pilot_xfoil.parquet"))
    cat = pd.DataFrame({"airfoil_id": df.airfoil_id.unique()})
    cat = F.assign_families(cat, uiuc.load_coordinates)
    df = attach_family_id(df, cat)

    trg, teg = new_geometry_split(df, test_frac=0.34)
    ok_g, bad = check_new_geometry(df, trg, teg)
    trc, tec = new_condition_split(df, test_frac=0.25)
    ok_c, shared = check_no_shared_rows(df, trc, tec)

    print(f"rows={len(df)}  families={df.family_id.nunique()}")
    print(f"new-geometry: train={trg.sum()} test={teg.sum()}  "
          f"L1 no-family-straddle={ok_g} (offending={bad})")
    print(f"new-condition: train={trc.sum()} test={tec.sum()}  "
          f"L2 no-shared-row={ok_c} (shared={shared})  "
          f"shapes-in-both={(pd.Series(df.airfoil_id[trc]).nunique() == pd.Series(df.airfoil_id[tec]).nunique())}")
