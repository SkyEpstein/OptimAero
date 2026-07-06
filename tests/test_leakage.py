"""Leakage regression tests (data-model §6).

The build must fail if any airfoil family straddles the new-geometry split (L1) or any exact
(shape, condition) row is shared across the new-condition split (L2). One test deliberately
constructs a leaky split to prove the checks are non-vacuous (they actually catch leakage).
"""
import os

import numpy as np
import pandas as pd
import pytest

from optimaero.datasets import uiuc
from optimaero import families as F
from optimaero import splits as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = os.path.join(REPO, "data", "processed", "pilot_xfoil.parquet")
BACKBONE = os.path.join(REPO, "data", "processed", "xfoil_backbone.parquet")


@pytest.fixture(scope="module")
def data_with_families():
    """Prefer the full backbone once it exists; fall back to the pilot artifact."""
    path = BACKBONE if os.path.exists(BACKBONE) else PILOT
    if not os.path.exists(path):
        pytest.skip("no dataset yet (run the sweep or pilot first)")
    df = pd.read_parquet(path)
    cat = pd.DataFrame({"airfoil_id": df.airfoil_id.unique()})
    cat = F.assign_families(cat, uiuc.load_coordinates)
    return S.attach_family_id(df, cat)


def test_new_geometry_no_family_straddle(data_with_families):
    df = data_with_families
    train, test = S.new_geometry_split(df, test_frac=0.2, seed=1)
    ok, offending = S.check_new_geometry(df, train, test)
    assert ok, f"L1 FAIL — families straddle train/test: {offending}"
    assert train.sum() > 0 and test.sum() > 0


def test_new_condition_no_shared_rows(data_with_families):
    df = data_with_families
    train, test = S.new_condition_split(df, test_frac=0.2, seed=1)
    ok, shared = S.check_no_shared_rows(df, train, test)
    assert ok, f"L2 FAIL — {shared} exact (shape,condition) rows shared"


def test_leakage_check_is_not_vacuous(data_with_families):
    """A deliberately leaky split MUST be flagged — else the guard is meaningless."""
    df = data_with_families
    leaky_family = df.family_id.iloc[0]
    fam0 = df.family_id.to_numpy() == leaky_family
    train = np.ones(len(df), dtype=bool)
    test = fam0.copy()  # family now present on BOTH sides
    ok, offending = S.check_new_geometry(df, train, test)
    assert not ok and leaky_family in offending, "leakage check failed to catch a straddle!"
