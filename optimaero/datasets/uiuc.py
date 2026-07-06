"""UIUC airfoil catalog, sourced from the AeroSandbox-bundled airfoil database.

AeroSandbox ships the UIUC coordinate database (~2,170 airfoils), so we read it locally
and reproducibly (version-pinned via requirements.txt) instead of scraping the web.

This module only *enumerates and loads* airfoils and classifies their design series
(descriptive statistics only). The leakage-critical `family_id` assignment — the hybrid
lineage + shape-space near-duplicate merge — lives in `optimaero.families`, per
`specs/2026-07-05-data-foundation/data-model.md §3`.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import aerosandbox as asb
import aerosandbox.geometry.airfoil as _asb_af


def database_dir() -> str:
    """Absolute path to the AeroSandbox-bundled UIUC .dat database."""
    return os.path.join(os.path.dirname(_asb_af.__file__), "airfoil_database")


def list_airfoils() -> list[str]:
    """Sorted list of airfoil names (one per bundled .dat file)."""
    paths = glob.glob(os.path.join(database_dir(), "*.dat"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


# --- Design-series classification (DESCRIPTIVE ONLY; not the leakage family) ---------
# NOTE: a coarse series tag (e.g. "all NACA 4-digit") is deliberately NOT used as the
# leakage family — naca0006 and naca4412 are genuinely different airfoils and may sit in
# different splits. The family unit (exact/near duplicates) is decided in families.py.
_SERIES_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("naca4", re.compile(r"^naca\d{4}$")),
    ("naca5", re.compile(r"^naca\d{5}$")),
    ("naca6", re.compile(r"^naca6\d")),
    ("selig", re.compile(r"^s\d{3,4}$")),
    ("eppler", re.compile(r"^e\d{3}")),
    ("wortmann_fx", re.compile(r"^fx")),
    ("goettingen", re.compile(r"^goe")),
    ("clark", re.compile(r"^clark")),
    ("ag", re.compile(r"^ag\d")),
    ("drela", re.compile(r"^(ah|dae)\d")),
]


def classify_series(name: str) -> str:
    n = name.lower()
    for tag, pat in _SERIES_PATTERNS:
        if pat.match(n):
            return tag
    return "other"


def normalized_name(name: str) -> str:
    """Collapse common source/rename suffixes for a first-pass exact-duplicate key.

    Coarse on purpose — the real near-duplicate guard is the shape-space merge in
    `optimaero.families`. Examples: 'naca0012-il' -> 'naca0012', 'n0012' -> 'naca0012'.
    """
    n = name.lower().strip()
    n = re.sub(r"[-_ ]?(il|jf|sm|ns)$", "", n)  # common database source suffixes
    n = re.sub(r"[^a-z0-9]", "", n)
    n = re.sub(r"^n(?=\d{4}$)", "naca", n)  # n0012 -> naca0012
    return n


def load_coordinates(name: str) -> np.ndarray | None:
    """Return the airfoil's coordinates as an (N, 2) array, or None if it won't load."""
    try:
        af = asb.Airfoil(name=name)
        c = np.asarray(af.coordinates, dtype=float)
        if c.ndim != 2 or c.shape[0] < 10 or not np.isfinite(c).all():
            return None
        return c
    except Exception:
        return None


def build_catalog(load_coords: bool = True) -> pd.DataFrame:
    """Enumerate the bundled UIUC airfoils into a catalog DataFrame.

    Columns: airfoil_id, source, series, normalized_name, n_coords, loadable.
    """
    rows = []
    for name in list_airfoils():
        rec = {
            "airfoil_id": name,
            "source": "uiuc",
            "series": classify_series(name),
            "normalized_name": normalized_name(name),
            "n_coords": None,
            "loadable": True,
        }
        if load_coords:
            c = load_coordinates(name)
            if c is None:
                rec["loadable"] = False
            else:
                rec["n_coords"] = int(c.shape[0])
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":  # quick self-check
    cat = build_catalog(load_coords=True)
    print(f"airfoils: {len(cat)}  loadable: {int(cat.loadable.sum())}")
    print("\nby series:\n", cat.series.value_counts())
    dupes = cat.normalized_name.value_counts()
    print(f"\nexact/rename-duplicate groups (normalized_name collisions): "
          f"{int((dupes > 1).sum())} groups covering {int(dupes[dupes > 1].sum())} airfoils")
