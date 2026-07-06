"""Hybrid airfoil-family assignment for the new-geometry leakage split.

`family_id` groups each airfoil with (a) its exact/renamed duplicates and (b) its
shape-space near-duplicates (scaled/rotated/thickened twins from different sources), so
that no shape or its near-duplicates can straddle a train/test split.
See `specs/2026-07-05-data-foundation/data-model.md §3`.

Pipeline:
  1. geometric signature per airfoil — fixed-length, chord-normalized (self-contained,
     no dependence on the CST fit, so leakage grouping is decoupled from storage choices).
  2. must-link exact/rename duplicates (shared normalized name) as a floor.
  3. union-find merge of any pair within Euclidean distance `tau` in signature space.
`tau` is chosen from the nearest-neighbour distance distribution (the gap between "same
shape, tiny numeric noise" and "genuinely different airfoil") and recorded in the data card.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Near-duplicate threshold in 80-dim signature space (Euclidean, fraction-of-chord units).
# Chosen empirically: the max distance between any two same-shape/different-name geometric
# twins in the UIUC DB is 0.00195 (verified), so 0.003 captures every known twin with ~50%
# margin while keeping merged families tiny (it merges near-identical duplicates, NOT genuine
# thickness/camber variants — those are legitimately different airfoils and may split). See
# data-model.md §3. Sensitivity across tau is reported in the data card.
FAMILY_TAU = 0.003


# ---------------------------------------------------------------- geometric signature
def geometric_signature(coords: np.ndarray, n_stations: int = 40) -> np.ndarray | None:
    """Fixed-length shape descriptor: upper & lower surface y at cosine-spaced x on unit
    chord. Robust to point count and ordering. Returns a (2*n_stations,) vector, or None.
    """
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] < 10:
        return None
    x, y = c[:, 0], c[:, 1]
    x0, x1 = float(x.min()), float(x.max())
    chord = x1 - x0
    if chord <= 0:
        return None
    xs = (x - x0) / chord
    ys = y / chord
    # Orientation guard: normalize loop direction to a consistent sign (signed area > 0)
    # so a loop-reversed twin from a future source (AirfRANS/external .dat) yields the SAME
    # signature and cannot straddle a split. No-op for the UIUC DB (uniformly oriented).
    if 0.5 * np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys) < 0:
        xs, ys = xs[::-1], ys[::-1]
    le = int(np.argmin(xs))  # leading edge = min-x point
    # split the loop into the two surfaces around the LE
    xs_u, ys_u = xs[: le + 1][::-1], ys[: le + 1][::-1]  # reorder LE->TE
    xs_l, ys_l = xs[le:], ys[le:]
    if len(xs_u) < 3 or len(xs_l) < 3:
        return None
    stations = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, n_stations)))  # cosine 0..1

    def _interp(sx, sy):
        order = np.argsort(sx)
        sx, sy = sx[order], sy[order]
        # collapse duplicate x for a valid monotonic interp
        uniq, idx = np.unique(sx, return_index=True)
        return np.interp(stations, uniq, sy[idx])

    return np.concatenate([_interp(xs_u, ys_u), _interp(xs_l, ys_l)])


def build_signatures(names, loader) -> tuple[list[str], np.ndarray]:
    """Compute signatures for names via `loader(name)->coords`. Skips ones that fail."""
    ok, sigs = [], []
    for name in names:
        coords = loader(name)
        if coords is None:
            continue
        s = geometric_signature(coords)
        if s is None:
            continue
        ok.append(name)
        sigs.append(s)
    return ok, np.asarray(sigs, dtype=float)


# ----------------------------------------------------------------------- union-find
class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster(signatures: np.ndarray, tau: float,
            must_link: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Return a family_id per row: merge rows within distance `tau` (+ any must_link pairs)."""
    n = signatures.shape[0]
    uf = _UnionFind(n)
    for a, b in (must_link or []):
        uf.union(a, b)
    tree = cKDTree(signatures)
    for i, j in tree.query_pairs(r=tau):
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    _, family_id = np.unique(roots, return_inverse=True)
    return family_id


def nn_distance_stats(signatures: np.ndarray) -> dict:
    """Nearest-neighbour distance distribution — used to pick `tau` defensibly."""
    tree = cKDTree(signatures)
    d, _ = tree.query(signatures, k=2)
    nn = d[:, 1]
    qs = np.quantile(nn, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.95])
    return {"min": float(qs[0]), "p01": float(qs[1]), "p05": float(qs[2]),
            "p10": float(qs[3]), "p25": float(qs[4]), "median": float(qs[5]),
            "p95": float(qs[6])}


def assign_families(catalog: pd.DataFrame, loader, tau: float = FAMILY_TAU,
                    name_col: str = "airfoil_id",
                    dup_key_col: str | None = "normalized_name") -> pd.DataFrame:
    """Attach `family_id` to a copy of `catalog`. Rows whose geometry won't load get -1."""
    names = list(catalog[name_col])
    ok, sigs = build_signatures(names, loader)
    pos = {name: i for i, name in enumerate(ok)}
    must_link = []
    if dup_key_col is not None and dup_key_col in catalog.columns:
        by_key: dict = {}
        for _, row in catalog.iterrows():
            if row[name_col] in pos:
                by_key.setdefault(row[dup_key_col], []).append(pos[row[name_col]])
        for idxs in by_key.values():
            for k in range(1, len(idxs)):
                must_link.append((idxs[0], idxs[k]))
    fam = cluster(sigs, tau=tau, must_link=must_link)
    fam_of = {ok[i]: int(fam[i]) for i in range(len(ok))}
    out = catalog.copy()
    out["family_id"] = out[name_col].map(lambda nm: fam_of.get(nm, -1))
    return out


if __name__ == "__main__":  # compact self-check on the real UIUC set
    from optimaero.datasets import uiuc

    cat = uiuc.build_catalog(load_coords=False)
    ok, sigs = build_signatures(cat.airfoil_id, uiuc.load_coordinates)
    print(f"signatures: {len(ok)}/{len(cat)}  dim={sigs.shape[1]}")
    stats = nn_distance_stats(sigs)
    print("nn-distance:", {k: round(v, 4) for k, v in stats.items()})
    for tau in (stats["p05"], stats["p10"], stats["p25"]):
        fam = cluster(sigs, tau=tau)
        nfam = len(np.unique(fam))
        merged = len(ok) - nfam
        print(f"  tau={tau:.4f} -> {nfam} families ({merged} airfoils merged)")
    # spot check: do the naca0012 variants land in one family at a mid tau?
    fam = cluster(sigs, tau=stats["p10"])
    fam_of = {ok[i]: fam[i] for i in range(len(ok))}
    v = sorted({n for n in ok if "0012" in n and n.startswith(("naca0012", "n0012"))})
    print("naca0012-ish:", v[:6], "-> families",
          sorted({int(fam_of[n]) for n in v}) if v else "none")
