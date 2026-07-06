"""Canonical airfoil geometry: normalization + CST/Kulfan fit.

The CST fit is the canonical *parametric storage* form (spec R2). We record the round-trip
residual for every fit — a poor fit is a data-quality flag, never silently accepted
(`data-model.md §2`). High-camber airfoils need more weights than low-camber ones, so the
weight count is configurable and the residual is always reported.
"""
from __future__ import annotations

import numpy as np
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_kulfan_parameters,
    get_kulfan_coordinates,
)

# A fit whose max round-trip error exceeds this (fraction of chord) is flagged "poor".
POOR_FIT_MAX_ERR = 0.01  # 1% of chord


def normalize(coords_or_name) -> asb.Airfoil:
    """Return a chord-normalized `asb.Airfoil` (chord 1, LE at origin)."""
    af = (coords_or_name if isinstance(coords_or_name, asb.Airfoil)
          else asb.Airfoil(name=coords_or_name))
    return af.normalize()


def _min_dist_to_polyline(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Per-point minimum Euclidean distance from `points` to the polyline `poly` (its
    segments). Vectorized; catches localized defects that station-resampling would miss."""
    p = np.asarray(points, dtype=float)
    a = np.asarray(poly, dtype=float)[:-1]
    b = np.asarray(poly, dtype=float)[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom == 0.0, 1e-30, denom)
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nsj,sj->ns", ap, ab) / denom, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(p[:, None, :] - proj, axis=2).min(axis=1)


def _residual(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[float, float]:
    """Symmetric max + one-way RMS between two airfoil coordinate loops, on RAW coordinates
    (not a resampled signature) so a localized sub-station defect cannot report zero error."""
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    if a.shape[0] < 3 or b.shape[0] < 3:
        return float("nan"), float("nan")
    d_ab = _min_dist_to_polyline(a, b)  # original vs reconstruction
    d_ba = _min_dist_to_polyline(b, a)  # reconstruction vs original (catches oscillations)
    max_err = float(max(d_ab.max(), d_ba.max()))
    rms_err = float(np.sqrt(np.mean(d_ab ** 2)))
    return max_err, rms_err


# Empirically, 12 weights/side is the sweet spot on the UIUC set: 191/200 fit under 1% chord.
# Higher orders (16, 20) destabilize the least-squares fit and blow up a chunk of airfoils
# (p95 max-err jumps to ~0.68), so we do NOT raise this blindly. Poorly-fit airfoils are
# flagged (fit_ok=False) and kept — raw coordinates remain the source of truth.
def cst_fit(coords: np.ndarray, n_weights_per_side: int = 12) -> dict:
    """Fit CST/Kulfan weights to `coords`; return weights + round-trip residual + a flag."""
    coords = np.asarray(coords, dtype=float)
    params = get_kulfan_parameters(coords, n_weights_per_side=n_weights_per_side)
    recon = get_kulfan_coordinates(
        lower_weights=params["lower_weights"],
        upper_weights=params["upper_weights"],
        leading_edge_weight=params["leading_edge_weight"],
        TE_thickness=params["TE_thickness"],
    )
    resid_max, resid_rms = _residual(coords, recon)
    return {
        "n_weights_per_side": n_weights_per_side,
        "upper_weights": np.asarray(params["upper_weights"], dtype=float),
        "lower_weights": np.asarray(params["lower_weights"], dtype=float),
        "leading_edge_weight": float(params["leading_edge_weight"]),
        "TE_thickness": float(params["TE_thickness"]),
        "resid_max": resid_max,
        "resid_rms": resid_rms,
        "fit_ok": bool(np.isfinite(resid_max) and resid_max <= POOR_FIT_MAX_ERR),
    }


if __name__ == "__main__":  # compact residual sweep to pick n_weights
    from optimaero.datasets import uiuc

    names = uiuc.list_airfoils()
    rng = np.random.default_rng(0)
    sample = list(rng.choice(names, size=200, replace=False))
    for nw in (8, 12, 16, 20):
        maxes, oks = [], 0
        for nm in sample:
            c = uiuc.load_coordinates(nm)
            if c is None:
                continue
            f = cst_fit(c, n_weights_per_side=nw)
            maxes.append(f["resid_max"])
            oks += int(f["fit_ok"])
        maxes = np.array(maxes)
        print(f"n_weights={nw:2d}: median max-err={np.median(maxes):.4f} "
              f"p95={np.quantile(maxes, 0.95):.4f}  fit_ok={oks}/{len(maxes)} (<1% chord)")
