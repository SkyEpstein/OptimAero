"""Wire the universal drag surrogate into the optimizer.

`aero_estimate` — accurate drag/Cd from the universal surrogate (falls back to the old physics estimate if
the surrogate is unavailable), with lift/L·D still from `body_aero` (the surrogate predicts drag only).

`optimize_universal` — a surrogate-driven optimizer for ANY shape: sample many additive/deformation
candidates, score them all with the surrogate in seconds (no CFD), then CFD-verify only a diverse top-K and
return the lowest-drag one (never worse than the input). This is the "works on everything" optimize path.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import trimesh

from optimaero.shapeopt.optimize import deform, body_aero, _flow_rotation
from optimaero.cfd.foam import cfd_label
from optimaero.universal.surrogate import available, load

# deform param space [elong, nose_taper, tail_taper, smooth, grow]; scaled by aggressiveness at call time
_LO = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
_HI = np.array([1.6, 0.9, 1.3, 0.9, 0.35])


def aero_estimate(mesh: trimesh.Trimesh, V: float, alpha_deg: float = 0.0, flow_axis: str = "x") -> dict:
    """Drag/Cd from the universal surrogate (≈4% on a real drone) + lift/L·D from body_aero. Falls back
    entirely to body_aero when the surrogate artifact is absent."""
    base = body_aero(mesh, V, alpha_deg=alpha_deg, flow_axis=flow_axis)
    if not available():
        return base
    try:
        drag, cd, _ = load().predict_drag(mesh, V, flow_axis=flow_axis)
        lift = base.get("lift", 0.0)
        return {"drag": float(drag), "lift": float(lift), "Cd": float(cd),
                "Cl": base.get("Cl", 0.0), "LD": (lift / drag if drag else 0.0)}
    except Exception:
        return base


@dataclass
class UniversalResult:
    optimized: trimesh.Trimesh
    mb: dict
    ma: dict
    params: dict
    improved: bool          # a deformed candidate beat the input (else the input is returned unchanged)
    baseline_ok: bool       # the input's own CFD succeeded → the % reduction is a real CFD comparison
    n_cfd: int
    n_scored: int


def _watertight_deform(mesh, p):
    try:
        m = deform(mesh, p)
        return m if (m is not None and m.is_watertight and m.volume > 0) else None
    except Exception:
        return None


def optimize_universal(mesh: trimesh.Trimesh, V: float, flow_axis: str = "x", alpha_deg: float = 0.0,
                       aggressiveness: float = 0.6, n_search: int = 1200, top_k: int = 5,
                       workers: int = 5, seed: int = 0, progress=None) -> UniversalResult:
    """Surrogate-driven optimization of any shape: score n_search streamlining deformations with the
    universal surrogate, CFD-verify a diverse top-K, return the lowest-drag one (never worse than the input).
    Works entirely in the +x flow frame (deform() streamlines along local X), then maps the result back."""
    sur = load()
    R = _flow_rotation(flow_axis); Rinv = np.linalg.inv(R)         # flow_axis → +x, and back
    work = mesh if flow_axis == "x" else mesh.copy().apply_transform(R)
    hi = _LO + (_HI - _LO) * float(np.clip(aggressiveness, 0.05, 1.0))
    rng = np.random.default_rng(seed)
    cand = _LO + (hi - _LO) * rng.random((n_search, 5))

    scored = []
    for i, p in enumerate(cand):
        if progress and i % 200 == 0:
            progress("score", i, n_search)
        m = _watertight_deform(work, p)
        if m is None:
            continue
        try:
            drag_pred, _, _ = sur.predict_drag(m, V, flow_axis="x")     # work is already in the +x frame
            scored.append((float(drag_pred), p))
        except Exception:
            continue
    scored.sort(key=lambda s: s[0])

    picks = []
    for _, p in scored:
        if all(np.linalg.norm(p - pk) > 0.15 for pk in picks):
            picks.append(p)
        if len(picks) >= top_k:
            break

    def cfd(args):
        idx, p = args
        m = work if p is None else _watertight_deform(work, p)         # +x frame; cfd_label expects flow +x
        if m is None:
            return (idx, p, 1e9, None)
        try:
            r = cfd_label(m.copy(), V, alpha_deg=alpha_deg, case_dir=f"/tmp/oa_univ_{idx}", refine=4, layers=2)
            d = r["drag"] if (r.get("drag") and r["drag"] > 0) else 1e9
        except Exception:
            d = 1e9
        if progress:
            progress("verify", idx, len(picks) + 1)
        return (idx, p, d, m)

    tasks = [(0, None)] + [(i + 1, p) for i, p in enumerate(picks)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        evals = list(ex.map(cfd, tasks))
    base = next(e for e in evals if e[0] == 0)
    d0 = base[2]; baseline_ok = d0 < 1e9
    improved_evals = [e for e in evals if e[0] != 0 and e[3] is not None and e[2] < 1e9]
    # accept a deformed pick ONLY if the baseline CFD succeeded AND the pick genuinely beats it
    best, improved = base, False
    if baseline_ok and improved_evals:
        b = min(improved_evals, key=lambda e: e[2])
        if b[2] < d0:
            best, improved = b, True

    opt_x = best[3] if best[3] is not None else work
    opt = opt_x if flow_axis == "x" else opt_x.copy().apply_transform(Rinv)   # back to the user's frame
    mb = aero_estimate(mesh, V, alpha_deg, flow_axis)
    ma = aero_estimate(opt, V, alpha_deg, flow_axis)
    if baseline_ok:
        mb["drag"] = float(d0)                                        # measured before
    if best[2] < 1e9:
        ma["drag"] = float(best[2])                                   # measured after
    return UniversalResult(optimized=opt, mb=mb, ma=ma, improved=improved, baseline_ok=baseline_ok,
                           params=({} if best[1] is None else
                                   {"elong": float(best[1][0]), "nose": float(best[1][1]),
                                    "tail": float(best[1][2]), "smooth": float(best[1][3]),
                                    "grow": float(best[1][4])}),
                           n_cfd=len(tasks), n_scored=len(scored))
