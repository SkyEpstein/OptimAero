"""Envelope-constrained inverse design.

Search CST shape space for an airfoil that best meets a `DesignRequirement` while fitting
inside a packaging `Envelope`, using any `Surrogate` for evaluation. Black-box optimizer
(scipy differential evolution) so it works with any bake-off winner (differentiable or not).

Per architecture-spec R3 the optimizer (a) enforces the envelope as a hard constraint,
(b) optimizes the requirement objective, and (c) prefers shapes the surrogate is *confident*
about — low-confidence/OOD optima are penalized and flagged, not silently returned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

from optimaero.surrogate import Surrogate
from optimaero.requirements import Envelope, DesignRequirement
from optimaero.families import geometric_signature
from optimaero import geometry as G
from optimaero.datasets import uiuc

# Physical drag floor (≈ the XFOIL backbone's minimum Cd). A prediction below this is the
# surrogate extrapolating to fake near-zero drag off-manifold — reject it.
PHYS_CD_FLOOR = 0.003


@dataclass
class DesignResult:
    coords: np.ndarray
    upper_weights: np.ndarray
    lower_weights: np.ndarray
    best_alpha: float
    Cl: float
    Cd: float
    LD: float
    thickness: float
    trusted: bool
    ood: bool
    feasible: bool          # envelope satisfied
    objective_value: float  # the metric maximized (e.g. best L/D)
    # Ground-truth verification (filled by optimize_verified): the REAL performance from XFOIL.
    xfoil_Cl: float | None = None
    xfoil_Cd: float | None = None
    xfoil_LD: float | None = None


def _coords_from_weights(uw, lw, n_points=80):
    return get_kulfan_coordinates(lower_weights=np.asarray(lw), upper_weights=np.asarray(uw),
                                  leading_edge_weight=0.0, TE_thickness=0.0,
                                  n_points_per_side=n_points)


def _max_thickness(coords) -> float:
    sig = geometric_signature(coords)
    if sig is None:
        return np.nan
    n = len(sig) // 2
    return float(np.max(sig[:n] - sig[n:]))


def _score_shape(surr: Surrogate, coords, req: DesignRequirement):
    """Return (metric, alpha*, Cl*, Cd*, pred*) for the shape under the requirement."""
    alphas = np.arange(req.alpha_range[0], req.alpha_range[1] + 1e-9, 1.0)
    preds = surr.predict_batch(coords, alphas, req.Re, req.mach)
    cl = np.array([p.Cl for p in preds])
    cd = np.array([p.Cd for p in preds])
    if req.objective == "max_LD":
        ld = np.where(cd > 1e-6, cl / cd, -1e9)
        i = int(np.argmax(ld))
        return float(ld[i]), float(alphas[i]), float(cl[i]), float(cd[i]), preds[i]
    if req.objective == "max_Cl":
        i = int(np.argmax(cl))
        return float(cl[i]), float(alphas[i]), float(cl[i]), float(cd[i]), preds[i]
    if req.objective == "min_Cd_at_Cl":
        j = int(np.argmin(np.abs(cl - req.target_Cl)))
        miss = abs(cl[j] - req.target_Cl)
        metric = -float(cd[j]) - 10.0 * miss  # penalize missing the target lift
        return metric, float(alphas[j]), float(cl[j]), float(cd[j]), preds[j]
    raise ValueError(f"unknown objective {req.objective}")


def optimize(surr: Surrogate, req: DesignRequirement, env: Envelope,
             n_weights: int = 8, maxiter: int = 25, popsize: int = 12,
             seed: int = 0, seed_airfoil: str = "naca2412") -> DesignResult:
    # design space: upper + lower CST weights
    bounds = [(0.0, 0.45)] * n_weights + [(-0.45, 0.20)] * n_weights
    start = G.cst_fit(uiuc.load_coordinates(seed_airfoil), n_weights_per_side=n_weights)
    x0 = np.concatenate([start["upper_weights"], start["lower_weights"]])
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    def objective(x):
        uw, lw = x[:n_weights], x[n_weights:]
        try:
            coords = _coords_from_weights(uw, lw)
        except Exception:
            return 1e6
        t = _max_thickness(coords)
        # HARD envelope constraint — an infeasible shape is rejected, never softly penalized
        # (a soft penalty is trivially dwarfed by an exploited metric like L/D=4e5).
        if not np.isfinite(t) or t > env.max_thickness or t < env.min_thickness:
            return 1e6
        metric, a_star, cl, cd, pred = _score_shape(surr, coords, req)
        # HARD gates on surrogate-extrapolation artifacts: an off-manifold (novel/OOD) shape
        # or a physically-impossible drag (Cd below the training floor → the model is
        # extrapolating to fake near-zero drag) is rejected outright. (constitution: trust>speed)
        if pred.ood or cd < PHYS_CD_FLOOR:
            return 1e3
        if not pred.trusted:
            metric -= 0.3 * abs(metric)   # discount low-confidence gains; can't win on size alone
        return -metric

    res = differential_evolution(objective, bounds, x0=x0, maxiter=maxiter,
                                 popsize=popsize, seed=seed, tol=1e-4, polish=True,
                                 init="sobol")
    uw, lw = res.x[:n_weights], res.x[n_weights:]
    coords = _coords_from_weights(uw, lw)
    t = _max_thickness(coords)
    metric, a_star, cl, cd, pred = _score_shape(surr, coords, req)
    return DesignResult(
        coords=coords, upper_weights=uw, lower_weights=lw, best_alpha=a_star,
        Cl=cl, Cd=cd, LD=(cl / cd if cd > 1e-6 else float("nan")), thickness=t,
        trusted=pred.trusted, ood=pred.ood,
        feasible=(env.min_thickness <= t <= env.max_thickness),
        objective_value=metric,
    )


def optimize_verified(surr: Surrogate, req: DesignRequirement, env: Envelope,
                      xfoil_path: str, n_seeds: int = 3, **kw):
    """Surrogate-driven search from several seeds, then VERIFY each optimum with the real
    XFOIL (the constitution's CFD-fallback). Returns (best_verified, all_candidates) — the
    surrogate's claimed number is never returned unverified."""
    import aerosandbox as asb

    best, cands = None, []
    for s in range(n_seeds):
        r = optimize(surr, req, env, seed=s, **kw)
        try:
            res = asb.XFoil(airfoil=asb.Airfoil(coordinates=r.coords), Re=req.Re,
                            mach=req.mach, xfoil_command=xfoil_path, timeout=60).alpha([r.best_alpha])
            if np.atleast_1d(res.get("alpha", [])).size:
                r.xfoil_Cl = float(np.atleast_1d(res["CL"])[0])
                r.xfoil_Cd = float(np.atleast_1d(res["CD"])[0])
                r.xfoil_LD = (r.xfoil_Cl / r.xfoil_Cd) if r.xfoil_Cd > 1e-6 else None
        except Exception:
            pass
        cands.append(r)
        if r.xfoil_LD is not None and (best is None or (best.xfoil_LD or -1) < r.xfoil_LD):
            best = r
    return best, cands


if __name__ == "__main__":  # end-to-end demo with the placeholder surrogate
    from optimaero.surrogate import NeuralFoilSurrogate

    surr = NeuralFoilSurrogate()
    env = Envelope(max_thickness=0.12, min_thickness=0.08)
    req = DesignRequirement(Re=1_000_000, objective="max_LD")

    # baseline: best L/D of the seed airfoil within the same envelope/condition
    base_coords = uiuc.load_coordinates("naca2412")
    base_metric, base_a, base_cl, base_cd, base_pred = _score_shape(surr, base_coords, req)
    print(f"baseline naca2412: best L/D={base_metric:.1f} at a={base_a:.0f}  "
          f"(t/c={_max_thickness(base_coords):.3f})  trusted={base_pred.trusted} "
          f"ood={base_pred.ood}")

    r = optimize(surr, req, env, maxiter=20, seed=0)
    print(f"optimized:         best L/D={r.LD:.1f} at a={r.best_alpha:.0f}  "
          f"(t/c={r.thickness:.3f})  Cl={r.Cl:.3f} Cd={r.Cd:.4f}")
    print(f"envelope [{env.min_thickness}-{env.max_thickness}] satisfied: {r.feasible} | "
          f"trusted={r.trusted} ood={r.ood}")
