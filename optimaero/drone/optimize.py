"""Autonomous drone aero-optimizer.

Given an imported drone, the PROGRAM decides the aerodynamic treatment itself. Two search modes share the
same CFD-verify + additive-only selection:
  • optimize_drone            — blind: sample N treatment forms, CFD-evaluate ALL of them.
  • optimize_drone_surrogate  — surrogate-driven (Sky's vision): score THOUSANDS of forms in ms with the
    drone surrogate, then CFD-verify only a diverse top-K. Far more tuning per CFD call.

Both verify every returned design is additive-only (constitution §5.8: output must contain the original and
never shrink it) — a design that fails is disqualified even if its drag is lower — and fall back to the
unmodified drone if nothing valid beats it. Never worse, never a broken mesh.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite

import numpy as np
import trimesh

from optimaero.shapeopt.optimize import _flow_rotation
from optimaero.drone.airfoil import airfoil_arms, add_tail, add_nose
from optimaero.cfd.foam import cfd_label

# expanded (high-dimensional) additive-treatment space — 6 knobs vs the base 3. In this bigger space a
# blind CFD search (a dozen samples) covers almost nothing, so a surrogate that scores thousands wins.
KNOBS_HD = ["tail_len", "tail_base", "arm_chord", "arm_thick", "nose_len", "nose_base"]
LO_HD = np.array([0.0, 0.6, 0.9, 0.5, 0.0, 0.6])
HI_HD = np.array([2.2, 1.0, 2.6, 1.1, 1.5, 1.0])


def _build_hd(drone, seg, flow_axis, p):
    """Build a drone form from the 6-knob expanded treatment: airfoil arms + boat-tail (length, base) +
    nose fairing (length, base). Additive-only; body detection uses the original drone via body_source."""
    tail_len, tail_base, chord_s, thick_s, nose_len, nose_base = (float(x) for x in p)
    m = airfoil_arms(drone, seg, flow_axis=flow_axis, chord=chord_s * seg["rmax"], thick_scale=thick_s)
    if tail_len > 0.15:
        m = add_tail(m, seg, flow_axis=flow_axis, tail_len_frac=tail_len, base_scale=tail_base,
                     body_source=drone)
    if nose_len > 0.10:
        m = add_nose(m, seg, flow_axis=flow_axis, nose_len_frac=nose_len, base_scale=nose_base,
                     body_source=drone)
    return m

LO = np.array([0.0, 0.9, 0.5])          # treatment knob bounds: tail_len, chord×rmax, thick
HI = np.array([2.2, 2.6, 1.1])
BARE = np.array([0.0, 0.0, 1.0])        # the "no treatment" form (index 0 in every verify batch)


def _build(drone, seg, flow_axis, p):
    tail_len, chord_s, thick_s = float(p[0]), float(p[1]), float(p[2])
    m = airfoil_arms(drone, seg, flow_axis=flow_axis, chord=chord_s * seg["rmax"], thick_scale=thick_s)
    if tail_len > 0.15:
        # body detection must use the ORIGINAL drone (matches seg's face labels); the cone unions onto m.
        m = add_tail(m, seg, flow_axis=flow_axis, tail_len_frac=tail_len, body_source=drone)
    return m


def additive_ok(out, original, vol_tol: float = 0.999, contain_tol: float = 0.995) -> bool:
    """True iff `out` plausibly CONTAINS `original` and did not shrink it (constitution §5.8).
    Gates: (1) volume(out) ≥ volume(original); (2) the boolean intersection recovers ≥99.5% of the
    original's volume. Any boolean/watertight failure returns False (fail-closed)."""
    try:
        if out is None or not out.is_watertight or out.volume <= 0:
            return False
        if out.volume < original.volume * vol_tol:
            return False
        inter = original.intersection(out)
        return inter is not None and inter.volume >= original.volume * contain_tol
    except Exception:
        return False


@dataclass
class DroneResult:
    optimized: trimesh.Trimesh
    drag_before: float
    drag_after: float
    params: dict
    all_evals: list
    metrics_before: dict | None = None
    metrics_after: dict | None = None
    contains_original: bool = True        # additive-only verified on the returned mesh (§5.8)
    baseline_ok: bool = True              # bare-drone CFD succeeded → drag_before is a real number
    improved: bool = False                # a fairing design beat the bare drone (else drone returned as-is)
    alpha_deg: float = 0.0                # AoA the CFD actually used
    mode: str = "blind"                   # "blind" | "surrogate"
    n_cfd: int = 0                        # how many CFD evaluations this run spent
    surrogate_meta: dict | None = None    # surrogate diagnostics when mode == "surrogate"


def _metric(d: dict | None, drag: float) -> dict:
    d = d or {}
    dr = d.get("drag") if d.get("drag") else drag
    lift = d.get("lift") or 0.0
    return {"drag": float(dr), "lift": float(lift), "Cd": d.get("Cd") or 0.0,
            "Cl": d.get("Cl") or 0.0, "LD": (lift / dr if dr else 0.0)}


def _cfd_eval_params(drone, seg, flow_axis, param_list, V, alpha_deg, workers, progress):
    """CFD-evaluate a list of treatment-param vectors in parallel. Index 0 is the bare drone.
    The mesh is built OUTSIDE the CFD try so a CFD failure never discards a real mesh."""
    lock = threading.Lock(); done = {"i": 0}
    ntot = len(param_list)

    def bump():
        with lock:
            done["i"] += 1
            return done["i"]

    def evalc(i):
        p = param_list[i]
        r = None
        drag = 1e6
        try:
            m = drone if i == 0 else _build(drone, seg, flow_axis, p)
        except Exception:
            m = None
        if m is not None:
            try:
                o = m.copy(); o.apply_transform(_flow_rotation(flow_axis))
                r = cfd_label(o, V, alpha_deg=alpha_deg, case_dir=f"/tmp/oa_optdrone_{i}",
                              refine=4, layers=2)
                drag = r["drag"] if (r and r.get("drag") and r["drag"] > 1.0) else 1e6
            except Exception:
                r, drag = None, 1e6
        k = bump()
        if progress:
            progress(k, ntot, drag)
        return {"params": {"tail": float(p[0]), "chord": float(p[1]), "thick": float(p[2])},
                "mesh": m, "drag": drag, "metrics": r, "bare": i == 0}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(evalc, range(ntot)))


def _result_from_evals(evals, drone, alpha_deg, mode="blind", surrogate_meta=None) -> DroneResult:
    """Never-worse + additive-only selection shared by both search modes."""
    bare = next((e for e in evals if e["bare"]), None)
    bare_ok = bare is not None and bare["mesh"] is not None and bare["drag"] < 1e6
    baseline = bare["drag"] if bare_ok else None

    cands = sorted((e for e in evals if not e["bare"] and e["mesh"] is not None and e["drag"] < 1e6),
                   key=lambda e: e["drag"])
    chosen = None
    for e in cands:
        if baseline is not None and e["drag"] >= baseline:
            break
        if additive_ok(e["mesh"], drone):
            chosen = e
            break

    improved = chosen is not None
    if chosen is None:
        chosen = bare
    opt_mesh = chosen["mesh"] if (chosen and chosen["mesh"] is not None) else drone
    contains_original = True if not improved else additive_ok(opt_mesh, drone)

    if bare_ok:
        drag_before = float(bare["drag"]); mb = _metric(bare["metrics"], bare["drag"])
    else:
        drag_before = float("nan"); mb = None
    if improved:
        drag_after = float(chosen["drag"]); ma = _metric(chosen["metrics"], chosen["drag"])
    else:
        drag_after = drag_before; ma = mb

    n_valid_cfd = sum(1 for e in evals if e["drag"] < 1e6)
    return DroneResult(optimized=opt_mesh, drag_before=drag_before, drag_after=drag_after,
                       params=(chosen["params"] if chosen else {"tail": 0.0, "chord": 0.0, "thick": 1.0}),
                       all_evals=[{"params": e["params"],
                                   "drag": (round(e["drag"], 1) if isfinite(e["drag"]) else None)}
                                  for e in evals],
                       metrics_before=mb, metrics_after=ma, contains_original=contains_original,
                       baseline_ok=bare_ok, improved=improved, alpha_deg=float(alpha_deg),
                       mode=mode, n_cfd=len(evals), surrogate_meta=surrogate_meta)


def optimize_drone(drone: trimesh.Trimesh, seg: dict, V: float, flow_axis: str = "z",
                   n: int = 12, workers: int = 5, seed: int = 0, alpha_deg: float = 0.0,
                   progress=None) -> DroneResult:
    """Blind search: CFD-evaluate N sampled treatment forms; return the lowest-drag additive-valid one."""
    rng = np.random.default_rng(seed)
    samples = LO + (HI - LO) * rng.random((n, 3))
    samples[0] = BARE
    evals = _cfd_eval_params(drone, seg, flow_axis, list(samples), V, alpha_deg, workers, progress)
    return _result_from_evals(evals, drone, alpha_deg, mode="blind")


def _diverse_topk(cand, order, k, min_dist=0.14):
    """Greedily pick k low-predicted-drag candidates that are spaced apart in the (normalized) knob space,
    so the CFD-verify batch explores distinct forms rather than a cluster around one predicted optimum."""
    rng = (HI - LO)
    picks = []
    for i in order:
        ci = (cand[i] - LO) / rng
        if all(np.linalg.norm(ci - (cand[j] - LO) / rng) > min_dist for j in picks):
            picks.append(int(i))
        if len(picks) >= k:
            break
    if not picks:
        picks = [int(order[0])]
    return picks


def optimize_drone_surrogate(drone: trimesh.Trimesh, seg: dict, V: float, flow_axis: str = "z",
                             surrogate=None, alpha_deg: float = 0.0, n_search: int = 8000,
                             top_k: int = 6, workers: int = 5, seed: int = 0,
                             progress=None) -> DroneResult:
    """Surrogate-driven search (Sky's vision): score n_search forms in ms with the drone surrogate, then
    CFD-verify a diverse top-K. Returns the best CFD-confirmed, additive-valid, never-worse design."""
    from optimaero.drone.surrogate import load_surrogate  # lazy (avoids import cost when unused)

    def _blind():
        return optimize_drone(drone, seg, V, flow_axis=flow_axis, n=max(8, top_k + 6),
                              workers=workers, seed=seed, alpha_deg=alpha_deg, progress=progress)

    if surrogate is None:
        try:
            surrogate = load_surrogate()                 # graceful: missing/corrupt artifact → blind search
        except Exception:
            return _blind()
    rng = np.random.default_rng(seed)
    cand = LO + (HI - LO) * rng.random((n_search, 3))
    # 3-knob rows only — the surrogate ranks forms by a speed-invariant metric (drag area), so no
    # constant condition features are fed; the final answer is CFD-verified at the real V/alpha below.
    rows = [{"tail_len": float(c[0]), "chord": float(c[1]), "thick": float(c[2])} for c in cand]
    try:
        cda_pred, _ = surrogate.predict(rows)
        cda_pred = np.asarray(cda_pred, float)
    except Exception:
        return _blind()
    finite = np.isfinite(cda_pred)
    if not finite.any():
        return _blind()
    cda_pred = np.where(finite, cda_pred, np.inf)        # push any NaN/inf predictions to the back
    order = np.argsort(cda_pred)
    picks = _diverse_topk(cand, order, top_k)
    verify = [BARE] + [cand[i] for i in picks]           # index 0 = bare baseline
    meta = {"n_search": int(n_search), "top_k_verified": len(picks),
            "surrogate_best_pred_cda": float(cda_pred[order[0]]),
            "surrogate_meta": getattr(surrogate, "meta", {})}
    evals = _cfd_eval_params(drone, seg, flow_axis, verify, V, alpha_deg, workers, progress)
    res = _result_from_evals(evals, drone, alpha_deg, mode="surrogate", surrogate_meta=meta)
    if res.improved:                                     # honesty diagnostic: predicted vs CFD-actual
        won = np.array([res.params["tail"], res.params["chord"], res.params["thick"]])
        j = int(np.argmin(np.linalg.norm(cand - won, axis=1)))
        meta["winner_surrogate_pred_cda"] = float(cda_pred[j])
        meta["winner_cfd_drag"] = float(res.drag_after)
    return res


def optimize_drone_general(drone: trimesh.Trimesh, seg: dict, V: float, flow_axis: str = "z",
                           surrogate=None, alpha_deg: float = 0.0, n_search: int = 8000,
                           top_k: int = 6, workers: int = 5, seed: int = 0,
                           progress=None) -> DroneResult:
    """Surrogate-driven search with the GENERAL surrogate — works on ANY multirotor. Computes the imported
    drone's shape descriptors (no CFD), predicts the reduction ratio for n_search treatments, and
    CFD-verifies a diverse top-K. Falls back to blind CFD if the surrogate is missing or errors."""
    from optimaero.drone.general_surrogate import load_general
    from optimaero.drone.generator import drone_descriptors

    def _blind():
        return optimize_drone(drone, seg, V, flow_axis=flow_axis, n=max(8, top_k + 6),
                              workers=workers, seed=seed, alpha_deg=alpha_deg, progress=progress)

    if surrogate is None:
        try:
            surrogate = load_general()
        except Exception:
            return _blind()
    try:
        desc = drone_descriptors(drone, seg, flow_axis)  # geometric, no CFD
    except Exception:
        return _blind()
    rng = np.random.default_rng(seed)
    cand = LO + (HI - LO) * rng.random((n_search, 3))
    rows = [{**desc, "tail_len": float(c[0]), "chord": float(c[1]), "thick": float(c[2])} for c in cand]
    try:
        ratio_pred, _ = surrogate.predict(rows)
        ratio_pred = np.asarray(ratio_pred, float)
    except Exception:
        return _blind()
    finite = np.isfinite(ratio_pred)
    if not finite.any():
        return _blind()
    ratio_pred = np.where(finite, ratio_pred, np.inf)    # lowest ratio = most drag reduction
    order = np.argsort(ratio_pred)
    picks = _diverse_topk(cand, order, top_k)
    verify = [BARE] + [cand[i] for i in picks]
    meta = {"n_search": int(n_search), "top_k_verified": len(picks),
            "surrogate_best_pred_ratio": float(ratio_pred[order[0]]),
            "surrogate_meta": getattr(surrogate, "meta", {})}
    evals = _cfd_eval_params(drone, seg, flow_axis, verify, V, alpha_deg, workers, progress)
    return _result_from_evals(evals, drone, alpha_deg, mode="general", surrogate_meta=meta)
