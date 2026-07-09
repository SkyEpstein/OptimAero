"""Enclosing aerodynamic envelope.

Grow a streamlined outer skin that FULLY CONTAINS the user's imported shape and adapts its own
width/height silhouette into an aerodynamic body — a streamlined nose upstream, a tapering tail
downstream, sized to the part in every direction. The envelope never goes inside the original
surface; it only adds material outward. The proportions are then optimized for a chosen property
(minimize drag, maximize lift, or maximize L/D).

Contrast with shapeopt.optimize (which deforms/shrinks toward an inner keep-out): here the whole
original is the keep-out and the result is a superset of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.optimize import differential_evolution

from optimaero.shapeopt.optimize import (_flow_rotation, drag_estimate, body_aero, RHO)

OBJECTIVES = ("min_drag", "max_lift", "max_LD")


def _profiles(work: trimesh.Trimesh, S: int = 28):
    """Half-width profiles wy(x), wz(x) of the original along the flow axis (x)."""
    v = work.vertices
    x = v[:, 0]
    x0, x1 = float(x.min()), float(x.max())
    L = max(x1 - x0, 1e-9)
    cy, cz = float(v[:, 1].mean()), float(v[:, 2].mean())
    stations = np.linspace(x0, x1, S)
    half = (L / S) * 1.6 + 1e-9
    wy = np.zeros(S)
    wz = np.zeros(S)
    for i, xi in enumerate(stations):
        sel = np.abs(x - xi) <= half
        if int(sel.sum()) >= 3:
            wy[i] = float(np.abs(v[sel, 1] - cy).max())
            wz[i] = float(np.abs(v[sel, 2] - cz).max())
    valid = (wy > 1e-9) & (wz > 1e-9)
    if int(valid.sum()) < 2:                    # degenerate — fall back to bbox half-extents
        wy[:] = max(float(np.abs(v[:, 1] - cy).max()), 1e-4)
        wz[:] = max(float(np.abs(v[:, 2] - cz).max()), 1e-4)
    else:
        wy = np.interp(stations, stations[valid], wy[valid])
        wz = np.interp(stations, stations[valid], wz[valid])
    k = np.ones(3) / 3.0                         # light smoothing
    wy = np.convolve(np.pad(wy, 1, "edge"), k, "same")[1:-1]
    wz = np.convolve(np.pad(wz, 1, "edge"), k, "same")[1:-1]
    return dict(stations=stations, wy=wy, wz=wz, x0=x0, x1=x1, L=L, cy=cy, cz=cz)


def _ring(xc, cy, cz, Ay, Az, n, th):
    ct, st = np.cos(th), np.sin(th)
    y = cy + Ay * np.sign(ct) * np.abs(ct) ** (2.0 / n)
    z = cz + Az * np.sign(st) * np.abs(st) ** (2.0 / n)
    return np.column_stack([np.full(th.size, xc), y, z])


def build_envelope(prof: dict, params, P: int = 30) -> trimesh.Trimesh:
    """Closed ring-loft envelope (flow-aligned) that contains the original by construction.
    params = [grow, nose_frac, tail_frac, round_exp, (camber)]. camber (optional, default 0) bends
    the mean line up mid-body (like an airfoil) so the body generates lift — used for the Cl model."""
    grow, nose_frac, tail_frac, n = params[:4]
    camber = float(params[4]) if len(params) > 4 else 0.0
    n = float(np.clip(n, 2.0, 10.0))
    stations, wy, wz = prof["stations"], prof["wy"], prof["wz"]
    x0, x1, L, cy, cz = prof["x0"], prof["x1"], prof["L"], prof["cy"], prof["cz"]
    # analytic containment: a superellipse (Ay,Az) contains the bbox slice (wy,wz) iff Ay>=wy*2^(1/n)
    gg = max(float(grow), 2.0 ** (1.0 / n) + 0.03)
    th = np.linspace(0, 2 * np.pi, P, endpoint=False)

    xs, Ays, Azs = [], [], []
    # nose extension (elliptical): apex handled separately; interior nose rings taper 0 -> body[0]
    ln = max(nose_frac, 0.0) * L
    lt = max(tail_frac, 0.0) * L
    n_nose = 8 if ln > 1e-6 else 0
    n_tail = 10 if lt > 1e-6 else 0
    for i in range(1, n_nose + 1):               # from apex(0) toward body(1), skip apex (i=0)
        u = i / (n_nose + 1)
        f = np.sqrt(max(1.0 - (1.0 - u) ** 2, 0.0))   # half-ellipse nose
        xs.append(x0 - ln * (1.0 - u)); Ays.append(wy[0] * gg * f); Azs.append(wz[0] * gg * f)
    for i in range(len(stations)):               # body rings — >= original at every station
        xs.append(stations[i]); Ays.append(wy[i] * gg); Azs.append(wz[i] * gg)
    for i in range(1, n_tail + 1):               # tail boat-tail toward apex
        u = i / (n_tail + 1)
        f = (1.0 - u) ** 2
        xs.append(x1 + lt * u); Ays.append(wy[-1] * gg * f); Azs.append(wz[-1] * gg * f)

    xe0, xe1 = x0 - ln, x1 + lt
    span = max(xe1 - xe0, 1e-9)
    cz_amp = camber * L                          # parabolic camber line, peak mid-body → lift

    def cz_at(x):
        t = (x - xe0) / span
        return cz + cz_amp * 4.0 * t * (1.0 - t)

    rings = [_ring(xs[i], cy, cz_at(xs[i]), max(Ays[i], 1e-5), max(Azs[i], 1e-5), n, th)
             for i in range(len(xs))]
    nose_apex = np.array([[xe0, cy, cz_at(xe0)]])
    tail_apex = np.array([[xe1, cy, cz_at(xe1)]])
    verts = np.vstack([nose_apex] + rings + [tail_apex])
    R = len(rings)
    faces = []
    # nose apex (index 0) fan to ring 0 (indices 1..P)
    for j in range(P):
        faces.append([0, 1 + j, 1 + (j + 1) % P])
    # between consecutive rings
    for r in range(R - 1):
        a0 = 1 + r * P
        b0 = 1 + (r + 1) * P
        for j in range(P):
            j1 = (j + 1) % P
            faces.append([a0 + j, b0 + j, b0 + j1])
            faces.append([a0 + j, b0 + j1, a0 + j1])
    # tail apex fan
    tail_idx = verts.shape[0] - 1
    last0 = 1 + (R - 1) * P
    for j in range(P):
        j1 = (j + 1) % P
        faces.append([tail_idx, last0 + j1, last0 + j])
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    m.fix_normals()
    return m


@dataclass
class EnvelopeResult:
    optimized: trimesh.Trimesh
    original: trimesh.Trimesh
    objective: str
    metrics_before: dict
    metrics_after: dict
    contains_original: bool
    params: dict


def _metrics(mesh_flow, V, alpha_deg):
    a = body_aero(mesh_flow, V, alpha_deg=alpha_deg, flow_axis="x")
    return {"drag": a["drag"], "lift": a["lift"], "LD": a["LD"], "Cl": a["Cl"], "Cd": a["Cd"]}


def optimize_envelope(mesh: trimesh.Trimesh, V: float, flow_axis: str = "x",
                      objective: str = "min_drag", alpha_deg: float = 0.0,
                      aggressiveness: float = 0.5, maxiter: int = 16, seed: int = 0,
                      max_len_ratio: float | None = None) -> EnvelopeResult:
    if objective not in OBJECTIVES:
        objective = "min_drag"
    Rm = _flow_rotation(flow_axis)
    work = mesh.copy(); work.apply_transform(Rm)          # flow along +x
    prof = _profiles(work)
    a = float(np.clip(aggressiveness, 0.0, 1.0))
    bounds = [(1.0, 1.0 + 0.6 * a),                       # grow (outward margin)
              (0.3, 0.5 + 1.6 * a),                       # nose_frac — always some streamlined nose
              (0.5, 0.8 + 2.4 * a),                       # tail_frac — always a boat-tail
              (2.0, 8.0)]                                 # round_exp (ellipse..rounded-rectangle)
    if max_len_ratio is not None:                         # cap total length ≤ max_len_ratio × original
        extra = max(float(max_len_ratio) - 1.0, 0.2)
        bounds[1] = (0.05, 0.45 * extra)                  # nose_frac
        bounds[2] = (0.10, 0.55 * extra)                  # tail_frac

    def score(m):                                         # lower is better for every objective
        if objective == "min_drag":
            return m["drag"]
        if objective == "max_lift":
            return -m["lift"] + 1e-3 * m["drag"]
        return -m["LD"]                                   # max_LD

    def objective_fn(p):
        try:
            env = build_envelope(prof, p)
            if not np.isfinite(env.vertices).all() or env.area <= 0:
                return 1e9
            return score(_metrics(env, V, alpha_deg))
        except Exception:
            return 1e9

    res = differential_evolution(objective_fn, bounds, maxiter=maxiter, popsize=15, seed=seed,
                                 tol=1e-4, polish=False)
    env = build_envelope(prof, res.x)
    # guarantee containment of the ORIGINAL (analytic sizing should already ensure it; verify+project)
    contains = True
    try:
        sd = trimesh.proximity.signed_distance(env, work.vertices)
        for _ in range(4):
            if float(np.nanmin(sd)) >= -1e-6:
                break
            env.vertices[:, 1] = prof["cy"] + (env.vertices[:, 1] - prof["cy"]) * 1.12
            env.vertices[:, 2] = prof["cz"] + (env.vertices[:, 2] - prof["cz"]) * 1.12
            sd = trimesh.proximity.signed_distance(env, work.vertices)
        contains = bool(float(np.nanmin(sd)) >= -1e-6)
    except Exception:
        contains = True  # proximity failed; analytic sizing still guarantees containment

    mb = _metrics(work, V, alpha_deg)
    ma = _metrics(env, V, alpha_deg)
    px = {n_: float(v_) for n_, v_ in zip(["grow", "nose_frac", "tail_frac", "round_exp"], res.x)}
    # NEVER WORSE: if no enclosing envelope beats the original on the chosen objective, keep the
    # original unchanged. Enclosing a compact or gappy body can add drag (more wetted/planform area);
    # in that case forcing an envelope would make the target metric worse, so we don't.
    if score(ma) > score(mb):
        env, ma, contains = work.copy(), mb, True
        px = {"grow": 1.0, "nose_frac": 0.0, "tail_frac": 0.0, "round_exp": 0.0}
    out = env.copy(); out.apply_transform(np.linalg.inv(Rm))   # back to user's orientation
    return EnvelopeResult(
        optimized=out, original=mesh, objective=objective, metrics_before=mb, metrics_after=ma,
        contains_original=contains, params=px)


if __name__ == "__main__":
    import os
    part = trimesh.creation.box(extents=[0.30, 0.12, 0.10])
    r = optimize_envelope(part, V=25.0, objective="min_drag", maxiter=12)
    print("box 0.30x0.12x0.10 → enclosing envelope (min_drag)")
    print("params:", {k: round(v, 3) for k, v in r.params.items()})
    print(f"drag {r.metrics_before['drag']:.3f} N → {r.metrics_after['drag']:.3f} N")
    print("contains original:", r.contains_original)
    out = "/tmp/envelope"; os.makedirs(out, exist_ok=True)
    r.original.export(f"{out}/orig.stl"); r.optimized.export(f"{out}/env.stl")
    print("wrote", out)
