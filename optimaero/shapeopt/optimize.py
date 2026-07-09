"""Aerodynamic shape optimization of an IMPORTED shape.

Deform the OUTER surface of the user's mesh to reduce drag, while keeping an inner keep-out
region (their parts / volume) fully inside. This optimizes the user's own geometry — the
output is a deformation of their shape, not a new body wrapped around a box.

Aero is a fast physics-informed estimate (frontal-area form drag that rewards streamlining +
wetted-area skin friction) — not CFD; a real solver plugs in later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.optimize import differential_evolution

RHO, NU = 1.225, 1.5e-5


def make_watertight(mesh: trimesh.Trimesh):
    """Best-effort convert an imported mesh into ONE watertight solid — the additive booleans need a
    volume, and CAD/STL exports often aren't watertight. Most common case: a multi-body export (e.g.
    Fusion's body + motor mounts as separate overlapping solids) reads as one non-watertight surface;
    boolean-unioning the solid components fixes it. Returns (mesh, note)."""
    if mesh.is_watertight:
        return mesh, "watertight"
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        parts = [mesh]
    wt = [p for p in parts if p.is_watertight]
    if wt:                                                   # union the solid bodies into one
        try:
            u = wt[0] if len(wt) == 1 else trimesh.boolean.union(wt)
            if u is not None and u.is_watertight and u.volume > 0:
                note = f"repaired: unioned {len(wt)} solid bodies"
                if len(wt) < len(parts):
                    note += f" (dropped {len(parts) - len(wt)} non-solid fragment(s))"
                return u, note
        except Exception:
            pass
    m = mesh.copy()                                          # fallback: clean up + fill holes
    try:
        m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(m); trimesh.repair.fix_normals(m)
    except Exception:
        pass
    return (m, "repaired: filled holes") if m.is_watertight else \
        (m, "WARNING: not watertight — the optimizer may be unable to add material")


def load_shape(path: str, units: str = "mm", repair: bool = True) -> trimesh.Trimesh:
    from optimaero.three_d.cad3d import UNIT_SCALE
    mesh = trimesh.load(path, force="mesh")
    mesh.apply_scale(UNIT_SCALE.get(units, 0.001))
    if repair:
        mesh, _ = make_watertight(mesh)
    return mesh


def drag_estimate(mesh: trimesh.Trimesh, V: float) -> float:
    """Fast physics-informed drag [N]: form drag (frontal area, lower for streamlined bodies)
    + skin friction (wetted area). Flow is along +x."""
    L = float(mesh.extents[0])
    try:
        A_front = float(mesh.projected([1, 0, 0]).area)
    except Exception:
        A_front = float(mesh.extents[1] * mesh.extents[2])
    # Bluff/gappy bodies (a drone's exposed arms, an open frame): at speed the flow separates around
    # the whole silhouette rather than passing cleanly through the gaps, so form/pressure drag acts on
    # the CONVEX-HULL frontal area. For an already-streamlined (convex) body this equals A_front.
    try:
        A_form = A_front if mesh.is_convex else max(A_front, float(mesh.convex_hull.projected([1, 0, 0]).area))
    except Exception:
        A_form = A_front
    A_wet = float(mesh.area)
    d = 2 * np.sqrt(A_form / np.pi) if A_form > 0 else max(L, 1e-6)
    fineness = max(L / max(d, 1e-6), 0.5)
    Cd_form = 0.10 + 0.90 * np.exp(-0.6 * (fineness - 1.0))   # bluff→~1.0, streamlined→~0.16
    Re = max(V * L / NU, 1.0)
    Cf = 0.074 / Re ** 0.2
    q = 0.5 * RHO * V ** 2
    return q * (Cd_form * A_form + Cf * A_wet)


def deform(mesh: trimesh.Trimesh, params) -> trimesh.Trimesh:
    """Deform the shape's outer surface: elongate along the flow + taper nose/tail toward the
    centerline + smooth. All are deformations OF the user's mesh (topology preserved)."""
    elong, nose, tail, smooth, grow = params
    m = mesh.copy()
    v = m.vertices.astype(float).copy()
    x = v[:, 0]
    x0, x1 = x.min(), x.max()
    L = max(x1 - x0, 1e-9)
    t = (x - x0) / L
    v[:, 0] = x0 + (x - x0) * elong
    cy, cz = v[:, 1].mean(), v[:, 2].mean()
    taper = 1.0 - nose * np.clip(1 - t / 0.4, 0, 1) ** 2 - tail * np.clip((t - 0.6) / 0.4, 0, 1) ** 2
    bulge = 1.0 + grow * np.sin(np.pi * np.clip(t, 0, 1))   # ADD material outside (mid-body fairing)
    s = np.clip(taper * bulge, 0.20, 4.0)                  # s>1 grows outward; never touches keep-out
    v[:, 1] = cy + (v[:, 1] - cy) * s
    v[:, 2] = cz + (v[:, 2] - cz) * s
    m.vertices = v
    if smooth >= 0.5:
        pre = m.copy()
        trimesh.smoothing.filter_laplacian(m, iterations=int(round(smooth)))
        if not np.isfinite(m.vertices).all():   # smoothing blew up (concave mesh) → revert
            m = pre
    return m


def keepout(mesh: trimesh.Trimesh, frac: float = 0.85) -> trimesh.Trimesh:
    """The inner region the parts occupy — the user's shape shrunk toward its centroid. The
    optimized outer surface must always contain this."""
    k = mesh.copy()
    k.vertices = mesh.centroid + (mesh.vertices - mesh.centroid) * frac
    return k


def _keepout_clearance(outer: trimesh.Trimesh, points: np.ndarray) -> float:
    """Min signed distance [m] of `points` inside `outer` (>0 inside, <0 outside). Robust on
    smoothed meshes where boolean `contains()` is flaky. A positive minimum ⇒ keep-out is
    safely inside the outer surface."""
    try:
        return float(trimesh.proximity.signed_distance(outer, points).min())
    except Exception:
        return -1e9


@dataclass
class ShapeResult:
    optimized: trimesh.Trimesh
    original: trimesh.Trimesh
    drag_before: float
    drag_after: float
    params: dict
    keepout_preserved: bool


def _flow_rotation(flow_axis: str) -> np.ndarray:
    """4x4 transform that rotates the chosen flow axis onto +x (so the x-based deform/drag
    code handles any flow direction). Inverse brings the result back to the user's orientation."""
    a = str(flow_axis).lower().lstrip("+-")
    if a == "y":
        return trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])  # y → x
    if a == "z":
        return trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])   # z → x
    return np.eye(4)                                                            # x → x


def optimize_shape(mesh: trimesh.Trimesh, V: float, flow_axis: str = "x",
                   keepout_frac: float = 0.85, aggressiveness: float = 0.5,
                   maxiter: int = 20, seed: int = 0) -> ShapeResult:
    R = _flow_rotation(flow_axis)
    work = mesh.copy()
    work.apply_transform(R)                       # align the flow direction with +x
    ko = keepout(work, keepout_frac)
    ko_pts = ko.vertices
    if len(ko_pts) > 300:
        ko_pts = ko_pts[np.random.default_rng(0).choice(len(ko_pts), 300, replace=False)]
    margin = 0.004 * float(work.extents.max())    # required inside-clearance for the keep-out
    d0 = drag_estimate(work, V)
    a = float(np.clip(aggressiveness, 0.0, 1.0))  # 0 = gentle tweaks, 1 = drastic reshaping
    bounds = [(1.0, 1.0 + 4.0 * a),               # elongate (drastic → up to 5× along the flow)
              (0.0, 0.5), (0.0, 0.6),             # nose, tail taper
              (0.0, 4.0),                         # smooth
              (0.0, 0.6 * a)]                     # grow — ADD material OUTSIDE (fairing)

    def objective(p):
        try:
            dm = deform(work, p)
            clr = _keepout_clearance(dm, ko_pts)   # HARD: keep-out must stay inside with margin
            if clr < margin:
                return 1e6 + 1e6 * (margin - clr)  # smooth penalty as it approaches the keep-out
            return drag_estimate(dm, V)
        except Exception:
            return 1e6

    res = differential_evolution(objective, bounds, maxiter=maxiter, popsize=10, seed=seed,
                                 tol=1e-3, polish=False)
    best_work = deform(work, res.x)
    d_after = drag_estimate(best_work, V)
    px = list(res.x)
    # HARD GUARANTEE: never return a shape worse than the input. If the search couldn't beat the
    # original (non-convergence, wrong flow axis, over-tight keep-out), keep the original unchanged.
    if not (d_after < d0):
        best_work, d_after, px = work.copy(), d0, [1.0, 0.0, 0.0, 0.0, 0.0]
    preserved = _keepout_clearance(best_work, ko_pts) >= 0.0
    best = best_work.copy()
    best.apply_transform(np.linalg.inv(R))        # back to the user's original orientation
    return ShapeResult(
        optimized=best, original=mesh, drag_before=d0, drag_after=d_after,
        params={n: float(v) for n, v in
                zip(["elongate", "nose_taper", "tail_taper", "smooth", "grow"], px)},
        keepout_preserved=preserved)


def _proj_area(mesh: trimesh.Trimesh, normal) -> float:
    """Area of the shape projected along `normal` (frontal or planform)."""
    try:
        return float(mesh.projected(normal).area)
    except Exception:
        e = mesh.extents
        idx = [i for i in range(3) if abs(normal[i]) < 0.5]
        return float(e[idx[0]] * e[idx[1]])


def body_aero(mesh: trimesh.Trimesh, V: float, alpha_deg: float = 0.0,
              flow_axis: str = "x") -> dict:
    """Fast lift / drag / L·D estimate at angle of attack `alpha_deg` (flow along the chosen
    axis, z is 'up'). Lift uses a thin lifting-body model: a symmetric body at 0° makes ~0
    lift; lift and induced drag grow with angle. Coefficients are referenced to planform area.
    This is an ESTIMATE — CFD gives the rigorous forces."""
    R = _flow_rotation(flow_axis)
    m = mesh.copy()
    m.apply_transform(R)                          # flow along +x, z is up
    A_front = _proj_area(m, [1, 0, 0])            # frontal area (drag ref)
    S_plan = _proj_area(m, [0, 0, 1])             # planform area (lift ref)
    q = 0.5 * RHO * V ** 2
    a = np.radians(alpha_deg)
    D_axial = drag_estimate(m, V)                 # streamlined axial drag at 0°
    lift = q * S_plan * 2.0 * np.sin(a) * np.cos(a)      # thin lifting-body lift
    drag = D_axial + q * S_plan * 2.0 * np.sin(a) ** 2   # axial + angle-of-attack pressure drag
    Sref = S_plan if S_plan > 0 else max(A_front, 1e-9)
    Cl = lift / (q * Sref)
    Cd = drag / (q * Sref)
    LD = lift / drag if drag > 1e-9 else 0.0
    return {"lift": float(lift), "drag": float(drag), "LD": float(LD),
            "Cl": float(Cl), "Cd": float(Cd), "Sref": float(Sref), "alpha_deg": float(alpha_deg)}


if __name__ == "__main__":  # prove it optimizes an imported shape, preserving the interior
    import os
    # a blocky "part" (blunt box) as the user's imported shape
    part = trimesh.creation.box(extents=[0.30, 0.12, 0.10])
    r = optimize_shape(part, V=25.0, maxiter=15)
    print(f"imported shape: box 0.30x0.12x0.10 m")
    print(f"optimized (deformed) params: { {k: round(v,3) for k,v in r.params.items()} }")
    print(f"drag: {r.drag_before:.3f} N -> {r.drag_after:.3f} N  "
          f"({(1-r.drag_after/r.drag_before)*100:.0f}% lower)")
    print(f"inner volume preserved: {r.keepout_preserved}")
    print(f"same mesh topology (a deformation, not a new body): "
          f"{len(r.optimized.vertices)==len(part.vertices) or 'smoothed'}")
    out = "/tmp/shapeopt"; os.makedirs(out, exist_ok=True)
    r.original.export(f"{out}/before.stl"); r.optimized.export(f"{out}/after.stl")
    print(f"wrote {out}/before.stl and after.stl")
