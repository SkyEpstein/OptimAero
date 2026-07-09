"""In-place aerodynamic streamlining of a segmented multirotor.

Reshapes the body / arm / pod surfaces to cut drag while keeping each rotor disk clear. It never
encloses the drone and never fills the rotor space — the result is the SAME drone, slipperier.
Drag uses a component buildup (body + arms + pods each get their own frontal area and shape factor)
so streamlining an individual part actually shows up — the whole-body estimate cannot resolve that.
"""
from __future__ import annotations

import numpy as np
import trimesh

from optimaero.drone.segment import AXES

RHO, NU = 1.225, 1.5e-5


def _submesh(mesh: trimesh.Trimesh, face_labels, pred):
    idx = [i for i, l in enumerate(face_labels) if pred(l)]
    if not idx:
        return None
    return mesh.submesh([idx], append=True)


def _comp_cd_area(sub: trimesh.Trimesh, fi: int):
    ext = np.asarray(sub.bounding_box.extents, float)
    Lf = float(ext[fi])
    perp = [i for i in range(3) if i != fi]
    n = [0.0, 0.0, 0.0]; n[fi] = 1.0
    try:
        A = float(sub.projected(n).area)
    except Exception:
        A = float(ext[perp[0]] * ext[perp[1]])
    d = 2 * np.sqrt(A / np.pi) if A > 0 else max(Lf, 1e-3)
    fineness = max(Lf / max(d, 1e-6), 0.5)
    Cd = 0.10 + 0.90 * np.exp(-0.6 * (fineness - 1.0))     # bluff→~1.0, streamlined→~0.16
    return Cd, A, float(sub.area)


def drone_drag(mesh: trimesh.Trimesh, seg: dict, V: float, flow_axis: str = "z"):
    """Component-buildup drag [N] at speed V along flow_axis. Returns (total, breakdown-by-part)."""
    fi = AXES[flow_axis]
    fl = seg["face_labels"]
    q = 0.5 * RHO * V ** 2
    parts = {"body": lambda l: l == "body",
             "arms": lambda l: l.startswith("arm"),
             "pods": lambda l: l.startswith("pod")}
    form = wet = 0.0
    breakdown = {}
    for name, pred in parts.items():
        sub = _submesh(mesh, fl, pred)
        if sub is None:
            breakdown[name] = 0.0
            continue
        Cd, A, Awet = _comp_cd_area(sub, fi)
        form += Cd * A
        wet += Awet
        breakdown[name] = q * Cd * A
    L = float(mesh.extents[fi])
    Cf = 0.074 / max(V * L / NU, 1.0) ** 0.2
    breakdown["friction"] = q * Cf * wet
    return q * form + q * Cf * wet, breakdown


def streamline_drone(mesh: trimesh.Trimesh, seg: dict, flow_axis: str = "z",
                     strength: float = 1.0) -> trimesh.Trimesh:
    """Boat-tail each motor pod on its leeward (downstream) side and taper it toward a tip — a local
    fairing that cuts pod base/form drag. Only leeward vertices move (downstream + inward), so nothing
    enters the rotor keep-clear disk on the windward side. Body/arms are left intact (already slim)."""
    fi = AXES[flow_axis]
    la, lb = [i for i in range(3) if i != fi]
    Vn = np.asarray(mesh.vertices, float).copy()
    labels = seg["labels"]
    s = float(np.clip(strength, 0.0, 1.5))
    for g in range(seg["n_arms"]):
        idx = [i for i in range(len(Vn)) if labels[i] == f"pod{g}"]
        if not idx:
            continue
        cen = Vn[idx].mean(0)
        lee = min(float(Vn[i, fi] - cen[fi]) for i in idx)   # most-leeward offset (negative)
        span = max(-lee, 1e-6)
        for i in idx:
            z = Vn[i, fi] - cen[fi]
            if z < 0:                                        # leeward half only → boat-tail
                t = min(-z / span, 1.0)
                Vn[i, fi] = cen[fi] + z * (1.0 + 0.8 * s)    # extend the tail downstream
                scale = 1.0 - 0.7 * s * t ** 1.5             # taper the cross-section toward the tip
                Vn[i, la] = cen[la] + (Vn[i, la] - cen[la]) * scale
                Vn[i, lb] = cen[lb] + (Vn[i, lb] - cen[lb]) * scale
    out = mesh.copy()
    out.vertices = Vn
    return out


def optimize_streamline(mesh: trimesh.Trimesh, seg: dict, V: float, flow_axis: str = "z"):
    """Pick the boat-tail strength that minimises component-buildup drag. Returns (best_mesh, d0, d1, strength)."""
    d0, _ = drone_drag(mesh, seg, V, flow_axis)
    best = (mesh, d0, 0.0)
    for s in np.linspace(0.2, 1.5, 14):
        m = streamline_drone(mesh, seg, flow_axis, strength=float(s))
        d, _ = drone_drag(m, seg, V, flow_axis)
        if d < best[1]:
            best = (m, d, float(s))
    return best[0], d0, best[1], best[2]
