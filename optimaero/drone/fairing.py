"""Reference-style additive drone fairing.

Turns a bare functional multirotor into a sleek high-speed drone the way a real one is built: a
streamlined FUSELAGE pod around the central body, a teardrop NACELLE faired around each motor, and
streamlined STRUTS around the arms — every piece ADDED around the baseline (volume ≥ original, the
drone preserved inside), with the rotor columns cut clear so the props still work. Matches the
fuselage + motor-nacelle + faired-arm look of a high-speed FPV drone.
"""
from __future__ import annotations

import numpy as np
import trimesh

from optimaero.drone.segment import AXES
from optimaero.drone.streamline import _submesh
from optimaero.shapeopt.envelope import build_envelope, _profiles, _flow_rotation


def _fair(sub: trimesh.Trimesh, Rm, Rinv, params) -> trimesh.Trimesh:
    """Streamlined teardrop envelope around a component sub-mesh, in the drone's frame."""
    w = sub.copy(); w.apply_transform(Rm)            # travel axis → +x
    env = build_envelope(_profiles(w), params)
    env.apply_transform(Rinv)                        # back to the drone frame
    return env


def fair_drone(drone: trimesh.Trimesh, seg: dict, flow_axis: str = "z",
               fuselage_p=(1.02, 0.35, 0.55, 5.0), nacelle_p=(1.1, 0.7, 0.9, 3.0),
               strut_p=(1.05, 0.3, 0.5, 4.0), prop_clear: bool = True) -> trimesh.Trimesh:
    """Each *_p is (grow, nose_frac, tail_frac, round_exp) for that component's fairing, or None to
    skip it. High round_exp + grow≈1 keeps the fairing TIGHT (doesn't widen the frontal footprint);
    bigger nose/tail_frac elongates it along the travel axis (streamlines without adding frontal area)."""
    axis = AXES[flow_axis]
    la, lb = [i for i in range(3) if i != axis]
    Rm = _flow_rotation(flow_axis); Rinv = np.linalg.inv(Rm)
    fl = seg["face_labels"]
    n = seg["n_arms"]
    parts = [drone]

    if fuselage_p is not None:
        body = _submesh(drone, fl, lambda l: l == "body")
        if body is not None:
            parts.append(_fair(body, Rm, Rinv, list(fuselage_p)))
    if nacelle_p is not None:
        for g in range(n):
            pod = _submesh(drone, fl, lambda l, g=g: l == f"pod{g}")
            if pod is not None:
                parts.append(_fair(pod, Rm, Rinv, list(nacelle_p)))
    if strut_p is not None:
        for g in range(n):
            arm = _submesh(drone, fl, lambda l, g=g: l == f"arm{g}")
            if arm is not None and len(arm.vertices) >= 6:
                parts.append(_fair(arm, Rm, Rinv, list(strut_p)))

    result = trimesh.boolean.union([p for p in parts if p is not None and p.is_watertight])

    if prop_clear and seg["rotor_disks"]:
        zlo = float(result.vertices[:, axis].min()) - 0.03
        zhi = float(result.vertices[:, axis].max()) + 0.03
        cyls = []
        for disk in seg["rotor_disks"]:
            cx, cy = disk["center_lat"]
            cyl = trimesh.creation.cylinder(radius=disk["radius"], height=(zhi - zlo), sections=40)
            if axis != 2:
                cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], np.eye(3)[axis]))
            c = np.zeros(3); c[la] = cx; c[lb] = cy; c[axis] = (zlo + zhi) / 2
            cyl.apply_translation(c); cyls.append(cyl)
        result = result.difference(trimesh.util.concatenate(cyls))
        result = result.union(drone)                 # restore any drone material the prop holes cut
    return result
