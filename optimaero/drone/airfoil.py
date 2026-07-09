"""Airfoil-ize a drone's arms (and streamline its pods) in place.

Give each round arm strut an AIRFOIL cross-section aligned with the flow — rounded leading edge,
tapered trailing edge, chord along the airflow, thin across it — so the strut streamlines (a
cylinder's Cd ~1.2 → an airfoil's ~0.1) WITHOUT growing the frontal footprint. Additive (the airfoil
is built around the arm line), props left clear. This is "add airfoils to the arms" as the reference
high-speed drones do it.
"""
from __future__ import annotations

import numpy as np
import trimesh

from optimaero.drone.segment import AXES
from optimaero.drone.streamline import _submesh


def _airfoil_polygon(chord: float, thick: float, n: int = 24):
    """Symmetric NACA-ish airfoil polygon in the (chord=x, thickness=y) plane, nose at x=0."""
    import shapely.geometry as sg
    x = np.linspace(0.0, 1.0, n)
    yt = (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    yt = yt / yt.max() * (thick / 2.0)
    top = [(float(xi * chord), float(yi)) for xi, yi in zip(x, yt)]
    bot = [(float(xi * chord), float(-yi)) for xi, yi in zip(x[::-1], yt[::-1])]
    return sg.Polygon(top + bot)


def _strut(chord, thick, span, theta, axis, la, lb, r_in, arm_pos):
    """An airfoil-section strut: chord along the flow axis, span along the arm (azimuth theta)."""
    poly = _airfoil_polygon(chord, thick)
    s = trimesh.creation.extrude_polygon(poly, height=span)   # airfoil in xy, extruded along +z(span)
    # center the chord on the leading-edge origin → shift so the chord straddles the arm line
    s.apply_translation([-chord * 0.35, 0.0, 0.0])            # put ~35% chord ahead of the arm line
    # map local (x=chord, y=thick, z=span) → world (flow axis, in-plane perp, radial@theta)
    ct, st = np.cos(theta), np.sin(theta)
    radial = np.zeros(3); radial[la] = ct; radial[lb] = st
    perp = np.zeros(3); perp[la] = -st; perp[lb] = ct
    flow = np.zeros(3); flow[axis] = 1.0
    R = np.eye(4)
    R[:3, 0] = flow; R[:3, 1] = perp; R[:3, 2] = radial
    s.apply_transform(R)
    # move outward so the span runs from r_in to r_in+span along the arm, at the arm's flow-position
    base = radial * r_in
    base[axis] = arm_pos
    s.apply_translation(base)
    return s


def add_tail(drone: trimesh.Trimesh, seg: dict, flow_axis: str = "z",
             tail_len_frac: float = 1.4, body_source: trimesh.Trimesh | None = None) -> trimesh.Trimesh:
    """Add a streamlined boat-tail to the body's downstream end so the wake closes instead of
    separating off a flat base — usually the biggest single drag reduction on a bluff-based body.
    The tail tapers inward from the base cross-section to a point, so it does NOT grow the footprint.

    `drone` is the mesh the cone is unioned onto (may already carry airfoil arms). `body_source`, if
    given, is the ORIGINAL drone whose faces match `seg['face_labels']` — body detection must use it,
    because after airfoil_arms the arm-carrying mesh has a different face count/ordering than seg."""
    axis = AXES[flow_axis]
    la, lb = [i for i in range(3) if i != axis]
    src = body_source if body_source is not None else drone
    body = _submesh(src, seg["face_labels"], lambda l: l == "body") or src
    bv = body.vertices
    zmin = float(bv[:, axis].min()); zmax = float(bv[:, axis].max())
    L = max(zmax - zmin, 1e-6)
    base = bv[bv[:, axis] < zmin + 0.2 * L]                 # the downstream (-flow) base ring
    cy = float(base[:, la].mean()); cz = float(base[:, lb].mean())
    ay = max(float(np.ptp(base[:, la])) / 2, 1e-3); az = max(float(np.ptp(base[:, lb])) / 2, 1e-3)
    tail_len = tail_len_frac * L
    cone = trimesh.creation.cone(radius=1.0, height=tail_len, sections=40)   # base z=0, apex +z
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))  # apex → −z (downstream)
    cone.apply_scale([ay, az, 1.0])                        # elliptical base matching the body base
    if axis != 2:
        cone.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], np.eye(3)[axis]))
    t = np.zeros(3); t[la] = cy; t[lb] = cz; t[axis] = zmin
    cone.apply_translation(t)
    return drone.union(cone) if cone.is_watertight else drone


def _arm_thickness(mesh, r_in, r_pod, theta, la, lb, span):
    """Measure an arm's true cross-section thickness by slicing perpendicular to it at mid-span, so the
    airfoil can be sized to the ARM (a proper teardrop that hugs it), not to the drone or the motor pod.
    Clamped to a sane band; falls back to a fraction of the span if the slice fails."""
    fallback = float(np.clip(0.18 * span, 0.006, 0.03))
    try:
        r_mid = 0.5 * (r_in + r_pod)
        normal = np.zeros(3); normal[la] = np.cos(theta); normal[lb] = np.sin(theta)
        sec = mesh.section(plane_origin=normal * r_mid, plane_normal=normal)
        if sec is not None:
            p2, _ = sec.to_planar()
            return float(np.clip(max(p2.extents), 0.006, 0.4 * span))
    except Exception:
        pass
    return fallback


def airfoil_arms(drone: trimesh.Trimesh, seg: dict, flow_axis: str = "z",
                 chord: float | None = None, thick_scale: float = 1.0,
                 prop_clear: bool = True) -> trimesh.Trimesh:
    axis = AXES[flow_axis]
    la, lb = [i for i in range(3) if i != axis]
    rmax = seg["rmax"]; r_in = seg["r_core"]
    chord_s = (chord / rmax) if chord else 1.0   # the optimizer's chord knob (≈0.9–2.6), decoupled from size
    parts = [drone]
    for disk in seg["rotor_disks"]:
        cx, cy = disk["center_lat"]
        theta = float(np.arctan2(cy, cx))
        r_pod = float(np.hypot(cx, cy))
        span = max(r_pod - r_in, 1e-3)
        # A PROPER airfoil sized to the arm: thickness hugs the round arm (never thinner, so the fairing
        # encloses it), chord = fineness × thickness (a real teardrop, ~4–6:1), NOT tied to arm length.
        arm_t = _arm_thickness(drone, r_in, r_pod, theta, la, lb, span)
        thick = arm_t * max(thick_scale, 1.05)
        fineness = float(np.clip(3.5 + chord_s, 3.5, 6.5))
        strut_chord = fineness * thick
        parts.append(_strut(strut_chord, thick, span, theta, axis, la, lb, r_in, disk["axis_pos"]))
    result = trimesh.boolean.union([p for p in parts if p.is_watertight])

    if prop_clear and seg["rotor_disks"]:
        zlo = float(result.vertices[:, axis].min()) - 0.03
        zhi = float(result.vertices[:, axis].max()) + 0.03
        cyls = []
        for disk in seg["rotor_disks"]:
            cx, cy = disk["center_lat"]
            cyl = trimesh.creation.cylinder(radius=disk["radius"], height=(zhi - zlo), sections=36)
            if axis != 2:
                cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], np.eye(3)[axis]))
            c = np.zeros(3); c[la] = cx; c[lb] = cy; c[axis] = (zlo + zhi) / 2
            cyl.apply_translation(c); cyls.append(cyl)
        result = result.difference(trimesh.util.concatenate(cyls)).union(drone)
    return result
