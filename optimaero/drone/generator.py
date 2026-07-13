"""Parametric multirotor generator — synthesize diverse watertight drones to train a GENERAL surrogate.

A drone here is: a central body (ellipsoid, elongated along the flow axis) + N radial arms + N motor pods
at the arm tips. Randomizing arm count, body fineness, arm length/thickness, and pod size spans the
multirotor design space so the surrogate learns to optimize ANY multirotor, not one. Each drone is a single
watertight solid (unioned) that `segment_multirotor` can segment and the airfoil/tail builder can treat.

Run:  python -m optimaero.drone.generator            # smoke: build a few, report segmentation
"""
from __future__ import annotations

import numpy as np
import trimesh

from optimaero.shapeopt.optimize import _flow_rotation
from optimaero.cfd.dataset import _area_dist_features
from optimaero.drone.segment import AXES

# design-space bounds (metres) — chosen to cover small FPV → larger multirotors
ARM_COUNTS = [3, 4, 6, 8]

# drone-SHAPE descriptors (constant per drone, purely geometric → computable at serve time with NO CFD).
# These condition the general surrogate so it predicts for ANY multirotor, not one.
DESCRIPTOR_FEATS = ["n_rotors", "rmax", "r_core", "r_core_ratio", "arm_len", "arm_r", "pod_r",
                    "arm_slender", "pod_arm", "A_front", "A_plan", "A_wet", "vol", "Dmax",
                    "body_fineness", "wet_front", "plan_front",
                    "prismatic", "x_maxarea", "area_smooth", "base_area", "nose_area"]


def drone_descriptors(drone: trimesh.Trimesh, seg: dict, flow_axis: str = "z") -> dict:
    """Geometric shape descriptors of a bare drone (flow along +x, no CFD). Used both to label the
    multi-drone dataset and to condition the surrogate at serve time on the imported drone's shape."""
    from optimaero.drone.airfoil import _arm_thickness
    o = drone.copy(); o.apply_transform(_flow_rotation(flow_axis))
    ext = np.asarray(o.bounding_box.extents, float); L = float(ext[0])
    try:
        A_front = float(o.projected([1, 0, 0]).area)
    except Exception:
        A_front = float(ext[1] * ext[2])
    try:
        A_plan = float(o.projected([0, 0, 1]).area)
    except Exception:
        A_plan = float(ext[0] * ext[1])
    A_wet = float(o.area); vol = float(o.volume)
    Dmax = 2 * np.sqrt(A_front / np.pi) if A_front > 0 else max(L, 1e-3)
    rmax = float(seg["rmax"]); r_core = float(seg["r_core"])
    disks = seg["rotor_disks"]
    pod_r = float(np.mean([d["radius"] for d in disks])) if disks else 0.01
    arm_len = max(rmax - r_core, 1e-4)
    axis = AXES[flow_axis]; la, lb = [i for i in range(3) if i != axis]
    ats = []
    for d in disks:
        cx, cy = d["center_lat"]; th = float(np.arctan2(cy, cx)); r_pod = float(np.hypot(cx, cy))
        ats.append(_arm_thickness(drone, r_core, r_pod, th, la, lb, max(r_pod - r_core, 1e-3)))
    arm_r = float(np.mean(ats)) if ats else 0.008
    feat = {"n_rotors": int(len(disks)), "rmax": rmax, "r_core": r_core,
            "r_core_ratio": r_core / max(rmax, 1e-6), "arm_len": arm_len, "arm_r": arm_r, "pod_r": pod_r,
            "arm_slender": arm_r / arm_len, "pod_arm": pod_r / arm_len,
            "A_front": A_front, "A_plan": A_plan, "A_wet": A_wet, "vol": vol, "Dmax": float(Dmax),
            "body_fineness": L / max(Dmax, 1e-6), "wet_front": A_wet / max(A_front, 1e-9),
            "plan_front": A_plan / max(A_front, 1e-9)}
    feat.update(_area_dist_features(o))
    return feat


def sample_params(rng) -> dict:
    n = int(rng.choice(ARM_COUNTS))
    body_r = float(rng.uniform(0.015, 0.040))            # body half-width (⊥ flow)
    body_hl = float(rng.uniform(0.030, 0.095))           # body half-length (along flow) → fineness varies
    arm_len = float(rng.uniform(0.055, 0.135))           # arm-tip radius
    arm_r = float(rng.uniform(0.004, 0.011))             # arm (strut) radius
    pod_r = float(rng.uniform(0.012, 0.024))             # motor-pod radius (must stay < half arm spacing)
    pod_hh = float(rng.uniform(0.008, 0.020))            # motor-pod half-height (along flow)
    phase = float(rng.uniform(0.0, np.pi / n))           # X vs + orientation
    # keep pods from overlapping: adjacent spacing = 2*arm_len*sin(pi/n) must exceed ~2.2*pod_r
    min_span = 2.2 * pod_r
    arm_len = max(arm_len, min_span / (2 * np.sin(np.pi / n)))
    return {"n_arms": n, "body_r": body_r, "body_hl": body_hl, "arm_len": arm_len,
            "arm_r": arm_r, "pod_r": pod_r, "pod_hh": pod_hh, "phase": phase}


def build_drone(p: dict) -> trimesh.Trimesh:
    """Build a watertight multirotor from params. Flow/up axis = +z (arms in the xy plane)."""
    parts = []
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([p["body_r"], p["body_r"], p["body_hl"]])      # ellipsoid elongated along z
    parts.append(body)
    n = p["n_arms"]
    for k in range(n):
        th = p["phase"] + 2 * np.pi * k / n
        d = np.array([np.cos(th), np.sin(th), 0.0])
        p0 = d * (0.5 * p["body_r"])                                # start inside the body
        p1 = d * p["arm_len"]                                       # arm tip
        arm = trimesh.creation.cylinder(radius=p["arm_r"], segment=[p0, p1], sections=20)
        parts.append(arm)
        pod = trimesh.creation.cylinder(radius=p["pod_r"], height=2 * p["pod_hh"], sections=28)
        pod.apply_translation(p1)                                   # motor pod at the arm tip, axis ∥ z
        parts.append(pod)
    drone = trimesh.boolean.union([m for m in parts if m.is_watertight])
    return drone


def random_multirotor(rng) -> tuple:
    """Return (watertight drone mesh, params). Retries a few times if a union isn't watertight."""
    for _ in range(6):
        p = sample_params(rng)
        try:
            d = build_drone(p)
            if d is not None and d.is_watertight and d.volume > 0:
                return d, p
        except Exception:
            continue
    return None, None


if __name__ == "__main__":
    from optimaero.drone.segment import segment_multirotor
    rng = np.random.default_rng(0)
    ok = 0
    for i in range(6):
        d, p = random_multirotor(rng)
        if d is None:
            print(f"drone {i}: FAILED to build watertight"); continue
        try:
            seg = segment_multirotor(d, up="z", n_arms=p["n_arms"])
            nr = len(seg["rotor_disks"])
            good = nr == p["n_arms"]
            ok += good
            print(f"drone {i}: n_arms={p['n_arms']} vol={d.volume*1e6:6.1f}cm3 wt={d.is_watertight} "
                  f"-> segmented rotors={nr} {'OK' if good else 'MISMATCH'}")
        except Exception as e:
            print(f"drone {i}: seg error {e}")
    print(f"segmentation matched arm count on {ok}/6")
