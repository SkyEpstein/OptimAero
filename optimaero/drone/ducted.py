"""Ducted aerodynamic shell for a multirotor.

Grow a streamlined shell AROUND the drone (contains it, volume >= original), cut a duct through the
shell at each rotor so the props still pull air, then UNION the drone back so its geometry is fully
preserved inside. Result: a faired, functional high-speed drone body — never shrinks or moves the
original, only adds material around it (Sky's universal rule).
"""
from __future__ import annotations

import numpy as np
import trimesh

from optimaero.drone.segment import AXES
from optimaero.shapeopt.envelope import optimize_envelope


def ducted_shell(drone: trimesh.Trimesh, seg: dict, V: float, flow_axis: str = "z",
                 aggressiveness: float = 0.6, duct_scale: float = 2.0,
                 max_len_ratio: float = 1.6):
    """Return (result, shell, metrics_before, metrics_after).

    duct_scale = duct radius as a multiple of the motor-pod radius (>1 so the prop/airflow clears the
    motor — a duct exactly pod-sized would be plugged by the pod).
    max_len_ratio caps the shell length to a practical multiple of the drone (a compact ogive, not a
    drag-minimizing needle)."""
    axis = AXES[flow_axis]
    la, lb = [i for i in range(3) if i != axis]

    r = optimize_envelope(drone, V, flow_axis=flow_axis, objective="min_drag",
                          aggressiveness=aggressiveness, max_len_ratio=max_len_ratio)
    shell = r.optimized

    zlo = float(min(drone.vertices[:, axis].min(), shell.vertices[:, axis].min())) - 0.03
    zhi = float(max(drone.vertices[:, axis].max(), shell.vertices[:, axis].max())) + 0.03
    height = zhi - zlo

    cyls = []
    for disk in seg["rotor_disks"]:
        cx, cy = disk["center_lat"]
        rad = disk["radius"] * duct_scale
        cyl = trimesh.creation.cylinder(radius=rad, height=height, sections=48)
        if axis != 2:                                   # cylinder is along z by default
            cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], np.eye(3)[axis]))
        center = np.zeros(3)
        center[la] = cx; center[lb] = cy; center[axis] = (zlo + zhi) / 2
        cyl.apply_translation(center)
        cyls.append(cyl)

    ducts = trimesh.util.concatenate(cyls)
    shell_ducted = shell.difference(ducts)              # open the rotor columns through the shell
    result = shell_ducted.union(drone)                  # add the drone back — fully preserved
    return result, shell, r.metrics_before, r.metrics_after
