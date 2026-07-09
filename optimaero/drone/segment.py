"""Structure-aware segmentation of a multirotor (quad/hex) drone mesh.

Splits the mesh into a central BODY, radial ARMS, and MOTOR PODS, and proposes the ROTOR
KEEP-CLEAR disks that must stay open. Heuristic — meant as a proposal the user confirms/corrects
(hybrid mode). Assumes arms radiate from a central axis (the travel/up axis) with motor pods at
the tips — the standard multirotor layout.
"""
from __future__ import annotations

import numpy as np
import trimesh

AXES = {"x": 0, "y": 1, "z": 2}


def _angular_kmeans(ang: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    cents = np.linspace(-np.pi, np.pi, k, endpoint=False) + float(np.arctan2(np.sin(ang).mean(),
                                                                            np.cos(ang).mean()))
    cents = (cents + np.pi) % (2 * np.pi) - np.pi
    for _ in range(iters):
        d = np.abs(((ang[:, None] - cents[None, :] + np.pi) % (2 * np.pi)) - np.pi)
        a = np.argmin(d, axis=1)
        for j in range(k):
            m = ang[a == j]
            if len(m):
                cents[j] = np.arctan2(np.sin(m).mean(), np.cos(m).mean())
    return cents


def segment_multirotor(mesh: trimesh.Trimesh, up: str = "z", n_arms: int = 4,
                       prop_radius: float = 0.0) -> dict:
    """Return per-vertex labels + rotor keep-clear disks. Labels: 'body', 'armK', 'podK'.
    prop_radius (metres, >0) sets the rotor disk radius explicitly (the prop's swept radius);
    0 = auto-detect from the motor-pod size."""
    axis = AXES[up]
    la, lb = [i for i in range(3) if i != axis]
    V = np.asarray(mesh.vertices, float)
    c = V.mean(0)
    p = V - c
    u = p[:, axis]                                   # height along the travel axis
    r = np.sqrt(p[:, la] ** 2 + p[:, lb] ** 2)       # radius from the central axis
    theta = np.arctan2(p[:, lb], p[:, la])
    rmax = float(r.max()) or 1.0

    # separate the central BODY (small radius, all heights) from the radial SPOKES (large radius).
    # Radius, not height, is the reliable discriminator: arms/pods live far from the central axis.
    r_core = 0.32 * rmax
    far = r > 0.5 * rmax
    u_arm = float(np.percentile(u[far], 95)) if int(far.sum()) >= 8 else float(u.max())

    labels = np.array(["body"] * len(V), dtype=object)     # per-vertex (body/pod; arms have no verts)
    spoke = np.where(r > r_core)[0]
    cents = np.linspace(-np.pi, np.pi, n_arms, endpoint=False)
    if len(spoke):
        cents = _angular_kmeans(theta[spoke], n_arms)
        d = np.abs(((theta[spoke, None] - cents[None, :] + np.pi) % (2 * np.pi)) - np.pi)
        grp = np.argmin(d, axis=1)
        for k, i in enumerate(spoke):
            if r[i] > 0.60 * rmax:                         # only pods are dense enough to have verts
                labels[i] = f"pod{int(grp[k])}"

    # FACE labels — the ARMS are thin spokes that exist only as faces bridging body-core to a pod
    # (there is a vertex gap between body and pod). Classify each face by its mean radius/azimuth.
    F = np.asarray(mesh.faces)
    fr = r[F].mean(1)
    fang = np.arctan2(p[F][:, :, lb].mean(1), p[F][:, :, la].mean(1))
    face_labels = np.array(["body"] * len(F), dtype=object)
    fspoke = np.where(fr > r_core)[0]
    if len(fspoke):
        d2 = np.abs(((fang[fspoke, None] - cents[None, :] + np.pi) % (2 * np.pi)) - np.pi)
        fgrp = np.argmin(d2, axis=1)
        for k, i in enumerate(fspoke):
            kind = "pod" if fr[i] > 0.62 * rmax else "arm"
            face_labels[i] = f"{kind}{int(fgrp[k])}"

    # rotor keep-clear disks: one per arm, centred on the pod. radius = the pod itself (Sky's choice)
    disks = []
    for g in range(n_arms):
        gi = [i for i in spoke if labels[i] == f"pod{g}"]
        if not gi:
            continue
        pv = V[gi]
        cen = [float(pv[:, la].mean()), float(pv[:, lb].mean())]
        pos = float(pv[:, axis].mean())
        rad = (prop_radius if prop_radius > 0 else
               max(float(np.ptp(pv[:, la])), float(np.ptp(pv[:, lb])), 1e-3) / 2.0)
        disks.append({"arm": g, "center_lat": cen, "axis_pos": pos, "radius": float(rad)})

    counts = {k: int((face_labels == k).sum()) for k in set(face_labels)}
    return {"labels": labels, "face_labels": face_labels, "axis": axis, "lat": [la, lb],
            "center": c.tolist(), "u_arm": u_arm, "r_core": r_core, "rmax": rmax,
            "n_arms": n_arms, "rotor_disks": disks, "counts": counts}


def component_of(seg: dict, label: str) -> str:
    """'body' | 'arm' | 'pod' for a vertex label."""
    if label.startswith("pod"):
        return "pod"
    if label.startswith("arm"):
        return "arm"
    return "body"
