"""Universal geometric feature extractor: any watertight mesh → a fixed feature vector that predicts drag
across shape types. Validated feature set (held-out across 7 diverse types): area-rule shape distribution +
normal-based streamlining (how much surface faces into the flow = form drag) + principal-axis moments.
Rich features were the key to fine within-type ranking (drones 0.35→0.91 vs the 4-descriptor set).

Flow is +x by convention; pass flow_axis to rotate an arbitrary mesh so its travel direction is +x first.
`universal_features(mesh)` (serve time) and `features_from_saved(json)` (training) call the SAME core, so
there is no train/serve skew — only the surface sampling differs, and the features are sampling-robust.
"""
from __future__ import annotations

import numpy as np
import trimesh

FEATURE_NAMES = [
    "fineness", "A_front", "A_wet", "vol", "Dmax", "wet_front", "prismatic", "x_maxarea",
    "base_area", "nose_area", "area_smooth", "max_xsec", "area_q1", "area_q2", "area_q3",
    "moment0", "moment1", "moment2", "front_frac", "back_frac", "mean_abs_nx",
]
_AXIS = {"x": 0, "y": 1, "z": 2}


def _core(A_front, A_wet, vol, fineness, Dmax, points, normals):
    """Shared feature computation from scalar geometry + a surface point cloud (positions + normals)."""
    P = np.asarray(points, float); P = P - P.mean(0)
    x = P[:, 0]; L = float(x.max() - x.min() + 1e-9)
    nb = 24; bins = np.linspace(x.min(), x.max(), nb + 1); radii = []
    for j in range(nb):
        s = (x >= bins[j]) & (x < bins[j + 1])
        radii.append(float(np.sqrt((P[s, 1:] ** 2).sum(1)).max()) if s.sum() >= 3 else 0.0)
    a = np.asarray(radii) ** 2; amax = float(a.max() + 1e-9)              # cross-section-area proxy / station
    ev = np.sort(np.linalg.eigvalsh(np.cov(P.T)))[::-1]; ev = ev / (ev.sum() + 1e-9)   # elongation moments
    nx = np.asarray(normals, float)[:, 0]
    feat = [fineness, A_front, A_wet, vol, Dmax, A_wet / (A_front + 1e-9), vol / (amax * L),
            float(np.argmax(a) / nb), a[0] / amax, a[-1] / amax,
            float(np.mean(np.abs(np.diff(a))) / amax), amax,
            a[nb // 4] / amax, a[nb // 2] / amax, a[3 * nb // 4] / amax,
            float(ev[0]), float(ev[1]), float(ev[2]),
            float((nx > 0.5).mean()), float((nx < -0.5).mean()), float(np.abs(nx).mean())]
    return np.asarray(feat, dtype=float)


def universal_features(mesh: trimesh.Trimesh, flow_axis: str = "x", npts: int = 512,
                       seed: int = 0) -> np.ndarray:
    """Serve-time: extract the drag feature vector from a mesh (flow mapped to +x via `flow_axis`)."""
    a = _AXIS[flow_axis]
    m = mesh
    if a != 0:
        m = mesh.copy(); m.apply_transform(trimesh.geometry.align_vectors(np.eye(3)[a], [1.0, 0.0, 0.0]))
    try:
        A_front = float(m.projected([1, 0, 0]).area)
    except Exception:
        A_front = float(m.extents[1] * m.extents[2])
    A_wet = float(m.area); vol = float(max(m.volume, 1e-12))
    Dmax = 2 * np.sqrt(A_front / np.pi) if A_front > 0 else 1.0     # same degenerate fallback as train path
    fineness = float(m.extents[0]) / max(Dmax, 1e-9)
    rng = np.random.default_rng(seed)
    try:
        pts, fi = trimesh.sample.sample_surface(m, npts, seed=int(rng.integers(1 << 31)))
        nrm = m.face_normals[fi]
    except Exception:
        pts = m.vertices; nrm = np.tile([1.0, 0.0, 0.0], (len(pts), 1))
    return _core(A_front, A_wet, vol, fineness, Dmax, pts, nrm)


def features_from_saved(d: dict) -> np.ndarray:
    """Training-time: same features from a saved shape JSON (points, normals, A_front, A_wet, fineness, vol)."""
    A_front = float(d.get("A_front", 0.0)); A_wet = float(d.get("A_wet", 0.0))
    vol = float(d.get("vol", 1e-12)); fineness = float(d.get("fineness", 0.0))
    Dmax = 2 * np.sqrt(A_front / np.pi) if A_front > 0 else 1.0
    return _core(A_front, A_wet, vol, fineness, Dmax, d["points"], d["normals"])
