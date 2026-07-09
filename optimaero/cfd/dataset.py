"""Generate a CFD-labeled dataset over the envelope parameter space, spanning speed regimes.

Each row = an envelope's geometric/silhouette features + condition (V, alpha, Re, Mach, speed_regime)
labeled with capped-CFD drag/lift. Checkpointed after every row (resumable). Feeds the bake-off +
confidence model. Speed regime is recorded so the bake-off can train per-regime models (Sky's ask —
one model per speed band, since a single model strains across the full Re/Mach range).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import trimesh

from optimaero.shapeopt.envelope import build_envelope, _profiles
from optimaero.cfd.foam import cfd_label

NU = 1.5e-5
SPEEDS = [15.0, 25.0, 40.0, 70.0, 100.0, 134.0]     # low → high (covers Re and Mach regimes)


def speed_regime(V: float) -> str:
    return "low" if V < 30 else ("mid" if V < 80 else "high")


def sample_base(rng, asymmetric_bias: bool = False) -> trimesh.Trimesh:
    """Diverse base geometries (enriched, 2026-07-07) so the surrogate spans a rich shape space.
    asymmetric_bias favors the asymmetric (box+lump) type for the lift-tailored dataset."""
    kind = 5 if (asymmetric_bias and rng.random() < 0.6) else int(rng.integers(0, 6))
    if kind == 0:                                        # box, varied aspect
        m = trimesh.creation.box(extents=[rng.uniform(0.10, 0.5), rng.uniform(0.04, 0.3),
                                          rng.uniform(0.04, 0.3)])
    elif kind == 1:                                      # ellipsoid (scaled sphere)
        m = trimesh.creation.icosphere(radius=0.1)
        m.apply_scale([rng.uniform(0.5, 3.0), rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.5)])
    elif kind == 2:                                      # cylinder
        m = trimesh.creation.cylinder(radius=float(rng.uniform(0.04, 0.15)),
                                      height=float(rng.uniform(0.12, 0.5)))
    elif kind == 3:                                      # cone
        m = trimesh.creation.cone(radius=float(rng.uniform(0.05, 0.16)),
                                  height=float(rng.uniform(0.15, 0.5)))
    elif kind == 4:                                      # capsule
        m = trimesh.creation.capsule(radius=float(rng.uniform(0.03, 0.1)),
                                     height=float(rng.uniform(0.1, 0.4)))
    else:                                                # box + offset lump (vertical camber → lift)
        a = trimesh.creation.box(extents=[rng.uniform(0.15, 0.45), rng.uniform(0.06, 0.2),
                                          rng.uniform(0.06, 0.2)])
        b = trimesh.creation.icosphere(radius=float(rng.uniform(0.03, 0.09)))
        b.apply_translation([rng.uniform(-0.1, 0.1), 0.0, rng.uniform(0.02, 0.09)])
        m = trimesh.util.concatenate([a, b])
    # NO random 3D rotation — it makes the lift direction arbitrary and Cl unlearnable (verifier
    # catch, 2026-07-07). Keep a CONSISTENT frame (flow +x, up +z) so lift has a consistent sign;
    # diversity comes from the 6 shape types + aspect ratios + params + angle of attack.
    return m


def _area_dist_features(mesh: trimesh.Trimesh, n: int = 14) -> dict:
    """Longitudinal cross-section-area distribution A(x) descriptors (area-rule shape) — these
    distinguish geometries that the coarse fineness/area features alias to the same Cd."""
    v = mesh.vertices
    x = v[:, 0]
    x0, x1 = float(x.min()), float(x.max())
    L = max(x1 - x0, 1e-9)
    edges = np.linspace(x0, x1, n + 1)
    A = np.zeros(n)
    for i in range(n):
        sel = (x >= edges[i]) & (x < edges[i + 1] if i < n - 1 else x <= edges[i + 1])
        if int(sel.sum()) >= 3:
            A[i] = np.pi / 4.0 * float(np.ptp(v[sel, 1])) * float(np.ptp(v[sel, 2]))
    Amax = float(A.max()) if A.max() > 0 else 1e-9
    dA = np.diff(A)
    return {"prismatic": float(mesh.volume) / (Amax * L) if Amax > 0 else 0.0,
            "x_maxarea": float(int(A.argmax()) / (n - 1)),          # where the body is fattest
            "area_smooth": float(np.std(dA) / Amax),                # area-rule smoothness
            "base_area": float(A[-1] / Amax),                       # aft (base-drag) area
            "nose_area": float(A[0] / Amax)}                        # fore area


def envelope_features(env: trimesh.Trimesh, params, V: float, alpha: float) -> dict:
    ext = np.asarray(env.bounding_box.extents, float)
    L = float(ext[0])                                # flow is along +x for envelopes
    try:
        A_front = float(env.projected([1, 0, 0]).area)
    except Exception:
        A_front = float(ext[1] * ext[2])
    try:
        A_plan = float(env.projected([0, 0, 1]).area)
    except Exception:
        A_plan = float(ext[0] * ext[1])
    Dmax = 2 * np.sqrt(A_front / np.pi) if A_front > 0 else max(L, 1e-3)
    grow, nose, tail, rnd = params[:4]
    camber = float(params[4]) if len(params) > 4 else 0.0
    A_wet = float(env.area)
    feat = {"L": L, "A_front": A_front, "A_plan": A_plan, "A_wet": A_wet,
            "Dmax": float(Dmax), "fineness": L / max(Dmax, 1e-6),
            "vol": float(env.volume), "wet_front": A_wet / max(A_front, 1e-9),
            "plan_front": A_plan / max(A_front, 1e-9),   # planform/frontal asymmetry
            "grow": float(grow), "nose_frac": float(nose), "tail_frac": float(tail),
            "round_exp": float(rnd), "camber": camber, "V": V, "Re": V * L / NU,
            "Mach": V / 343.0, "alpha_deg": alpha, "speed_regime": speed_regime(V)}
    feat.update(_area_dist_features(env))            # richer shape descriptors (area-rule)
    return feat


def generate(n: int, out_path: str, seed: int = 0, resume: bool = True,
             case_dir: str = "/tmp/oa_cfd_ds", mode: str = "cd") -> str:
    """mode='cd': diverse symmetric bodies, AoA 0-10, mesh refine 3 (drag model).
    mode='cl': cambered + asymmetric bodies, AoA 0-15, finer mesh refine 4 (lift model, Sky's
    tailored-data plan) — camber is the real lift generator so lift becomes strongly learnable."""
    cl = (mode == "cl")
    rng = np.random.default_rng(seed)
    rows = []
    if resume and os.path.exists(out_path):
        rows = pd.read_parquet(out_path).to_dict("records")
    for i in range(len(rows), n):
        base = sample_base(rng, asymmetric_bias=cl)
        prof = _profiles(base)
        camber = float(rng.uniform(0.02, 0.14)) if cl else 0.0
        # nose/tail capped (2026-07-07) so fineness stays realistic (< ~10) — the old wide ranges
        # produced fineness-21 needles with Cd≈0 that were under-resolved noise (~9% of the Cd set).
        params = [rng.uniform(1.0, 1.4), rng.uniform(0.2, 0.9),
                  rng.uniform(0.3, 1.2), rng.uniform(2.0, 8.0), camber]
        env = build_envelope(prof, params)
        V = float(rng.choice(SPEEDS))
        alpha = float(rng.choice([0., 3., 6., 9., 12., 15.] if cl else [0., 2., 4., 6., 8., 10.]))
        feat = envelope_features(env, params, V, alpha)
        lab = cfd_label(env, V, alpha, case_dir=case_dir, refine=4 if cl else 3)
        feat.update({"drag": lab["drag"], "lift": lab["lift"], "Cd": lab["Cd"],
                     "Cl": lab["Cl"], "converged": lab["converged"]})
        rows.append(feat)
        pd.DataFrame(rows).to_parquet(out_path)             # checkpoint every row
        ok = sum(1 for r in rows if r.get("converged"))
        print(f"[{i + 1}/{n}] V={V:.0f} regime={feat['speed_regime']} "
              f"drag={lab['drag']} converged_total={ok}/{len(rows)}", flush=True)
    return out_path


def generate_anchor(n: int, out_path: str, seed: int = 0, resume: bool = True,
                    case_dir: str = "/tmp/oa_cfd_anchor") -> str:
    """Paired CFD per geometry: coarse (refine 3, no layers) AND fine (refine 4 + 3 layers).
    Stores both so we can learn the coarse→fine correction that de-biases the cheap 14k backbone.
    `Cd`/`Cl` = the FINE (truth) labels; `Cd_coarse`/`Cl_coarse` = the cheap-mesh values."""
    rng = np.random.default_rng(seed)
    rows = []
    if resume and os.path.exists(out_path):
        rows = pd.read_parquet(out_path).to_dict("records")
    for i in range(len(rows), n):
        base = sample_base(rng)
        prof = _profiles(base)
        params = [rng.uniform(1.0, 1.4), rng.uniform(0.2, 0.9),
                  rng.uniform(0.3, 1.2), rng.uniform(2.0, 8.0), 0.0]
        env = build_envelope(prof, params)
        V = float(rng.choice(SPEEDS))
        alpha = float(rng.choice([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))
        feat = envelope_features(env, params, V, alpha)
        coarse = cfd_label(env, V, alpha, case_dir=case_dir + "_c", refine=3, layers=0)
        fine = cfd_label(env, V, alpha, case_dir=case_dir + "_f", refine=4, layers=3)
        feat.update({"Cd_coarse": coarse["Cd"], "Cl_coarse": coarse["Cl"],
                     "Cd": fine["Cd"], "Cl": fine["Cl"],
                     "converged": bool(coarse["converged"] and fine["converged"])})
        rows.append(feat)
        pd.DataFrame(rows).to_parquet(out_path)
        print(f"[{i + 1}/{n}] V={V:.0f} Cd coarse={coarse['Cd']} fine={fine['Cd']} "
              f"conv={feat['converged']}", flush=True)
    return out_path


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/envelope_cfd.parquet"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    case_dir = sys.argv[4] if len(sys.argv) > 4 else "/tmp/oa_cfd_ds"
    mode = sys.argv[5] if len(sys.argv) > 5 else "cd"
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    if mode == "anchor":
        generate_anchor(n, out, seed=seed, case_dir=case_dir)
    else:
        generate(n, out, seed=seed, case_dir=case_dir, mode=mode)
