"""Drone-form CFD dataset generator (for the drone surrogate).

Samples the additive-treatment space [tail_len, chord×rmax, thick] of a base drone, builds each form
(airfoil_arms + add_tail — the exact geometry the optimizer searches), runs real CFD, and records the
form's geometric + area-rule features (flow along +x, the same recipe as the envelope dataset MINUS the
envelope-only params, PLUS the 3 treatment knobs and base-drone descriptors) alongside the drag/Cd label.

Resumable and sharded: each completed form is written to shard_dir/form_<i>.json; a rerun skips indices
already on disk. `merge()` concatenates the shards into a parquet for training.

Run:  python -m optimaero.drone.dataset <base_stl> <n> [shard_dir]
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import trimesh

from optimaero.shapeopt.optimize import load_shape, _flow_rotation
from optimaero.cfd.dataset import NU, speed_regime, _area_dist_features
from optimaero.cfd.foam import cfd_label, RHO
from optimaero.drone.segment import segment_multirotor
from optimaero.drone.optimize import _build, additive_ok, _build_hd, LO_HD, HI_HD, KNOBS_HD
from optimaero.drone.generator import random_multirotor, drone_descriptors

# the treatment search space (matches optimize_drone's knobs): tail_len, chord×rmax, thick
LO = np.array([0.0, 0.9, 0.5])
HI = np.array([2.2, 2.6, 1.1])

# features the drone surrogate trains on (geometric + area-rule + the 3 knobs + base-drone descriptors);
# deliberately EXCLUDES the envelope-only grow/nose_frac/tail_frac/round_exp.
DRONE_FEATURES = ["fineness", "A_front", "A_plan", "A_wet", "Dmax", "vol", "wet_front", "plan_front",
                  "tail_len", "chord", "thick", "rmax", "r_core", "n_rotors",
                  "Re", "Mach", "alpha_deg",
                  "prismatic", "x_maxarea", "area_smooth", "base_area", "nose_area"]


def drone_form_features(o: trimesh.Trimesh, seg: dict, params, V: float, alpha: float) -> dict:
    """Features of a drone form. `o` must already be rotated so the flow runs along +x (as CFD sees it),
    matching the envelope feature convention so the shared geometric features are directly comparable."""
    ext = np.asarray(o.bounding_box.extents, float)
    L = float(ext[0])
    try:
        A_front = float(o.projected([1, 0, 0]).area)
    except Exception:
        A_front = float(ext[1] * ext[2])
    try:
        A_plan = float(o.projected([0, 0, 1]).area)
    except Exception:
        A_plan = float(ext[0] * ext[1])
    Dmax = 2 * np.sqrt(A_front / np.pi) if A_front > 0 else max(L, 1e-3)
    A_wet = float(o.area)
    tail_len, chord, thick = float(params[0]), float(params[1]), float(params[2])
    feat = {"L": L, "A_front": A_front, "A_plan": A_plan, "A_wet": A_wet,
            "Dmax": float(Dmax), "fineness": L / max(Dmax, 1e-6), "vol": float(o.volume),
            "wet_front": A_wet / max(A_front, 1e-9), "plan_front": A_plan / max(A_front, 1e-9),
            "tail_len": tail_len, "chord": chord, "thick": thick,
            "rmax": float(seg["rmax"]), "r_core": float(seg["r_core"]),
            "n_rotors": int(len(seg["rotor_disks"])),
            "V": V, "Re": V * L / NU, "Mach": V / 343.0, "alpha_deg": alpha,
            "speed_regime": speed_regime(V)}
    feat.update(_area_dist_features(o))
    return feat


def sample_forms(n: int, seed: int = 0) -> np.ndarray:
    """Latin-hypercube-ish sample of the treatment space; a handful of anchors pin the corners."""
    rng = np.random.default_rng(seed)
    s = LO + (HI - LO) * rng.random((n, 3))
    if n >= 1:
        s[0] = [0.0, 0.0, 1.0]                       # bare drone (no treatment)
    anchors = [[2.2, 2.6, 0.5], [2.2, 0.9, 0.5], [0.0, 2.6, 0.5], [1.4, 2.4, 0.6]]
    for j, a in enumerate(anchors, start=1):
        if j < n:
            s[j] = a
    return s


def _one(i, drone, seg, flow_axis, params, V, alpha, shard_dir):
    out = os.path.join(shard_dir, f"form_{i}.json")
    if os.path.exists(out):
        return "skip"
    rec = {"i": int(i), "tail_len": float(params[0]), "chord": float(params[1]),
           "thick": float(params[2]), "V": float(V), "alpha_deg": float(alpha)}
    try:
        m = drone if (params[0] <= 0 and params[1] <= 0) else _build(drone, seg, flow_axis, params)
        o = m.copy(); o.apply_transform(_flow_rotation(flow_axis))
        feat = drone_form_features(o, seg, params, V, alpha)
        r = cfd_label(o, V, alpha_deg=alpha, case_dir=f"/tmp/oa_droneform_{i}", refine=4, layers=2)
        rec.update(feat)
        rec.update({"drag": r.get("drag"), "lift": r.get("lift"), "Cd": r.get("Cd"),
                    "Cl": r.get("Cl"), "converged": bool(r.get("converged")),
                    "additive_ok": bool(additive_ok(m, drone))})
    except Exception as e:  # noqa
        rec.update({"drag": None, "Cd": None, "converged": False, "error": str(e)[:160]})
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, out)                              # atomic — a killed run never leaves a half file
    return "done" if rec.get("converged") else "fail"


def generate(base_stl: str, n: int, shard_dir: str = "data/processed/drone_form_shards",
             flow_axis: str = "z", V: float = 134.11, alpha: float = 0.0, units: str = "mm",
             n_arms: int = 4, workers: int = 5, seed: int = 0, progress=None):
    """Generate n drone-form CFD samples (resumable). Writes one json per form into shard_dir."""
    os.makedirs(shard_dir, exist_ok=True)
    drone = load_shape(base_stl, units=units)
    seg = segment_multirotor(drone, up=flow_axis, n_arms=n_arms)
    samples = sample_forms(n, seed=seed)
    done = {"n": 0}
    lock_free_counter = []  # noqa - counting via list append is atomic under GIL for our purposes

    def run(i):
        st = _one(i, drone, seg, flow_axis, samples[i], V, alpha, shard_dir)
        lock_free_counter.append(1)
        if progress:
            progress(len(lock_free_counter), n, st)
        return st

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run, range(n)))
    return merge(shard_dir)


def merge(shard_dir: str = "data/processed/drone_form_shards",
          out_path: str = "data/processed/drone_form.parquet") -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "form_*.json"))):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(out_path, index=False)
    return df


# ----- MULTI-DRONE dataset (for the GENERAL surrogate) -----------------------------------------------

def _treatment_samples(m: int, rng, lo, hi, anchor) -> np.ndarray:
    s = lo + (hi - lo) * rng.random((m, len(lo)))
    if m >= 1:
        s[0] = anchor                                 # a solid mid-treatment anchor per drone
    return s


def _multi_one(key, drone, seg, desc, flow_axis, params, V, alpha, shard_dir, is_bare, build_fn, names):
    out = os.path.join(shard_dir, f"{key}.json")
    if os.path.exists(out):
        return "skip"
    rec = {"drone_id": key.split("_")[0], "is_bare": bool(is_bare), "V": float(V), "alpha_deg": float(alpha)}
    rec.update(desc)                                  # drone-shape descriptors (constant per drone)
    if is_bare:
        rec.update({n: 0.0 for n in names})
    else:
        rec.update({n: float(v) for n, v in zip(names, params)})
    try:
        m = drone if is_bare else build_fn(drone, seg, flow_axis, params)
        o = m.copy(); o.apply_transform(_flow_rotation(flow_axis))
        r = cfd_label(o, V, alpha_deg=alpha, case_dir=f"/tmp/oa_md_{key}", refine=4, layers=2)
        q = 0.5 * RHO * V ** 2
        cda = (r["drag"] / q) if (r.get("drag") and r["drag"] > 0) else None
        rec.update({"drag": r.get("drag"), "cda": cda, "Cd": r.get("Cd"),
                    "converged": bool(r.get("converged")),
                    "additive_ok": bool(is_bare or additive_ok(m, drone))})
    except Exception as e:  # noqa
        rec.update({"drag": None, "cda": None, "converged": False, "error": str(e)[:150]})
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, out)
    return "done" if rec.get("converged") else "fail"


def generate_multi(n_drones: int, m_treatments: int = 18,
                   shard_dir: str = "data/processed/multidrone_shards", flow_axis: str = "z",
                   V: float = 134.11, alpha: float = 0.0, workers: int = 5, seed: int = 0,
                   hd: bool = False, progress=None):
    """Generate a multi-drone × treatment CFD dataset for the GENERAL surrogate. Each drone is randomly
    synthesized (deterministic per index → resume-safe), CFD'd bare + across treatments; rows carry the
    drone's shape descriptors + treatment knobs + drag-area (cda). Sharded, resumable.
    hd=True uses the 6-knob expanded treatment space (adds nose fairing + tail/nose shape knobs)."""
    if hd:
        lo, hi, names, build_fn = LO_HD, HI_HD, KNOBS_HD, _build_hd
        anchor = np.array([1.6, 1.0, 1.6, 0.7, 0.8, 1.0])          # mid form with a nose fairing
    else:
        lo, hi, names, build_fn = LO, HI, ["tail_len", "chord", "thick"], _build
        anchor = np.array([1.6, 1.6, 0.7])
    os.makedirs(shard_dir, exist_ok=True)
    tasks = []
    for did in range(n_drones):
        drone, p = random_multirotor(np.random.default_rng(seed * 10000 + did))
        if drone is None:
            continue
        try:
            seg = segment_multirotor(drone, up=flow_axis, n_arms=p["n_arms"])
            if not seg.get("rotor_disks"):
                continue
            desc = drone_descriptors(drone, seg, flow_axis)
        except Exception:
            continue
        dkey = f"d{did:03d}"
        trs = _treatment_samples(m_treatments, np.random.default_rng(seed * 77 + did), lo, hi, anchor)
        tasks.append((f"{dkey}_bare", drone, seg, desc, None, True))
        for j, t in enumerate(trs):
            tasks.append((f"{dkey}_t{j:02d}", drone, seg, desc, t, False))
    total = len(tasks); done = []

    def run(task):
        key, drone, seg, desc, params, is_bare = task
        st = _multi_one(key, drone, seg, desc, flow_axis, params, V, alpha, shard_dir, is_bare,
                        build_fn, names)
        done.append(1)
        if progress:
            progress(len(done), total, st)
        return st

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run, tasks))
    out = shard_dir.replace("_shards", "") + ".parquet"      # e.g. .../multidrone_hd_shards -> multidrone_hd.parquet
    return merge_multi(shard_dir, out)


def merge_multi(shard_dir: str = "data/processed/multidrone_shards",
                out_path: str = "data/processed/multidrone.parquet") -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "d*.json"))):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("multi", "multihd"):
        hd = sys.argv[1] == "multihd"
        n_drones = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        m = int(sys.argv[3]) if len(sys.argv) > 3 else 18
        sd = sys.argv[4] if len(sys.argv) > 4 else "data/processed/multidrone_shards"
        df = generate_multi(n_drones, m_treatments=m, shard_dir=sd, hd=hd,
                            progress=lambda k, tot, st: print(f"  {k}/{tot} {st}", flush=True))
        ok = int((df["converged"] == True).sum()) if "converged" in df else 0   # noqa: E712
        nd = df["drone_id"].nunique() if "drone_id" in df else 0
        print(f"generated {len(df)} rows across {nd} drones ({ok} converged), hd={hd} -> {sd}")
    else:
        base = sys.argv[1] if len(sys.argv) > 1 else "/Users/skyepstein/Downloads/High-Speed Drone (Model).stl"
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        sd = sys.argv[3] if len(sys.argv) > 3 else "data/processed/drone_form_shards"
        df = generate(base, n, shard_dir=sd,
                      progress=lambda k, tot, st: print(f"  {k}/{tot} {st}", flush=True))
        ok = int((df["converged"] == True).sum()) if "converged" in df else 0   # noqa: E712
        print(f"generated {len(df)} rows ({ok} converged) -> data/processed/drone_form.parquet")
