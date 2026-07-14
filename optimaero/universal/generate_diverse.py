"""Generate + CFD-label diverse canonical-family shapes to fix the wing/streamlined ranking ceiling.
Reuses benchmarks.py generators with parameter sweeps; EXCLUDES the exact 9 benchmark tuples (leakage
guard). Matches the training JSON schema. Resumable (skips existing). Usage:
    python gen_diverse_v3.py [LIMIT] [WORKERS]
LIMIT=0 -> all. Writes data/processed/xtype_{wing,bluff,bodies}/c_*.json.
"""
import os, sys, json
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor, as_completed
from optimaero.universal import benchmarks as B
from optimaero.cfd.foam import cfd_label, RHO

V = 134.11
np.random.seed(0)

# ---- benchmark tuples to EXCLUDE from training (leakage guard) ----
def _is_benchmark(kind, params):
    x = params
    if kind == "sphere":     return abs(x["r"] - 0.05) < 1e-4
    if kind == "cylinder":   return abs(x["r"] - 0.03) < 1e-4 and abs(x["L"] - 0.12) < 1e-4
    if kind == "cube":       return abs(x["s"] - 0.08) < 1e-4
    if kind == "ahmed":      return abs(x["scale"] - 1.0) < 1e-3 and abs(x["slant"] - 25.0) < 1e-2
    if kind == "streambody": return abs(x["r"] - 0.018) < 1e-4 and abs(x["L"] - 0.11) < 1e-4
    if kind == "wing":       return x["code"] in ("0012", "2412", "4412") and abs(x["c"] - 0.08) < 1e-4 and abs(x["s"] - 0.12) < 1e-4
    if kind == "onera":      return abs(x["rc"] - 0.0806) < 1e-4 and abs(x["taper"] - 0.56) < 1e-3 and abs(x["ss"] - 0.12) < 1e-4
    return False


def _ellipsoid(rx, ry, rz):
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    m.apply_transform(np.diag([rx, ry, rz, 1.0]))
    return m


def build_specs():
    """Return list of (typ, name, kind, params, builder_thunk)."""
    S = []
    i = 0
    def add(typ, kind, params, thunk):
        nonlocal i
        if _is_benchmark(kind, params):
            return
        S.append((typ, f"c_{i:04d}", kind, params, thunk)); i += 1

    # --- WINGS (straight) : the main target type ---
    codes = ["0006", "0008", "0010", "0013", "0015", "0018", "0021", "1408", "1410", "2408",
             "2410", "2411", "2415", "3410", "3412", "4409", "4411", "4415", "4418", "6409", "6412", "5410"]
    sizes = [(0.06, 0.11), (0.07, 0.15), (0.09, 0.10), (0.09, 0.17), (0.11, 0.13)]
    for k, code in enumerate(codes):
        for c, s in [sizes[k % len(sizes)], sizes[(k + 2) % len(sizes)]]:
            add("wing", "wing", {"code": code, "c": c, "s": s},
                (lambda code=code, c=c, s=s: B.naca_wing(code, chord=c, span=s)))

    # --- WINGS (swept/tapered onera-like) ---
    for k in range(16):
        rc = 0.06 + 0.006 * (k % 5)
        taper = 0.4 + 0.08 * (k % 4)
        ss = 0.10 + 0.02 * (k % 3)
        sweep = 15.0 + 6.0 * (k % 5)
        thick = 0.08 + 0.03 * (k % 3)
        add("wing", "onera", {"rc": rc, "taper": taper, "ss": ss, "sweep": sweep, "t": thick},
            (lambda rc=rc, taper=taper, ss=ss, sweep=sweep, thick=thick:
             B.onera_m6(root_chord=rc, taper=taper, semispan=ss, le_sweep_deg=sweep, thick=thick)))

    # --- STREAMLINED BODIES (type=bodies) ---
    for r in (0.012, 0.016, 0.020, 0.024, 0.028):
        for L in (0.08, 0.11, 0.14, 0.17):
            add("bodies", "streambody", {"r": r, "L": L},
                (lambda r=r, L=L: B.streamlined_body(radius=r, length=L)))
    # --- ELLIPSOIDS (type=bodies) : streamwise-stretched -> low drag; sphere-ish -> higher ---
    for k in range(10):
        rx = 0.04 + 0.012 * (k % 5)
        ry = 0.02 + 0.006 * (k % 3)
        rz = 0.02 + 0.006 * ((k + 1) % 3)
        add("bodies", "ellipsoid", {"rx": rx, "ry": ry, "rz": rz},
            (lambda rx=rx, ry=ry, rz=rz: _ellipsoid(rx, ry, rz)))

    # --- BLUFF: spheres, cylinders, boxes, ahmed ---
    for r in (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.030, 0.040, 0.060, 0.070):
        add("bluff", "sphere", {"r": r}, (lambda r=r: B.sphere(radius=r)))
    for r in (0.02, 0.025, 0.035, 0.04):
        for L in (0.09, 0.14):
            add("bluff", "cylinder", {"r": r, "L": L}, (lambda r=r, L=L: B.cylinder_crossflow(radius=r, length=L)))
    for k in range(10):
        lx = 0.05 + 0.01 * (k % 4); ly = 0.05 + 0.012 * (k % 3); lz = 0.05 + 0.012 * ((k + 1) % 3)
        add("bluff", "box", {"lx": lx, "ly": ly, "lz": lz}, (lambda lx=lx, ly=ly, lz=lz: trimesh.creation.box(extents=[lx, ly, lz])))
    for scale in (0.8, 0.9, 1.1, 1.2):
        for slant in (15.0, 20.0, 30.0, 35.0):
            add("bluff", "ahmed", {"scale": scale, "slant": slant},
                (lambda scale=scale, slant=slant: B.ahmed_body(scale=scale, slant_deg=slant)))
    return S


def label_and_save(spec):
    typ, name, kind, params, thunk = spec
    outdir = f"data/processed/xtype_{typ}"
    outpath = f"{outdir}/{name}.json"
    if os.path.exists(outpath):
        return (name, "exists")
    try:
        m = thunk()
        if m is None or not m.is_watertight or m.volume <= 0:
            return (name, f"skip: not watertight ({kind})")
        A_front = float(m.projected([1, 0, 0]).area)
        if A_front <= 0:
            return (name, "skip: zero frontal area")
        A_wet = float(m.area); vol = float(m.volume)
        Dmax = 2 * np.sqrt(A_front / np.pi)
        fineness = float(m.extents[0]) / max(Dmax, 1e-9)
        pts, fi = trimesh.sample.sample_surface(m, 512, seed=0)
        nrm = m.face_normals[fi]
        r = cfd_label(m.copy(), V, alpha_deg=0.0, case_dir=f"/tmp/oa_gen_{name}", refine=4, layers=2)
        drag = r.get("drag") if r else None
        if not drag or drag <= 0 or not np.isfinite(drag):
            return (name, f"skip: cfd fail ({kind})")
        Cd = drag / (0.5 * RHO * V ** 2 * A_front)
        if not (0 < Cd < 6):
            return (name, f"skip: Cd={Cd:.2f} out of range ({kind})")
        d = {"type": typ, "kind": kind, "Cd": float(Cd), "drag": float(drag),
             "points": np.asarray(pts, float).tolist(), "normals": np.asarray(nrm, float).tolist(),
             "A_front": A_front, "A_wet": A_wet, "fineness": fineness, "vol": vol}
        os.makedirs(outdir, exist_ok=True)
        json.dump(d, open(outpath, "w"))
        return (name, f"ok {kind:<10} Cd={Cd:.3f} drag={drag:.1f}")
    except Exception as e:
        return (name, f"ERROR {kind}: {e}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    specs = build_specs()
    if limit:
        specs = specs[:limit]
    print(f"generating {len(specs)} shapes (workers={workers}); by type:",
          {t: sum(1 for s in specs if s[0] == t) for t in ("wing", "bodies", "bluff")})
    done = ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(label_and_save, s): s[1] for s in specs}
        for fut in as_completed(futs):
            name, msg = fut.result()
            done += 1
            if msg.startswith("ok") or msg == "exists":
                ok += 1
            print(f"[{done}/{len(specs)}] {name}: {msg}", flush=True)
    print(f"DONE: {ok}/{len(specs)} labeled ok")
