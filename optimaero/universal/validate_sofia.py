"""Reproducible real-CAD validation: predict + optimize a third-party NASA airframe.

Downloads the public-domain NASA SOFIA (Boeing 747SP) 3D-print parts from NASA-3D-Resources, boolean-unions
them into one airframe (watertight IN-MEMORY — note: STL export loses connectivity), scales to the training
size regime, then:
  1) predicts drag from geometry alone with the frozen universal surrogate (no CFD), and
  2) CFD-verifies the prediction, and
  3) runs the universal streamlining optimizer (CFD-verified, never-worse).

HONESTY: the CFD is this project's own COARSE steady RANS (stated ±30–50%); a frontal-area RANS Cd≈0.4 at
0.15 m / 134 m/s is NOT a real 747 cruise Cd (~0.03, wing-area, compressible). This validates "does the model
predict what our RANS would say for a real unseen shape," on n=1. Run:  python -m optimaero.universal.validate_sofia
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request

import numpy as np
import trimesh

RAW = "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/3D%20Printing/SOFIA"
PARTS = ["Nose section.stl", "Fuselage top.stl", "Telescope cavity closed.stl",
         "Tail section.stl", "Left wing.stl", "Right wing.stl"]        # external aero surfaces only
V = 134.11
LENGTH_M = 0.15                                                        # scale to the ~0.1 m training regime


def download(cache_dir: str) -> list[str]:
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for name in PARTS:
        dst = os.path.join(cache_dir, name)
        if not os.path.exists(dst):
            tmp = dst + ".part"                                   # atomic: never cache a truncated file
            try:
                urllib.request.urlretrieve(f"{RAW}/{urllib.parse.quote(name)}", tmp)
                os.replace(tmp, dst)
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise RuntimeError(f"could not download '{name}' from {RAW} — NASA-3D-Resources may have "
                                   f"moved it (RAW pins the 'master' branch, not a commit SHA): {e}") from e
        paths.append(dst)
    return paths


def assemble(paths: list[str]) -> trimesh.Trimesh:
    """Boolean-union the parts into one airframe (watertight in-memory), flow +x (nose already at min x),
    scaled to LENGTH_M. Do NOT round-trip through STL — that drops the union's connectivity."""
    u = trimesh.boolean.union([trimesh.load(p, force="mesh") for p in paths])
    u.apply_translation(-u.centroid)
    u.apply_scale(LENGTH_M / u.extents[0])
    return u


def main(cache_dir: str = "/tmp/oa_sofia_parts"):
    from optimaero.universal.surrogate import load, available
    from optimaero.cfd.foam import cfd_label, RHO

    u = assemble(download(cache_dir))
    try:
        A_front = float(u.projected([1, 0, 0]).area)             # guarded like surrogate.py / foam.py
    except Exception:
        A_front = float(u.extents[1] * u.extents[2])             # bbox fallback
    print(f"airframe: watertight(in-mem)={u.is_watertight} faces={len(u.faces)} "
          f"length={u.extents[0]:.3f} m  A_front={A_front*1e4:.2f} cm^2")

    if not available():
        print("universal surrogate artifact missing — train it first"); return
    _, cd_pred, conf = load().predict_drag(u, V, flow_axis="x")
    print(f"\n1) PREDICTED (geometry only, no CFD): Cd={cd_pred:.3f}  conf_err={conf:.3f}")

    r = cfd_label(u.copy(), V, alpha_deg=0.0, case_dir="/tmp/oa_sofia_val", refine=4, layers=2)
    if not (r and r.get("drag") and r["drag"] > 0):
        print("   CFD did not converge"); return
    cd_cfd = r["drag"] / (0.5 * RHO * V ** 2 * A_front)
    print(f"   CFD (coarse RANS): Cd={cd_cfd:.3f}  ->  prediction error {100*(cd_pred-cd_cfd)/cd_cfd:+.1f}% (Cd)")
    print("   (n=1; lead with Cd not Newtons; this matches our OWN coarse RANS, not real-world drag)")

    from optimaero.universal.optimize import optimize_universal
    print("\n2) OPTIMIZE (universal streamlining, CFD-verify top-3, never-worse)...")
    res = optimize_universal(u, V, flow_axis="x", alpha_deg=0.0, aggressiveness=0.5,
                             n_search=400, top_k=3, workers=3, seed=0)
    if res.baseline_ok and res.improved:
        red = 100 * (res.mb["drag"] - res.ma["drag"]) / res.mb["drag"]
        print(f"   CFD drag {res.mb['drag']:.3f} N -> {res.ma['drag']:.3f} N  ({red:.1f}% lower)  params={res.params}")
    elif res.baseline_ok:
        print("   no deformation beat it -> returned unchanged (never-worse)")
    else:
        print("   baseline CFD did not converge — cannot report a reduction")


if __name__ == "__main__":
    main()
