"""Ingest the AirfRANS RANS dataset into the OptimAero unified schema (higher-fidelity anchor).

AirfRANS (Bonnet et al., NeurIPS 2022) = incompressible RANS over NACA 4/5-digit airfoils.
Each simulation has an airfoil geometry, an inlet velocity + angle of attack (-> Reynolds,
Mach ~ 0), and INTEGRATED force coefficients (drag C_D, lift C_L).

============================================================================================
SIZE ASSESSMENT (done BEFORE downloading -- this drove the source choice):
------------------------------------------------------------------------------------------
The `airfrans` PyPI package (airfrans.dataset.download) offers ONLY two monolithic zips,
both far over a 3 GB budget and both requiring flow-field integration to get coefficients:
    * Dataset.zip     = 10,029,067,577 bytes  (~9.34 GiB)  cropped .vtu/.vtp fields
    * OF_dataset.zip  = 71,310,335,730 bytes  (~66.4 GiB)  raw OpenFOAM
  (sizes from HTTP HEAD on data.isir.upmc.fr; the `task` arg of airfrans.dataset.load only
   selects which ALREADY-downloaded sims to load, it does NOT shrink the download; and
   force_coefficient() must integrate the full internal .vtu, so fields can't be skipped.)

INSTEAD we use the PLAID mirror on Hugging Face, which ships the SAME 1000 OpenFOAM RANS
simulations but with the integrated coefficients PRECOMPUTED as scalars, at a fraction of
the size:
    * PLAID-datasets/AirfRANS_remeshed = 610,767,168 bytes (~0.61 GB) in 3 parquet shards
      -> UNDER the 3 GB budget, so this is the DEFAULT source.  (out_scalars = C_D, C_L;
         in_scalars = angle_of_attack [radians], inlet_velocity [m/s]. License: ODbL-1.0.)
    * PLAID-datasets/AirfRANS_original  ~15.9 GB (full fields) -- not needed here.
============================================================================================

Reproducible recipe (default = remeshed HF, ~0.61 GB):
  1. Install:   /Users/skyepstein/OptimAero/.venv/bin/pip install airfrans   # for the pkg path
                (the remeshed path below needs only pyarrow + pandas, already installed.)
  2. Sizes:     python scripts/ingest_airfrans.py --check-size    # HTTP HEAD, no download
  3. Ingest:    python scripts/ingest_airfrans.py --hf-remeshed
                -> downloads 3 parquet shards, extracts scalars, writes
                   data/processed/airfrans_anchor.parquet

Full-package path (ONLY if you accept the ~9.34 GiB download and want NACA-digit ids):
    python scripts/ingest_airfrans.py --force-download --root <dir>
    python scripts/ingest_airfrans.py --parse-package --root <dir> --task scarce

Output schema (matches specs/2026-07-05-data-foundation/data-model.md sec 1, and the on-disk
columns of data/processed/pilot_xfoil.parquet):
  airfoil_id, source, fidelity, alpha_deg, Re, Mach, Cl, Cd, Cm, converged, regime_flag
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import sys

import numpy as np

REPO = "/Users/skyepstein/OptimAero"
OUT_PARQUET = osp.join(REPO, "data", "processed", "airfrans_anchor.parquet")
SCHEMA_COLS = ["airfoil_id", "source", "fidelity", "alpha_deg", "Re", "Mach",
               "Cl", "Cd", "Cm", "converged", "regime_flag"]

# ---- source URLs -------------------------------------------------------------------------
PKG_URLS = {  # the only two the airfrans package exposes (airfrans/dataset.py)
    "Dataset": "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip",
    "OF_dataset": "https://data.isir.upmc.fr/extrality/NeurIPS_2022/OF_dataset.zip",
}
HF_REMESHED_BASE = ("https://huggingface.co/datasets/PLAID-datasets/"
                    "AirfRANS_remeshed/resolve/main/data")
HF_REMESHED_SHARDS = [f"all_samples-{i:05d}-of-00003.parquet" for i in range(3)]

SIZE_STOP_BYTES = 3 * 1024**3  # task rule: stop above ~3 GB

# ---- air properties (airfrans defaults at T = 298.15 K; incompressible) ------------------
T_KELVIN = 298.15
NU = (-3.400747e-6 + 3.452139e-8 * T_KELVIN
      + 1.00881778e-10 * T_KELVIN**2 - 1.363528e-14 * T_KELVIN**3)  # ~1.5498e-5 m^2/s
CHORD = 1.0  # AirfRANS airfoils are unit chord


def human(n: int) -> str:
    return f"{n:,} bytes (~{n/1024**3:.2f} GiB / {n/1000**3:.2f} GB)"


def _head_size(url: str) -> int:
    import urllib.request
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", "0"))


def check_size() -> None:
    print("airfrans package zips (integration required, no coeff-only file):")
    for name, url in PKG_URLS.items():
        n = _head_size(url)
        gate = "OVER 3GB budget" if n > SIZE_STOP_BYTES else "ok"
        print(f"  {name:12s} {human(n)}  [{gate}]")
    print("HF PLAID remeshed shards (coefficients precomputed -- DEFAULT source):")
    total = 0
    for s in HF_REMESHED_SHARDS:
        n = _head_size(f"{HF_REMESHED_BASE}/{s}")
        total += n
        print(f"  {s}  {human(n)}")
    gate = "OVER 3GB budget" if total > SIZE_STOP_BYTES else "ok"
    print(f"  TOTAL remeshed {human(total)}  [{gate}]")


# ---- default path: HF PLAID remeshed (precomputed coefficients) --------------------------
def _download(url: str, dest: str) -> None:
    import urllib.request
    if osp.exists(dest) and osp.getsize(dest) > 0:
        return
    urllib.request.urlretrieve(url, dest)


def ingest_hf_remeshed(cache_dir: str, out: str) -> "pd.DataFrame":
    """Download the 3 remeshed parquet shards, deobj the PLAID samples, pull scalars.

    Each parquet row is a single `sample` binary column = a pickled PLAID dict whose
    ['scalars'] holds {'C_D','C_L','angle_of_attack' (rad),'inlet_velocity' (m/s)}.
    No plaid/CGNS library needed -- the scalars live in a plain dict.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    os.makedirs(cache_dir, exist_ok=True)
    rows = []
    gidx = 0  # stable global simulation index across shards -> airfoil_id
    for shard in HF_REMESHED_SHARDS:
        dest = osp.join(cache_dir, shard)
        _download(f"{HF_REMESHED_BASE}/{shard}", dest)
        table = pq.read_table(dest)
        for raw in table.column("sample").to_pylist():
            sc = pickle.loads(raw)["scalars"]
            sc = {str(k): float(v) for k, v in sc.items()}
            u = sc["inlet_velocity"]
            alpha_deg = float(np.degrees(sc["angle_of_attack"]))  # scalar is in radians
            rows.append(dict(
                airfoil_id=f"airfrans_remeshed_{gidx:04d}",  # 1 unique NACA shape per sim;
                source="airfrans",                            # remeshed source carries no
                fidelity="rans-airfrans",                     # NACA digits, so id = sim idx
                alpha_deg=round(alpha_deg, 4),
                Re=float(u * CHORD / NU),
                Mach=0.0,                       # incompressible RANS by construction
                Cl=float(sc["C_L"]),
                Cd=float(sc["C_D"]),
                Cm=float("nan"),                # AirfRANS provides no moment coefficient
                converged=True,                 # only converged solutions ship
                regime_flag="ok",
            ))
            gidx += 1
    df = _finalize(pd.DataFrame(rows))
    os.makedirs(osp.dirname(out), exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    _report(df, out)
    return df


# ---- optional path: full airfrans package (~9.34 GiB; NACA-digit ids) --------------------
def force_download_package(root: str, file_name: str = "Dataset") -> None:
    import airfrans.dataset as afd
    os.makedirs(root, exist_ok=True)
    afd.download(root=root, file_name=file_name, unzip=True, OpenFOAM=False)


def _naca_id_from_name(name: str) -> str:
    # airfrans sim name = airFoil2D_SST_<vel>_<angle>_<naca digits...>_<param>
    parts = name.split("_")
    digits = parts[4:-1]  # matches simulation.boundary_layer's name.split('_')[4:-1]
    compact = "".join(str(int(float(d))) if float(d).is_integer() else str(d) for d in digits)
    return f"naca{compact}"


def parse_package(root: str, task: str = "scarce", train: bool = True, out: str = OUT_PARQUET):
    """Integrate C_D/C_L from the shipped fields (requires the ~9.34 GiB download)."""
    import json
    import pandas as pd
    from airfrans.simulation import Simulation

    with open(osp.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    taskk = "full" if (task == "scarce" and not train) else task
    split = "train" if train else "test"
    names = manifest[f"{taskk}_{split}"]

    rows = []
    for name in names:
        sim = Simulation(root=root, name=name)
        (cd, _, _), (cl, _, _) = sim.force_coefficient(reference=True)
        u = float(sim.inlet_velocity)
        rows.append(dict(
            airfoil_id=_naca_id_from_name(name),
            source="airfrans", fidelity="rans-airfrans",
            alpha_deg=round(float(sim.angle_of_attack) * 180.0 / np.pi, 4),
            Re=float(u * CHORD / NU), Mach=0.0,
            Cl=float(cl), Cd=float(cd), Cm=float("nan"),
            converged=True, regime_flag="ok",
        ))
    df = _finalize(pd.DataFrame(rows))
    os.makedirs(osp.dirname(out), exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    _report(df, out)
    return df


# ---- shared helpers ----------------------------------------------------------------------
def _finalize(df):
    for col in ("airfoil_id", "source", "fidelity", "regime_flag"):
        df[col] = df[col].astype("string")  # match pilot_xfoil.parquet dtypes
    return df[SCHEMA_COLS]


def _report(df, out) -> None:
    print(f"Wrote {len(df)} rows ({df.airfoil_id.nunique()} unique airfoils) -> {out}")
    print(f"  Cl range: {df.Cl.min():.4f} .. {df.Cl.max():.4f}")
    print(f"  Cd range: {df.Cd.min():.5f} .. {df.Cd.max():.5f}")
    print(f"  Re range: {df.Re.min():.3e} .. {df.Re.max():.3e}")
    print(f"  alpha_deg range: {df.alpha_deg.min():.3f} .. {df.alpha_deg.max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-size", action="store_true", help="HTTP HEAD all sources; no DL")
    ap.add_argument("--hf-remeshed", action="store_true",
                    help="DEFAULT: ingest the 0.61 GB HF PLAID remeshed mirror")
    ap.add_argument("--force-download", action="store_true",
                    help="download the ~9.34 GiB airfrans package zip (overrides budget)")
    ap.add_argument("--parse-package", action="store_true",
                    help="parse an already-downloaded airfrans package dataset")
    ap.add_argument("--root", default=osp.join(REPO, "data", "raw", "airfrans"),
                    help="package download dir / cache dir")
    ap.add_argument("--cache-dir", default=osp.join(REPO, "data", "raw", "airfrans_remeshed"))
    ap.add_argument("--task", default="scarce", choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--test-split", action="store_true")
    ap.add_argument("--out", default=OUT_PARQUET)
    args = ap.parse_args()

    if args.check_size:
        check_size()
    if args.force_download:
        force_download_package(args.root)
    if args.parse_package:
        parse_package(args.root, task=args.task, train=not args.test_split, out=args.out)
    if args.hf_remeshed:
        ingest_hf_remeshed(args.cache_dir, args.out)
    if not any([args.check_size, args.force_download, args.parse_package, args.hf_remeshed]):
        ap.print_help()


if __name__ == "__main__":
    main()
