"""XFOIL generation sweep: (airfoil, alpha, Re, Mach) -> (Cl, Cd, Cm), fidelity='xfoil'.

The ground-truth backbone. XFOIL is driven headlessly through AeroSandbox against the
locally-built binary. Non-converged points are simply absent (every returned row is
converged); `regime_flag` marks low-Re and likely post-stall rows, where XFOIL is
approximate and must never be trusted silently (constitution §5, data-model §1).
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import aerosandbox as asb

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XFOIL_PATH = os.environ.get("OPTIMAERO_XFOIL", os.path.join(_REPO, "tools", "xfoil", "xfoil"))

# Constitution R6 grid (v1 backbone: low-speed, Mach 0 first; Mach axis added later).
ALPHAS = np.round(np.arange(-8.0, 18.0 + 1e-9, 1.0), 3)
RES = [5e4, 1e5, 2e5, 5e5, 1e6]
MACHS = [0.0]

# Per-(airfoil, Re) XFOIL session timeout. A sweep that hasn't converged in this long is
# stuck on a hard alpha; cutting it keeps whatever converged and avoids wasting minutes.
DEFAULT_TIMEOUT = int(os.environ.get("OPTIMAERO_TIMEOUT", "45"))


def regime_flag(alpha: float, Re: float) -> str:
    if Re < 1e5:
        return "low_re"
    if abs(alpha) > 12.0:  # XFOIL is approximate near/after stall
        return "post_stall"
    return "ok"


def sweep_airfoil(name: str, alphas=ALPHAS, Re: float = 1e6, mach: float = 0.0,
                  xfoil_path: str = XFOIL_PATH, timeout: int = 120) -> pd.DataFrame:
    """Run one airfoil at one (Re, Mach) over an alpha list. Returns converged rows only."""
    try:
        af = asb.Airfoil(name)
        r = asb.XFoil(airfoil=af, Re=Re, mach=mach, xfoil_command=xfoil_path,
                      timeout=timeout).alpha(list(alphas))
    except Exception:
        return pd.DataFrame()
    a = np.asarray(r.get("alpha", []), dtype=float)
    if a.size == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "airfoil_id": name,
        "source": "uiuc",
        "fidelity": "xfoil",
        "alpha_deg": a,
        "Re": float(Re),
        "Mach": float(mach),
        "Cl": np.asarray(r["CL"], dtype=float),
        "Cd": np.asarray(r["CD"], dtype=float),
        "Cm": np.asarray(r["CM"], dtype=float),
        "converged": True,
        "regime_flag": [regime_flag(x, Re) for x in a],
    })


def run_sweep(names, res=RES, machs=MACHS, xfoil_path: str = XFOIL_PATH,
              progress: bool = True) -> tuple[pd.DataFrame, dict]:
    """Sweep every (airfoil, Re, Mach). Returns (rows, stats). `stats` reports the
    convergence yield honestly (requested vs returned alpha points)."""
    frames = []
    requested = returned = 0
    total = len(names) * len(res) * len(machs)
    done = 0
    t0 = time.time()
    for name in names:
        for Re in res:
            for mach in machs:
                df = sweep_airfoil(name, Re=Re, mach=mach, xfoil_path=xfoil_path)
                requested += len(ALPHAS)
                returned += len(df)
                if len(df):
                    frames.append(df)
                done += 1
                if progress:
                    print(f"  {done}/{total} sweeps  {time.time() - t0:.0f}s  "
                          f"rows={returned}", end="\r")
    if progress:
        print()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    stats = {
        "sweeps": total, "rows": int(len(out)),
        "alpha_yield": round(returned / requested, 3) if requested else 0.0,
        "seconds": round(time.time() - t0, 1),
    }
    return out, stats


# ------------------------------------------------ parallel full sweep with checkpointing
def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _sweep_airfoil_all(name, res, machs, shard_dir, xfoil_path):
    """Sweep one airfoil across all (Re, Mach); write its shard. Idempotent per airfoil."""
    frames = []
    for Re in res:
        for mach in machs:
            df = sweep_airfoil(name, Re=Re, mach=mach, xfoil_path=xfoil_path)
            if len(df):
                frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # empty shard is still written (as a marker) so a resume does not retry dead airfoils
    out.to_parquet(os.path.join(shard_dir, f"{_safe(name)}.parquet"), index=False)
    return name, int(len(out))


def _worker(args):
    return _sweep_airfoil_all(*args)


def run_full_sweep(names, res=RES, machs=MACHS, shard_dir: str | None = None,
                   xfoil_path: str = XFOIL_PATH, n_workers: int | None = None) -> str:
    """Parallel sweep over `names`, one parquet shard per airfoil. Resumes by skipping
    airfoils that already have a shard (crash-safe)."""
    shard_dir = shard_dir or os.path.join(_REPO, "data", "processed", "xfoil_shards")
    os.makedirs(shard_dir, exist_ok=True)
    todo = [n for n in names
            if not os.path.exists(os.path.join(shard_dir, f"{_safe(n)}.parquet"))]
    print(f"full sweep: {len(names)} airfoils | {len(names) - len(todo)} already done | "
          f"{len(todo)} to go", flush=True)
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 1)
    args = [(n, res, machs, shard_dir, xfoil_path) for n in todo]
    t0, done, rows = time.time(), 0, 0
    with mp.Pool(n_workers) as pool:
        for name, nr in pool.imap_unordered(_worker, args):
            done += 1
            rows += nr
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (len(todo) - done) / rate / 60 if rate > 0 else 0
            print(f"  {done}/{len(todo)}  {name} ({nr})  total_rows={rows}  "
                  f"{el:.0f}s  eta~{eta:.0f}m", flush=True)
    return shard_dir


def concat_shards(shard_dir: str | None = None, out_path: str | None = None):
    """Concatenate all per-airfoil shards into the unified backbone table."""
    shard_dir = shard_dir or os.path.join(_REPO, "data", "processed", "xfoil_shards")
    frames = [pd.read_parquet(f) for f in glob.glob(os.path.join(shard_dir, "*.parquet"))]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = out_path or os.path.join(_REPO, "data", "processed", "xfoil_backbone.parquet")
    df.to_parquet(out_path, index=False)
    return df, out_path


if __name__ == "__main__":  # end-to-end pilot on a diverse handful
    pilot = ["naca0012", "naca4412", "naca2412", "s1223", "e387", "clarky"]
    df, stats = run_sweep(pilot, res=[1e5, 1e6])
    print("pilot stats:", stats)
    print("rows per airfoil:\n", df.groupby("airfoil_id").size())
    # physical spot-check: cambered NACA 4412 near alpha 0 at Re=1e6
    m = df[(df.airfoil_id == "naca4412") & (df.Re == 1e6) & (df.alpha_deg == 0)]
    if len(m):
        print(f"sanity: naca4412 Cl(0,Re=1e6) = {m.Cl.iloc[0]:.3f} (expect ~0.4-0.5)")
    os.makedirs(os.path.join(_REPO, "data", "processed"), exist_ok=True)
    out_path = os.path.join(_REPO, "data", "processed", "pilot_xfoil.parquet")
    df.to_parquet(out_path, index=False)
    print("wrote", out_path)
