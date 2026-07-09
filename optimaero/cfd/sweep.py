"""Parallel CFD sweep driver — run N capped workers concurrently to build the dataset in hours.

Each worker gets its own container caps (OA_CFD_MEM / OA_CFD_CPUS), case dir, seed, and shard file,
and checkpoints after every row (resumable). `merge` concatenates the shards. Sized so N workers fit
the machine safely (e.g. 5 × 3 cpus / 4 GB = 15 cores, 20 GB — well within the host).

Launch:  .venv/bin/python -m optimaero.cfd.sweep <total> [workers]
Merge:   .venv/bin/python -c "from optimaero.cfd.sweep import merge; print(merge())"
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

import pandas as pd

SHARD_DIR = "data/processed/envelope_shards"
MERGED = "data/processed/envelope_cfd_v2.parquet"
ANCHOR_SHARD_DIR = "data/processed/anchor_shards"
ANCHOR_MERGED = "data/processed/envelope_anchor.parquet"
CL_SHARD_DIR = "data/processed/cl_shards"
CL_MERGED = "data/processed/envelope_cl.parquet"


def launch(total: int, workers: int = 5, mem: str = "4g", cpus: str = "3", seed0: int = 1000,
           mode: str = "cd", shard_dir: str = SHARD_DIR):
    os.makedirs(shard_dir, exist_ok=True)
    per = (total + workers - 1) // workers
    env = dict(os.environ, OA_CFD_MEM=mem, OA_CFD_CPUS=cpus,
               PYTHONPATH=os.path.abspath(os.getcwd()))
    procs = []
    for k in range(workers):
        shard = os.path.join(shard_dir, f"shard_{k}.parquet")
        log = open(os.path.join(shard_dir, f"shard_{k}.log"), "a")
        p = subprocess.Popen([sys.executable, "-m", "optimaero.cfd.dataset",
                              str(per), shard, str(seed0 + k), f"/tmp/oa_cfd_w{k}", mode],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
    return procs


def status(shard_dir: str = SHARD_DIR) -> dict:
    rows, conv = 0, 0
    shards = glob.glob(os.path.join(shard_dir, "shard_*.parquet"))
    for f in shards:
        try:
            d = pd.read_parquet(f)
            rows += len(d); conv += int(d["converged"].sum())
        except Exception:
            pass
    return {"rows": rows, "converged": conv, "shards": len(shards)}


def merge(out: str = MERGED, shard_dir: str = SHARD_DIR) -> int:
    frames = []
    for f in glob.glob(os.path.join(shard_dir, "shard_*.parquet")):
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            pass
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out)
    return len(df)


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    mode = sys.argv[3] if len(sys.argv) > 3 else "cd"
    sd = {"anchor": ANCHOR_SHARD_DIR, "cl": CL_SHARD_DIR}.get(mode, SHARD_DIR)
    mem = "6g" if mode == "anchor" else "4g"          # finer mesh (layers) needs more headroom
    cpus = "3" if mode == "anchor" else "2"           # leave room for the co-running sweep
    procs = launch(total, workers, mem=mem, cpus=cpus, mode=mode, shard_dir=sd)
    print(f"launched {workers} {mode} workers toward {total} rows (PIDs {[p.pid for p in procs]})")
    print(f"shards in {sd}/ ; status(shard_dir='{sd}') / merge(out=..., shard_dir='{sd}')")
