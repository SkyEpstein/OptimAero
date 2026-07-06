"""Full XFOIL backbone sweep over the UIUC database, then concat shards to one table.

Runs unattended and resumes on restart (per-airfoil shards). Env knobs (for testing/tuning):
  OPTIMAERO_SWEEP_LIMIT   cap number of airfoils (0 = all)
  OPTIMAERO_WORKERS       worker processes (0 = cpu_count-1)
  OPTIMAERO_SHARD_DIR     shard directory override
  OPTIMAERO_OUT           unified output parquet path override
"""
import os

import numpy as np

from optimaero.datasets import uiuc
from optimaero import generate

if __name__ == "__main__":
    names = uiuc.list_airfoils()
    sample = int(os.environ.get("OPTIMAERO_SWEEP_SAMPLE", "0"))
    if sample > 0:  # random seeded sample — representative timing, not the weird first-N
        names = list(np.random.default_rng(0).choice(names, size=sample, replace=False))
    limit = int(os.environ.get("OPTIMAERO_SWEEP_LIMIT", "0"))
    if limit > 0:
        names = names[:limit]
    workers = int(os.environ.get("OPTIMAERO_WORKERS", "0")) or None
    shard_dir = os.environ.get("OPTIMAERO_SHARD_DIR") or None
    out_path = os.environ.get("OPTIMAERO_OUT") or None

    generate.run_full_sweep(names, shard_dir=shard_dir, n_workers=workers)
    df, path = generate.concat_shards(shard_dir=shard_dir, out_path=out_path)
    print(f"DONE: {len(df)} rows, {df.airfoil_id.nunique() if len(df) else 0} airfoils -> {path}")
    if len(df):
        print("regime:", df.regime_flag.value_counts().to_dict())
        print("Re values:", sorted(df.Re.unique().tolist()))
