"""Generate an honest data card for the OptimAero datasets (spec AC3).

Reports what we actually have — sources, counts, fidelity mix, convergence yield, family/leakage
stats with tau sensitivity, and known gaps — with no inflation. Writes docs/DATA_CARD.md.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from optimaero.datasets import uiuc
from optimaero import families as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(_REPO, "data", "processed")


def _fmt(n):
    return f"{n:,}"


def generate() -> str:
    lines = ["# OptimAero — Data Card", "", "*Honest inventory (spec AC3). Auto-generated.*", ""]

    # --- XFOIL backbone ---
    bb = pd.read_parquet(os.path.join(PROC, "xfoil_backbone.parquet"))
    reg = bb.regime_flag.value_counts().to_dict()
    lines += [
        "## XFOIL backbone (`fidelity=xfoil`)", "",
        f"- **Rows:** {_fmt(len(bb))} | **airfoils:** {bb.airfoil_id.nunique()} "
        f"| **Re:** {[int(r) for r in sorted(bb.Re.unique())]} "
        f"| **Mach:** {[float(m) for m in sorted(bb.Mach.unique())]} (low-speed v1)",
        f"- **Alpha yield:** {len(bb) / (27 * 5 * 2174):.1%} of the theoretical grid "
        f"(non-converged points are absent, not imputed).",
        f"- **Regime mix:** ok {_fmt(reg.get('ok', 0))} / low_re {_fmt(reg.get('low_re', 0))} "
        f"/ post_stall {_fmt(reg.get('post_stall', 0))}. "
        f"*post_stall & low_re rows are flagged — XFOIL is approximate there.*",
        f"- **Target ranges:** Cl [{bb.Cl.min():.2f}, {bb.Cl.max():.2f}], "
        f"Cd [{bb.Cd.min():.4f}, {bb.Cd.max():.3f}], Cm [{bb.Cm.min():.2f}, {bb.Cm.max():.2f}].",
        "",
    ]

    # --- Family / leakage stats with tau sensitivity ---
    ok, sigs = F.build_signatures(sorted(bb.airfoil_id.unique()), uiuc.load_coordinates)
    stats = F.nn_distance_stats(sigs)
    lines += ["## Leakage control (airfoil families)", "",
              f"- **Signatures:** {len(ok)} airfoils, 80-dim, orientation-guarded.",
              f"- **tau = {F.FAMILY_TAU}** (locked). Nearest-neighbour distance p05={stats['p05']:.4f}, "
              f"median={stats['median']:.4f}.", "",
              "| tau | families | largest | merged |", "|---|---|---|---|"]
    for tau in (0.002, F.FAMILY_TAU, 0.005, 0.008):
        fam = F.cluster(sigs, tau=tau)
        _, counts = np.unique(fam, return_counts=True)
        lines.append(f"| {tau} | {len(np.unique(fam))} | {counts.max()} | {len(ok) - len(np.unique(fam))} |")
    lines += ["", "*tau=0.003 captures all known same-shape/different-name twins "
              "(max twin distance 0.00195) while the largest family stays tiny — it merges "
              "duplicates, not genuine thickness/camber variants.*", ""]

    # --- AirfRANS anchor ---
    ap = os.path.join(PROC, "airfrans_anchor.parquet")
    if os.path.exists(ap):
        ar = pd.read_parquet(ap)
        lines += ["## AirfRANS anchor (`fidelity=rans-airfrans`)", "",
                  f"- **Rows:** {_fmt(len(ar))} RANS sims | Re [{ar.Re.min():.2e}, {ar.Re.max():.2e}] "
                  f"| Cl [{ar.Cl.min():.2f}, {ar.Cl.max():.2f}].",
                  "- **Caveats (honest):** Re 2–6M does **not** overlap the XFOIL backbone (≤1M) — "
                  "it extends coverage, it does not directly validate XFOIL. The mirror lacks "
                  "per-airfoil geometry and Cm, so it is **not yet training-integrable**.", ""]

    lines += ["## Known gaps", "",
              "- Mach = 0 only (compressibility / prop-tip transonic is future work).",
              "- AirfRANS geometry not yet recovered (high-Re reference only for now).",
              "- post_stall / very-low-Re XFOIL rows are approximate (flagged, not trusted blindly).",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    card = generate()
    os.makedirs(os.path.join(_REPO, "docs"), exist_ok=True)
    out = os.path.join(_REPO, "docs", "DATA_CARD.md")
    with open(out, "w") as f:
        f.write(card)
    print(f"wrote {out} ({len(card)} chars)")
