# OptimAero — Data Card

*Honest inventory (spec AC3). Auto-generated.*

## XFOIL backbone (`fidelity=xfoil`)

- **Rows:** 213,406 | **airfoils:** 2169 | **Re:** [50000, 100000, 200000, 500000, 1000000] | **Mach:** [0.0] (low-speed v1)
- **Alpha yield:** 72.7% of the theoretical grid (non-converged points are absent, not imputed).
- **Regime mix:** ok 140,911 / low_re 43,495 / post_stall 29,000. *post_stall & low_re rows are flagged — XFOIL is approximate there.*
- **Target ranges:** Cl [-1.37, 2.88], Cd [0.0030, 0.437], Cm [-0.41, 0.21].

## Leakage control (airfoil families)

- **Signatures:** 2169 airfoils, 80-dim, orientation-guarded.
- **tau = 0.003** (locked). Nearest-neighbour distance p05=0.0026, median=0.0197.

| tau | families | largest | merged |
|---|---|---|---|
| 0.002 | 2124 | 2 | 45 |
| 0.003 | 2103 | 4 | 66 |
| 0.005 | 2047 | 17 | 122 |
| 0.008 | 1939 | 64 | 230 |

*tau=0.003 captures all known same-shape/different-name twins (max twin distance 0.00195) while the largest family stays tiny — it merges duplicates, not genuine thickness/camber variants.*

## AirfRANS anchor (`fidelity=rans-airfrans`)

- **Rows:** 1,000 RANS sims | Re [2.02e+06, 6.04e+06] | Cl [-0.53, 1.89].
- **Caveats (honest):** Re 2–6M does **not** overlap the XFOIL backbone (≤1M) — it extends coverage, it does not directly validate XFOIL. The mirror lacks per-airfoil geometry and Cm, so it is **not yet training-integrable**.

## Known gaps

- Mach = 0 only (compressibility / prop-tip transonic is future work).
- AirfRANS geometry not yet recovered (high-Re reference only for now).
- post_stall / very-low-Re XFOIL rows are approximate (flagged, not trusted blindly).
