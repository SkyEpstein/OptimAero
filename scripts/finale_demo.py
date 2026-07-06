"""OptimAero end-to-end demo on the REAL trained model (no placeholder).

Flow: envelope + requirements → inverse-design optimizer (TrainedSurrogate) → optimized
airfoil → BEMT propeller performance → CAD STEP/STL export. Confidence-gated throughout —
this is the whole product running on OUR surrogate + OUR confidence model.
"""
import os

import numpy as np

from optimaero.bakeoff.deploy import TrainedSurrogate
from optimaero.requirements import Envelope, DesignRequirement
from optimaero.optimize.inverse_design import optimize_verified, _score_shape

_XFOIL = "/Users/skyepstein/OptimAero/tools/xfoil/xfoil"
from optimaero.datasets import uiuc
from optimaero.cad import io as cad
from optimaero.physics import bemt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    surr = TrainedSurrogate()
    print(f"SURROGATE: {surr.name}\n" + "=" * 60)

    # 1) Section aero with confidence
    print("\n[1] Section aero — naca4412 @ Re=1e6 (our model + our confidence):")
    for a in (0, 4, 8):
        p = surr.predict(uiuc.load_coordinates("naca4412"), a, 1e6)
        print(f"    a={a:>2}°  Cl={p.Cl:+.3f}±{p.Cl_err:.3f}  Cd={p.Cd:.4f}±{p.Cd_err:.4f}  "
              f"trusted={p.trusted} ood={p.ood}")

    # 2) Envelope-constrained inverse design — XFOIL-VERIFIED (surrogate searches, XFOIL confirms)
    print("\n[2] Inverse design — max L/D @ Re=1e6, envelope t/c ∈ [0.08, 0.12] (XFOIL-verified):")
    env = Envelope(max_thickness=0.12, min_thickness=0.08)
    req = DesignRequirement(Re=1_000_000, objective="max_LD")
    bm, *_ = _score_shape(surr, uiuc.load_coordinates("naca2412"), req)
    r, cands = optimize_verified(surr, req, env, xfoil_path=_XFOIL, n_seeds=3, maxiter=20)
    print(f"    baseline naca2412 (surrogate):  L/D≈{bm:.0f}")
    print(f"    surrogate claims for optimum:   L/D={r.LD:.0f} @ a={r.best_alpha:.0f}° "
          f"(t/c={r.thickness:.3f}, feasible={r.feasible})")
    if r.xfoil_LD:
        print(f"    >>> XFOIL-VERIFIED optimum:     L/D={r.xfoil_LD:.0f}  "
              f"Cl={r.xfoil_Cl:.3f} Cd={r.xfoil_Cd:.4f}  "
              f"(surrogate over-promised {r.LD / r.xfoil_LD:.1f}× — verification caught it)")

    # 3) Propeller performance (optimized section as the blade) via BEMT
    print("\n[3] Propeller (BEMT) — optimized section, R=0.15m, 2 blades, 5000 rpm:")
    prop = bemt.Propeller(radius=0.15, n_blades=2, section_coords=r.coords,
                          chord=lambda x: 0.02, twist_deg=lambda x: 28 - 18 * x, hub_frac=0.15)
    for V in (5.0, 10.0):
        pr = bemt.solve(surr, prop, V=V, rpm=5000, n_elements=12)
        eff = f"{pr.efficiency:.3f}" if np.isfinite(pr.efficiency) else "n/a"
        print(f"    V={V:>4}m/s J={pr.J:.2f}: thrust={pr.thrust:+.2f}N  eff={eff}  "
              f"frac_trusted={pr.frac_trusted:.2f} any_ood={pr.any_ood}")

    # 4) CAD export of the optimized section
    print("\n[4] CAD export (neutral formats):")
    out = os.path.join(_REPO, "data", "processed", "finale")
    paths = cad.export_airfoil(r.coords, out, name="optimaero_optimized", chord=0.2, span=1.0)
    for k, p in paths.items():
        print(f"    {k.upper()}: {p} ({os.path.getsize(p):,} bytes)")

    print("\n" + "=" * 60 + "\nEnd-to-end pipeline ran on the trained model. Placeholder gone.")
