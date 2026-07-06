"""Quantify the surrogate-exploitation gap: optimize, then verify the optimizer's shape with
the REAL XFOIL. The gap between surrogate-claimed and XFOIL-true performance is the finding."""
import numpy as np
import aerosandbox as asb

from optimaero.bakeoff.deploy import TrainedSurrogate
from optimaero.requirements import Envelope, DesignRequirement
from optimaero.optimize.inverse_design import optimize

XF = "/Users/skyepstein/OptimAero/tools/xfoil/xfoil"

if __name__ == "__main__":
    surr = TrainedSurrogate()
    env = Envelope(max_thickness=0.12, min_thickness=0.08)
    req = DesignRequirement(Re=1_000_000, objective="max_LD")
    r = optimize(surr, req, env, maxiter=20, seed=0)
    print(f"surrogate claims:  L/D={r.LD:.1f}  a={r.best_alpha:.0f}  Cl={r.Cl:.3f} Cd={r.Cd:.4f}  "
          f"trusted={r.trusted} ood={r.ood}")
    af = asb.Airfoil(coordinates=r.coords)
    res = asb.XFoil(airfoil=af, Re=1e6, mach=0, xfoil_command=XF, timeout=60).alpha([r.best_alpha])
    a = np.atleast_1d(res.get("alpha", []))
    if a.size:
        cl, cd = float(np.atleast_1d(res["CL"])[0]), float(np.atleast_1d(res["CD"])[0])
        print(f"XFOIL truth:       L/D={cl/cd:.1f}  a={r.best_alpha:.0f}  Cl={cl:.3f} Cd={cd:.4f}")
        print(f"=> surrogate over-promised L/D by {r.LD/(cl/cd):.1f}x")
    else:
        print("XFOIL did NOT converge on the optimized shape — itself a red flag that the "
              "optimizer left the physically-sensible region.")
