"""Stage A: aerodynamic-enclosure optimization around a required volume (fast method).

Given a packaging volume (a box the user's components must fit inside), build a streamlined
3D body that PROVABLY contains it and minimize its drag with the fast AeroSandbox solver.
Containment is a hard constraint; the optimizer trades body length (friction drag) against
fineness (form drag) and finds the low-drag streamlined shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import aerosandbox as asb
from scipy.optimize import differential_evolution

RHO_AIR = 1.225   # kg/m^3
NU_AIR = 1.5e-5   # m^2/s


@dataclass
class Box:
    """The packaging volume the enclosure must contain (metres)."""
    lx: float  # length (along the flow axis)
    ly: float  # width
    lz: float  # height


def _profile(xi, p):
    """Cross-section scale f(xi) in [0,1] along the body: elliptical nose (0..p) and
    elliptical tail (p..1), peaking at 1 at xi=p. Smooth teardrop."""
    xi = np.asarray(xi, dtype=float)
    f = np.empty_like(xi)
    nose = xi <= p
    f[nose] = np.sqrt(np.clip(1 - ((p - xi[nose]) / p) ** 2, 0, 1))
    f[~nose] = np.sqrt(np.clip(1 - ((xi[~nose] - p) / (1 - p)) ** 2, 0, 1))
    return f


def build_body(L, w_max, h_max, p, n_sec=25) -> asb.Fuselage:
    """Streamlined body: length L, max half-width/height, max-section location p (fraction)."""
    xi = np.linspace(0, 1, n_sec)
    f = _profile(xi, p)
    xsecs = [asb.FuselageXSec(xyz_c=[float(L * xi[i]), 0, 0],
                              width=float(max(2 * w_max * f[i], 1e-4)),
                              height=float(max(2 * h_max * f[i], 1e-4)))
             for i in range(n_sec)]
    return asb.Fuselage(name="enclosure", xsecs=xsecs)


def containment_margin(L, w_max, h_max, p, box: Box, box_x0) -> float:
    """Min clearance (m) between the body's cross-section and the box over the box's extent.
    >= 0 means the box is fully contained."""
    x = np.linspace(box_x0, box_x0 + box.lx, 15)
    f = _profile(x / L, p)
    return float(min((w_max * f - box.ly / 2).min(), (h_max * f - box.lz / 2).min()))


def drag(body: asb.Fuselage, V, rho=RHO_AIR) -> float:
    """Drag force [N] at airspeed V via the fast AeroSandbox buildup (inviscid + friction)."""
    ap = asb.Airplane(fuselages=[body])
    cd = float(asb.AeroBuildup(ap, asb.OperatingPoint(velocity=V, alpha=0)).run()["CD"])
    return 0.5 * rho * V ** 2 * ap.s_ref * cd


@dataclass
class EnclosureResult:
    L: float
    w_max: float
    h_max: float
    p: float
    box_x0: float
    drag: float
    contains: bool
    fineness: float  # L / max diameter — the streamlining ratio
    body: asb.Fuselage


def optimize_enclosure(box: Box, V: float, maxiter: int = 30, seed: int = 0) -> EnclosureResult:
    by, bz = box.ly / 2, box.lz / 2
    bounds = [(box.lx * 1.5, box.lx * 8),  # L
              (by, box.ly * 3),            # w_max
              (bz, box.lz * 3),            # h_max
              (0.2, 0.5),                  # p (max-section location)
              (0.0, 1.0)]                  # box position fraction within (L - lx)

    def objective(x):
        L, w_max, h_max, p, frac = x
        box_x0 = frac * max(L - box.lx, 0.0)
        m = containment_margin(L, w_max, h_max, p, box, box_x0)
        if m < 0:                       # HARD containment constraint
            return 1e3 + 1e3 * (-m)
        try:
            return drag(build_body(L, w_max, h_max, p), V)
        except Exception:
            return 1e6

    res = differential_evolution(objective, bounds, maxiter=maxiter, popsize=12,
                                 seed=seed, tol=1e-4, polish=True)
    L, w_max, h_max, p, frac = res.x
    box_x0 = frac * max(L - box.lx, 0.0)
    body = build_body(L, w_max, h_max, p)
    return EnclosureResult(
        L=L, w_max=w_max, h_max=h_max, p=p, box_x0=box_x0,
        drag=drag(body, V), contains=containment_margin(L, w_max, h_max, p, box, box_x0) >= -1e-9,
        fineness=L / (2 * max(w_max, h_max)), body=body)


if __name__ == "__main__":  # demo: enclose a component box, minimize drag at 30 m/s
    box = Box(lx=0.30, ly=0.10, lz=0.08)  # 30cm x 10cm x 8cm of components
    V = 30.0
    r = optimize_enclosure(box, V, maxiter=25)
    print(f"box to enclose: {box.lx}x{box.ly}x{box.lz} m  (volume {box.lx*box.ly*box.lz*1e3:.2f} L)")
    print(f"optimized enclosure: L={r.L:.3f}m  max {2*r.w_max:.3f}x{2*r.h_max:.3f}m  "
          f"fineness L/D={r.fineness:.2f}  (streamlined optimum ~4-6)")
    print(f"drag @ {V} m/s = {r.drag:.3f} N   contains box: {r.contains}")
    # context: a bluff box of the same frontal area has far more drag (Cd~1 vs our streamlined)
    q = 0.5 * RHO_AIR * V ** 2
    bluff = q * (box.ly * box.lz) * 1.05  # flat-plate-ish Cd~1.05 on the box frontal area
    print(f"vs a bare bluff box (~Cd 1.05): {bluff:.3f} N  ->  ~{bluff/r.drag:.1f}x more drag")
