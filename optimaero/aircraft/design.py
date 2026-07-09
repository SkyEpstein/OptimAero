"""Multi-mode airframe designer.

Given a payload volume + an aircraft type + a chosen mission objective, DESIGN and tune the
aerodynamic features (wings, body, arms) with real aero (AeroSandbox buildup) and iterate to
optimize the objective. Aircraft types are pluggable via the TYPES registry — airplane and
quadcopter now, more by adding a builder.

Not a shape wrapper: airplane mode designs a real wing that generates lift; quadcopter mode
tunes a low-drag frame. Different objective -> genuinely different airframe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import aerosandbox as asb
from scipy.optimize import differential_evolution

from optimaero.three_d.enclosure import Box, optimize_enclosure

RHO = 1.225  # kg/m^3

OBJECTIVES = {
    "max_LD": "Maximize lift-to-drag (efficient cruise / long range)",
    "max_lift": "Maximize lift (max payload)",
    "lift_target_min_drag": "Lift a target weight at minimum drag",
    "min_drag": "Minimize drag (low-lift frames, e.g. multirotor)",
}


@dataclass
class DesignSpec:
    box: Box                        # payload volume the body must contain
    V: float                        # airspeed (m/s)
    aircraft_type: str = "airplane"
    objective: str = "max_LD"
    target_lift_N: float = 0.0
    airfoil: str = "naca2412"
    rotor_clearance: float = 0.12   # quad: min arm reach for the rotors (m)


@dataclass
class AircraftDesign:
    airplane: asb.Airplane
    aircraft_type: str
    objective: str
    params: dict
    CL: float
    CD: float
    lift_N: float
    drag_N: float
    LD: float
    ref_area: float
    alpha: float


# ---------------------------------------------------------------- aero (whole aircraft)
def _aero(ap: asb.Airplane, V, alpha):
    """AeroBuildup: wing lift + induced + profile drag + FUSELAGE/arm drag + friction."""
    r = asb.AeroBuildup(airplane=ap, op_point=asb.OperatingPoint(velocity=V, alpha=alpha)).run()
    CL = float(np.asarray(r["CL"]).ravel()[0])
    CD = float(np.asarray(r["CD"]).ravel()[0])
    S = ap.s_ref
    q = 0.5 * RHO * V ** 2
    return CL, CD, S, q * S * CL, q * S * CD


# ---------------------------------------------------------------- aircraft type builders
def _build_airplane(spec, fuse, x_le, params):
    chord, span, alpha = params
    af = asb.Airfoil(spec.airfoil)
    wing = asb.Wing(name="wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[x_le, 0, 0], chord=chord, twist=0, airfoil=af),
        asb.WingXSec(xyz_le=[x_le + 0.25 * chord, span / 2, 0], chord=chord * 0.6, twist=0,
                     airfoil=af)])
    return asb.Airplane(name="airplane", wings=[wing], fuselages=[fuse]), float(alpha)


def _build_quadcopter(spec, fuse, x_le, params):
    arm_len, arm_thick = params
    arms = []
    for ang in (45, 135, 225, 315):
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        xs = [asb.FuselageXSec(xyz_c=[x_le + r * c, r * s, 0],
                               width=float(arm_thick), height=float(arm_thick))
              for r in np.linspace(0.01, arm_len, 6)]
        arms.append(asb.Fuselage(name=f"arm{ang}", xsecs=xs))
    return asb.Airplane(name="quadcopter", fuselages=[fuse] + arms), 0.0  # level forward flight


TYPES = {
    "airplane": dict(build=_build_airplane,
                     bounds=[(0.05, 0.45), (0.4, 2.6), (-2.0, 9.0)],
                     names=["wing_chord", "wing_span", "alpha_deg"],
                     objectives=["max_LD", "max_lift", "lift_target_min_drag", "min_drag"]),
    "quadcopter": dict(build=_build_quadcopter,
                       bounds=[(None, None), (0.006, 0.03)],  # arm_len bound set from clearance
                       names=["arm_len", "arm_thick"],
                       objectives=["min_drag"]),
}


# ---------------------------------------------------------------- type-aware evaluation
def _evaluate(spec, fuse, x_le, params):
    """Return (airplane, alpha, CL, CD, ref_area, lift_N, drag_N) with type-appropriate aero."""
    ap, alpha = TYPES[spec.aircraft_type]["build"](spec, fuse, x_le, params)
    if spec.aircraft_type == "quadcopter":
        # Buildup on the streamlined BODY only (the thin arm tubes are degenerate fuselages
        # that make the buildup return NaN). Arms add drag as 4 cylinders in ~crossflow.
        _, _, Sb, _, body_drag = _aero(asb.Airplane(fuselages=[fuse]), spec.V, 0.0)
        arm_len, arm_thick = params
        q = 0.5 * RHO * spec.V ** 2
        arm_drag = 4 * 1.0 * q * (arm_thick * arm_len)   # Cd~1.0 cylinder, frontal = thick*len
        return ap, 0.0, 0.0, float("nan"), Sb, 0.0, body_drag + arm_drag
    CL, CD, S, lift, drag = _aero(ap, spec.V, alpha)
    return ap, alpha, CL, CD, S, lift, drag


# ---------------------------------------------------------------- generic designer
def design(spec: DesignSpec, maxiter: int = 12, seed: int = 0) -> AircraftDesign:
    T = TYPES[spec.aircraft_type]
    if spec.objective not in T["objectives"]:
        spec.objective = T["objectives"][0]
    fr = optimize_enclosure(spec.box, spec.V, maxiter=12)        # containing body (verified)
    fuse, x_le = fr.body, fr.box_x0 + spec.box.lx * 0.5

    bounds = list(T["bounds"])
    if spec.aircraft_type == "quadcopter":                       # arm reach >= rotor clearance
        bounds[0] = (spec.rotor_clearance, spec.rotor_clearance * 2.2)

    def objective(x):
        try:
            ap, alpha, CL, CD, S, lift, drag = _evaluate(spec, fuse, x_le, x)
        except Exception:
            return 1e6
        if not np.isfinite(drag) or drag <= 0:
            return 1e6
        if spec.objective == "max_LD":
            return -(lift / drag)
        if spec.objective == "max_lift":
            return -lift
        if spec.objective == "min_drag":
            return drag
        if spec.objective == "lift_target_min_drag":
            return drag + 100.0 * max(0.0, spec.target_lift_N - lift)
        return 1e6

    res = differential_evolution(objective, bounds, maxiter=maxiter, popsize=10, seed=seed,
                                 tol=1e-3, polish=True)
    ap, alpha, CL, CD, S, lift, drag = _evaluate(spec, fuse, x_le, res.x)
    return AircraftDesign(
        airplane=ap, aircraft_type=spec.aircraft_type, objective=spec.objective,
        params={n: float(v) for n, v in zip(T["names"], res.x)},
        CL=CL, CD=CD, lift_N=lift, drag_N=drag,
        LD=(lift / drag if drag > 0 else float("nan")), ref_area=S, alpha=alpha)


if __name__ == "__main__":  # both modes, real aero
    box = Box(lx=0.30, ly=0.10, lz=0.08)
    print("AIRPLANE mode:")
    for obj in ("max_LD", "max_lift", "lift_target_min_drag"):
        s = DesignSpec(box, 20.0, "airplane", obj, target_lift_N=20.0)
        d = design(s, maxiter=10)
        print(f"  [{obj:20s}] {d.params}  lift {d.lift_N:6.1f} N  drag {d.drag_N:5.2f} N  "
              f"L/D {d.LD:5.1f}")
    print("QUADCOPTER mode:")
    d = design(DesignSpec(box, 12.0, "quadcopter", "min_drag"), maxiter=10)
    print(f"  [min_drag] {d.params}  drag {d.drag_N:5.2f} N @ 12 m/s "
          f"(4 arms + body, containing the payload)")
