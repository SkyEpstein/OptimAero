"""Blade-Element Momentum Theory (BEMT) propeller solver.

Constitution Phase 3 — physics coupling. This module assembles the section-level
`Surrogate` (Cl/Cd per airfoil section) into propeller vehicle-level metrics
(thrust, torque, power, efficiency), with the surrogate's confidence propagated
section->vehicle (constitution §3, §4).

Standard low-speed BEMT
-----------------------
The blade is discretized into radial elements from hub to tip. Each element is a 2D
airfoil section seeing a local resultant velocity `W`. Momentum theory (annulus mass/
momentum balance) and blade-element theory (section forces) are reconciled by iterating
the axial induction `a` and swirl (tangential) induction `a'` to convergence, with a
Prandtl tip-loss factor `F` accounting for the finite blade count.

Sign / geometry conventions (the part that bites)
-------------------------------------------------
- `V` is the axial freestream (advance) velocity, positive downstream.
- `Omega = 2*pi*rpm/60` is the shaft angular rate; the blade section moves tangentially
  at `Omega*r`.
- Axial flow through the disc is accelerated: `V*(1+a)`  (a >= 0 for a thrusting prop).
- Tangential flow at the section is reduced by swirl: `Omega*r*(1-a')`.
- Inflow angle `phi = atan2(V*(1+a), Omega*r*(1-a'))`, measured from the disc plane.
- Angle of attack `alpha = twist - phi` (twist = local geometric pitch angle from disc
  plane). At high advance ratio phi grows, alpha drops, and thrust falls — the physical
  behaviour we validate.
- Section Cl/Cd rotate into thrust/torque directions:
      Cn = Cl*cos(phi) - Cd*sin(phi)   (axial / thrust direction)
      Ct = Cl*sin(phi) + Cd*cos(phi)   (tangential / torque direction)
  `Cd` always subtracts from thrust and always adds to torque — the drag signs above are
  what enforce that.

Nondimensionalization uses the propeller convention (n in rev/s, D = diameter):
    J  = V / (n*D)          advance ratio
    CT = T / (rho*n^2*D^4)
    CP = P / (rho*n^3*D^5)
    eta = T*V / P  ( = CT/CP * J )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from optimaero.surrogate import Surrogate

# Constitution-fixed sea-level air properties (constitution scope: low-speed regime).
RHO_AIR = 1.225      # kg/m^3
NU_AIR = 1.5e-5      # m^2/s (kinematic viscosity)


@dataclass
class Propeller:
    """A propeller geometry. `chord` and `twist_deg` are callables of x = r/R in [0,1]."""
    radius: float                              # tip radius R [m]
    n_blades: int
    section_coords: np.ndarray                 # (N,2) unit-chord airfoil coordinates
    chord: Callable[[float], float]            # chord [m] as a function of r/R
    twist_deg: Callable[[float], float]        # geometric pitch angle [deg] as a fn of r/R
    hub_frac: float = 0.15                     # blade starts at hub_frac * R

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


@dataclass
class PropResult:
    """Vehicle-level propeller performance at one operating point, with propagated trust."""
    thrust: float          # T [N]
    torque: float          # Q [N·m]
    power: float           # P [W] = Q*Omega
    efficiency: float      # eta = T*V/P  (nan when P<=0 or V==0)
    J: float               # advance ratio V/(n*D)
    CT: float              # thrust coefficient T/(rho n^2 D^4)
    CP: float              # power coefficient P/(rho n^3 D^5)
    any_ood: bool          # any blade element flagged out-of-distribution by the surrogate
    frac_trusted: float    # fraction of blade elements the surrogate trusts
    converged: bool = True # every element's induction loop converged
    # Diagnostics (per radial element), useful for debugging the induction/sign logic.
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    dT: np.ndarray = field(default_factory=lambda: np.array([]))
    dQ: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    a: np.ndarray = field(default_factory=lambda: np.array([]))
    aprime: np.ndarray = field(default_factory=lambda: np.array([]))


def _prandtl_tip_loss(B: int, r: float, R: float, phi: float, r_hub: float) -> float:
    """Prandtl tip- and hub-loss factor F in (0,1]. Guards tiny/degenerate phi."""
    sphi = abs(np.sin(phi))
    if sphi < 1e-6:
        return 1.0
    # Tip loss.
    f_tip = B * (R - r) / (2.0 * r * sphi)
    F_tip = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), 0.0, 1.0))
    # Hub loss (mirror of the tip loss at the root).
    f_hub = B * (r - r_hub) / (2.0 * r_hub * sphi) if r_hub > 0 else np.inf
    F_hub = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_hub), 0.0, 1.0))
    F = F_tip * F_hub
    return float(max(F, 1e-4))


def _solve_element(
    surrogate: Surrogate, prop: Propeller, x: float, V: float, Omega: float,
    mach: float, max_iter: int, tol: float, relax: float,
):
    """Iterate (a, a') to convergence for one blade element at r/R = x.

    Returns a dict with the converged local state and the section AeroPrediction.
    """
    R = prop.radius
    r = x * R
    r_hub = prop.hub_frac * R
    B = prop.n_blades
    c = float(prop.chord(x))
    twist = np.radians(float(prop.twist_deg(x)))
    coords = prop.section_coords

    # Local solidity of the annulus: sigma' = B c / (2 pi r).
    sigma = B * c / (2.0 * np.pi * r)

    a, ap = 0.0, 0.0          # induction factors — start from momentum-free guess
    converged = False
    pred = None
    phi = np.arctan2(V, Omega * r)  # fallback if the loop never assigns

    for _ in range(max_iter):
        # Inflow angle from current inductions (propeller sign convention).
        Vax = V * (1.0 + a)              # accelerated axial flow
        Vtan = Omega * r * (1.0 - ap)    # swirl-reduced tangential flow
        phi = np.arctan2(Vax, max(Vtan, 1e-9))

        alpha = twist - phi              # section angle of attack
        W = np.hypot(Vax, Vtan)          # resultant velocity magnitude
        Re = max(W * c / NU_AIR, 1.0)

        pred = surrogate.predict(coords, np.degrees(alpha), Re, mach)
        Cl, Cd = pred.Cl, pred.Cd

        cphi, sphi = np.cos(phi), np.sin(phi)
        Cn = Cl * cphi - Cd * sphi       # -> thrust direction
        Ct = Cl * sphi + Cd * cphi       # -> torque direction

        F = _prandtl_tip_loss(B, r, R, phi, r_hub)

        # Blade-element ↔ momentum balance for the new inductions.
        # Axial:      a/(1+a)   = sigma*Cn / (4 F sin^2 phi)
        # Tangential: ap/(1-ap) = sigma*Ct / (4 F sin phi cos phi)
        s2 = max(sphi * sphi, 1e-9)
        sc = sphi * cphi
        sc = np.sign(sc) * max(abs(sc), 1e-9)

        ka = sigma * Cn / (4.0 * F * s2)          # = a/(1+a)
        kt = sigma * Ct / (4.0 * F * sc)          # = ap/(1-ap)

        # Invert. Clip ka<1 to keep a finite/physical (a<0 allowed: windmill/brake state).
        ka = min(ka, 0.99)
        a_new = ka / (1.0 - ka)
        ap_new = kt / (1.0 + kt)

        # Keep inductions in a sane band; relax to damp oscillation.
        a_new = float(np.clip(a_new, -0.5, 1.5))
        ap_new = float(np.clip(ap_new, -0.5, 0.9))
        a_new = a + relax * (a_new - a)
        ap_new = ap + relax * (ap_new - ap)

        if abs(a_new - a) < tol and abs(ap_new - ap) < tol:
            a, ap = a_new, ap_new
            converged = True
            break
        a, ap = a_new, ap_new

    # Final recompute at converged state for consistent force reporting.
    Vax = V * (1.0 + a)
    Vtan = Omega * r * (1.0 - ap)
    phi = np.arctan2(Vax, max(Vtan, 1e-9))
    alpha = twist - phi
    W = np.hypot(Vax, Vtan)
    Re = max(W * c / NU_AIR, 1.0)
    if pred is None:
        pred = surrogate.predict(coords, np.degrees(alpha), Re, mach)
    Cl, Cd = pred.Cl, pred.Cd
    cphi, sphi = np.cos(phi), np.sin(phi)
    Cn = Cl * cphi - Cd * sphi
    Ct = Cl * sphi + Cd * cphi

    q = 0.5 * RHO_AIR * W * W            # dynamic pressure on the section
    # Force per unit span, times B blades, integrated over dr later.
    dT_dr = B * q * c * Cn               # thrust per unit radius [N/m]
    dQ_dr = B * q * c * Ct * r           # torque per unit radius [N·m/m]

    return {
        "r": r, "a": a, "aprime": ap, "phi": phi, "alpha": alpha,
        "dT_dr": dT_dr, "dQ_dr": dQ_dr, "pred": pred, "converged": converged,
    }


def solve(
    surrogate: Surrogate,
    prop: Propeller,
    V: float,
    rpm: float,
    n_elements: int = 12,
    mach: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    relax: float = 0.5,
) -> PropResult:
    """Solve propeller performance at freestream `V` [m/s] and shaft speed `rpm`.

    Integrates the blade from `hub_frac*R` to `R` over `n_elements` radial stations,
    querying `surrogate` for section Cl/Cd at each, and reconciles blade-element and
    momentum theory via the (a, a') iteration with Prandtl tip loss. Confidence from
    the per-element surrogate predictions is propagated to `any_ood` / `frac_trusted`.
    """
    R = prop.radius
    D = prop.diameter
    Omega = 2.0 * np.pi * rpm / 60.0
    n_rev = rpm / 60.0                   # rev/s, for J/CT/CP

    # Radial stations at element mid-points (hub_frac..1), with their widths.
    edges = np.linspace(prop.hub_frac, 1.0, n_elements + 1)
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges) * R              # element radial width [m]

    r_arr, dT_arr, dQ_arr = [], [], []
    alpha_arr, phi_arr, a_arr, ap_arr = [], [], [], []
    trusted_flags, ood_flags = [], []
    all_conv = True

    for x, drw in zip(x_mid, dr):
        el = _solve_element(surrogate, prop, float(x), V, Omega, mach,
                            max_iter, tol, relax)
        r_arr.append(el["r"])
        dT_arr.append(el["dT_dr"] * drw)      # element thrust [N]
        dQ_arr.append(el["dQ_dr"] * drw)      # element torque [N·m]
        alpha_arr.append(np.degrees(el["alpha"]))
        phi_arr.append(np.degrees(el["phi"]))
        a_arr.append(el["a"])
        ap_arr.append(el["aprime"])
        pred = el["pred"]
        trusted_flags.append(bool(pred.trusted))
        ood_flags.append(bool(pred.ood))
        all_conv = all_conv and el["converged"]

    T = float(np.sum(dT_arr))
    Q = float(np.sum(dQ_arr))
    P = Q * Omega                         # shaft power [W]

    J = V / (n_rev * D) if n_rev > 0 else np.nan
    CT = T / (RHO_AIR * n_rev**2 * D**4) if n_rev > 0 else np.nan
    CP = P / (RHO_AIR * n_rev**3 * D**5) if n_rev > 0 else np.nan
    # Propulsive efficiency = useful power out (T*V) / shaft power in (P). It is only a
    # meaningful (0,1) quantity where the prop is actually PRODUCING thrust on shaft
    # power (T>0 and P>0). At braking/windmill points (T<=0, or P<=0) the ratio is not a
    # propulsive efficiency, so we report nan rather than a misleading number.
    eta = (T * V / P) if (T > 0.0 and P > 0.0 and V > 0.0) else float("nan")

    frac_trusted = float(np.mean(trusted_flags)) if trusted_flags else 0.0
    any_ood = bool(np.any(ood_flags))

    return PropResult(
        thrust=T, torque=Q, power=P, efficiency=eta, J=J, CT=CT, CP=CP,
        any_ood=any_ood, frac_trusted=frac_trusted, converged=all_conv,
        r=np.array(r_arr), dT=np.array(dT_arr), dQ=np.array(dQ_arr),
        alpha_deg=np.array(alpha_arr), phi_deg=np.array(phi_arr),
        a=np.array(a_arr), aprime=np.array(ap_arr),
    )


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":  # VALIDATION — sweep J, confirm the propeller physics.
    from optimaero.surrogate import NeuralFoilSurrogate
    from optimaero.datasets.uiuc import load_coordinates

    coords = load_coordinates("naca4412")

    # Test propeller: R=0.15 m, 2 blades, ~constant chord, twist 28°(root)->10°(tip).
    def chord_fn(x):        # ~0.02 m, gently tapering
        return 0.022 - 0.008 * x

    def twist_fn(x):        # linear washout, root 28° -> tip 10°
        return 28.0 - 18.0 * x

    prop = Propeller(
        radius=0.15, n_blades=2, section_coords=coords,
        chord=chord_fn, twist_deg=twist_fn, hub_frac=0.15,
    )

    surr = NeuralFoilSurrogate()
    rpm = 5000.0
    n_rev = rpm / 60.0
    D = prop.diameter

    # Sweep V so that J = V/(n D) spans ~0.1 .. 0.9.
    J_targets = np.linspace(0.1, 0.9, 9)
    V_sweep = J_targets * n_rev * D

    print(f"Test prop: R={prop.radius} m, B={prop.n_blades}, section=naca4412, "
          f"rpm={rpm:.0f}, D={D:.3f} m, n={n_rev:.1f} rev/s")
    print(f"{'J':>6} {'V(m/s)':>8} {'T(N)':>9} {'Q(N·m)':>9} {'P(W)':>9} "
          f"{'eta':>7} {'CT':>7} {'CP':>7} {'ood':>4} {'trust':>6}")
    rows = []
    for V in V_sweep:
        res = solve(surr, prop, float(V), rpm, n_elements=10)
        rows.append(res)
        eta_s = f"{res.efficiency:7.3f}" if np.isfinite(res.efficiency) else "    nan"
        print(f"{res.J:6.2f} {V:8.2f} {res.thrust:9.3f} {res.torque:9.4f} "
              f"{res.power:9.2f} {eta_s} {res.CT:7.3f} {res.CP:7.3f} "
              f"{str(res.any_ood):>4} {res.frac_trusted:6.2f}")

    # ---- Physics checks (acceptance criteria) ----
    Js = np.array([r.J for r in rows])
    Ts = np.array([r.thrust for r in rows])
    etas = np.array([r.efficiency for r in rows])

    print("\n--- PHYSICS VALIDATION ---")
    # AC1: thrust positive at low J and monotonically decreasing across the sweep.
    thrust_low_pos = Ts[0] > 0
    thrust_decreasing = bool(np.all(np.diff(Ts) < 1e-6))
    print(f"[AC1] thrust>0 at low J: {thrust_low_pos} (T={Ts[0]:.3f} N); "
          f"thrust decreasing with J: {thrust_decreasing} "
          f"(T goes {Ts[0]:.3f} -> {Ts[-1]:.3f} N)")

    # AC2: efficiency bounded in (0,1) wherever it is defined and thrust is producing.
    finite = np.isfinite(etas)
    eta_bounded = bool(np.all((etas[finite] >= 0.0) & (etas[finite] <= 1.0)))
    print(f"[AC2] eta in [0,1] where defined: {eta_bounded} "
          f"(max eta = {np.nanmax(etas):.3f})")

    # AC3: efficiency peaks at an intermediate J (interior maximum).
    valid = finite & (etas > 0)
    if valid.sum() >= 3:
        peak_i = int(np.nanargmax(np.where(valid, etas, -np.inf)))
        interior_peak = 0 < peak_i < len(etas) - 1
        print(f"[AC3] efficiency peak at intermediate J: {interior_peak} "
              f"(peak eta={etas[peak_i]:.3f} at J={Js[peak_i]:.2f})")
    else:
        interior_peak = False
        print("[AC3] not enough valid efficiency points to locate a peak")

    physical = thrust_low_pos and thrust_decreasing and eta_bounded and interior_peak
    print(f"\nCURVE IS PHYSICAL: {physical}")
    n_ood = sum(r.any_ood for r in rows)
    print(f"Operating points with any OOD element: {n_ood}/{len(rows)}; "
          f"min frac_trusted across sweep = {min(r.frac_trusted for r in rows):.2f}")
