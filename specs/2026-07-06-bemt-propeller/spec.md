# Spec — BEMT propeller model (condensed, Phase 3)

Date: 2026-07-06 · Owner: Sky · SDD ceremony: condensed (single obvious approach, no design fork)

## Problem
Constitution Phase 3 needs physics coupling that assembles the section surrogate into
vehicle-level metrics, with uncertainty propagated section→vehicle (constitution §3, §4).
First deliverable: a propeller Blade-Element Momentum Theory (BEMT) solver that consumes the
`Surrogate` interface for section aero.

## Scope
- ONE new module `optimaero/physics/bemt.py` + package `__init__.py`. No other modules touched.
- Standard low-speed BEMT (incompressible, small-UAV regime — matches constitution scope).

## Approach (chosen by reasoning; textbook algorithm, no bake-off needed)
Discretize blade hub→tip into N elements. Per element iterate axial `a` and swirl `a'` to
convergence with a Prandtl tip-loss `F`:
- phi = atan2(V(1+a), Omega r (1-a')); alpha = twist - phi; W = resultant velocity.
- Re = W·chord/nu; query surrogate → Cl, Cd.
- Cn = Cl cos phi - Cd sin phi ; Ct = Cl sin phi + Cd cos phi.
- Update a, a' from blade-element ↔ momentum balance with F; relax; iterate.
- Integrate dT, dQ; power = Q·Omega; efficiency = T·V/P; J = V/(nD); CT, CP nondimensional.
Confidence: any element ood → any_ood True; frac_trusted = mean(trusted) over elements.

Constants: rho=1.225 kg/m^3, nu=1.5e-5 m^2/s.

## Acceptance criteria (VALIDATION — required, not optional)
Test prop: R=0.15 m, 2 blades, chord≈0.02 m ~const, twist 25–30°(root)→8–12°(tip), naca4412.
At fixed 5000 rpm, sweep V so J∈~[0.1,0.9]. The eta(J), thrust(J) curve MUST show:
1. Thrust positive at low J, decreasing as J rises (eventually negative). [AC1]
2. eta = T·V/P ∈ (0,1) for producing points; NEVER > 1. [AC2]
3. eta rises then PEAKS at intermediate J (~0.4–0.7) then falls — the efficiency hump. [AC3]
If not physical → debug induction/sign logic until it is. Do not ship a broken model.

## Verify (result — 2026-07-06)
Ran `python -m optimaero.physics.bemt` (10 elements, J∈[0.1,0.9], NeuralFoil placeholder):
- AC1 PASS: T = 6.44 N at J=0.1, monotonic ↓ to −2.95 N at J=0.9.
- AC2 PASS: η∈(0,1) at every producing point; max η≈0.62.
- AC3 PASS: interior efficiency hump, peak η≈0.62 near J≈0.45 (finer-grid confirm).
- Cross-check: η via T·V/P vs CT/CP·J agree to 4 decimals; all elements converged.
- Fix made: η defined only where T>0 ∧ P>0 (else nan); braking/windmill points no longer
  emit a meaningless negative "efficiency". Reporting fix, not a physics change.
- OOD: only J=0.9 (deep windmill) trips any_ood; frac_trusted=1.0 through the useful range.
Model reports `CURVE IS PHYSICAL: True`. Verdict: physically correct.
