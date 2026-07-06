# 3D Aerodynamic-Enclosure Optimization (spec)

*SDD: specify. Governed by `memory/constitution.md` (§2 core target, revised 2026-07-06).*
Date: 2026-07-06

## Problem

The user supplies a **3D volume** their internal components must fit inside and selects a
**purpose**; OptimAero produces an **optimized aerodynamic outer shape** that (a) fully
contains the volume and (b) minimizes drag / meets the purpose's aero objective, and exports
it as CAD (STEP/STL). This is the true product vision — replacing the 2D-airfoil scope.

The 2D pipeline (surrogate + confidence bake-off + inverse design + verify-against-truth + CAD
+ GUI) is the validated blueprint; this spec re-aims it at 3D.

## Approach — hybrid, staged (data pick made by Claude, confirmed by Sky)

- **Fast 3D method (workhorse):** AeroSandbox 3D aero (panel / component buildup) estimates a
  body's drag in ~seconds → immediate working pipeline + high-volume data.
- **3D CFD (anchor + training truth):** OpenFOAM at feasible scale (hundreds–low-thousands of
  cases on 16 cores over days — NOT the 213k of 2D) → the ML genuinely replaces CFD; the
  confidence + verify-against-CFD story holds. Mirrors the 2D XFOIL+AirfRANS pattern.

### Stage A — working 3D tool (fast method)
1. **Volume input** — a packaging volume (box dims, or an imported STL/STEP).
2. **Enclosure geometry** — a parameterized smooth 3D body (streamlined fuselage-like body via
   cross-sections along an axis) constrained to **contain the volume** at every station.
3. **Fast aero** — AeroSandbox drag (form + skin friction + base) at the operating speed.
4. **Optimizer** — black-box search over enclosure params; **hard containment constraint**;
   objective from the selected purpose.
5. **CAD export** — optimized enclosure → STEP/STL (loft the cross-sections).
6. **GUI** — volume + purpose + speed in → enclosure preview + drag + STEP out.

### Stage B — CFD-backed surrogate (research depth)
7. **3D shape parameterization + sampling** → generate an OpenFOAM RANS dataset (feasible scale).
8. **Bake-off** — predictor + confidence on the 3D data (a 3D geometric representation:
   parameters / point cloud / SDF — a bake-off dimension), group-aware splits, honest metrics.
9. **Verify-in-the-loop** — the optimizer's optimum confirmed by OpenFOAM (the CFD-fallback).

## Requirements

- **R1.** Containment is a hard guarantee — the exported shape provably encloses the user volume.
- **R2.** "Purpose" selects the aero objective + operating condition (e.g. drone body → min drag
  at cruise speed; a lifting body → min drag at a lift target).
- **R3.** Inputs in the user's language — a real speed (m/s → Reynolds internally) and absolute
  forces where meaningful, not just coefficients.
- **R4.** Same honesty discipline as 2D: no leakage, R²+RMSE together, verify-against-CFD, honest
  data card and limitations.

## Open clarifications (later)

- **[3D-GEOM]** enclosure parameterization family (body-of-revolution vs elliptical
  cross-sections vs free lofted sections) — decide when building Stage A.
- **[3D-REP]** surrogate's 3D geometry representation (parameters vs point-cloud/SDF) — a
  Stage-B bake-off dimension.
- **[3D-PURPOSE]** the initial purpose list and each one's objective/condition.

## Feasibility gates (verify before building on them)

- **F1.** AeroSandbox can build an arbitrary streamlined body and return a physically-sane drag
  fast (the fast-method backbone). *Probe before Stage A.*
- **F2.** OpenFOAM can mesh + solve one enclosure shape end-to-end on this Mac at acceptable
  cost (the Stage-B backbone). *Probe before committing Stage-B compute.*
