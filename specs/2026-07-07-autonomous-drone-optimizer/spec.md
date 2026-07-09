# Autonomous drone optimizer (condensed spec)

**Date:** 2026-07-07 · **Status:** implemented, CFD-verified

## Problem
Sky's directive: *"YOU AREN'T MEANT TO HAVE ANY INPUT. THE PROGRAM SHOULD DO IT ITSELF."*
The drone path required hand-picking the aero treatment (which fairing, what chord). It must instead
decide the treatment autonomously: import a drone → the program figures out the tail + airfoils + the
amounts by itself → lower drag, additive-only (constitution §5.8).

## Approach (chosen by reasoning, not arbitrarily)
- Search the **additive-treatment parameter space** — `[boat-tail length, arm-airfoil chord, thickness]`
  — with Latin-hypercube sampling (candidate 0 = the bare drone as the control).
- **Judge every candidate with real CFD** (OpenFOAM, refine 4 + boundary layers), not the fast estimate
  (which was 3.4× wrong on this drone). Evaluate candidates in parallel, resource-capped.
- Return the **lowest-drag** design; **never-worse-than-bare** fallback guarantees no regression.
- One button in the GUI ("Optimize drone (automatic, CFD)"), zero aero params exposed.

## Acceptance criteria
1. `optimize_drone` returns a design with `drag_after ≤ drag_before` at identical CFD fidelity. ✅ (137.7→59.9 N)
2. Additive-only: original preserved, volume ≥ input — **runtime-verified per candidate** by
   `additive_ok()` (watertight + volume≥input + boolean-intersection recovers ≥99.5% of the original),
   and a design that fails is disqualified even if its drag is lower. ✅
3. Before/after are the SAME regime (both refine 4 + layers, same run) — honest comparison. ✅
4. Wired as the default GUI strategy; live progress; reports real CFD Cd/Cl/L·D⁻¹; Docker check. ✅
5. Sky's tail claim tested independently: tail alone −36%, tail+airfoil −42%, autonomous −57%. ✅
6. Never returns a None/broken mesh; honest baseline when the bare-drone CFD fails. ✅

## Verify (adversarial)
- **A 4-lens adversarial-review workflow (44 agents) found 16 confirmed defects; all fixed.** Highlights:
  a *critical* never-worse fallback that returned `None` and crashed the GUI when the bare CFD failed
  (the `min()` baseline made `best>=d0` always true → overwrote the real winner with a nulled bare);
  the additive-only guarantee was asserted but **never checked**; `add_tail` received the original
  segmentation but a changed (airfoiled) mesh; CFD ran at 0° AoA yet the readout was labeled with the
  user's AoA; lift/L·D at 0° (mesh noise on a symmetric body) shown as concrete performance.
- Fixes: build the mesh outside the CFD try (never lose a real mesh); walk candidates by drag and take
  the first that beats the baseline AND passes `additive_ok`; honest baseline (`drag_before=nan`,
  `metrics_before=None`) when bare CFD fails; `add_tail(..., body_source=drone)`; AoA plumbed through
  and labeled; near-zero lift flagged as noise; Docker check off the Tk main thread; non-multirotor
  guard; atomic progress counter; `res['ok']` = the real `contains_original`, not a hardcoded literal.
- Unit test (`scratchpad/unit_fallback.py`) drives all four fallback paths (bare-fails, all-worse,
  additive-gate, total-failure) — no None mesh, additive gating and honest baseline confirmed.
- `additive_ok` validated on real geometry: the −57% result and a fresh build pass; a shrunk/moved
  drone fails (`scratchpad/additive_real.py`).

## Open / next
- Horizontal-cruise drone mode (flow along the fuselage, not +z) — chord currently runs vertical at +z.
- Widen the search (nacelle fairings, arm count) once the ML surrogate can pre-screen candidates cheaply.
