# Spec — Enclosing aerodynamic envelope

## Problem
The current shape optimizer only *shrinks* the imported shape toward an inner keep-out (taper ≤ 1)
and bulges symmetrically in the mid-body. It cannot add material below the object or downstream,
so it looks "afraid to add anything." Sky wants it to build an **enclosing envelope** that adapts
his shape into an aerodynamic form.

## Decision (Sky, 2026-07-06, MCQ)
> "An enclosing envelope that adapts the original shape to make it more aerodynamic / targeted for
> a specific property."

## Scope / requirements
1. **Contains the original.** The optimized outer surface must fully enclose the entire imported
   mesh — it grows OUTWARD only, never inside the original surface. (Guaranteed, not best-effort.)
2. **Adapts the original.** The envelope follows the original's own width/height silhouette per
   station (a flat wide part → flat wide streamlined body; a round part → round teardrop). Not a
   generic ellipsoid dropped around a box.
3. **Adds material in all directions**, especially a streamlined nose (upstream) and a tapering
   tail (downstream / away from motion), bounded by the "drastic changes" slider.
4. **Targets a property.** Selectable objective: minimize drag, maximize lift, or maximize L/D
   (lift/L·D use the angle-of-attack aero model).
5. Respects the chosen flow direction. Exports to CAD. Reports before/after drag, lift, L/D + coeffs.

## Approach (plan)
- Extract the original's half-width profiles wy(x), wz(x) along the flow axis.
- Build the envelope as a closed ring-loft: streamlined nose extension → body rings sized to
  `(wy·g, wz·g)` (g ≥ analytic containment factor so each superellipse ring contains the original
  bbox slice) → tapering tail extension → apex points at both ends (watertight).
- Containment is guaranteed analytically during the search (sizing factor) — a single
  signed-distance check verifies the winner and projects outward in the rare violation.
- Optimize params [grow, nose_frac, tail_frac, round_exp] with differential_evolution for the
  selected objective; bounds scale with aggressiveness.

## Acceptance criteria
- For box/sphere/cylinder/flat-plate/L-shape: envelope **contains 100% of original vertices**
  (independent signed-distance check ≥ 0).
- min_drag envelope has drag ≤ original bluff drag for a blunt import (streamlining wins).
- max_lift / max_LD at AoA return positive lift / finite L/D and beat min_drag on that metric.
- No NaN/crash; exports valid STL/OBJ; flow_axis y/z handled; GUI end-to-end works.

## Verification
Multi-agent sweep (containment, objective correctness, robustness) + adversarial verifier, same as
the current test workflow.
