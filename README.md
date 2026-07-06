# OptimAero

**Import a CAD volume your parts must fit inside → get back an optimized aerodynamic
enclosure as a CAD file.**

OptimAero grows a drag-minimized 3D aerodynamic skin around a required packaging volume and
exports it (STEP/STL). The workflow is exactly: **import CAD → aerodynamic optimization →
export CAD**, in a plain desktop GUI with a 3D viewer. Containment (your parts provably fit
inside) is a hard, verified guarantee.

Under the hood it is an *uncertainty-aware ML-surrogate-for-CFD* system: a fast aerodynamic
evaluator, a **confidence model** that knows when to defer to a real solver, and
inverse-design with **verify-against-truth** so the reported performance is never the
surrogate's unverified guess.

## Status

- **3D enclosure tool (fast method) — working.** Import a CAD volume → optimized enclosure →
  CAD out, in `optimaero.gui3d` (Tkinter, with a rotatable 3D viewer). Drag in Newtons,
  containment verified (ellipse circumscribes the box).
- **2D airfoil foundation — validated methodology.** A 213k-row XFOIL surrogate
  (new-geometry Cl R² = 0.984), a learned confidence model, and inverse design with XFOIL
  verification. This *proved* the methodology now being applied in 3D. See
  `docs/METHODS_AND_RESULTS.md`.
- **In progress — Stage B:** a 3D **CFD-trained** surrogate + confidence (OpenFOAM), for
  higher aerodynamic accuracy than the fast method.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m optimaero.gui3d      # the 3D enclosure tool (import CAD → aero → CAD)
.venv/bin/python -m optimaero.gui        # the 2D airfoil inverse-design tool
```

## Governing docs

- `memory/constitution.md` — mission, scope, non-negotiable principles.
- `specs/` — the spec-driven-development record for each phase.
- `docs/METHODS_AND_RESULTS.md` — the honest methods + results (with figures).

## Non-negotiable principles

no data leakage · report R² **and** RMSE together · **trust before speed** (verify against
real solvers) · bake-offs, not hunches · honest limitations, always.
