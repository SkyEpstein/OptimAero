"""Universal aerodynamic drag surrogate — works on ANY shape (drones, bodies, fuselages, wings, planes…).

The lesson of the drone-only surrogate: a per-archetype model overfits (rank capped ~0.75). Trained across
DIVERSE shape types with a rich geometric feature set (area-rule + normal-based streamlining + moments),
one drag surrogate generalizes across shape families — 0.95+ rank overall, 0.9 within-type on shapes with
real drag variation. This package holds that surrogate: `features.universal_features` (any mesh → feature
vector) and `surrogate.UniversalDragSurrogate` (features → Cd + confidence)."""
