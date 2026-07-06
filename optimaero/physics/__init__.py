"""Physics coupling: assemble section-surrogate predictions into vehicle-level metrics.

Constitution Phase 3. Uncertainty is propagated section->vehicle (constitution §3, §4):
every coupled result carries the trust view of the section evaluations it was built from.
"""
from __future__ import annotations

from optimaero.physics.bemt import Propeller, PropResult, solve

__all__ = ["Propeller", "PropResult", "solve"]
