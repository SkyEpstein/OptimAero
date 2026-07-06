"""User-facing design intent: the packaging envelope a shape must fit inside, and the
performance requirements. This is what a CAD user supplies alongside their shape constraint.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Envelope:
    """Packaging constraint the airfoil MUST fit inside (a hard constraint).

    v1 models the envelope as thickness bounds (t/c). A full 2D bounding contour imported
    from CAD (`cad/io.py`) is a later extension that supersedes `max_thickness`.
    """
    max_thickness: float = 0.15  # max thickness / chord the shape may occupy
    min_thickness: float = 0.05  # structural floor


@dataclass
class DesignRequirement:
    """Performance targets at an operating condition (top speed → Re, plus lift/drag intent).

    objective:
      - "max_LD"        : maximize best lift-to-drag ratio over the alpha range
      - "min_Cd_at_Cl"  : hit `target_Cl` with minimum drag
      - "max_Cl"        : maximize achievable lift (e.g. high-lift section)
    """
    Re: float
    objective: str = "max_LD"
    target_Cl: float | None = None
    alpha_range: tuple[float, float] = (-3.0, 12.0)
    mach: float = 0.0

    def __post_init__(self):
        if self.objective == "min_Cd_at_Cl" and self.target_Cl is None:
            raise ValueError("objective 'min_Cd_at_Cl' requires target_Cl")
