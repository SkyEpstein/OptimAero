"""Neutral-format CAD I/O (constitution: STEP/STL/IGES, no lock-in).

Export an optimized airfoil section as a 3D solid to STEP (parametric B-rep — what
SolidWorks/Fusion/Onshape import) and STL (mesh). Import a packaging envelope from a
STEP/STL file and reduce it to the thickness bound the inverse-design optimizer consumes.

Built on CadQuery / OpenCASCADE.
"""
from __future__ import annotations

import os

import numpy as np
import cadquery as cq

from optimaero.requirements import Envelope


def airfoil_to_solid(coords: np.ndarray, chord: float = 1.0, span: float = 1.0):
    """Extrude a 2D airfoil profile (unit-chord coords) into a 3D wing segment solid."""
    c = np.asarray(coords, dtype=float) * chord
    pts = [(float(x), float(y)) for x, y in c]
    if pts[0] != pts[-1]:
        pts.append(pts[0])  # close the loop
    return cq.Workplane("XY").polyline(pts).close().extrude(span)


def export_step(solid, path: str) -> str:
    cq.exporters.export(solid, path)
    return path


def export_stl(solid, path: str, tolerance: float = 1e-3) -> str:
    cq.exporters.export(solid, path, tolerance=tolerance)
    return path


def export_airfoil(coords: np.ndarray, out_dir: str, name: str = "optimaero_section",
                   chord: float = 1.0, span: float = 1.0) -> dict:
    """Export an airfoil to both STEP and STL. Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    solid = airfoil_to_solid(coords, chord=chord, span=span)
    return {
        "step": export_step(solid, os.path.join(out_dir, f"{name}.step")),
        "stl": export_stl(solid, os.path.join(out_dir, f"{name}.stl")),
    }


def import_envelope(path: str, chord: float | None = None) -> Envelope:
    """Read a STEP/STL packaging shape and reduce it to a thickness envelope (t/c bounds).

    v1 uses the imported shape's bounding-box thickness as `max_thickness`; a richer 2D
    bounding-contour constraint is a later extension.
    """
    shape = cq.importers.importStep(path) if path.lower().endswith((".step", ".stp")) \
        else cq.Workplane(obj=cq.importers.importShape("STL", path))
    bb = shape.val().BoundingBox()
    c = chord if chord is not None else bb.xlen
    max_tc = (bb.ylen / c) if c > 0 else 0.15
    return Envelope(max_thickness=float(max_tc), min_thickness=float(max_tc) * 0.4)


if __name__ == "__main__":  # round-trip demo
    from optimaero.datasets import uiuc

    coords = uiuc.load_coordinates("naca2412")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(repo, "data", "processed", "cad_demo")
    paths = export_airfoil(coords, out, name="naca2412_wing", chord=0.2, span=1.0)
    for k, p in paths.items():
        print(f"{k}: {p}  ({os.path.getsize(p)} bytes)")
    env = import_envelope(paths["step"], chord=0.2)
    print(f"imported envelope from STEP: max t/c = {env.max_thickness:.3f} (naca2412 ~0.12)")
