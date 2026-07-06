"""Export a 3D enclosure body to neutral CAD (STEP/STL) by lofting its elliptical
cross-sections with CadQuery/OpenCASCADE."""
from __future__ import annotations

import os

import numpy as np
import cadquery as cq
from cadquery import Solid, Wire, Vector


def _ellipse_wire(x: float, w: float, h: float, n: int = 48) -> Wire:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = [Vector(float(x), float(w * np.cos(a)), float(h * np.sin(a))) for a in ang]
    pts.append(pts[0])
    return Wire.makePolygon(pts)


def enclosure_solid(result) -> Solid:
    """Loft the body's cross-sections into a solid (tips clamped to a tiny radius so it closes)."""
    wires = []
    for xs in result.body.xsecs:
        w = max(xs.width / 2, 1e-4)
        h = max(xs.height / 2, 1e-4)
        wires.append(_ellipse_wire(xs.xyz_c[0], w, h))
    return Solid.makeLoft(wires, ruled=True)


def export_enclosure(result, out_dir: str, name: str = "enclosure3d") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    solid = enclosure_solid(result)
    wp = cq.Workplane(obj=solid)
    step = os.path.join(out_dir, f"{name}.step")
    stl = os.path.join(out_dir, f"{name}.stl")
    cq.exporters.export(wp, step)
    cq.exporters.export(wp, stl, tolerance=1e-3)
    return {"step": step, "stl": stl, "volume_m3": float(solid.Volume())}


if __name__ == "__main__":
    from optimaero.three_d.enclosure import Box, optimize_enclosure

    box = Box(lx=0.30, ly=0.10, lz=0.08)
    r = optimize_enclosure(box, V=30.0, maxiter=20)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = export_enclosure(r, os.path.join(repo, "data", "processed", "enclosure_demo"))
    box_vol = box.lx * box.ly * box.lz
    print(f"exported STEP: {out['step']} ({os.path.getsize(out['step']):,} bytes)")
    print(f"exported STL:  {out['stl']} ({os.path.getsize(out['stl']):,} bytes)")
    print(f"enclosure volume {out['volume_m3']*1e3:.2f} L  >  box volume {box_vol*1e3:.2f} L "
          f"(encloses it: {out['volume_m3'] > box_vol})")
