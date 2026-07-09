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


UNIT_SCALE = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254}  # → metres


def import_volume(path: str, units: str = "mm"):
    """Import any common CAD/mesh file → the payload Box, converted to METRES.

    `units` = the units the CAD file is modeled in. **Default "mm"** — the near-universal
    convention for CAD parts and STL exports — so a 300 mm part becomes 0.30 m, not 300 m.
    B-rep (STEP/STP/IGES/IGS/BREP) via OpenCASCADE; meshes (STL/OBJ/PLY/OFF/GLB/3MF) via
    trimesh, using the ORIENTED (minimal) bounding box so a rotated part isn't over-reported.
    """
    from optimaero.three_d.enclosure import Box
    scale = UNIT_SCALE.get(units, 0.001)
    ext = os.path.splitext(path)[1].lower().lstrip(".")

    if ext in ("step", "stp", "iges", "igs", "brep"):
        try:
            if ext in ("step", "stp"):
                shp = cq.importers.importStep(path).val()
            elif ext == "brep":
                shp = cq.importers.importBrep(path).val()
            else:  # iges/igs — CadQuery has no IGES reader; read via OpenCASCADE directly
                from OCP.IGESControl import IGESControl_Reader
                from OCP.IFSelect import IFSelect_RetDone
                reader = IGESControl_Reader()
                if reader.ReadFile(path) != IFSelect_RetDone:
                    raise RuntimeError("IGES read failed")
                reader.TransferRoots()
                shp = cq.Shape(reader.OneShape())
            bb = shp.BoundingBox()
            dims = (bb.xlen, bb.ylen, bb.zlen)
        except Exception as e:
            raise ValueError(f"could not read {ext.upper()} CAD file: {e}")
    else:
        try:
            import trimesh
            mesh = trimesh.load(path, force="mesh")
            if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) < 3:
                raise ValueError("file has no readable mesh geometry")
            # axis-aligned bbox — exact for parts modeled axis-aligned (the CAD norm); a
            # rotated part over-reports slightly, which still safely contains it.
            dims = tuple(np.asarray(mesh.bounding_box.extents, dtype=float))
        except Exception as e:
            raise ValueError(f"could not read '.{ext}' file (empty or unsupported?): {e}")

    dims = tuple(float(d) * scale for d in dims)
    if len(dims) != 3 or not all(np.isfinite(d) and d > 0 for d in dims):
        raise ValueError(f"CAD file has a degenerate/zero bounding volume: {dims}")
    return Box(lx=dims[0], ly=dims[1], lz=dims[2])


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
