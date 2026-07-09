"""Export a designed aircraft to CAD. The aircraft is a mesh (wings + fuselage/arms), so we
export universal mesh formats (STL/OBJ/PLY/GLB) that any CAD/CAM/slicer/printer accepts.
"""
from __future__ import annotations

import os

import numpy as np


def aircraft_mesh(design):
    """Combined triangular mesh of the aircraft's fuselage(s)/arms + wings."""
    import trimesh
    ap = design.airplane
    verts, faces, off = [], [], 0
    for comp in list(ap.fuselages) + list(ap.wings):
        p, f = comp.mesh_body()
        p = np.asarray(p, dtype=float)
        f = np.asarray(f)
        verts.append(p)
        faces.append(f + off)
        off += len(p)
    return trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(faces))


def export_aircraft(design, path: str) -> str:
    """Write the designed aircraft to `path`. Mesh formats: .stl/.obj/.ply/.glb/.off."""
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        path += ".stl"
    aircraft_mesh(design).export(path)
    return path


if __name__ == "__main__":
    from optimaero.three_d.enclosure import Box
    from optimaero.aircraft.design import design, DesignSpec
    d = design(DesignSpec(Box(0.30, 0.10, 0.08), 20.0, "airplane", "max_LD"), maxiter=8)
    out = export_aircraft(d, "/tmp/designed_aircraft.stl")
    print(f"exported {out} ({os.path.getsize(out):,} bytes)")
