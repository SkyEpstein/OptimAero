"""OptimAero workflow demo: CAD file IN → aero → CAD file OUT (Sky's vision).

1. (stand-in for the user's imported CAD) make a component 'part' STEP
2. import it as the packaging volume
3. grow a drag-minimized aerodynamic enclosure around it
4. export the enclosure as the final CAD product
"""
import os

import cadquery as cq

from optimaero.three_d import cad3d
from optimaero.three_d.enclosure import optimize_enclosure

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    out = os.path.join(_REPO, "data", "processed", "workflow_demo")
    os.makedirs(out, exist_ok=True)

    # (1) stand-in for the user's CAD file: a component part ~0.30 x 0.10 x 0.08 m
    part_path = os.path.join(out, "user_part.step")
    cq.exporters.export(cq.Workplane("XY").box(0.30, 0.10, 0.08), part_path)
    print(f"[1 IN ] imported CAD file:  {part_path}")

    # (2) import it as the packaging volume
    box = cad3d.import_volume(part_path)
    print(f"        -> volume to enclose: {box.lx:.3f} x {box.ly:.3f} x {box.lz:.3f} m")

    # (3) aero optimization
    r = optimize_enclosure(box, V=30.0, maxiter=25)
    print(f"[2 AERO] enclosure L={r.L:.3f} m,  drag {r.drag:.3f} N @ 30 m/s,  "
          f"contains part: {r.contains}")

    # (4) export the final product
    res = cad3d.export_enclosure(r, out, name="final_enclosure")
    print(f"[3 OUT] final product CAD:  {res['step']} ({os.path.getsize(res['step']):,} bytes)")
    print(f"        enclosure volume {res['volume_m3']*1e3:.2f} L wraps the "
          f"{box.lx*box.ly*box.lz*1e3:.2f} L part.")
