"""Quick self-test — confirms the quadcopter designer and CAD import work in THIS environment.

Run:  .venv/bin/python -m scripts.selftest
If everything says [OK], the code is fine and any GUI failure means a stale window — quit it
and relaunch. If anything says [FAIL], copy the line to Claude and it will be fixed.
"""
import trimesh

from optimaero.aircraft.design import design, DesignSpec
from optimaero.three_d.enclosure import Box
from optimaero.three_d import cad3d


def main():
    print("OptimAero self-test")
    print("=" * 34)

    try:
        d = design(DesignSpec(Box(0.30, 0.10, 0.08), 12.0, "quadcopter", "min_drag"), maxiter=6)
        print(f"[OK]   quadcopter design      -> drag {d.drag_N:.3f} N (finite: {d.drag_N == d.drag_N})")
    except Exception as e:  # noqa
        print(f"[FAIL] quadcopter design      -> {e!r}")

    for ext in ("stl", "obj", "ply", "off", "step", "iges"):
        p = f"/tmp/oa_selftest.{ext}"
        try:
            if ext in ("step",):
                import cadquery as cq
                cq.exporters.export(cq.Workplane("XY").box(0.30, 0.10, 0.08), p)
            elif ext == "iges":
                import cadquery as cq
                from OCP.IGESControl import IGESControl_Writer
                w = IGESControl_Writer()
                w.AddShape(cq.Workplane("XY").box(0.30, 0.10, 0.08).val().wrapped)
                w.Write(p)
            else:
                trimesh.creation.box(extents=[0.30, 0.10, 0.08]).export(p)
            b = cad3d.import_volume(p, units="m")  # our test boxes are modeled in metres
            got = sorted((round(b.lx, 3), round(b.ly, 3), round(b.lz, 3)))
            ok = got == sorted((0.08, 0.10, 0.30))
            print(f"[{'OK' if ok else 'FAIL'}] import .{ext:<4}            -> "
                  f"{round(b.lx,3)} x {round(b.ly,3)} x {round(b.lz,3)}")
        except Exception as e:  # noqa
            print(f"[FAIL] import .{ext:<4}            -> {e!r}")


if __name__ == "__main__":
    main()
