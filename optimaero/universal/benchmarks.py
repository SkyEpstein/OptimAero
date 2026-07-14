"""Canonical benchmark geometries for validating the universal drag surrogate on out-of-distribution
shapes it never trained on.

Every generator returns a watertight `trimesh.Trimesh` oriented so the flow is +x (the convention
`cfd_label` and `predict_drag(flow_axis="x")` expect), scaled to the training size regime (~0.1 m) so the
Reynolds number matches the data the surrogate learned from (V = 134.11 m/s, RHO = 1.225).

These are the *textbook* proportions — a real sphere, the Ahmed reference body, real NACA sections, the
ONERA M6 planform — as opposed to the random parametric shapes in `data/processed/xtype_*`. Where a
published frontal-area drag coefficient exists it is attached as `cd_lit` so the caller can judge whether
steady RANS is even a trustworthy ground truth for that shape (it is poor for separated bluff wakes).
"""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon


# ---- airfoil sections -------------------------------------------------------------------------------

def naca4(m: float, p: float, t: float, c: float = 0.08, n: int = 90) -> np.ndarray:
    """NACA 4-digit section as a closed (x, y) loop, chord `c` along +x. m,p are camber fraction/position
    (0 for symmetric), t is thickness fraction. Blunt trailing edge (standard −0.1015 coeff) so the loop
    is a valid closed polygon."""
    beta = np.linspace(0.0, np.pi, n)
    x = c * (1 - np.cos(beta)) / 2                      # cosine spacing, dense at LE/TE
    xc = np.clip(x / c, 1e-9, 1.0)
    yt = 5 * t * c * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                      + 0.2843 * xc**3 - 0.1015 * xc**4)
    if p > 0 and m > 0:
        yc = np.where(xc < p, m / p**2 * (2 * p * xc - xc**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2)) * c
        dyc = np.where(xc < p, 2 * m / p**2 * (p - xc), 2 * m / (1 - p)**2 * (p - xc))
    else:
        yc = np.zeros_like(x); dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu = x - yt * np.sin(th); yu = yc + yt * np.cos(th)
    xl = x + yt * np.sin(th); yl = yc - yt * np.cos(th)
    upper = np.column_stack([xu, yu])
    lower = np.column_stack([xl, yl])[::-1]
    loop = np.vstack([upper, lower[1:-1]])              # drop duplicate LE/TE endpoints
    return loop


def naca_wing(code: str = "0012", chord: float = 0.08, span: float = 0.12) -> trimesh.Trimesh:
    """A constant-chord NACA wing: chord along +x, thickness along y, span along z (flow +x sees the thin
    edge). `code` is a 4-digit NACA designation."""
    m = int(code[0]) / 100.0; p = int(code[1]) / 10.0; t = int(code[2:]) / 100.0
    loop = naca4(m, p, t, c=chord)
    poly = Polygon(loop)
    if not poly.is_valid:
        poly = poly.buffer(0)
    mesh = trimesh.creation.extrude_polygon(poly, height=span)
    mesh.apply_translation(-mesh.centroid)
    return mesh


# ---- bluff bodies -----------------------------------------------------------------------------------

def sphere(radius: float = 0.05) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    return m


def cylinder_crossflow(radius: float = 0.03, length: float = 0.12) -> trimesh.Trimesh:
    """Circular cylinder with its axis along z (perpendicular to the +x flow) — the classic crossflow
    bluff body."""
    return trimesh.creation.cylinder(radius=radius, height=length)


def cube(side: float = 0.08) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[side, side, side])


def ahmed_body(scale: float = 1.0, slant_deg: float = 25.0) -> trimesh.Trimesh:
    """Ahmed reference body (¹⁄₁₀ scale, no support legs), flow +x. Rounded fore-body (top & bottom edges)
    and a 25° rear slant — the canonical automotive bluff-with-tapered-rear. Frontal-area Cd_lit ≈ 0.29.
    Built as a side profile (x–z) extruded across the width y, so the fore-body side edges stay square
    (a documented simplification of the true ellipsoidal nose)."""
    L, W, H = 0.1044 * scale, 0.0389 * scale, 0.0288 * scale
    r = 0.010 * scale                                   # fore-body edge radius
    ls = 0.0222 * scale                                 # slant length
    phi = np.radians(slant_deg)
    lsx, lsz = ls * np.cos(phi), ls * np.sin(phi)
    xs = L - lsx
    arc = np.linspace(0, np.pi / 2, 8)
    # closed side profile, CCW: front verticals + rounded top/bottom front corners, top edge, rear slant,
    # rear vertical, bottom edge.
    top_arc = np.column_stack([r - r * np.cos(arc), (H - r) + r * np.sin(arc)])   # (0,H-r)->(r,H)
    bot_arc = np.column_stack([r - r * np.sin(arc), r - r * np.cos(arc)])         # (r,0)->(0,r)
    pts = np.vstack([
        [[0, r]],                                       # start of front vertical
        [[0, H - r]],
        top_arc,                                        # top-front round
        [[xs, H]],                                      # top edge
        [[L, H - lsz]],                                 # rear slant
        [[L, 0]],                                       # rear vertical
        [[r, 0]],                                       # bottom edge
        bot_arc,                                        # bottom-front round
    ])
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    # extrude along z, then rotate so the extrusion axis becomes the width y and the profile spans x–z
    prism = trimesh.creation.extrude_polygon(poly, height=W)   # profile in x–y, thickness in z
    R = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    prism.apply_transform(R)                            # y(profile-height)->z, z(width)->-y  → flow +x
    prism.apply_translation(-prism.centroid)
    return prism


def streamlined_body(radius: float = 0.018, length: float = 0.11) -> trimesh.Trimesh:
    """A rounded slender body (capsule) with its long axis along the +x flow — a low-drag streamlined
    reference at the opposite extreme from the sphere."""
    m = trimesh.creation.capsule(height=length, radius=radius)   # axis along z
    R = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    m.apply_transform(R)                                # z-axis -> x-axis (flow-aligned)
    m.apply_translation(-m.centroid)
    return m


# ---- swept wing -------------------------------------------------------------------------------------

def onera_m6(root_chord: float = 0.0806, taper: float = 0.56, semispan: float = 0.12,
             le_sweep_deg: float = 30.0, thick: float = 0.10) -> trimesh.Trimesh:
    """ONERA M6 planform (¹⁄₁₀ scale): swept, tapered half-wing with a symmetric section (NACA-00xx
    approximation of the ONERA D airfoil). Built as the convex hull of the root and tip sections — valid
    because a symmetric section is convex — giving a faithful straight-tapered swept wing."""
    tip_chord = root_chord * taper
    dx = semispan * np.tan(np.radians(le_sweep_deg))    # LE sweep offset at the tip
    root = naca4(0, 0, thick, c=root_chord)             # (x, y) with y = thickness
    tip = naca4(0, 0, thick, c=tip_chord)
    # place sections in 3D: x = chordwise (+flow), y = spanwise, z = thickness
    root3 = np.column_stack([root[:, 0], np.zeros(len(root)), root[:, 1]])
    tip3 = np.column_stack([tip[:, 0] + dx, np.full(len(tip), semispan), tip[:, 1]])
    hull = trimesh.Trimesh(vertices=np.vstack([root3, tip3])).convex_hull
    hull.apply_translation(-hull.centroid)
    return hull


# ---- registry ---------------------------------------------------------------------------------------

def all_benchmarks() -> list[dict]:
    """Build the full benchmark set. Each entry: {name, mesh, cd_lit, note}. cd_lit is a published
    frontal-area drag coefficient where a trustworthy one exists, else None."""
    specs = [
        ("sphere", sphere, 0.47, "subcritical sphere; steady RANS poor for separated wake"),
        ("cylinder", cylinder_crossflow, 1.10, "crossflow cylinder; unsteady wake, RANS approximate"),
        ("cube", cube, 1.05, "sharp-edged bluff reference"),
        ("ahmed_25deg", ahmed_body, 0.29, "Ahmed body 25deg slant, no legs (frontal-area Cd)"),
        ("streamlined_body", streamlined_body, None, "low-drag rounded slender body"),
        ("naca0012_wing", lambda: naca_wing("0012"), None, "symmetric wing, alpha=0"),
        ("naca2412_wing", lambda: naca_wing("2412"), None, "cambered wing, alpha=0"),
        ("naca4412_wing", lambda: naca_wing("4412"), None, "high-camber wing, alpha=0"),
        ("onera_m6_wing", onera_m6, None, "swept tapered wing, symmetric section"),
    ]
    out = []
    for name, fn, cd_lit, note in specs:
        try:
            m = fn()
            if m is None or not m.is_watertight or m.volume <= 0:
                out.append({"name": name, "mesh": None, "cd_lit": cd_lit,
                            "note": note + " [BUILD FAILED: not watertight]"})
                continue
            out.append({"name": name, "mesh": m, "cd_lit": cd_lit, "note": note})
        except Exception as e:                          # noqa: BLE001 — report, never crash the whole set
            out.append({"name": name, "mesh": None, "cd_lit": cd_lit,
                        "note": note + f" [BUILD ERROR: {e}]"})
    return out


if __name__ == "__main__":
    for b in all_benchmarks():
        m = b["mesh"]
        if m is None:
            print(f"{b['name']:<18} FAILED  {b['note']}")
            continue
        Afront = float(m.projected([1, 0, 0]).area)
        ext = m.bounding_box.extents.round(4)
        print(f"{b['name']:<18} wt={m.is_watertight!s:<5} faces={len(m.faces):<6} "
              f"bbox={ext} A_front={Afront:.5f} vol={m.volume:.6f}  cd_lit={b['cd_lit']}")
