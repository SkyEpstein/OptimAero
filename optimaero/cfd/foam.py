"""Capped OpenFOAM CFD labeler for envelope bodies.

Steady incompressible external-flow CFD (simpleFoam, k-omega SST) around a watertight body mesh,
run HARD resource-capped in Docker with a host-memory watchdog so it can never crash the Mac.
Returns drag/lift [N] + Cd/Cl. Coarse mesh (fast, ±30-50%) — the honest "truth" for the surrogate.

cfd_label(mesh, V, alpha_deg) is the entry point.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import numpy as np
import trimesh

IMAGE = "opencfd/openfoam-default:latest"
MEM = os.environ.get("OA_CFD_MEM", "12g")     # lower per-worker for parallel sweeps (e.g. 4g)
CPUS = os.environ.get("OA_CFD_CPUS", "6")     # lower per-worker for parallel sweeps (e.g. 3)
RHO, NU = 1.225, 1.5e-5


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _head(cls, obj):
    return (f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n"
            f"    object {obj};\n}}\n")


def write_case(mesh: trimesh.Trimesh, case_dir: str, V: float, alpha_deg: float = 0.0,
               A_ref: float = 0.01, refine: int = 3, layers: int = 0):
    """Write a minimal, robust simpleFoam + snappyHexMesh external-flow case."""
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    ext = np.asarray(mesh.bounding_box.extents, float)
    c = np.asarray(mesh.centroid, float)
    L = float(ext[0])
    ex, ey, ez = float(ext[0]), float(ext[1]), float(ext[2])
    # per-direction domain + cells (proportional to EACH body extent) so a flat/thin body is still
    # resolved by the background mesh — sizing the lateral cells by the big dimension made snappy
    # skip thin bodies (0 cells, no body patch, 0 forces).
    x0, x1 = c[0] - 3 * ex, c[0] + 8 * ex
    y0, y1 = c[1] - 4.5 * ey, c[1] + 4.5 * ey
    z0, z1 = c[2] - 4.5 * ez, c[2] + 4.5 * ez
    nx = max(24, int((x1 - x0) / (0.35 * ex)))
    ny = max(20, int((y1 - y0) / (0.35 * ey)))    # ~3 cells across the body in y before refinement
    nz = max(20, int((z1 - z0) / (0.35 * ez)))    # ~3 cells across the body in z (catches thin bodies)
    a = np.radians(alpha_deg)
    Ux, Uz = V * np.cos(a), V * np.sin(a)
    loc = (c[0] - 2.0 * ex, c[1] + 0.13 * ey, c[2] + 0.11 * ez)  # in fluid, upstream, strictly inside
    # turbulence inlet (I=5%, mixing length ~ 0.1 body)
    k = 1.5 * (0.05 * V) ** 2
    omega = k ** 0.5 / (0.09 ** 0.25 * 0.1 * max(L, 1e-3))

    tri_dir = os.path.join(case_dir, "constant", "triSurface")
    os.makedirs(tri_dir, exist_ok=True)
    mesh.export(os.path.join(tri_dir, "body.stl"))

    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    vtext = "\n".join(f"    ({vx} {vy} {vz})" for vx, vy, vz in verts)
    _w(os.path.join(case_dir, "system", "blockMeshDict"),
       "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
       "    object blockMeshDict;\n}\n\n"
       "scale 1;\n\n"
       f"vertices\n(\n{vtext}\n);\n\n"
       f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)\n);\n\n"
       "edges\n(\n);\n\n"
       "boundary\n(\n"
       "    inlet   { type patch; faces ( (0 4 7 3) ); }\n"
       "    outlet  { type patch; faces ( (1 2 6 5) ); }\n"
       "    walls   { type patch; faces ( (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }\n"
       ");\n\n"
       "mergePatchPairs\n(\n);\n")

    _w(os.path.join(case_dir, "system", "surfaceFeatureExtractDict"),
       "FoamFile{version 2.0;format ascii;class dictionary;object surfaceFeatureExtractDict;}\n"
       "body.stl{extractionMethod extractFromSurface;extractFromSurfaceCoeffs{includedAngle 150;}}\n")

    add_layers = "true" if layers > 0 else "false"
    layers_block = (f"layers {{ body {{ nSurfaceLayers {layers}; }} }}" if layers > 0
                    else "layers { }")
    _w(os.path.join(case_dir, "system", "snappyHexMeshDict"),
       "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
       "    object snappyHexMeshDict;\n}\n\n"
       f"castellatedMesh true;\nsnap true;\naddLayers {add_layers};\n\n"
       "geometry\n{\n    body { type triSurfaceMesh; file \"body.stl\"; }\n}\n\n"
       "castellatedMeshControls\n{\n"
       "    maxLocalCells 1000000;\n    maxGlobalCells 2000000;\n    minRefinementCells 10;\n"
       "    maxLoadUnbalance 0.10;\n    nCellsBetweenLevels 3;\n    features ( );\n"
       f"    refinementSurfaces\n    {{\n        body {{ level ({refine} {refine}); }}\n    }}\n"
       "    resolveFeatureAngle 30;\n    refinementRegions { }\n"
       f"    locationInMesh ({loc[0]} {loc[1]} {loc[2]});\n    allowFreeStandingZoneFaces true;\n}}\n\n"
       "snapControls\n{\n    nSmoothPatch 3;\n    tolerance 2.0;\n    nSolveIter 30;\n"
       "    nRelaxIter 5;\n}\n\n"
       "addLayersControls\n{\n    relativeSizes true;\n    " + layers_block + "\n    expansionRatio 1.2;\n"
       "    finalLayerThickness 0.5;\n    minThickness 0.05;\n    nGrow 0;\n    featureAngle 60;\n"
       "    nRelaxIter 3;\n    nSmoothSurfaceNormals 1;\n    nSmoothNormals 3;\n"
       "    nSmoothThickness 10;\n    maxFaceThicknessRatio 0.5;\n    maxThicknessToMedialRatio 0.3;\n"
       "    minMedialAxisAngle 90;\n    nBufferCellsNoExtrude 0;\n    nLayerIter 50;\n}\n\n"
       "meshQualityControls\n{\n    maxNonOrtho 65;\n    maxBoundarySkewness 20;\n"
       "    maxInternalSkewness 4;\n    maxConcave 80;\n    minVol 1e-13;\n    minTetQuality 1e-15;\n"
       "    minArea -1;\n    minTwist 0.02;\n    minDeterminant 0.001;\n    minFaceWeight 0.02;\n"
       "    minVolRatio 0.01;\n    minTriangleTwist -1;\n    nSmoothScale 4;\n    errorReduction 0.75;\n}\n\n"
       "mergeTolerance 1e-6;\n")

    ff = ("functions\n{\n"
          "    forceCoeffs\n    {\n"
          "        type forceCoeffs;\n        libs (forces);\n        patches (body);\n"
          f"        rho rhoInf;\n        rhoInf {RHO};\n"
          f"        liftDir ({-np.sin(a)} 0 {np.cos(a)});\n"
          f"        dragDir ({np.cos(a)} 0 {np.sin(a)});\n"
          "        CofR (0 0 0);\n        pitchAxis (0 1 0);\n"
          f"        magUInf {V};\n        lRef {L};\n        Aref {A_ref};\n"
          "        writeControl timeStep;\n        writeInterval 50;\n    }\n"
          "    forces\n    {\n"
          "        type forces;\n        libs (forces);\n        patches (body);\n"
          f"        rho rhoInf;\n        rhoInf {RHO};\n        CofR (0 0 0);\n"
          "        writeControl timeStep;\n        writeInterval 50;\n    }\n}\n")
    _w(os.path.join(case_dir, "system", "controlDict"),
       "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
       "    object controlDict;\n}\n\n"
       "application simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
       "endTime 1200;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1200;\npurgeWrite 1;\n"
       "writeFormat ascii;\nwritePrecision 7;\nwriteCompression off;\ntimeFormat general;\n"
       "timePrecision 6;\nrunTimeModifiable true;\n\n" + ff)

    _w(os.path.join(case_dir, "system", "fvSchemes"),
       "FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\n"
       "ddtSchemes{default steadyState;}\n"
       "gradSchemes{default Gauss linear;}\n"
       "divSchemes{default none;div(phi,U) bounded Gauss linearUpwind grad(U);"
       "div(phi,k) bounded Gauss upwind;div(phi,omega) bounded Gauss upwind;"
       "div((nuEff*dev2(T(grad(U))))) Gauss linear;}\n"
       "laplacianSchemes{default Gauss linear corrected;}\n"
       "interpolationSchemes{default linear;}\n"
       "snGradSchemes{default corrected;}\n"
       "wallDist{method meshWave;}\n")

    _w(os.path.join(case_dir, "system", "fvSolution"),
       "FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\n"
       "solvers{p{solver GAMG;tolerance 1e-6;relTol 0.05;smoother GaussSeidel;}"
       "\"(U|k|omega)\"{solver smoothSolver;smoother GaussSeidel;tolerance 1e-7;relTol 0.05;}}\n"
       "SIMPLE{nNonOrthogonalCorrectors 1;consistent yes;"
       "residualControl{p 1e-5;U 1e-5;\"(k|omega)\" 1e-5;}}\n"
       "relaxationFactors{equations{U 0.9;\".*\" 0.9;}}\n")

    _w(os.path.join(case_dir, "constant", "transportProperties"),
       "FoamFile{version 2.0;format ascii;class dictionary;object transportProperties;}\n"
       f"transportModel Newtonian;nu {NU};\n")
    _w(os.path.join(case_dir, "constant", "turbulenceProperties"),
       "FoamFile{version 2.0;format ascii;class dictionary;object turbulenceProperties;}\n"
       "simulationType RAS;RAS{RASModel kOmegaSST;turbulence on;printCoeffs on;}\n")

    def field(obj, cls, dims, internal, bcs):
        return (f"FoamFile{{version 2.0;format ascii;class {cls};object {obj};}}\n"
                f"dimensions {dims};internalField {internal};boundaryField{{{bcs}}}\n")
    _w(os.path.join(case_dir, "0", "U"), field(
        "U", "volVectorField", "[0 1 -1 0 0 0 0]", f"uniform ({Ux} 0 {Uz})",
        f"inlet{{type fixedValue;value uniform ({Ux} 0 {Uz});}}"
        "outlet{type inletOutlet;inletValue uniform (0 0 0);value uniform (0 0 0);}"
        "walls{type slip;}body{type noSlip;}"))
    _w(os.path.join(case_dir, "0", "p"), field(
        "p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0",
        "inlet{type zeroGradient;}outlet{type fixedValue;value uniform 0;}"
        "walls{type slip;}body{type zeroGradient;}"))
    _w(os.path.join(case_dir, "0", "k"), field(
        "k", "volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k}",
        f"inlet{{type fixedValue;value uniform {k};}}"
        f"outlet{{type inletOutlet;inletValue uniform {k};value uniform {k};}}"
        "walls{type slip;}body{type kqRWallFunction;value uniform " + f"{k};}}"))
    _w(os.path.join(case_dir, "0", "omega"), field(
        "omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {omega}",
        f"inlet{{type fixedValue;value uniform {omega};}}"
        f"outlet{{type inletOutlet;inletValue uniform {omega};value uniform {omega};}}"
        "walls{type slip;}body{type omegaWallFunction;value uniform " + f"{omega};}}"))
    _w(os.path.join(case_dir, "0", "nut"), field(
        "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0",
        "inlet{type calculated;value uniform 0;}outlet{type calculated;value uniform 0;}"
        "walls{type calculated;value uniform 0;}body{type nutkWallFunction;value uniform 0;}"))
    return dict(L=L, A_ref=A_ref)


def _mem_free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        try:                                        # macOS: available ≈ free+spec+inactive+purgeable
            out = subprocess.check_output(["vm_stat"]).decode()
            page, vals = 4096, {}
            for line in out.splitlines():
                if "page size of" in line:
                    page = int(line.split("of")[1].split("bytes")[0])
                for key in ("Pages free", "Pages speculative", "Pages inactive", "Pages purgeable"):
                    if line.startswith(key):
                        vals[key] = int(line.split(":")[1].strip().rstrip("."))
            return sum(vals.values()) * page / 1e9
        except Exception:
            return 99.0


def run_case(case_dir: str, timeout: int = 1500, min_free_gb: float = 3.0) -> bool:
    """Run blockMesh + snappyHexMesh + simpleFoam in a HARD-capped container with a host-memory
    watchdog. Returns True on completion. Never lets host free RAM fall below min_free_gb."""
    name = "oa_cfd_" + str(abs(hash(case_dir)) % 10 ** 8)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    script = ("set -e; cd /case; "
              "surfaceFeatureExtract > log.sfe 2>&1 || true; "
              "blockMesh > log.blockMesh 2>&1; "
              "snappyHexMesh -overwrite > log.snappy 2>&1; "
              "simpleFoam > log.simpleFoam 2>&1; echo DONE")
    cmd = ["docker", "run", "--rm", "--name", name,
           "--memory", MEM, "--memory-swap", MEM, "--cpus", CPUS,
           "-v", f"{os.path.abspath(case_dir)}:/case", IMAGE, "bash", "-lc", script]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    killed = {"v": False}

    def watchdog():
        t0 = time.time()
        while proc.poll() is None:
            if _mem_free_gb() < min_free_gb:
                killed["v"] = True
                subprocess.run(["docker", "kill", name], capture_output=True)
                break
            if time.time() - t0 > timeout:
                subprocess.run(["docker", "kill", name], capture_output=True)
                break
            time.sleep(3)

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()
    try:
        proc.wait(timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return not killed["v"]


def read_forces(case_dir: str):
    """Parse the last forceCoeffs row → (Cd, Cl). Searches postProcessing for coefficient.dat."""
    root = os.path.join(case_dir, "postProcessing", "forceCoeffs")
    if not os.path.isdir(root):
        return None
    dat = None
    for sub in sorted(os.listdir(root)):
        for fn in ("coefficient.dat", "forceCoeffs.dat"):
            p = os.path.join(root, sub, fn)
            if os.path.exists(p):
                dat = p
    if not dat:
        return None
    cols, last = None, None
    with open(dat) as f:
        for line in f:
            if line.startswith("#"):
                if "Time" in line or "Cd" in line:
                    cols = line.lstrip("#").split()
                continue
            if line.strip():
                last = line.split()
    if not last or not cols:
        return None
    idx = {c: i for i, c in enumerate(cols)}
    try:
        cd = float(last[idx.get("Cd", 1)])
        cl = float(last[idx.get("Cl", 3 if "Cl" not in idx else idx["Cl"])])
        return cd, cl
    except Exception:
        return None


def cfd_label(mesh: trimesh.Trimesh, V: float, alpha_deg: float = 0.0, case_dir: str | None = None,
              refine: int = 3, layers: int = 0):
    """Full pipeline: write case → capped run → parse. Returns dict(drag,lift,Cd,Cl,converged).
    refine = snappyHexMesh refinement level; layers = boundary-layer count (0 = wall functions only,
    >0 resolves the near-wall flow for accurate skin-friction drag)."""
    case_dir = case_dir or os.path.join("/tmp", "oa_cfd_case")
    try:
        A_front = float(mesh.projected([1, 0, 0]).area)
    except Exception:
        e = mesh.bounding_box.extents
        A_front = float(e[1] * e[2])
    info = write_case(mesh, case_dir, V, alpha_deg, A_ref=max(A_front, 1e-6), refine=refine,
                      layers=layers)
    ok = run_case(case_dir)
    forces = read_forces(case_dir) if ok else None
    if not forces:
        return {"drag": None, "lift": None, "Cd": None, "Cl": None, "converged": False}
    cd, cl = forces
    if not (np.isfinite(cd) and np.isfinite(cl)) or abs(cd) > 10.0 or abs(cl) > 10.0:
        # diverged case that still wrote a value — physically impossible coefficients; reject it
        return {"drag": None, "lift": None, "Cd": None, "Cl": None, "converged": False}
    q = 0.5 * RHO * V ** 2
    return {"drag": cd * q * info["A_ref"], "lift": cl * q * info["A_ref"],
            "Cd": cd, "Cl": cl, "converged": True, "A_ref": info["A_ref"]}
