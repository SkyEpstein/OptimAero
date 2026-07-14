"""Guard tests for the universal feature extractor and benchmark generators (no CFD).

The key invariant: FEATURE_NAMES length must equal the vector _core returns. A mismatch silently corrupts
every feature past the mismatch (the model still trains, but predicts on misaligned columns at serve time).
"""
import numpy as np
import trimesh

from optimaero.universal.features import universal_features, features_from_saved, FEATURE_NAMES
from optimaero.universal.benchmarks import all_benchmarks


def _sample_saved(mesh, npts=512, seed=0):
    """Build a saved-JSON-style dict from a mesh (mirrors the training-data schema)."""
    A_front = float(mesh.projected([1, 0, 0]).area)
    pts, fi = trimesh.sample.sample_surface(mesh, npts, seed=seed)
    return {"points": np.asarray(pts, float).tolist(),
            "normals": np.asarray(mesh.face_normals[fi], float).tolist(),
            "A_front": A_front, "A_wet": float(mesh.area), "vol": float(mesh.volume),
            "fineness": float(mesh.extents[0]) / max(2 * np.sqrt(A_front / np.pi), 1e-9)}


def test_feature_vector_matches_names_length():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    v = universal_features(m, "x")
    assert len(v) == len(FEATURE_NAMES), (len(v), len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate feature names"


def test_serve_and_train_paths_agree_in_shape_and_are_finite():
    m = trimesh.creation.box(extents=[0.1, 0.05, 0.05])
    serve = universal_features(m, "x")
    train = features_from_saved(_sample_saved(m))
    assert serve.shape == train.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(serve)) and np.all(np.isfinite(train))


def test_features_finite_on_degenerate_meshes():
    # thin plate (near-flat area distribution) and a tiny mesh — the divisions must stay guarded
    thin = trimesh.creation.box(extents=[0.12, 0.001, 0.1])
    tiny = trimesh.creation.icosphere(subdivisions=1, radius=1e-4)
    for m in (thin, tiny):
        v = universal_features(m, "x")
        assert v.shape == (len(FEATURE_NAMES),)
        assert np.all(np.isfinite(v)), f"non-finite feature on {m.extents}"


def test_flow_axis_invariance_of_feature_count():
    m = trimesh.creation.capsule(height=0.11, radius=0.02)
    for axis in ("x", "y", "z"):
        assert universal_features(m, axis).shape == (len(FEATURE_NAMES),)


def test_all_benchmarks_build_watertight_or_fail_closed():
    bms = all_benchmarks()
    assert len(bms) == 9
    for b in bms:
        assert set(b) >= {"name", "mesh", "cd_lit", "note"}
        if b["mesh"] is not None:
            assert b["mesh"].is_watertight and b["mesh"].volume > 0, b["name"]
