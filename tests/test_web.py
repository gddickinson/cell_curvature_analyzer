"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific -------------------------------------------------------------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402

from modules.analysis_defaults import DEFAULTS  # noqa: E402
from modules.synthetic_cell import make_demo_stack  # noqa: E402
from modules.web.analysis import analyze_stack, decode, encode  # noqa: E402

FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "edge_fixture.json").read_text())


def _close(a, b, what=""):
    """Recursive comparison, rtol 1e-6 / atol 1e-9 (libm differs in the last digit between macOS and Linux)."""
    if isinstance(b, dict):
        assert set(a) == set(b), what
        for k in b:
            _close(a[k], b[k], f"{what}.{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), what
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{what}[{i}]")
    elif isinstance(b, float) and a is not None:
        assert abs(a - b) <= 1e-9 + 1e-6 * abs(b), f"{what}: {a} vs {b}"
    else:
        assert a == b, f"{what}: {a} vs {b}"


# Curvature signs and the valid-point choice are discrete decisions on float maths: on another platform a value
# within ~1e-15 of a threshold may flip, so the fixture comparisons are skipped on CI (the node test still runs).
ON_CI = pytest.mark.skipif(bool(os.environ.get("CI")), reason="discrete choices from float maths (KIT rule)")


@ON_CI
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_fixture_is_current(case):
    """The committed fixture still matches the Python core (regenerate: python scripts/make_edge_fixture.py)."""
    shape = case["shape"]
    im = decode(case["image"], case["image_dtype"], shape).astype(float)
    m = decode(case["mask_u8"], "u1", shape)
    _close(analyze_stack(im, m, case["params"]), case["expected"], case["name"])


@ON_CI
def test_analyze_endpoint_matches_core():
    case = next(c for c in FIXTURE["cases"] if c["name"] == "wound_with_empty_frame")
    im = decode(case["image"], case["image_dtype"], case["shape"]).astype(np.float32)
    body = {"shape": case["shape"], "image_f32": encode(im, "f4"), "mask_u8": case["mask_u8"], "params": case["params"]}
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200, r.text
    _close(r.json(), case["expected"])
    assert r.json()["frames"][1]["error"] == "No contour found"


def test_analyze_bad_inputs():
    assert client.post("/api/analyze", json={"shape": [1, 4, 4]}).status_code == 400
    ok = {"shape": [1, 4, 4], "image_f32": encode(np.zeros((1, 4, 4)), "f4"), "mask_u8": encode(np.zeros((1, 4, 4)), "u1")}
    assert client.post("/api/analyze", json={**ok, "params": {"n_points": 1}}).status_code == 400
    assert client.post("/api/analyze", json={**ok, "shape": [1, 5, 4]}).status_code == 400
    assert client.post("/api/analyze", json={**ok, "shape": [10000, 10000, 1000]}).status_code in (400, 413)
    assert client.post("/api/analyze", json=ok).json()["frames"][0]["error"] == "No contour found"


def test_demo_is_synthetic_and_reproducible():
    d = client.get("/api/demo").json()
    assert "synthetic" in d["source"]
    a, m = make_demo_stack()
    np.testing.assert_array_equal(decode(d["image_u16"], "u2", d["shape"]), a)
    assert d["result"]["params"] == dict(DEFAULTS)
    # the enrichment is built in at concave edge: the regression on normalized curvature has a negative slope
    assert d["result"]["stats"]["normalized"]["slope"] < 0


def test_info_defaults_are_the_single_source():
    assert client.get("/api/info").json()["defaults"] == dict(DEFAULTS)


def test_web_stack_analysis_matches_desktop_process_stack(tmp_path):
    """analyze_stack gives the same statistics as CurvatureAnalyzer.process_stack (the desktop app's path)."""
    pytest.importorskip("PyQt5")
    pytest.importorskip("matplotlib")
    import tifffile
    from modules.curvature_analysis import CurvatureAnalyzer
    images, masks = make_demo_stack()
    tifffile.imwrite(tmp_path / "i.tif", images)
    tifffile.imwrite(tmp_path / "m.tif", masks * 255)
    desk = CurvatureAnalyzer().process_stack(str(tmp_path / "i.tif"), str(tmp_path / "m.tif"), str(tmp_path / "o"))
    web = analyze_stack(images, masks)
    for key, which in (("sign_curvature_correlation", "sign"), ("normalized_curvature_correlation", "normalized")):
        for k, v in desk["statistics"][key].items():
            assert web["stats"][which][k] == pytest.approx(v, rel=1e-12)
    assert [m["type"] for m in web["movement"]] == desk["movement_types"]
