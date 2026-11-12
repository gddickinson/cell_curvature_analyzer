"""FastAPI web app for Cell Curvature Analyzer: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8208, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.
POST /api/analyze runs the repo's Python analysis (modules/edge_analysis.py via web/analysis.py); on the static
site static_api.js answers it with the JS port (edge_engine.js), which edge_engine.test.mjs checks against Python.
"""
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..analysis_defaults import DEFAULTS
from ..synthetic_cell import DEMO_SPEC, make_demo_stack
from .analysis import analyze_stack, decode, encode

STATIC = Path(__file__).resolve().parent / "static"
MAX_PIXELS = 40_000_000  # frames x rows x cols accepted by POST /api/analyze

app = FastAPI(title="Cell Curvature Analyzer")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary and the analysis defaults (modules/analysis_defaults.py) for the pages."""
    return {
        "title": "Cell Curvature Analyzer",
        "description": "Membrane curvature along the cell edge, fluorescence sampled just inside it, and how the two "
                       "relate, from a fluorescence image and a binary cell mask.",
        "repo": "cell_curvature_analyzer",
        "defaults": dict(DEFAULTS),
        "curvature_measures": ["sign (+1 convex, -1 concave or straight)", "magnitude (|cross| / chord^2, >= 0)",
                               "normalized (sign x magnitude / frame max, -1..1)"],
        "desktop_tabs": ["Visualization", "Analysis Results", "Correlation Analysis", "Temporal Analysis",
                         "Edge Movement", "Export Results", "Settings", "Log"],
        "sampling_methods": {"web": ["equidistant along the contour (desktop 'Standard')"],
                             "desktop_only": ["angular", "axis-based", "x-axis"]},
    }


@app.get("/api/demo")
def demo():
    """The synthetic demo stack (modules/synthetic_cell.py) and the Python analysis of it with default parameters."""
    images, masks = make_demo_stack()
    spec = {k: (list(v) if isinstance(v, tuple) else v) for k, v in DEMO_SPEC.items()}
    return {"shape": list(images.shape), "image_u16": encode(images, "u2"), "mask_u8": encode(masks, "u1"),
            "spec": spec, "source": "synthetic: modules/synthetic_cell.py (seeded), not measured data",
            "result": analyze_stack(images, masks)}


@app.post("/api/analyze")
def analyze(payload: dict = Body(...)):
    """{shape: [T, H, W], image_f32: base64 float32 LE, mask_u8: base64 uint8, params: {...}} -> analysis JSON."""
    try:
        shape = [int(v) for v in payload["shape"]]
        if len(shape) != 3 or min(shape) < 1 or shape[1] < 2 or shape[2] < 2:
            raise ValueError("shape must be [frames, rows, cols] with at least 2 x 2 pixels")
        if shape[0] * shape[1] * shape[2] > MAX_PIXELS:
            raise HTTPException(413, "Stack too large for the web app; use the desktop app.")
        images = decode(payload["image_f32"], "f4", shape)
        masks = decode(payload["mask_u8"], "u1", shape)
        return analyze_stack(images, masks, payload.get("params") or {})
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, f"Bad request: {e}")


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/demo"]
