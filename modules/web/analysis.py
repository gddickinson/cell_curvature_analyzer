"""Stack analysis for the web app, on top of the Qt-free core (modules.edge_analysis).

analyze_stack follows CurvatureAnalyzer.process_stack (equidistant sampling): per-frame curvature and intensity,
edge movement for each frame with a cell after the first, and the intensity-vs-curvature regression pooled over
frames for the sign and the normalised curvature. The browser port (static/edge_engine.js::analyzeStack) returns
the same JSON.
"""
import base64

import numpy as np

from .. import edge_analysis as E
from ..analysis_defaults import DEFAULTS

PARAM_KEYS = ("n_points", "depth", "width", "min_cell_coverage", "try_rotation", "exclude_endpoints",
              "movement_threshold", "point_spacing", "smoothing_px", "straight_threshold")


def clean_params(params=None):
    """Defaults from analysis_defaults.DEFAULTS, overridden by known keys; validated."""
    p = dict(DEFAULTS)
    for k, v in (params or {}).items():
        if k in PARAM_KEYS and v is not None:
            p[k] = type(DEFAULTS[k])(v)
    if not (3 <= p["n_points"] <= 2000):
        raise ValueError("n_points must be between 3 and 2000")
    if p["depth"] <= 0 or p["width"] <= 0:
        raise ValueError("depth and width must be positive")
    if not (0 <= p["min_cell_coverage"] <= 1):
        raise ValueError("min_cell_coverage must be between 0 and 1")
    if p["point_spacing"] not in ("equal", "legacy"):
        raise ValueError("point_spacing must be 'equal' or 'legacy'")
    if not (0 <= p["smoothing_px"] <= 50):
        raise ValueError("smoothing_px must be between 0 (none, legacy) and 50")
    if not (0 <= p["straight_threshold"] <= 1):
        raise ValueError("straight_threshold must be between 0 and 1 (1/px)")
    return p


def _list(a):
    return np.asarray(a, dtype=float).tolist()


def _frame_json(t, r):
    if "error" in r:
        return {"frame": t, "error": r["error"]}
    sign, mag, norm = r["curvatures"]
    return {"frame": t, "contour": _list(r["contour_data"]), "points": _list(r["points"]),
            "sign": _list(sign), "magnitude": _list(mag), "normalized": _list(norm), "normals": _list(r["normals"]),
            "intensities": [None if np.isnan(v) else float(v) for v in r["intensities"]],
            "regions": [_list(v) for v in r["sampling_regions"]], "valid": [bool(v) for v in r["valid_points"]],
            "recovered": [bool(v) for v in r["recovered_by_rotation"]], "closed": bool(r["closed"])}


def analyze_stack(images, masks, params=None):
    """images [T, H, W] (any numeric), masks [T, H, W] (>0 = cell). Returns a JSON-ready dict."""
    p = clean_params(params)
    images = np.asarray(images, dtype=float)
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    if images.ndim == 2:
        images, masks = images[None], masks[None] if masks.ndim == 2 else masks
    if images.shape != masks.shape:
        raise ValueError(f"image stack {images.shape} and mask stack {masks.shape} differ in shape")
    kw = {k: p[k] for k in ("depth", "width", "min_cell_coverage", "try_rotation", "exclude_endpoints",
                            "smoothing_px", "straight_threshold")}
    frames, movement, curv_s, curv_n, ints, valid = [], [], [], [], [], []
    for t in range(images.shape[0]):
        r = E.analyze_frame(images[t], masks[t], p["n_points"], p["point_spacing"], **kw)
        frames.append(_frame_json(t, r))
        if "error" in r:
            continue
        curv_s.append(r["curvatures"][0]); curv_n.append(r["curvatures"][2])
        ints.append(r["intensities"]); valid.append(r["valid_points"])
        if t > 0:
            score, kind, mmap = E.detect_edge_movement(masks[t], masks[t - 1], p["movement_threshold"])
            movement.append({"frame": t, "score": float(score), "type": kind,
                             "extended_px": int(np.sum(mmap == 1)), "retracted_px": int(np.sum(mmap == -1))})
    return {"shape": list(images.shape[1:]), "n_frames": int(images.shape[0]), "params": p, "frames": frames,
            "movement": movement,
            "stats": {"sign": E.summary_correlation_stats(curv_s, ints, valid),
                      "normalized": E.summary_correlation_stats(curv_n, ints, valid)}}


def encode(a, dtype):
    """Little-endian base64 of an array cast to dtype (for the JSON API and fixtures)."""
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()).decode()


def decode(s, dtype, shape):
    return np.frombuffer(base64.b64decode(s), dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape)
