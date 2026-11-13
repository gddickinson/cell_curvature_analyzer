"""Default analysis parameters: the single place the GUI, the pipeline and the web app read them from.

smoothing_px and straight_threshold were chosen with scripts/curvature_benchmark.py (seeded masks of known curvature
with 0-0.6 px segmentation noise): sigma 6 px gives the best worst-shape sign agreement averaged over noise levels,
and 0.01/px the best balance of straight runs called straight vs curved points (|kappa| >= 1/100 px) signed right.
Edges with a radius above 100 px count as straight: lower the threshold for very large cells / fine pixel sizes.
Legacy (pre-2026-11) results: point_spacing "legacy", smoothing_px 0, straight_threshold 0.

Units: `depth` and `width` are in pixels (the analysis has no physical scale; multiply by the recording's
um_per_px to report them in micrometres). `min_cell_coverage` is a fraction 0-1.
"""
from types import MappingProxyType

DEFAULTS = MappingProxyType({
    "n_points": 100,            # equidistant sample points along the cell edge
    "depth": 20,                # px, how far the sampling rectangle extends into the cell
    "width": 5,                 # px, width of the sampling rectangle along the edge
    "min_cell_coverage": 0.8,   # fraction of the rectangle that must lie inside the mask
    "try_rotation": False,      # retry rejected rectangles pointing the other way (180 degrees)
    "exclude_endpoints": False, # skip the first and last points (useful for open, border-touching edges)
    "movement_threshold": 0.1,  # |normalised movement score| above this is extending/retracting
    "point_spacing": "equal",   # "equal" arc-length steps, or "legacy" (pre-2026-11 sampler; see edge_curvature)
    "smoothing_px": 6.0,        # px, Gaussian sigma along the edge before curvature; 0 = none (legacy 3-point)
    "straight_threshold": 0.01, # 1/px (radius 100 px), |curvature| below this is straight (sign 0); 0 = no dead-band
})
