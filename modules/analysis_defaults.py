"""Default analysis parameters: the single place the GUI, the pipeline and the web app read them from.

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
})
