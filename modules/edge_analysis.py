"""Qt-free core of the curvature analysis: cell edge, equidistant points, curvature, inward normals,
intensity sampling, edge movement and the curvature-intensity regression.

`CurvatureAnalyzer` (curvature_analysis.py) delegates to these functions; the web app imports them directly,
and the browser port (modules/web/static/edge_engine.js) is tested against them.

Coordinates are (row, col) pixel centres, as returned by skimage.measure.find_contours.

Point sampling, open/closed edges and curvature (options point_spacing, smoothing_px, straight_threshold, and
the sign convention: +1 convex / bulging out of the cell, -1 concave / bending into it, 0 straight) are documented
in edge_curvature.py, which also holds the code.
"""
import numpy as np
from scipy import stats
from skimage import draw, measure

from .analysis_defaults import DEFAULTS
from .edge_curvature import arc_positions, edge_curvature, is_closed, sample_edge_points, three_point_curvature


def detect_cell_edge(mask):
    """Longest 0.5-level contour of a binary mask, (K, 2) array of (row, col); empty array if none."""
    contours = measure.find_contours(mask, 0.5)
    if len(contours) > 0:
        return max(contours, key=len)
    return np.array([])


def sample_equidistant_points(contour, n_points=None, point_spacing=None):
    """n_points along the contour, starting at contour[0] (edge_curvature.sample_edge_points without positions)."""
    return sample_edge_points(contour, n_points, point_spacing)[0]


def measure_curvature(points, closed=True, straight_threshold=0.0):
    """(sign, magnitude, normalized) by the three-point rule (smoothing 'none'); see edge_curvature."""
    return three_point_curvature(points, closed, straight_threshold)


def calculate_inward_normal(points, mask, closed=True):
    """Unit normal at each point, chosen to point into the cell (tested 3 px along each direction).

    The tangent is the central difference of the neighbouring points; on an open edge the end points use their
    one neighbour (no wrap-around across the image).
    """
    n_points = len(points)
    normals = np.zeros((n_points, 2))
    mask_shape = mask.shape

    def at(j):
        return points[j % n_points] if closed else points[min(max(j, 0), n_points - 1)]
    for i in range(n_points):
        p = points[i]
        tangent = at(i + 1) - at(i - 1)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm
        else:
            tangent = at(i + 2) - at(i - 2)
            tangent_norm = np.linalg.norm(tangent)
            tangent = tangent / tangent_norm if tangent_norm > 1e-10 else np.array([1.0, 0.0])
        normal1 = np.array([-tangent[1], tangent[0]])
        normal2 = -normal1
        test_point1 = np.clip(p + 3 * normal1, [0, 0], [mask_shape[0] - 1, mask_shape[1] - 1])
        test_point2 = np.clip(p + 3 * normal2, [0, 0], [mask_shape[0] - 1, mask_shape[1] - 1])
        try:
            val1 = mask[int(round(test_point1[0])), int(round(test_point1[1]))]
            val2 = mask[int(round(test_point2[0])), int(round(test_point2[1]))]
            if val1 == 1:
                normals[i] = normal1
            elif val2 == 1:
                normals[i] = normal2
            else:
                normals[i] = normal2  # neither test point inside: default
        except (IndexError, ValueError):  # NaN points
            normals[i] = normal1
    return normals


def _rectangle(p, normal, depth, width):
    end_point = p + depth * normal
    perp = np.array([-normal[1], normal[0]])
    corner1, corner2 = p + width / 2 * perp, p - width / 2 * perp
    corner3, corner4 = end_point - width / 2 * perp, end_point + width / 2 * perp
    return np.array([corner1, corner4, corner3, corner2])


def _coverage(vertices, mask, image_shape):
    """(rr, cc, coverage) of the in-image pixels of a polygon; coverage None when it has no pixels."""
    rr, cc = draw.polygon(vertices[:, 0], vertices[:, 1], image_shape)
    ok = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
    if not np.any(ok):
        return rr, cc, None
    rr, cc = rr[ok], cc[ok]
    return rr, cc, np.sum(mask[rr, cc]) / len(rr)


def measure_intensity(image, mask, points, normals, depth=None, width=None, min_cell_coverage=None,
                      try_rotation=None, exclude_endpoints=None):
    """Mean image intensity in a depth x width rectangle reaching into the cell from each point.

    A point is valid when at least `min_cell_coverage` of the rectangle's pixels are inside the mask; with
    try_rotation a rejected rectangle is retried pointing the other way. Returns
    (intensities [NaN where invalid], sampling_regions [4x2 vertices], valid_points, recovered_by_rotation).
    """
    d = DEFAULTS
    depth = d["depth"] if depth is None else depth
    width = d["width"] if width is None else width
    min_cell_coverage = d["min_cell_coverage"] if min_cell_coverage is None else min_cell_coverage
    try_rotation = d["try_rotation"] if try_rotation is None else try_rotation
    exclude_endpoints = d["exclude_endpoints"] if exclude_endpoints is None else exclude_endpoints
    n_points = len(points)
    intensities = np.full(n_points, np.nan)
    sampling_regions = []
    valid_points = np.zeros(n_points, dtype=bool)
    recovered_by_rotation = np.zeros(n_points, dtype=bool)
    image_shape = image.shape
    for i in range(n_points):
        if exclude_endpoints and (i == 0 or i == n_points - 1):
            sampling_regions.append(np.zeros((4, 2)))
            continue
        p, normal = points[i], normals[i]
        vertices = _rectangle(p, normal, depth, width)
        rr, cc, cov = _coverage(vertices, mask, image_shape)
        if cov is not None:
            if cov >= min_cell_coverage:
                intensities[i] = np.mean(image[rr, cc])
                valid_points[i] = True
            elif try_rotation:
                # flipped rectangle, keeping the original perpendicular (same corner order as the desktop app)
                perp = np.array([-normal[1], normal[0]])
                end = p + depth * -normal
                rotated = np.array([p + width / 2 * perp, end + width / 2 * perp,
                                    end - width / 2 * perp, p - width / 2 * perp])
                rrr, rcc, rcov = _coverage(rotated, mask, image_shape)
                if rcov is not None and rcov >= min_cell_coverage:
                    intensities[i] = np.mean(image[rrr, rcc])
                    valid_points[i] = True
                    recovered_by_rotation[i] = True
                    vertices = rotated
        sampling_regions.append(vertices)
    return intensities, sampling_regions, valid_points, recovered_by_rotation


def analyze_points(image, mask, contour, points, depth=None, width=None, min_cell_coverage=None,
                   try_rotation=None, exclude_endpoints=None, smoothing_px=None, straight_threshold=None, arc=None):
    """Curvature, inward normals and intensity for already-sampled edge points (one frame).

    arc: arc position of each point along the contour (computed by projection when not given). On an open edge
    (one that runs off the image) the two end points are always excluded from intensity sampling.
    """
    closed = is_closed(contour)
    if arc is None:
        arc = arc_positions(contour, points)
    curvatures = edge_curvature(contour, points, arc, smoothing_px, straight_threshold)
    normals = calculate_inward_normal(points, mask, closed)
    exclude_endpoints = DEFAULTS["exclude_endpoints"] if exclude_endpoints is None else exclude_endpoints
    intensities, regions, valid, recovered = measure_intensity(
        image, mask, points, normals, depth, width, min_cell_coverage, try_rotation, exclude_endpoints or not closed)
    return {"points": points, "curvatures": curvatures, "normals": normals, "intensities": intensities,
            "sampling_regions": regions, "valid_points": valid, "contour_data": contour,
            "recovered_by_rotation": recovered, "closed": closed}


def analyze_frame(image, mask, n_points=None, point_spacing=None, **params):
    """Equidistant-sampling analysis of one frame (the desktop app's "Standard" method)."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    contour = detect_cell_edge(mask)
    if len(contour) == 0:
        return {"error": "No contour found"}
    points, arc = sample_edge_points(contour, n_points, point_spacing)
    return analyze_points(image, mask, contour, points, arc=arc, **params)


def detect_edge_movement(current_mask, previous_mask, threshold=None):
    """(normalised score, 'extending'|'retracting'|'stable', movement map) from two binary masks.

    score = (gained - lost pixels) / (gained + lost pixels), in [-1, 1]; 0 when nothing changed.
    """
    threshold = DEFAULTS["movement_threshold"] if threshold is None else threshold
    movement_map = np.zeros_like(current_mask, dtype=float)
    diff_mask = current_mask.astype(int) - previous_mask.astype(int)
    extension, retraction = diff_mask == 1, diff_mask == -1
    movement_map[extension] = 1.0
    movement_map[retraction] = -1.0
    ext, ret = np.sum(extension), np.sum(retraction)
    total = ext + ret
    score = (ext - ret) / total if total > 0 else 0.0
    kind = "extending" if score > threshold else "retracting" if score < -threshold else "stable"
    return score, kind, movement_map


def summary_correlation_stats(all_curvatures, all_intensities, all_valid_points):
    """Linear regression of intensity on curvature, pooled over frames (valid points only).

    Returns r_squared, p_value (two-sided), sample_size, slope, intercept, standard_error; the statistics are
    None when fewer than 2 points or when every curvature value is the same (regression undefined).
    """
    xs, ys = [], []
    for c, v, ok in zip(all_curvatures, all_intensities, all_valid_points):
        xs.extend(np.asarray(c)[ok])
        ys.extend(np.asarray(v)[ok])
    x, y = np.array(xs, dtype=float), np.array(ys, dtype=float)
    out = {"r_squared": None, "p_value": None, "sample_size": int(len(x))}
    if len(x) < 2 or np.all(x == x[0]):
        return out
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    out.update({"r_squared": float(r_value ** 2), "p_value": float(p_value), "slope": float(slope),
                "intercept": float(intercept), "standard_error": float(std_err)})
    return out
