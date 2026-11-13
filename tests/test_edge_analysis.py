"""Tests for the Qt-free analysis core (modules/edge_analysis.py), including the curvature sign convention."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import edge_analysis as E  # noqa: E402
from modules.analysis_defaults import DEFAULTS  # noqa: E402


def disc(shape=(120, 120), center=(60, 60), radius=40):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 < radius ** 2).astype(np.uint8)


def test_contour_runs_clockwise_with_the_cell_on_its_right():
    """find_contours orientation that the sign convention relies on (screen view, y downwards)."""
    c = E.detect_cell_edge(disc())
    assert np.allclose(c[0], c[-1])                      # closed
    x, y = c[:, 1], c[:, 0]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])   # > 0 is clockwise on screen (y down)
    assert signed_area > 0


@pytest.mark.parametrize("radius,n_points", [(40, 24), (25, 12), (50, 30)])
def test_circle_is_convex_everywhere(radius, n_points):
    """Convention: a convex edge (bulging out of the cell) has sign +1, and on a circle every point is convex.

    Needs sample spacing well above the pixel staircase of the mask (here >= 8 px); with closer points the
    sign of single points is dominated by pixelation (see test_close_points_on_a_pixel_circle_are_noisy).
    """
    mask = disc(radius=radius)
    pts = E.sample_equidistant_points(E.detect_cell_edge(mask), n_points)
    sign, mag, norm = E.measure_curvature(pts)
    assert np.all(sign == 1)
    # magnitude = (distance from chord) / chord = tan(theta/2)/2 for a circle, theta = 2 pi / n_points
    expected = np.tan(np.pi / n_points) / 2
    assert abs(np.median(mag) - expected) / expected < 0.1
    assert np.all((norm > 0) & (norm <= 1)) and norm.max() == 1


def test_close_points_on_a_pixel_circle_are_noisy():
    """Why smoothing is the default: with the legacy three-point rule (no smoothing), at ~4 px spacing some points on
    a pixel circle get sign -1 (tests/test_edge_curvature.py covers the smoothed default)."""
    pts = E.sample_equidistant_points(E.detect_cell_edge(disc(radius=40)), 60)
    sign = E.measure_curvature(pts)[0]
    assert 0.5 < np.mean(sign == 1) < 1


def test_circle_magnitude_is_scale_free():
    """Same number of points on a bigger circle gives the same magnitude (the measure is dimensionless)."""
    m1 = np.median(E.measure_curvature(E.sample_equidistant_points(E.detect_cell_edge(disc(radius=25)), 16))[1])
    m2 = np.median(E.measure_curvature(E.sample_equidistant_points(E.detect_cell_edge(disc(radius=50)), 16))[1])
    assert abs(m1 - m2) / m1 < 0.05


def test_hole_in_the_cell_is_concave():
    """A circular notch cut into the cell edge bends into the cell: sign -1 there, +1 on the convex rest."""
    mask = disc(radius=45)
    yy, xx = np.mgrid[:120, :120]
    mask[(yy - 60) ** 2 + (xx - 105) ** 2 < 18 ** 2] = 0       # bite out of the right-hand side
    pts = E.sample_equidistant_points(E.detect_cell_edge(mask), 40)
    sign = E.measure_curvature(pts)[0]
    inside_bite = (pts[:, 0] - 60) ** 2 + (pts[:, 1] - 105) ** 2 < 19.5 ** 2
    far = (pts[:, 1] < 40)
    assert inside_bite.sum() >= 5 and np.all(sign[inside_bite][1:-1] == -1)  # ends sit on the convex corners
    assert np.all(sign[far] == 1)


def test_wound_edge_bump_is_positive():
    """Open edge (cell below, wound above): a bump into the wound is +1 at its tip, the dip next to a bump -1."""
    yy, xx = np.mgrid[:100, :160]
    edge = 50 - 14 * np.exp(-((xx - 80) / 12.0) ** 2) + 10 * np.exp(-((xx - 30) / 10.0) ** 2)
    mask = (yy > edge).astype(np.uint8)
    pts = E.sample_equidistant_points(E.detect_cell_edge(mask), 50)
    sign = E.measure_curvature(pts)[0]
    tip = np.argmin(np.abs(pts[:, 1] - 80) + 100 * (pts[:, 0] > 45))
    dip = np.argmin(np.abs(pts[:, 1] - 30) + 100 * (pts[:, 0] < 55))
    assert sign[tip] == 1 and sign[dip] == -1


def test_straight_run_counts_as_minus_one():
    """Legacy rule (threshold 0): a collinear point (cross product 0) gets sign -1 and magnitude 0; the dead-band
    (straight_threshold > 0, the default) makes it 0 (tests/test_edge_curvature.py)."""
    pts = np.column_stack([np.zeros(5), np.arange(5.0)])
    sign, mag, _ = E.measure_curvature(pts)
    assert sign[2] == -1 and mag[2] == 0


def test_inward_normals_point_into_the_cell():
    mask = disc()
    pts = E.sample_equidistant_points(E.detect_cell_edge(mask), 30)
    normals = E.calculate_inward_normal(pts, mask)
    to_centre = np.array([60.0, 60.0]) - pts
    assert np.all(np.sum(normals * to_centre, axis=1) > 0)


def test_analyze_frame_samples_inside_and_uses_defaults():
    mask = disc()
    image = np.where(mask > 0, 200.0, 10.0)
    r = E.analyze_frame(image, mask)
    assert len(r["points"]) == DEFAULTS["n_points"]
    assert r["valid_points"].all() and np.all(r["intensities"] >= 0.8 * 200 + 0.2 * 10)


def test_rotation_recovers_flipped_regions():
    """A one-pixel-thick sliver fails coverage both ways; a wide cell with outward normals recovers by rotation."""
    mask = disc()
    image = np.ones(mask.shape)
    pts = E.sample_equidistant_points(E.detect_cell_edge(mask), 20)
    outward = -E.calculate_inward_normal(pts, mask)
    _, _, valid, rec = E.measure_intensity(image, mask, pts, outward, 10, 3, 0.8, try_rotation=True)
    assert valid.all() and rec.all()
    _, _, valid, _ = E.measure_intensity(image, mask, pts, outward, 10, 3, 0.8, try_rotation=False)
    assert not valid.any()


def test_edge_movement_and_constant_curvature_stats():
    a, b = disc(radius=30), disc(radius=35)
    assert E.detect_edge_movement(b, a)[1] == "extending"
    assert E.detect_edge_movement(a, b)[1] == "retracting"
    assert E.detect_edge_movement(a, a)[:2] == (0.0, "stable")
    s = E.summary_correlation_stats([np.ones(5)], [np.arange(5.0)], [np.ones(5, bool)])
    assert s["r_squared"] is None and s["sample_size"] == 5


def test_regression_matches_scipy():
    from scipy import stats
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=40), rng.normal(size=40)
    s = E.summary_correlation_stats([x[:20], x[20:]], [y[:20], y[20:]], [np.ones(20, bool)] * 2)
    ref = stats.linregress(x, y)
    assert s["sample_size"] == 40 and np.isclose(s["p_value"], ref.pvalue) and np.isclose(s["slope"], ref.slope)


def test_curvature_analyzer_delegates_to_core():
    pytest.importorskip("PyQt5")
    from modules.curvature_analysis import CurvatureAnalyzer
    rng = np.random.default_rng(0)
    mask = disc()
    image = rng.random(mask.shape) * 100
    r1 = CurvatureAnalyzer().analyze_frame(image, mask, n_points=30, depth=8, width=3)
    r2 = E.analyze_frame(image, mask, n_points=30, depth=8, width=3)
    for k in ("points", "normals", "intensities", "valid_points"):
        np.testing.assert_array_equal(r1[k], r2[k])
