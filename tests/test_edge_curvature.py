"""Edge geometry options (modules/edge_curvature.py): open edges, point spacing, smoothing and the straight dead-band."""
import numpy as np
import pytest

from modules import edge_analysis as E
from modules import edge_curvature as C
from modules.analysis_defaults import DEFAULTS

LEGACY = {"point_spacing": "legacy", "smoothing_px": 0.0, "straight_threshold": 0.0}


def disc(h=120, w=120, cy=60, cx=60, radius=40):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2).astype(np.uint8)


def wound_mask(far_end_bump=0.0):
    yy, xx = np.mgrid[:100, :160]
    edge = 50 - 14 * np.exp(-((xx - 80) / 12.0) ** 2) + far_end_bump * np.exp(-((xx - 150) / 6.0) ** 2)
    return (yy > edge).astype(np.uint8)


# --- the pre-2026-11 code, verbatim, to pin that the legacy options reproduce it -------------------------------
def old_sample(contour, n_points):
    contour_length = 0
    for i in range(len(contour) - 1):
        contour_length += np.linalg.norm(contour[i + 1] - contour[i])
    step_size = contour_length / n_points
    sampled_points = [contour[0]]
    current_length = 0
    i = 0
    while len(sampled_points) < n_points and i < len(contour) - 1:
        dist = np.linalg.norm(contour[i + 1] - contour[i])
        if current_length + dist >= step_size:
            t = (step_size - current_length) / dist
            sampled_points.append(contour[i] + t * (contour[i + 1] - contour[i]))
            current_length = 0
        else:
            current_length += dist
            i += 1
    return np.array(sampled_points)


def old_curvature(points):
    n = len(points)
    sign, mag = np.zeros(n), np.zeros(n)
    for i in range(n):
        p, pl, pr = points[i], points[(i - 1) % n], points[(i + 1) % n]
        v, w = pr - pl, p - pl
        vl = np.linalg.norm(v)
        cross = v[0] * w[1] - v[1] * w[0]
        mag[i] = abs(cross) / (vl * vl) if vl > 0 else 0
        sign[i] = 1 if cross > 0 else -1
    return sign, mag


@pytest.mark.parametrize("radius,n", [(40, 100), (25, 37), (50, 12)])
def test_legacy_options_reproduce_the_old_results_on_closed_edges(radius, n):
    contour = E.detect_cell_edge(disc(radius=radius))
    pts, s = C.sample_edge_points(contour, n, "legacy")
    np.testing.assert_array_equal(pts, old_sample(contour, n))
    np.testing.assert_allclose(C.arc_positions(contour, pts), s, atol=1e-9)
    sign, mag, _ = C.edge_curvature(contour, pts, s, 0.0, 0.0)
    os_, om = old_curvature(pts)
    np.testing.assert_array_equal(sign, os_)
    np.testing.assert_array_equal(mag, om)


def test_legacy_spacing_leaves_one_large_gap():
    """The bug the 'equal' spacing fixes: 100 points on a 40 px circle are ~2 px apart, then a ~42 px gap."""
    contour = E.detect_cell_edge(disc(radius=40))
    pts = C.sample_edge_points(contour, 100, "legacy")[0]
    gaps = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    assert gaps[-1] > 10 * np.median(gaps[:-1])


def test_equal_spacing_closed_and_open():
    contour = E.detect_cell_edge(disc(radius=40))
    pts, s = C.sample_edge_points(contour, 100, "equal")
    L = C.arc_length(contour)[-1]
    np.testing.assert_allclose(np.diff(np.append(s, L)), L / 100)
    chords = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    assert chords.max() - chords.min() < 0.25      # chords of equal arcs on a pixel circle
    open_c = E.detect_cell_edge(wound_mask())
    assert not C.is_closed(open_c) and C.is_closed(contour)
    pts, s = C.sample_edge_points(open_c, 30, "equal")
    np.testing.assert_allclose(pts[[0, -1]], open_c[[0, -1]])  # both ends of an open edge are sampled
    with pytest.raises(ValueError):
        C.sample_edge_points(contour, 10, "bogus")


def test_open_edge_does_not_wrap_round():
    """Open (border-touching) edge: the end points are not neighbours. Moving the far end of the edge leaves the
    curvature (three-point and smoothed) and the normals at the start unchanged."""
    contour = E.detect_cell_edge(wound_mask())
    assert not C.is_closed(contour)
    moved = contour.copy()
    moved[-30:, 0] -= 6.0                                   # far (right-hand) end of the edge bends up
    pts, s = C.sample_edge_points(contour, 40)
    pts2 = pts.copy()
    pts2[-1, 0] -= 6.0
    for k in (0, 1):
        assert C.three_point_curvature(pts, closed=False)[k][:3] == pytest.approx(
            C.three_point_curvature(pts2, closed=False)[k][:3], abs=0)
    # (only the resampling step L / (n - 1) changes with the total length: differences ~1e-6, not a wrap-around)
    assert C.smoothed_curvature(contour, s[:5], 6.0) == pytest.approx(C.smoothed_curvature(moved, s[:5], 6.0), abs=1e-5)
    mask = wound_mask()
    np.testing.assert_array_equal(E.calculate_inward_normal(pts, mask, closed=False)[:3],
                                  E.calculate_inward_normal(pts2, mask, closed=False)[:3])


@pytest.mark.parametrize("smoothing", [0.0, DEFAULTS["smoothing_px"]])
def test_open_edge_end_points(smoothing):
    """End points of an open edge are not sampled for intensity, normals point into the cell (below), and the
    three-point rule gives them no curvature."""
    image = np.random.default_rng(0).random((100, 160))
    a = E.analyze_frame(image, wound_mask(), 40, smoothing_px=smoothing)
    assert a["closed"] is False
    assert not a["valid_points"][0] and not a["valid_points"][-1] and np.isnan(a["intensities"][[0, -1]]).all()
    assert np.all(a["normals"][:, 0] > 0)
    if smoothing == 0:
        assert a["curvatures"][0][0] == 0 and a["curvatures"][0][-1] == 0


def test_old_code_paired_the_ends_of_an_open_edge():
    """What the fix changes: the old three-point rule used the last point as the first point's neighbour."""
    pts = C.sample_edge_points(E.detect_cell_edge(wound_mask()), 40, "legacy")[0]
    p0, pl, pr = pts[0], pts[-1], pts[1]
    assert np.linalg.norm(pl - p0) > 100   # the "neighbour" is across the image
    assert old_curvature(pts)[1][0] != 0 and C.three_point_curvature(pts, closed=False)[1][0] == 0


@pytest.mark.parametrize("radius", [20, 40, 80])
def test_smoothed_curvature_of_a_circle(radius):
    mask = disc(h=2 * radius + 40, w=2 * radius + 40, cy=radius + 20, cx=radius + 20, radius=radius)
    contour = E.detect_cell_edge(mask)
    pts, s = C.sample_edge_points(contour, 100)
    sign, mag, norm = C.edge_curvature(contour, pts, s, straight_threshold=0.0)
    assert np.all(sign == 1)
    if 1 / radius > 2 * DEFAULTS["straight_threshold"]:   # well above the dead-band: convex with the defaults too
        assert np.all(C.edge_curvature(contour, pts, s)[0] == 1)
    # |kappa| ~ 1/R in 1/px; smoothing shrinks a small circle a little (about exp(sigma^2 / 2R^2)), so R = 20 reads ~8 % high
    assert abs(np.median(mag) * radius - 1) < 0.1
    rev = contour[::-1]
    assert np.all(C.smoothed_curvature(rev, s, DEFAULTS["smoothing_px"]) < 0)  # reversed walk flips the sign


def test_smoothed_notch_is_concave():
    mask = disc(radius=45)
    yy, xx = np.mgrid[:120, :120]
    mask[(yy - 60) ** 2 + (xx - 105) ** 2 < 18 ** 2] = 0
    contour = E.detect_cell_edge(mask)
    pts, s = C.sample_edge_points(contour, 100)
    sign = C.edge_curvature(contour, pts, s)[0]
    bite = (pts[:, 0] - 60) ** 2 + (pts[:, 1] - 105) ** 2 < 18.5 ** 2
    centre = bite & (np.abs(pts[:, 0] - 60) < 10)
    assert centre.sum() >= 3 and np.all(sign[centre] == -1)
    assert np.all(sign[pts[:, 1] < 40] == 1)


def test_tiny_cell_does_not_blow_up():
    """A one-pixel cell (contour 2.8 px long) with sigma 6 px: sigma is capped at L / 2 pi, curvature stays finite."""
    one = np.zeros((10, 10), np.uint8)
    one[4, 5] = 1
    contour = E.detect_cell_edge(one)
    pts, s = C.sample_edge_points(contour, 5)
    k = C.smoothed_curvature(contour, s, 6.0)
    assert np.all(np.isfinite(k)) and np.all(k > 0) and np.all(k < 5)


def test_straight_dead_band():
    """Straight runs: legacy rule gives -1; with a threshold they are 0 (straight)."""
    pts = np.column_stack([np.zeros(5), np.arange(5.0)])
    assert C.three_point_curvature(pts)[0][2] == -1
    assert C.three_point_curvature(pts, straight_threshold=1e-6)[0][2] == 0
    sq = np.zeros((80, 80), np.uint8)
    sq[10:70, 15:65] = 1
    contour = E.detect_cell_edge(sq)
    pts, s = C.sample_edge_points(contour, 60)
    sign, mag, _ = C.edge_curvature(contour, pts, s)
    side = (np.abs(pts[:, 0] - 40) < 6) | (np.abs(pts[:, 1] - 40) < 6)   # middles of the four sides
    assert np.all(sign[side] == 0) and np.all(sign[~side] >= 0)


def test_defaults_meet_the_benchmark():
    """Seeded benchmark (scripts/curvature_benchmark.py) at realistic segmentation noise (sd 0.3 px), 3 seeds.

    Defaults: the sign is right on >= 85 % of curved points of every shape except the R = 80 px circle, whose
    curvature (0.0125/px) sits just above the 0.01/px dead-band, and >= 80 % of straight points are called straight.
    The legacy options get <= 70 % of curved points right on the smooth shapes (the reason for the change).
    """
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("bench", Path(__file__).parent.parent / "scripts" / "curvature_benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.NOISE_SD = (0.3,)
    res = bench.evaluate({"default": (DEFAULTS["smoothing_px"], DEFAULTS["straight_threshold"]), "legacy": (0, 0)},
                         seeds=3)
    curved = {n: v[0.3][0] for n, v in res["default"].items()}
    assert min(v for n, v in curved.items() if n != "circle R80") >= 0.85, curved
    assert curved["circle R80"] >= 0.5, curved
    assert res["default"]["rounded rect"][0.3][1] >= 0.8
    assert max(v[0.3][0] for n, v in res["legacy"].items() if n != "rounded rect") < 0.7
