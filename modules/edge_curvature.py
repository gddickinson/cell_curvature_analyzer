"""Edge geometry for the curvature analysis: open/closed edges, point sampling along the edge, and curvature.

Coordinates are (row, col) pixel centres, as returned by skimage.measure.find_contours. The browser port is
modules/web/static/curvature_engine.js (tested against this module).

Open and closed edges: find_contours returns a closed contour (last point == first point) for a cell inside the
image and an open one for an edge that runs off the image (e.g. a wound edge). Closed edges wrap round; open edges
do not: their two end points have a neighbour on one side only.

Point spacing (`point_spacing`):
  "equal"   points at equal arc-length steps: L/n for a closed edge (n points round the loop), L/(n-1) for an open
            edge (both end points included).
  "legacy"  the pre-2026-11 sampler. It restarted its arc-length count at the start of the segment holding the last
            point, so spacing was up to one contour segment short each step; the shortfall piles up as one large gap
            between the last and first points (e.g. 2.0 px steps and a 42 px gap for 100 points on a 40 px circle).

Curvature (`smoothing_px`, `straight_threshold`):
  smoothing_px > 0  the edge is resampled every ~0.5 px, smoothed with a Gaussian of sigma = smoothing_px along the
                    arc length (wrapping for closed edges, edge values repeated for open ones), and the signed
                    curvature kappa = (c' r'' - r' c'') / |(r', c')|^3 (central differences, 1/px) is read off at
                    each sample point. magnitude = |kappa| in 1/px (1/radius). For a closed edge sigma is capped at
                    L / (2 pi) (the radius of a circle of the same length), so tiny cells do not shrink to a point.
  smoothing_px = 0  "none (legacy)": three-point estimate from each sample point and its neighbours p_left, p_right:
                    cross = (p_right - p_left) x (p - p_left) in (row, col); magnitude = |cross| / |p_right - p_left|^2
                    (dimensionless, about kappa * spacing / 4). For the dead-band its curvature is the three-point
                    (Menger) curvature 2 cross / (|p - p_left| |p_right - p| |p_right - p_left|).
  Sign: find_contours walks the edge with the cell on its right-hand side (clockwise round a cell as seen on screen,
  y downwards), so kappa (and cross) is positive where the edge bulges OUT of the cell (convex, e.g. a protrusion
  into a wound) and negative where it bends INTO the cell (concave).
  sign = +1 if kappa > straight_threshold, -1 if kappa < -straight_threshold, else 0 (straight).
  straight_threshold = 0 with smoothing_px = 0 is the legacy rule: +1 if cross > 0 else -1 (straight runs -> -1).
  normalized = sign * magnitude / max(magnitude) per frame, in [-1, 1].
  The two end points of an open edge have no curvature in the three-point estimate (sign 0, magnitude 0).
"""
import numpy as np

from .analysis_defaults import DEFAULTS

RESAMPLE_PX = 0.5   # arc-length step of the resampled edge used for smoothing


def is_closed(contour):
    """True when the contour is a closed loop (find_contours repeats the first point at the end)."""
    contour = np.asarray(contour)
    return len(contour) > 2 and bool(np.all(contour[0] == contour[-1]))


def arc_length(contour):
    """Cumulative arc length at each contour vertex (starts at 0)."""
    seg = np.sqrt(np.sum(np.diff(np.asarray(contour, dtype=float), axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _at_arc(contour, cum, s):
    """Points at arc positions s on the polyline."""
    return np.column_stack([np.interp(s, cum, contour[:, 0]), np.interp(s, cum, contour[:, 1])])


def sample_legacy(contour, n_points):
    """The pre-2026-11 sampler (see module docstring); returns (points, arc positions)."""
    contour = np.asarray(contour, dtype=float)
    cum = arc_length(contour)
    step_size = cum[-1] / n_points
    sampled, s = [contour[0]], [0.0]
    current_length = 0
    i = 0
    while len(sampled) < n_points and i < len(contour) - 1:
        dist = np.linalg.norm(contour[i + 1] - contour[i])
        if current_length + dist >= step_size:
            t = (step_size - current_length) / dist
            sampled.append(contour[i] + t * (contour[i + 1] - contour[i]))
            s.append(cum[i] + t * dist)
            current_length = 0  # (the bug: counting restarts at contour[i], not at the new point)
        else:
            current_length += dist
            i += 1
    return np.array(sampled), np.array(s)


def sample_edge_points(contour, n_points=None, point_spacing=None):
    """(points, arc positions) of n_points along the contour; see the module docstring for the spacing rules."""
    n_points = DEFAULTS["n_points"] if n_points is None else n_points
    point_spacing = DEFAULTS["point_spacing"] if point_spacing is None else point_spacing
    if point_spacing == "legacy":
        return sample_legacy(contour, n_points)
    if point_spacing != "equal":
        raise ValueError(f"point_spacing must be 'equal' or 'legacy', not {point_spacing!r}")
    contour = np.asarray(contour, dtype=float)
    cum = arc_length(contour)
    length = cum[-1]
    if is_closed(contour):
        s = np.arange(n_points) * (length / n_points)
    else:
        s = np.arange(n_points) * (length / (n_points - 1)) if n_points > 1 else np.zeros(1)
        s[-1] = length if n_points > 1 else 0.0
    return _at_arc(contour, cum, s), s


def arc_positions(contour, points):
    """Arc position of each point projected onto its nearest contour segment (for the alternative samplers)."""
    contour = np.asarray(contour, dtype=float)
    cum = arc_length(contour)
    a, d = contour[:-1], np.diff(contour, axis=0)
    dd = np.maximum(np.sum(d * d, axis=1), 1e-300)
    out = np.zeros(len(points))
    for k, p in enumerate(np.asarray(points, dtype=float)):
        t = np.clip(np.sum((p - a) * d, axis=1) / dd, 0, 1)
        dist = np.sum((a + t[:, None] * d - p) ** 2, axis=1)
        j = int(np.argmin(dist))
        out[k] = cum[j] + t[j] * np.sqrt(dd[j])
    return out


def gaussian_smooth(values, sigma, closed):
    """Gaussian filter along a 1-D sequence (sigma in samples, kernel radius int(4 sigma + 0.5)).

    Wraps for closed sequences, repeats the end values for open ones (scipy.ndimage 'wrap' / 'nearest').
    """
    n = len(values)
    radius = int(4.0 * sigma + 0.5)
    k = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w = w / w.sum()
    idx = np.arange(n)[:, None] + k[None, :]
    idx = np.mod(idx, n) if closed else np.clip(idx, 0, n - 1)
    return np.sum(values[idx] * w[None, :], axis=1)


def smoothed_curvature(contour, s, smoothing_px):
    """Signed curvature (1/px) of the Gaussian-smoothed edge at arc positions s."""
    contour = np.asarray(contour, dtype=float)
    closed = is_closed(contour)
    cum = arc_length(contour)
    length = cum[-1]
    if length <= 0:
        return np.zeros(len(s))
    if closed:
        n = max(int(round(length / RESAMPLE_PX)), 8)
        ds = length / n
        grid = np.arange(n) * ds
    else:
        n = max(int(round(length / RESAMPLE_PX)), 2) + 1
        ds = length / (n - 1)
        grid = np.arange(n) * ds
        grid[-1] = length
    xy = _at_arc(contour, cum, grid)
    if closed:  # a loop much shorter than the kernel would shrink to a point: sigma at most its "radius" L / 2 pi
        smoothing_px = min(smoothing_px, length / (2 * np.pi))
    r = gaussian_smooth(xy[:, 0], smoothing_px / ds, closed)
    c = gaussian_smooth(xy[:, 1], smoothing_px / ds, closed)
    nxt, prv = np.roll(np.arange(n), -1), np.roll(np.arange(n), 1)
    r1, c1 = (r[nxt] - r[prv]) / (2 * ds), (c[nxt] - c[prv]) / (2 * ds)
    r2, c2 = (r[nxt] - 2 * r + r[prv]) / ds ** 2, (c[nxt] - 2 * c + c[prv]) / ds ** 2
    speed = (r1 * r1 + c1 * c1) ** 1.5
    kappa = np.where(speed > 0, (c1 * r2 - r1 * c2) / np.where(speed > 0, speed, 1), 0.0)
    if closed:
        return np.interp(np.mod(s, length), np.append(grid, length), np.append(kappa, kappa[0]))
    kappa[0], kappa[-1] = kappa[1], kappa[-2]  # one-sided ends: repeat the neighbouring value
    return np.interp(s, grid, kappa)


def _signs(kappa, straight_threshold):
    return np.where(kappa > straight_threshold, 1.0, np.where(kappa < -straight_threshold, -1.0, 0.0))


def _normalize(sign, magnitude):
    mx = np.max(magnitude) if len(magnitude) else 0
    return sign * (magnitude / mx) if mx > 0 else sign * 0


def three_point_curvature(points, closed=True, straight_threshold=0.0):
    """(sign, magnitude, normalized) by the legacy three-point rule (smoothing 'none')."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    sign, mag = np.zeros(n), np.zeros(n)
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue  # an open edge's end point has a neighbour on one side only: no curvature
        p, pl, pr = points[i], points[(i - 1) % n], points[(i + 1) % n]
        v, w = pr - pl, p - pl
        vl = np.linalg.norm(v)
        cross = v[0] * w[1] - v[1] * w[0]
        mag[i] = abs(cross) / (vl * vl) if vl > 0 else 0
        if straight_threshold > 0:
            den = np.linalg.norm(w) * np.linalg.norm(pr - p) * vl
            kappa = 2 * cross / den if den > 0 else 0.0
            sign[i] = _signs(np.array([kappa]), straight_threshold)[0]
        else:
            sign[i] = 1 if cross > 0 else -1  # legacy: +1 convex, straight (cross == 0) -> -1
    return sign, mag, _normalize(sign, mag)


def edge_curvature(contour, points, s, smoothing_px=None, straight_threshold=None):
    """(sign, magnitude, normalized) at sample points (arc positions s) of a contour; see the module docstring."""
    smoothing_px = DEFAULTS["smoothing_px"] if smoothing_px is None else smoothing_px
    straight_threshold = DEFAULTS["straight_threshold"] if straight_threshold is None else straight_threshold
    if smoothing_px <= 0:
        return three_point_curvature(points, is_closed(contour), straight_threshold)
    kappa = smoothed_curvature(contour, s, smoothing_px)
    sign = _signs(kappa, straight_threshold)
    mag = np.abs(kappa)
    return sign, mag, _normalize(sign, mag)
