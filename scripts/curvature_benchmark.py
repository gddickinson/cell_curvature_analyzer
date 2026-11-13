"""Seeded benchmark of the curvature options on masks of known curvature (chooses smoothing_px / straight_threshold).

Shapes (px, realistic for cells at 0.1-0.3 um/px): circles R = 20, 40, 80; an ellipse 70 x 35; a disc R = 60 with
a circular notch r = 10 or 20 (concave); a rounded rectangle 160 x 100 with 15 px corners (straight sides); an open
wound edge (sine, amplitude 12, wavelength 120, running off the image). Each shape is rasterised with a random
sub-pixel offset and rotation, after displacing its outline along the normal by smooth random noise
(segmentation noise: sd 0, 0.3 or 0.6 px, correlation length 2 px). True curvature comes from the noise-free
shape at the nearest outline point. 10 seeds per shape and noise level, n_points = 100 (the default).

Reports, per curvature option: sign agreement on curved points (|kappa_true| >= 1/100 px, away from notch corners),
the fraction of straight points (rounded-rectangle sides) called straight (sign 0), and the 95th percentile of
|kappa| on straight sides (what the dead-band has to exceed).

    python scripts/curvature_benchmark.py            # full table (about a minute)
    python scripts/curvature_benchmark.py --quick    # 3 seeds
"""
import sys
from pathlib import Path

import numpy as np
from skimage import draw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import edge_curvature as C  # noqa: E402
from modules.edge_analysis import detect_cell_edge  # noqa: E402

KAPPA_CURVED = 1 / 100   # |kappa_true| at least this counts as curved
NOISE_SD = (0.0, 0.3, 0.6)
CORR_PX = 2.0


def circle(R, n=4000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([R * np.sin(t), R * np.cos(t)]), np.full(n, 1 / R), True


def ellipse(a, b, n=4000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = a * b / (a * a * np.sin(t) ** 2 + b * b * np.cos(t) ** 2) ** 1.5
    return np.column_stack([b * np.sin(t), a * np.cos(t)]), k, True


def notched(R, r, n=4000):
    """Disc of radius R with a circular bite of radius r centred on its edge (at angle 0); returns kappa_true
    (+1/R on the disc arc, -1/r on the bite) and NaN within 3 px of the two corners."""
    phi = 2 * np.arcsin(r / (2 * R))                           # half-angle of the bite seen from the disc centre
    t = np.linspace(phi, 2 * np.pi - phi, n)
    arc = np.column_stack([R * np.sin(t), R * np.cos(t)])
    c0 = np.array([0.0, R])
    a0 = np.arctan2(arc[-1, 0] - c0[0], arc[-1, 1] - c0[1])
    a1 = np.arctan2(arc[0, 0] - c0[0], arc[0, 1] - c0[1])
    if a1 > a0:
        a1 -= 2 * np.pi
    u = np.linspace(a0, a1, n // 4)
    bite = np.column_stack([c0[0] + r * np.sin(u), c0[1] + r * np.cos(u)])
    pts = np.vstack([arc, bite])
    k = np.concatenate([np.full(len(arc), 1 / R), np.full(len(bite), -1 / r)])
    corners = np.array([arc[0], arc[-1]])
    near = np.min(np.linalg.norm(pts[:, None, :] - corners[None], axis=2), axis=1) < 3
    k[near] = np.nan
    return pts, k, True


def _with_sides(arcs, karcs):
    """Join arcs (in order round the loop) with densely sampled straight sides (kappa 0)."""
    out, kout = [], []
    for i, (arc, k) in enumerate(zip(arcs, karcs)):
        out.append(arc); kout.append(k)
        a, b = arc[-1], arcs[(i + 1) % len(arcs)][0]
        m = int(np.linalg.norm(b - a) * 4)
        if m >= 1:
            f = np.linspace(0, 1, m + 2)[1:-1, None]
            out.append(a + f * (b - a)); kout.append(np.zeros(m))
    return np.vstack(out), np.concatenate(kout)


def rounded_rect(w, h, rc, n=200):
    """Rounded rectangle traversed like circle() (row = sin, col = cos); straight sides have kappa 0."""
    arcs = []
    for k, (sy, sx) in enumerate([(1, 1), (1, -1), (-1, -1), (-1, 1)]):
        th = np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, n)
        arcs.append(np.column_stack([sy * (h / 2 - rc) + rc * np.sin(th), sx * (w / 2 - rc) + rc * np.cos(th)]))
    pts, k = _with_sides(arcs, [np.full(n, 1 / rc)] * 4)
    return pts, k, True


def finger(R=50, hw=5, length=25, n=4000):
    """Disc R with a finger (half-width hw, rounded tip) sticking out: tip +1/hw, disc +1/R, sides 0, base NaN."""
    phi = np.arcsin(hw / R)
    t = np.linspace(phi, 2 * np.pi - phi, n)
    disc = np.column_stack([R * np.sin(t), R * np.cos(t)])
    th = np.linspace(-np.pi / 2, np.pi / 2, 400)
    tip = np.column_stack([hw * np.sin(th), R + length + hw * np.cos(th)])
    pts, k = _with_sides([disc, tip], [np.full(n, 1 / R), np.full(400, 1 / hw)])
    corners = np.array([disc[0], disc[-1]])
    k[np.min(np.linalg.norm(pts[:, None, :] - corners[None], axis=2), axis=1) < 3] = np.nan
    return pts, k, True


def wound(width=220, amp=12, lam=120, n=4000):
    x = np.linspace(-20, width + 20, n)
    w = 2 * np.pi / lam
    y = amp * np.sin(w * x)
    k = -amp * w * w * np.sin(w * x) / (1 + (amp * w * np.cos(w * x)) ** 2) ** 1.5  # y'' / (1 + y'^2)^1.5: > 0 where the
    # edge bulges up (smaller row) into the wound
    return np.column_stack([y, x]), k, False


SHAPES = {"circle R20": lambda: circle(20), "circle R40": lambda: circle(40), "circle R80": lambda: circle(80),
          "ellipse 70x35": lambda: ellipse(70, 35), "notch r10": lambda: notched(60, 10),
          "notch r20": lambda: notched(60, 20), "rounded rect": lambda: rounded_rect(160, 100, 15),
          "finger hw5": finger,
          "wound edge": wound}


def _normals(pts, closed):
    nxt = np.roll(pts, -1, axis=0) if closed else np.vstack([pts[1:], 2 * pts[-1] - pts[-2]])
    prv = np.roll(pts, 1, axis=0) if closed else np.vstack([2 * pts[0] - pts[1], pts[:-1]])
    t = nxt - prv
    t /= np.linalg.norm(t, axis=1)[:, None]
    return np.column_stack([-t[:, 1], t[:, 0]])


def make_mask(name, sd, rng):
    pts, k, closed = SHAPES[name]()
    if name != "wound edge":
        ang = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pts = pts @ rot.T
    if sd > 0:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean()
        noise = rng.normal(size=len(pts))
        noise = C.gaussian_smooth(noise, CORR_PX / seg, closed)
        noise *= sd / noise.std()
        noisy = pts + noise[:, None] * _normals(pts, closed)
    else:
        noisy = pts
    if closed:
        h = w = int(2 * np.max(np.abs(pts)) + 30)
        off = np.array([h / 2, w / 2]) + rng.uniform(0, 1, 2)
        rr, cc = draw.polygon(noisy[:, 0] + off[0], noisy[:, 1] + off[1], (h, w))
    else:
        h, w = 120, 220
        off = np.array([60 + rng.uniform(0, 1), 0.0])
        poly = np.vstack([noisy + off, [[h + 5, w + 25], [h + 5, -25]]])   # cell below the edge
        rr, cc = draw.polygon(poly[:, 0], poly[:, 1], (h, w))
    mask = np.zeros((h, w), np.uint8)
    mask[rr, cc] = 1
    return mask, pts + off, k


def evaluate(options, seeds=10, n_points=100):
    """{option label: {shape: {sd: (sign agreement on curved, straight->0 fraction, p95 |kappa| straight)}}}."""
    out = {}
    for label, (smoothing, thr) in options.items():
        res = out.setdefault(label, {})
        for name in SHAPES:
            for sd in NOISE_SD:
                rng = np.random.default_rng([sum(map(ord, name)), int(sd * 10)])
                agree = tot = st0 = stn = 0
                stk = []
                for _ in range(seeds):
                    mask, truth_pts, k = make_mask(name, sd, rng)
                    contour = detect_cell_edge(mask)
                    p, s = C.sample_edge_points(contour, n_points, "equal")
                    sign, mag, _ = C.edge_curvature(contour, p, s, smoothing, thr)
                    kappa = sign * mag
                    near = np.argmin(np.sum((p[:, None, :] - truth_pts[None]) ** 2, axis=2), axis=1)
                    kt = k[near]
                    if not C.is_closed(contour):
                        kt = kt.copy(); kt[[0, -1]] = np.nan
                    curved = np.isfinite(kt) & (np.abs(kt) >= KAPPA_CURVED)
                    agree += np.sum(sign[curved] == np.sign(kt[curved])); tot += np.sum(curved)
                    straight = np.isfinite(kt) & (kt == 0)
                    st0 += np.sum(sign[straight] == 0); stn += np.sum(straight)
                    if smoothing > 0:
                        stk.extend(np.abs(kappa[straight]))
                    else:  # legacy: compare the three-point (Menger) curvature with the dead-band
                        stk.extend(np.abs(_menger(p, C.is_closed(contour))[straight]))
                res.setdefault(name, {})[sd] = (agree / tot if tot else np.nan, st0 / stn if stn else np.nan,
                                                float(np.percentile(stk, 95)) if stk else np.nan)
    return out


def _menger(p, closed):
    n = len(p)
    out = np.zeros(n)
    for i in range(n):
        if not closed and i in (0, n - 1):
            continue
        a, b, c = p[(i - 1) % n], p[i], p[(i + 1) % n]
        den = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a)
        cr = (c - a)[0] * (b - a)[1] - (c - a)[1] * (b - a)[0]
        out[i] = 2 * cr / den if den > 0 else 0
    return out


def main(argv):
    seeds = 3 if "--quick" in argv else 10
    opts = {"none (legacy)": (0, 0)}
    opts.update({f"sigma {s:g} px": (s, 0) for s in (2, 3, 4, 5, 6, 8)})
    if "--threshold" in argv:
        sig = float(argv[argv.index("--threshold") + 1])
        opts = {f"sigma {sig:g}, thr {t:g}": (sig, t) for t in (0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02)}
        opts["none (legacy), thr 0.01"] = (0, 0.01)
    res = evaluate(opts, seeds)
    for sd in NOISE_SD:
        print(f"\nsegmentation noise sd {sd} px: sign agreement on curved points (|kappa| >= 1/100 px)")
        print(f"{'option':24s}" + "".join(f"{n:>14s}" for n in SHAPES) + f"{'min':>8s}")
        for label, r in res.items():
            v = [r[n][sd][0] for n in SHAPES]
            print(f"{label:24s}" + "".join(f"{x:14.3f}" for x in v) + f"{np.nanmin(v):8.3f}")
        print("straight sides of the rounded rectangle: fraction called straight, 95th percentile |kappa| (1/px); "
              "balanced score = (straight called straight + mean curved agreement) / 2")
        for label, r in res.items():
            st = r['rounded rect'][sd][1]
            curved = np.nanmean([r[n][sd][0] for n in SHAPES])
            print(f"  {label:24s} {st:.3f}  {r['rounded rect'][sd][2]:.4f}  balanced {(st + curved) / 2:.3f}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
