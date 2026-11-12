"""Write tests/fixtures/edge_fixture.json: inputs plus the repo's Python results, for the JS port's node test.

Cases run modules.web.analysis.analyze_stack (i.e. modules.edge_analysis) on: two frames of the synthetic demo,
a disc, a disc with a notch, random blobs (several contours and holes), an open wound edge over three frames
with an empty frame, a lattice-aligned square and a one-pixel cell, with different parameters. It also stores
raw skimage.measure.find_contours output on random binary arrays, skimage.draw.polygon on random and
lattice-aligned rectangles, and scipy.stats.linregress on random data.

Run: python scripts/make_edge_fixture.py      (checked by tests/test_web.py::test_fixture_is_current)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, stats
from skimage import draw, measure

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.synthetic_cell import make_demo_stack  # noqa: E402
from modules.web.analysis import analyze_stack, encode  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "edge_fixture.json"
TIFF_DIR = ROOT / "tests" / "fixtures" / "tiff"


def disc(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2).astype(np.uint8)


def stack_cases(rng):
    im, m = make_demo_stack()
    yield "demo_2_frames", im[:2], m[:2], {}
    d = disc(80, 80, 40, 38, 28)
    yield "disc", (rng.random(d.shape) * 200 + 300.0 * d.astype(float))[None], d[None], {"n_points": 30, "depth": 8, "width": 4}
    notch = disc(90, 90, 45, 45, 34)
    notch[disc(90, 90, 45, 78, 14) > 0] = 0
    yield "notch_rotation", (rng.random(notch.shape) * 500)[None], notch[None], \
        {"n_points": 45, "depth": 12, "width": 5, "try_rotation": True, "min_cell_coverage": 0.9}
    blobs = (ndimage.gaussian_filter(rng.random((70, 90)), 3.5) > 0.5).astype(np.uint8)
    yield "blobs", (rng.random(blobs.shape) * 1000)[None], blobs[None], \
        {"n_points": 60, "depth": 6, "width": 3, "min_cell_coverage": 0.6, "try_rotation": True,
         "exclude_endpoints": True}
    yy, xx = np.mgrid[:60, :100]
    wound = []
    for t, shift in enumerate([0, 0, -4]):
        wound.append((yy > 30 + shift - 8 * np.exp(-((xx - 50) / 9.0) ** 2) + 5 * np.sin(xx / 7.0)).astype(np.uint8))
    wound[1][:] = 0
    wim = np.stack([rng.integers(0, 4000, (60, 100)) for _ in range(3)])
    yield "wound_with_empty_frame", wim, np.stack(wound), {"n_points": 25, "exclude_endpoints": True, "depth": 10}
    sq = np.zeros((40, 40), np.uint8)
    sq[10:30, 8:28] = 1
    yield "square", (np.arange(1600).reshape(40, 40) % 97)[None].astype(float), sq[None], \
        {"n_points": 16, "depth": 6, "width": 4, "min_cell_coverage": 0.5}
    one = np.zeros((10, 10), np.uint8)
    one[4, 5] = 1
    yield "one_pixel", np.ones((1, 10, 10)), one[None], {"n_points": 5, "depth": 2, "width": 1}


def write_tiffs(rng):
    """Small TIFFs written by tifffile in the layouts tiff_reader.js must read, with their expected pixels."""
    import tifffile
    d = TIFF_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = []
    a16 = rng.integers(0, 65535, (3, 12, 10)).astype(np.uint16)
    tifffile.imwrite(d / "u16_imagej_3frames.tif", a16, imagej=True, resolution=(1 / 0.16, 1 / 0.16),
                     metadata={"unit": "um"})
    out.append({"file": "u16_imagej_3frames.tif", "data": a16.tolist(), "um_per_px": 0.16})
    a8 = (rng.random((2, 17, 23)) > 0.5).astype(np.uint8) * 255
    tifffile.imwrite(d / "u8_lzw_2frames.tif", a8, compression="lzw")
    out.append({"file": "u8_lzw_2frames.tif", "data": a8.tolist(), "um_per_px": None})
    f32 = rng.normal(100, 30, (20, 16)).astype(np.float32)
    tifffile.imwrite(d / "f32_zlib_predictor.tif", f32, compression="zlib", predictor=True)
    out.append({"file": "f32_zlib_predictor.tif", "data": f32.astype(float).tolist(), "um_per_px": None})
    u16p = rng.integers(0, 4000, (33, 40)).astype(np.uint16)
    tifffile.imwrite(d / "u16_tiled_packbits_be.tif", u16p, tile=(16, 16), compression="packbits", byteorder=">")
    out.append({"file": "u16_tiled_packbits_be.tif", "data": u16p.tolist(), "um_per_px": None})
    u16z = rng.integers(0, 65535, (18, 21)).astype(np.uint16)
    tifffile.imwrite(d / "u16_zlib_predictor.tif", u16z, compression="zlib", predictor=True, rowsperstrip=5)
    out.append({"file": "u16_zlib_predictor.tif", "data": u16z.tolist(), "um_per_px": None})
    four = rng.integers(0, 255, (4, 14, 11)).astype(np.uint8)
    tifffile.imwrite(d / "u8_4frames_saved_as_planar.tif", four)  # tifffile stores (4, H, W) as 4 planar samples
    out.append({"file": "u8_4frames_saved_as_planar.tif", "data": four.tolist(), "um_per_px": None})
    return out


def main():
    rng = np.random.default_rng(2026)
    cases = []
    for name, im, m, params in stack_cases(rng):
        im = np.asarray(im, dtype=float)
        kind = "u2" if np.all(im == np.round(im)) and im.min() >= 0 and im.max() < 65536 else "f8"
        cases.append({"name": name, "shape": list(m.shape), "image_dtype": kind, "image": encode(im, kind),
                      "mask_u8": encode(m, "u1"),
                      "params": params, "expected": analyze_stack(im, m, params)})
    contours = []
    for k in range(6):
        a = (rng.random((25, 30)) > (0.4 + 0.05 * k)).astype(float)
        contours.append({"shape": list(a.shape), "mask_u8": encode(a, "u1"),
                         "contours": [c.tolist() for c in measure.find_contours(a, 0.5)]})
    polys = []
    for k in range(60):
        if k % 3 == 0:  # lattice-aligned: vertices on integer / half-integer positions
            r0, c0 = rng.integers(-3, 20, 2) + rng.choice([0, 0.5], 2)
            hh, ww = rng.integers(1, 9, 2) + rng.choice([0, 0.5], 2)
            r, c = np.array([r0, r0, r0 + hh, r0 + hh]), np.array([c0, c0 + ww, c0 + ww, c0])
        else:
            p, ang = rng.uniform(-2, 22, 2), rng.uniform(0, 2 * np.pi)
            n, perp = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
            dep, wid = rng.uniform(1, 10), rng.uniform(1, 6)
            e = p + dep * n
            v = np.array([p + wid / 2 * perp, e + wid / 2 * perp, e - wid / 2 * perp, p - wid / 2 * perp])
            r, c = v[:, 0], v[:, 1]
        rr, cc = draw.polygon(r, c, (20, 20))
        polys.append({"r": r.tolist(), "c": c.tolist(), "rr": rr.tolist(), "cc": cc.tolist()})
    regs = []
    for n in [3, 5, 12, 40, 200]:
        x = rng.normal(size=n)
        y = 0.3 * x + rng.normal(size=n)
        res = stats.linregress(x, y)
        regs.append({"x": x.tolist(), "y": y.tolist(), "slope": res.slope, "intercept": res.intercept,
                     "r": res.rvalue, "p": res.pvalue, "stderr": res.stderr})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases, "contours": contours, "polygons": polys, "linregress": regs,
                               "tiffs": write_tiffs(rng)}))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB): {len(cases)} stack cases, {len(contours)} contour, "
          f"{len(polys)} polygon, {len(regs)} regression cases")


if __name__ == "__main__":
    main()
