"""Synthetic demo data: a seeded image + mask stack of a cell edge facing a wound (scratch assay layout).

Not measured data. Used by the web app's demo and by tests. The cell fills the lower part of the field; its
edge (row position h(x, t)) carries protrusions that grow over the frames. The fluorescence image is
background + cytoplasm + Gaussian noise, plus an edge-localised enrichment that is, BY CONSTRUCTION, stronger
where the edge is concave (bends into the cell). The analysis should therefore find intensity decreasing with
signed curvature on this demo; that relationship is built in, not a finding.
"""
import numpy as np

DEMO_SPEC = {
    "shape": (160, 220),          # rows, cols
    "n_frames": 4,
    "seed": 7,
    "edge_row": 70.0,             # mean edge position (row) at frame 0
    "bumps": [                    # (centre col, width px, amplitude px at frame 0, growth px/frame); + = into wound
        (40, 14, 10, 2.0), (105, 10, -9, -1.0), (165, 16, 14, 2.5),
    ],
    "background": 100.0, "cytoplasm": 300.0, "enrichment": 500.0, "enrichment_depth": 8.0, "noise_sd": 25.0,
}


def edge_profile(cols, frame, spec=DEMO_SPEC):
    """Edge row position for each column at a frame (smaller row = further into the wound, which is at the top)."""
    h = np.full(cols.shape, spec["edge_row"], dtype=float) - 1.5 * frame
    for c0, w, amp, growth in spec["bumps"]:
        h -= (amp + growth * frame) * np.exp(-((cols - c0) / w) ** 2)
    return h


def make_demo_stack(spec=DEMO_SPEC):
    """(images uint16 [T, H, W], masks uint8 [T, H, W]); masks are 1 inside the cell."""
    rows, cols = spec["shape"]
    rng = np.random.default_rng(spec["seed"])
    yy, xx = np.mgrid[:rows, :cols].astype(float)
    x = np.arange(cols, dtype=float)
    images, masks = [], []
    for t in range(spec["n_frames"]):
        h = edge_profile(x, t, spec)
        mask = (yy > h[None, :]).astype(np.uint8)
        # h is a row position: a dip into the cell is a local maximum of h, so h'' < 0 marks concave edge
        h2 = np.gradient(np.gradient(h))
        concave = np.clip(-h2 / (np.abs(h2).max() or 1.0), 0, 1)
        depth = yy - h[None, :]
        rim = np.exp(-np.clip(depth, 0, None) / spec["enrichment_depth"]) * (0.25 + 0.75 * concave[None, :])
        img = np.where(mask > 0, spec["cytoplasm"] + spec["enrichment"] * rim, spec["background"])
        img = img + rng.normal(0, spec["noise_sd"], img.shape)
        images.append(np.clip(np.round(img), 0, 65535).astype(np.uint16))
        masks.append(mask)
    return np.stack(images), np.stack(masks)
