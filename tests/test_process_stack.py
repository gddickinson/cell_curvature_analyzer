"""Regression tests for CurvatureAnalyzer.process_stack on stacks with frames that have no cell."""
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.curvature_analysis import CurvatureAnalyzer  # noqa: E402


def _write_stack(tmp_path, masks):
    rng = np.random.default_rng(0)
    images = (rng.random(masks.shape) * 50 + 100 * masks).astype(np.uint16)
    img_path, mask_path = str(tmp_path / "img.tif"), str(tmp_path / "mask.tif")
    tifffile.imwrite(img_path, images)
    tifffile.imwrite(mask_path, (masks * 255).astype(np.uint8))
    return img_path, mask_path


def test_stack_with_two_empty_frames_before_the_cell(tmp_path):
    """Frames without a contour used to shift the contour list, so frame_idx - 1 indexed past its end (IndexError)."""
    yy, xx = np.mgrid[:64, :64]
    disc = ((yy - 32) ** 2 + (xx - 32) ** 2 < 20 ** 2).astype(np.uint8)
    masks = np.stack([np.zeros_like(disc), np.zeros_like(disc), disc, disc])
    img_path, mask_path = _write_stack(tmp_path, masks)
    res = CurvatureAnalyzer().process_stack(img_path, mask_path, str(tmp_path / "out"), n_points=20, depth=5, width=3)
    st = res["statistics"]
    assert st["processed_frames"] == 2 and st["frames_with_no_contour"] == 2
    # movement is measured for frames with a cell after the first (frames 2 and 3)
    assert len(res["movement_scores"]) == 2
