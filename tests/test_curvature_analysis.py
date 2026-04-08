"""
Tests for modules/curvature_analysis.py using synthetic edge data.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.curvature_analysis import CurvatureAnalyzer


def make_circular_mask(shape=(100, 100), center=(50, 50), radius=30):
    """Create a binary mask with a filled circle."""
    mask = np.zeros(shape, dtype=np.uint8)
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    circle = (rr - center[0])**2 + (cc - center[1])**2 < radius**2
    mask[circle] = 1
    return mask


def make_circle_contour(n_points=200, center=(50, 50), radius=30):
    """Create contour points on a circle."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    contour = np.column_stack([
        center[0] + radius * np.cos(t),
        center[1] + radius * np.sin(t)
    ])
    return contour


class TestDetectCellEdge:
    """Tests for detect_cell_edge."""

    def setup_method(self):
        self.analyzer = CurvatureAnalyzer()

    def test_detects_circular_edge(self):
        mask = make_circular_mask()
        contour = self.analyzer.detect_cell_edge(mask)
        assert len(contour) > 0
        assert contour.shape[1] == 2

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        contour = self.analyzer.detect_cell_edge(mask)
        assert len(contour) == 0

    def test_full_mask_returns_empty(self):
        """A fully filled mask has no 0.5-level contour (no boundary)."""
        mask = np.ones((100, 100), dtype=np.uint8)
        contour = self.analyzer.detect_cell_edge(mask)
        # find_contours at 0.5 won't find a contour in an all-1 mask
        assert len(contour) == 0


class TestSampleEquidistantPoints:
    """Tests for sample_equidistant_points."""

    def setup_method(self):
        self.analyzer = CurvatureAnalyzer()

    def test_returns_requested_number_of_points(self):
        contour = make_circle_contour(n_points=500)
        sampled = self.analyzer.sample_equidistant_points(contour, n_points=50)
        assert len(sampled) == 50

    def test_points_are_roughly_equidistant(self):
        contour = make_circle_contour(n_points=1000, radius=40)
        sampled = self.analyzer.sample_equidistant_points(contour, n_points=20)
        # Calculate distances between consecutive points
        distances = []
        for i in range(len(sampled) - 1):
            d = np.linalg.norm(sampled[i+1] - sampled[i])
            distances.append(d)
        distances = np.array(distances)
        # Standard deviation should be small relative to mean
        if len(distances) > 1:
            assert distances.std() / distances.mean() < 0.5


class TestMeasureCurvature:
    """Tests for measure_curvature."""

    def setup_method(self):
        self.analyzer = CurvatureAnalyzer()

    def test_returns_three_arrays(self):
        points = make_circle_contour(n_points=50, radius=30)
        result = self.analyzer.measure_curvature(points)
        assert len(result) == 3
        sign_curv, mag_curv, norm_curv = result
        assert len(sign_curv) == 50
        assert len(mag_curv) == 50
        assert len(norm_curv) == 50

    def test_circle_has_uniform_curvature(self):
        """A circle should have roughly uniform curvature magnitude."""
        points = make_circle_contour(n_points=100, radius=30)
        _, mag_curv, _ = self.analyzer.measure_curvature(points)
        # Remove any outliers and check consistency
        nonzero = mag_curv[mag_curv > 0]
        if len(nonzero) > 10:
            assert nonzero.std() / nonzero.mean() < 0.5

    def test_straight_line_has_zero_curvature(self):
        """Straight line points should have ~zero curvature."""
        points = np.column_stack([
            np.linspace(0, 100, 20),
            np.zeros(20)
        ])
        sign_curv, mag_curv, _ = self.analyzer.measure_curvature(points)
        # Interior points (not endpoints which wrap) should be near zero
        interior = mag_curv[2:-2]
        assert np.allclose(interior, 0, atol=1e-10)


class TestAnalyzeFrame:
    """Integration test for analyze_frame."""

    def setup_method(self):
        self.analyzer = CurvatureAnalyzer()

    def test_analyze_frame_returns_expected_keys(self):
        mask = make_circular_mask()
        # Create a synthetic fluorescence image with some signal inside the cell
        image = np.random.rand(100, 100).astype(np.float32) * 50
        image[mask > 0] += 100  # Brighter inside cell

        result = self.analyzer.analyze_frame(
            image, mask, n_points=20, depth=5, width=3,
            min_cell_coverage=0.5)

        assert 'points' in result
        assert 'curvatures' in result
        assert 'normals' in result
        assert 'intensities' in result
        assert 'valid_points' in result


class TestFileIO:
    """Tests for file I/O methods."""

    def setup_method(self):
        self.analyzer = CurvatureAnalyzer()

    def test_load_tiff_stack(self, tmp_path):
        """Test loading synthetic TIFF stacks."""
        import tifffile

        # Create synthetic image and mask stacks
        images = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint16)
        masks = np.zeros((3, 64, 64), dtype=np.uint8)
        rr, cc = np.ogrid[:64, :64]
        masks[:, (rr - 32)**2 + (cc - 32)**2 < 20**2] = 255

        img_path = str(tmp_path / "images.tif")
        mask_path = str(tmp_path / "masks.tif")
        tifffile.imwrite(img_path, images)
        tifffile.imwrite(mask_path, masks)

        loaded_images, loaded_masks = self.analyzer.load_tiff_stack(
            img_path, mask_path)

        assert loaded_images.shape == (3, 64, 64)
        assert loaded_masks.shape == (3, 64, 64)
        assert loaded_masks.max() == 1  # Binary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
