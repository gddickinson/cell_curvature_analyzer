"""CurvatureAnalyzer: stack / frame driver used by the desktop app.

Per-frame maths is in edge_analysis (shared with the web app); the alternative sampling methods are in
sampling_methods.py and the debug plots in sampling_debug.py (mixins).
"""
import numpy as np
from PyQt5.QtWidgets import QMessageBox
import os
import datetime
import tifffile

from . import edge_analysis
from .analysis_defaults import DEFAULTS
from .sampling_methods import AlternativeSamplingMixin
from .sampling_debug import SamplingDebugMixin

class CurvatureAnalyzer(AlternativeSamplingMixin, SamplingDebugMixin):
    """
    Class for analyzing cell curvature, intensity, and edge movement.
    Provides methods for both full stack processing and single frame analysis.
    """

    def __init__(self):
        """Initialize CurvatureAnalyzer"""
        pass

    def process_stack(self, image_path, mask_path, output_dir, n_points=DEFAULTS["n_points"],
                     depth=DEFAULTS["depth"], width=DEFAULTS["width"],
                     min_cell_coverage=DEFAULTS["min_cell_coverage"], try_rotation=DEFAULTS["try_rotation"],
                     exclude_endpoints=DEFAULTS["exclude_endpoints"], angular_distribution=False,
                     axis_based_distribution=False, x_axis_distribution=False,
                     progress_callback=None, point_spacing=None, smoothing_px=None, straight_threshold=None):
        """
        Process a TIFF stack of microscope images and masks

        Parameters:
        -----------
        image_path : str
            Path to microscope image stack
        mask_path : str
            Path to binary mask stack
        output_dir : str
            Directory to save results
        n_points : int
            Number of points along the contour
        depth : int
            Depth of the sampling rectangle (how far into the cell)
        width : int
            Width of the sampling rectangle
        min_cell_coverage : float
            Minimum fraction of the rectangle that must be inside the cell
        try_rotation : bool
            Whether to try rotating rejected sampling regions
        exclude_endpoints : bool
            Whether to exclude the first and last points in the analysis
        angular_distribution : bool
            Whether to distribute points evenly by angle
        axis_based_distribution : bool
            Whether to distribute points evenly along image axes
        x_axis_distribution : bool
            Whether to distribute points evenly along the x-axis
        progress_callback : function
            Callback function for progress updates (percentage, message)
        point_spacing, smoothing_px, straight_threshold :
            Point spacing and curvature options (None = analysis_defaults.DEFAULTS); see edge_curvature.py

        Returns:
        --------
        dict
            Dictionary of analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        point_spacing = DEFAULTS["point_spacing"] if point_spacing is None else point_spacing
        smoothing_px = DEFAULTS["smoothing_px"] if smoothing_px is None else smoothing_px
        straight_threshold = DEFAULTS["straight_threshold"] if straight_threshold is None else straight_threshold

        # Store parameters for metadata
        parameters = {
            "n_points": n_points,
            "depth": depth,
            "width": width,
            "min_cell_coverage": min_cell_coverage,
            "try_rotation": try_rotation,
            "exclude_endpoints": exclude_endpoints,
            "angular_distribution": angular_distribution,
            "axis_based_distribution": axis_based_distribution,
            "x_axis_distribution": x_axis_distribution,
            "point_spacing": point_spacing,
            "smoothing_px": smoothing_px,
            "straight_threshold": straight_threshold,
        }


        # Initialize statistics dictionary
        statistics = {
            "total_frames": 0,
            "processed_frames": 0,
            "frames_with_no_contour": 0,
            "total_points": 0,
            "valid_points": 0,
            "invalid_points": 0,
            "average_valid_points_per_frame": 0,
            "sign_curvature_correlation": {
                "r_squared": None,
                "p_value": None,
                "sample_size": 0
            },
            "normalized_curvature_correlation": {
                "r_squared": None,
                "p_value": None,
                "sample_size": 0
            },
            "edge_movement": {
                "extending_frames": 0,
                "retracting_frames": 0,
                "stable_frames": 0
            }
        }

        # Load images and masks
        try:
            images, masks = self.load_tiff_stack(image_path, mask_path)
            statistics["total_frames"] = images.shape[0]

            if progress_callback:
                progress_callback(5, f"Loaded {images.shape[0]} frames")
        except Exception as e:
            statistics["error_loading_files"] = str(e)
            if progress_callback:
                progress_callback(100, f"Error loading files: {e}")
            return {"error": str(e)}

        # Store data for all frames
        results = {}
        all_curvatures = []
        all_intensities = []
        all_valid_points = []
        all_masks = []
        all_contours = []

        # Store edge movement data
        all_movement_scores = []
        all_movement_types = []

        # Process each frame
        for frame_idx in range(images.shape[0]):
            if progress_callback:
                percentage = 5 + 70 * frame_idx / images.shape[0]
                progress_callback(int(percentage), f"Processing frame {frame_idx+1}/{images.shape[0]}")

            image = images[frame_idx]
            mask = masks[frame_idx]

            # Detect cell edge
            contour = self.detect_cell_edge(mask)

            if len(contour) == 0:
                statistics["frames_with_no_contour"] += 1
                continue

            # Store mask and contour for region-specific analysis
            all_masks.append(mask)
            all_contours.append(contour)

            # Analyze frame
            frame_results = self.analyze_frame(
                image, mask, n_points, depth, width, min_cell_coverage,
                try_rotation, exclude_endpoints, angular_distribution,
                axis_based_distribution, x_axis_distribution, point_spacing=point_spacing,
                smoothing_px=smoothing_px, straight_threshold=straight_threshold)

            # Store results
            results[frame_idx] = frame_results

            # Extract key data for statistics
            all_curvatures.append(frame_results["curvatures"])
            all_intensities.append(frame_results["intensities"])
            all_valid_points.append(frame_results["valid_points"])

            # Update statistics
            statistics["processed_frames"] += 1
            statistics["total_points"] += len(frame_results["points"])
            valid_count = np.sum(frame_results["valid_points"])
            statistics["valid_points"] += valid_count
            statistics["invalid_points"] += len(frame_results["points"]) - valid_count

            # Analyze edge movement (for frame 1 and onward)
            if frame_idx > 0:
                previous_mask = masks[frame_idx - 1]
                # The contour list skips frames without a cell, so look the previous contour up by frame
                # (detect_edge_movement compares the masks; the contours are passed for reference only).
                previous_contour = results[frame_idx - 1]["contour_data"] if (frame_idx - 1) in results else np.array([])
                movement_score, movement_type, movement_map = self.detect_edge_movement(
                    contour, previous_contour, mask, previous_mask
                )

                # Store movement data
                all_movement_scores.append(movement_score)
                all_movement_types.append(movement_type)

                # Update movement statistics
                if movement_type == 'extending':
                    statistics["edge_movement"]["extending_frames"] += 1
                elif movement_type == 'retracting':
                    statistics["edge_movement"]["retracting_frames"] += 1
                else:
                    statistics["edge_movement"]["stable_frames"] += 1

        # Calculate average valid points per frame
        if statistics["processed_frames"] > 0:
            statistics["average_valid_points_per_frame"] = statistics["valid_points"] / statistics["processed_frames"]

        # Add movement data to results
        results["movement_scores"] = all_movement_scores
        results["movement_types"] = all_movement_types

        # Calculate correlation statistics
        if all_curvatures and all_intensities and all_valid_points:
            if progress_callback:
                progress_callback(75, "Calculating statistics")

            # Get summary correlation statistics using both sign and normalized curvature
            all_sign_curvatures = [curvature_tuple[0] for curvature_tuple in all_curvatures]
            all_normalized_curvatures = [curvature_tuple[2] for curvature_tuple in all_curvatures]

            # Calculate statistics for sign curvature
            sign_stats = self.calculate_summary_correlation_stats(
                all_sign_curvatures, all_intensities, all_valid_points)
            statistics["sign_curvature_correlation"] = sign_stats

            # Calculate statistics for normalized curvature
            normalized_stats = self.calculate_summary_correlation_stats(
                all_normalized_curvatures, all_intensities, all_valid_points)
            statistics["normalized_curvature_correlation"] = normalized_stats

        # Add movement analysis
        if all_movement_types:
            statistics["edge_movement"]["predominant_behavior"] = max(
                ["extending", "retracting", "stable"],
                key=lambda x: statistics["edge_movement"][f"{x}_frames"]
            )

        # Add statistics to results
        results["statistics"] = statistics
        results["parameters"] = parameters

        # Add analysis timestamp
        results["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if progress_callback:
            progress_callback(100, "Analysis completed")

        return results

    def analyze_frame(self, image, mask, n_points=DEFAULTS["n_points"], depth=DEFAULTS["depth"],
                     width=DEFAULTS["width"], min_cell_coverage=DEFAULTS["min_cell_coverage"],
                     try_rotation=DEFAULTS["try_rotation"], exclude_endpoints=DEFAULTS["exclude_endpoints"],
                     angular_distribution=False,
                     axis_based_distribution=False, x_axis_distribution=False, point_spacing=None,
                     smoothing_px=None, straight_threshold=None):
        """
        Analyze a single frame

        Parameters:
        -----------
        image : ndarray
            Microscope image
        mask : ndarray
            Binary mask
        n_points : int
            Number of points to sample
        depth : int
            Depth of the sampling rectangle (how far into the cell)
        width : int
            Width of the sampling rectangle
        min_cell_coverage : float
            Minimum fraction of the rectangle that must be inside the cell
        try_rotation : bool
            Whether to try rotating rejected sampling regions
        exclude_endpoints : bool
            Whether to exclude the first and last points in the analysis
        angular_distribution : bool
            Whether to distribute points evenly by angle
        axis_based_distribution : bool
            Whether to distribute points evenly along image axes
        x_axis_distribution : bool
            Whether to distribute points evenly along the x-axis
        point_spacing, smoothing_px, straight_threshold :
            Point spacing ("Standard" method only) and curvature options; None = DEFAULTS (see edge_curvature.py)

        Returns:
        --------
        dict
            Dictionary of analysis results for the frame
        """
        # Detect cell edge
        contour = self.detect_cell_edge(mask)

        if len(contour) == 0:
            return {"error": "No contour found"}

        # Sample points based on selected method
        if x_axis_distribution:
            print(f"DEBUG: Using x-axis distribution of sample points")
            points = self.sample_x_axis_points(contour, n_points)
            arc = None
            method_name = "X-Axis"
        elif axis_based_distribution:
            print(f"DEBUG: Using axis-based distribution of sample points")
            points = self.sample_axis_based_points(contour, n_points)
            arc = None
            method_name = "Axis-Based"
        elif angular_distribution:
            print(f"DEBUG: Using angular distribution of sample points")
            points = self.sample_angular_points(contour, n_points)
            arc = None
            method_name = "Angular"
        else:
            print(f"DEBUG: Using standard equidistant sampling along contour")
            points, arc = edge_analysis.sample_edge_points(contour, n_points, point_spacing)
            method_name = "Standard"

        # Generate debug visualization
        #self.visualize_sampling(image, mask, points, method_name=method_name)

        # Curvature, inward normals and intensity (open edges: no wrap-around, end points not sampled)
        r = edge_analysis.analyze_points(image, mask, contour, points, depth, width, min_cell_coverage, try_rotation,
                                         exclude_endpoints, smoothing_px, straight_threshold, arc=arc)
        curvatures, normals, intensities = r["curvatures"], r["normals"], r["intensities"]
        sampling_regions, valid_points = r["sampling_regions"], r["valid_points"]
        recovered_by_rotation = r["recovered_by_rotation"]

        # Store results
        results = {
            "points": points,
            "curvatures": curvatures,
            "normals": normals,
            "intensities": intensities,
            "sampling_regions": sampling_regions,
            "valid_points": valid_points,
            "contour_data": contour,
            "recovered_by_rotation": recovered_by_rotation,
            "exclude_endpoints": exclude_endpoints,
            "angular_distribution": angular_distribution,
            "axis_based_distribution": axis_based_distribution,
            "x_axis_distribution": x_axis_distribution,
            "closed": r["closed"]
        }

        return results

    def add_contours_to_results(self):
        """Add contour data to existing results if missing"""
        # Check if data is loaded
        if not self.loaded_data or 'results' not in self.loaded_data or not self.loaded_data['results']:
            QMessageBox.warning(self, "Warning", "No analysis results found. Please run analysis first.")
            return False

        # Check if masks are loaded
        if 'masks' not in self.loaded_data or self.loaded_data['masks'] is None:
            QMessageBox.warning(self, "Warning", "No mask data found. Please load masks first.")
            return False

        results = self.loaded_data['results']
        masks = self.loaded_data['masks']

        # Count missing contours
        frames_updated = 0

        # Update each frame's results
        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            # Skip if contour already exists
            if 'contour_data' in frame_results and frame_results['contour_data'] is not None:
                continue

            # Get mask for this frame
            if frame_idx < masks.shape[0]:
                mask = masks[frame_idx]

                # Detect contour
                contour = self.curvature_analyzer.detect_cell_edge(mask)

                if len(contour) > 0:
                    # Add contour to results
                    frame_results['contour_data'] = contour
                    frames_updated += 1

        if frames_updated > 0:
            self.log_console.log(f"Added contours to {frames_updated} frames")
            return True
        else:
            self.log_console.log("No frames needed contour updates")
            return False


    def load_tiff_stack(self, image_path, mask_path):
        """
        Load a TIFF stack of microscope images and binary masks.

        Returns:
        --------
        (images, masks) : tuple
            Tuple of ndarray containing images and masks
        """
        # Load images
        images = tifffile.imread(image_path)

        # Load masks
        masks = tifffile.imread(mask_path)

        # Ensure masks are binary (0 = background, 1 = cell)
        masks = (masks > 0).astype(np.uint8)

        # Handle single image case
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]

        return images, masks

    def detect_cell_edge(self, mask):
        """Longest 0.5-level contour of a binary mask (see edge_analysis.detect_cell_edge)."""
        return edge_analysis.detect_cell_edge(mask)

    def sample_equidistant_points(self, contour, n_points=DEFAULTS["n_points"], point_spacing=None):
        """Equidistant points along a contour (see edge_analysis.sample_equidistant_points)."""
        return edge_analysis.sample_equidistant_points(contour, n_points, point_spacing)


    def measure_curvature(self, points, closed=True, straight_threshold=0.0):
        """(sign, magnitude, normalized) by the three-point rule; sign convention documented in edge_curvature."""
        return edge_analysis.measure_curvature(points, closed, straight_threshold)

    def calculate_inward_normal(self, points, mask, closed=True):
        """Unit normals pointing into the cell (see edge_analysis.calculate_inward_normal)."""
        return edge_analysis.calculate_inward_normal(points, mask, closed)

    def measure_intensity(self, image, mask, points, normals, depth=DEFAULTS["depth"], width=DEFAULTS["width"],
                          min_cell_coverage=DEFAULTS["min_cell_coverage"], try_rotation=DEFAULTS["try_rotation"],
                          exclude_endpoints=DEFAULTS["exclude_endpoints"]):
        """Mean intensity in rectangles reaching into the cell (see edge_analysis.measure_intensity).

        Returns (intensities, sampling_regions, valid_points, recovered_by_rotation).
        """
        return edge_analysis.measure_intensity(image, mask, points, normals, depth, width, min_cell_coverage,
                                               try_rotation, exclude_endpoints)

    def detect_edge_movement(self, current_contour, previous_contour, current_mask, previous_mask):
        """(normalised score, movement type, movement map) from two masks; the contours are not used.

        See edge_analysis.detect_edge_movement.
        """
        return edge_analysis.detect_edge_movement(current_mask, previous_mask)

    def calculate_summary_correlation_stats(self, all_curvatures, all_intensities, all_valid_points):
        """Pooled regression of intensity on curvature (see edge_analysis.summary_correlation_stats)."""
        return edge_analysis.summary_correlation_stats(all_curvatures, all_intensities, all_valid_points)


