import numpy as np
from PyQt5.QtWidgets import QMessageBox
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure, morphology, draw
from scipy import stats
import os
import datetime
import json
import tifffile

from . import edge_analysis
from .analysis_defaults import DEFAULTS

class CurvatureAnalyzer:
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
                     progress_callback=None):
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

        Returns:
        --------
        dict
            Dictionary of analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

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
            "x_axis_distribution": x_axis_distribution
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
                axis_based_distribution, x_axis_distribution)

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
                     axis_based_distribution=False, x_axis_distribution=False):
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
            method_name = "X-Axis"
        elif axis_based_distribution:
            print(f"DEBUG: Using axis-based distribution of sample points")
            points = self.sample_axis_based_points(contour, n_points)
            method_name = "Axis-Based"
        elif angular_distribution:
            print(f"DEBUG: Using angular distribution of sample points")
            points = self.sample_angular_points(contour, n_points)
            method_name = "Angular"
        else:
            print(f"DEBUG: Using standard equidistant sampling along contour")
            points = self.sample_equidistant_points(contour, n_points)
            method_name = "Standard"

        # Generate debug visualization
        #self.visualize_sampling(image, mask, points, method_name=method_name)

        # Measure curvature
        curvatures = self.measure_curvature(points)

        # Calculate inward normals
        normals = self.calculate_inward_normal(points, mask)

        # Measure intensity with cell coverage check
        intensities, sampling_regions, valid_points, recovered_by_rotation = self.measure_intensity(
            image, mask, points, normals, depth, width, min_cell_coverage, try_rotation, exclude_endpoints
        )

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
            "x_axis_distribution": x_axis_distribution
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

    def sample_equidistant_points(self, contour, n_points=DEFAULTS["n_points"]):
        """Equidistant points along a contour (see edge_analysis.sample_equidistant_points)."""
        return edge_analysis.sample_equidistant_points(contour, n_points)


    def sample_angular_points(self, contour, n_points=100):
        """
        Sample points around the contour at equal angular intervals.

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        sampled_points : ndarray
            Array of sampled points distributed evenly by angle
        """
        # Find the centroid of the cell (approximate center)
        center = np.mean(contour, axis=0)

        print(f"DEBUG: Cell centroid is at {center}")

        # Convert contour points to polar coordinates
        # Calculate the angle of each point relative to center
        polar_coords = []
        for point in contour:
            # Calculate vector from center to point
            vec = point - center
            # Calculate angle in radians
            angle = np.arctan2(vec[1], vec[0])
            # Calculate radius (distance from center)
            radius = np.linalg.norm(vec)
            polar_coords.append((angle, radius, point))

        # Sort by angle
        polar_coords.sort(key=lambda x: x[0])

        # Convert angles to cumulative form (unwrapped)
        unwrapped_coords = []
        prev_angle = polar_coords[0][0]
        cumulative_angle = prev_angle

        for angle, radius, point in polar_coords:
            # Handle angle wrap-around
            delta = angle - prev_angle
            if delta < -np.pi:
                delta += 2*np.pi
            elif delta > np.pi:
                delta -= 2*np.pi

            cumulative_angle += delta
            unwrapped_coords.append((cumulative_angle, radius, point))
            prev_angle = angle

        # Get min and max of cumulative angle
        min_angle = unwrapped_coords[0][0]
        max_angle = unwrapped_coords[-1][0]

        # Create target angles
        target_angles = np.linspace(min_angle, max_angle, n_points)

        # Find closest points to each target angle
        sampled_points = []
        for target in target_angles:
            closest_idx = min(range(len(unwrapped_coords)),
                              key=lambda i: abs(unwrapped_coords[i][0] - target))
            sampled_points.append(unwrapped_coords[closest_idx][2])

        # Check for duplicate points and adjust if necessary
        unique_points = []
        for i, point in enumerate(sampled_points):
            if i > 0 and np.array_equal(point, unique_points[-1]):
                # Find the next closest point to this target angle
                print(f"DEBUG: Found duplicate point at index {i}, finding alternative")
                target = target_angles[i]
                distances = [abs(coord[0] - target) for coord in unwrapped_coords]
                sorted_indices = np.argsort(distances)

                # Try the second closest point
                for idx in sorted_indices[1:5]:  # Try up to 5 alternative points
                    alt_point = unwrapped_coords[idx][2]
                    if not any(np.array_equal(alt_point, p) for p in unique_points):
                        unique_points.append(alt_point)
                        print(f"DEBUG: Found alternative point")
                        break
                else:
                    # If we can't find a unique point, slightly perturb the current one
                    perturbed = point + np.random.normal(0, 0.1, 2)
                    unique_points.append(perturbed)
                    print(f"DEBUG: Using perturbed point")
            else:
                unique_points.append(point)

        sampled_points = np.array(unique_points)

        print(f"DEBUG: Sampled {len(sampled_points)} points using angular distribution")

        # Check for points that are too close to each other
        for i in range(len(sampled_points)):
            next_i = (i + 1) % len(sampled_points)  # Wrap around to the first point
            p1 = sampled_points[i]
            p2 = sampled_points[next_i]
            dist = np.linalg.norm(p2 - p1)
            if dist < 1e-10:  # Very small distance
                print(f"DEBUG: Points {i} and {next_i} are very close: distance={dist}")

        return np.array(sampled_points)

    def sample_axis_based_points(self, contour, n_points=100):
        """
        Sample points around the contour with even spacing along the image axes.
        This helps reduce bunching in extrusions by ensuring points are distributed
        evenly in screen space rather than by contour distance or angle.

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        sampled_points : ndarray
            Array of sampled points distributed evenly by axis position
        """
        # Get the bounding box of the contour
        min_y, min_x = np.min(contour, axis=0)
        max_y, max_x = np.max(contour, axis=0)

        print(f"DEBUG: Contour bounding box: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")

        # We'll use a combination of x-axis and y-axis sampling
        # First, let's determine how many points to sample from each axis
        # We'll split them based on the aspect ratio of the bounding box
        width = max_x - min_x
        height = max_y - min_y

        # Calculate how many points to allocate to each axis based on perimeter
        perimeter = 2 * (width + height)
        x_portion = 2 * width / perimeter
        y_portion = 2 * height / perimeter

        n_x_points = int(n_points * x_portion)
        n_y_points = n_points - n_x_points

        print(f"DEBUG: Sampling {n_x_points} points from x-axis edges, {n_y_points} points from y-axis edges")

        # Create arrays of x and y coordinates for sampling
        x_values = []
        y_values = []

        # Sample points from the top and bottom edges
        if n_x_points > 0:
            x_step = width / (n_x_points / 2)

            # Top edge (constant y = min_y)
            for i in range(n_x_points // 2):
                x = min_x + i * x_step
                x_values.append(x)
                y_values.append(min_y)

            # Bottom edge (constant y = max_y)
            for i in range(n_x_points // 2):
                x = max_x - i * x_step
                x_values.append(x)
                y_values.append(max_y)

        # Sample points from the left and right edges
        if n_y_points > 0:
            y_step = height / (n_y_points / 2)

            # Left edge (constant x = min_x)
            for i in range(n_y_points // 2):
                y = min_y + i * y_step
                x_values.append(min_x)
                y_values.append(y)

            # Right edge (constant x = max_x)
            for i in range(n_y_points // 2):
                y = max_y - i * y_step
                x_values.append(max_x)
                y_values.append(y)

        # Create grid points from these coordinates
        grid_points = np.column_stack((y_values, x_values))

        # Find the closest contour point to each grid point
        sampled_points = []
        for grid_point in grid_points:
            # Calculate distances to all contour points
            distances = np.sqrt(np.sum((contour - grid_point)**2, axis=1))

            # Find the closest point
            closest_idx = np.argmin(distances)
            closest_point = contour[closest_idx]

            # Add to sampled points, checking for duplicates
            if not any(np.array_equal(closest_point, p) for p in sampled_points):
                sampled_points.append(closest_point)

        # If we don't have enough points (due to duplicates), add more
        if len(sampled_points) < n_points // 2:  # If we have far too few points
            print(f"DEBUG: Only found {len(sampled_points)} unique points, adding more")

            # Sort contour points by their perimeter distance
            distances = [0]
            for i in range(1, len(contour)):
                distances.append(distances[-1] + np.linalg.norm(contour[i] - contour[i-1]))

            total_length = distances[-1]

            # Sample additional points evenly by distance
            additional_points_needed = n_points - len(sampled_points)
            for i in range(additional_points_needed):
                target_dist = (i + 1) * total_length / (additional_points_needed + 1)

                # Find the closest point by distance
                idx = np.searchsorted(distances, target_dist)
                if idx >= len(contour):
                    idx = len(contour) - 1

                point = contour[idx]

                # Add if not already included
                if not any(np.array_equal(point, p) for p in sampled_points):
                    sampled_points.append(point)

        # Convert to numpy array
        sampled_points = np.array(sampled_points)

        print(f"DEBUG: Sampled {len(sampled_points)} points using axis-based distribution")

        return sampled_points

    def sample_x_axis_points(self, contour, n_points=100):
        """
        Sample points along the contour with even spacing along the x-axis.
        This distributes points at regular x-intervals to prevent bunching
        in areas with membrane extrusions.

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        sampled_points : ndarray
            Array of sampled points distributed evenly along x-axis
        """
        # Sort contour points by x-coordinate
        x_sorted_indices = np.argsort(contour[:, 1])
        x_sorted_contour = contour[x_sorted_indices]

        # Get min and max x values
        min_x = x_sorted_contour[0, 1]
        max_x = x_sorted_contour[-1, 1]
        x_range = max_x - min_x

        print(f"DEBUG: X-axis range: [{min_x:.1f}, {max_x:.1f}]")

        # Calculate x-positions with even spacing
        x_positions = np.linspace(min_x, max_x, n_points)

        # For each x-position, find the closest point on the contour
        sampled_points = []
        for x_pos in x_positions:
            # Find the closest point by x-coordinate
            x_distances = np.abs(contour[:, 1] - x_pos)
            closest_idx = np.argmin(x_distances)
            closest_point = contour[closest_idx]

            # Add point to our sampled list
            sampled_points.append(closest_point)

        # Convert to numpy array
        sampled_points = np.array(sampled_points)

        print(f"DEBUG: Sampled {len(sampled_points)} points based on x-axis positions")

        return sampled_points

    def measure_curvature(self, points):
        """(sign, magnitude, normalized) curvature; sign convention documented in edge_analysis."""
        return edge_analysis.measure_curvature(points)

    def calculate_inward_normal(self, points, mask):
        """Unit normals pointing into the cell (see edge_analysis.calculate_inward_normal)."""
        return edge_analysis.calculate_inward_normal(points, mask)

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


    def debug_visualize_sampling_methods(self, contour, n_points=100):
        """
        Create a debug visualization comparing standard and angular sampling methods

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Debug visualization figure
        """
        import matplotlib.pyplot as plt

        # Get both types of sampled points
        std_points = self.sample_equidistant_points(contour, n_points)
        ang_points = self.sample_angular_points(contour, n_points)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot standard sampling
        ax1.plot(contour[:, 1], contour[:, 0], 'b-', alpha=0.5, label='Contour')
        ax1.scatter(std_points[:, 1], std_points[:, 0], c='r', s=30, label='Sampled Points')
        ax1.set_title('Standard Equidistant Sampling')
        ax1.legend()
        ax1.axis('equal')

        # Plot angular sampling
        ax2.plot(contour[:, 1], contour[:, 0], 'b-', alpha=0.5, label='Contour')
        ax2.scatter(ang_points[:, 1], ang_points[:, 0], c='g', s=30, label='Sampled Points')
        ax2.set_title('Angular Sampling')
        ax2.legend()
        ax2.axis('equal')

        plt.tight_layout()

        # Save comparison to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"sampling_comparison_{timestamp}.png", dpi=150)

        print(f"DEBUG: Saved sampling method comparison to sampling_comparison_{timestamp}.png")

        return fig

    def visualize_sampling(self, image, mask, points, method_name=""):
        """
        Create a visualization of the sampled points on the cell contour
        """
        import matplotlib.pyplot as plt

        # Extract the contour
        contour = self.detect_cell_edge(mask)

        # Create figure
        plt.figure(figsize=(10, 8))

        # Show original image
        plt.imshow(image, cmap='gray')

        # Plot the contour
        if len(contour) > 0:
            plt.plot(contour[:, 1], contour[:, 0], 'b-', alpha=0.5, linewidth=1)

        # Plot the sampled points
        plt.scatter(points[:, 1], points[:, 0], c='r', s=30)

        # Add point indices (every 10th point)
        for i in range(0, len(points), 10):
            plt.text(points[i, 1] + 3, points[i, 0] + 3, str(i), color='white', fontsize=9,
                     bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

        # If using X-axis distribution, add vertical lines to show x-positions
        if "X-Axis" in method_name:
            # Get the min and max y values to define line lengths
            min_y = np.min(contour[:, 0])
            max_y = np.max(contour[:, 0])
            y_range = max_y - min_y

            # Draw vertical lines at each x-position
            for point in points:
                x = point[1]  # x-coordinate
                # Draw a vertical line from above to below the cell
                plt.plot([x, x], [min_y - 0.1*y_range, max_y + 0.1*y_range],
                        'g--', alpha=0.3, linewidth=0.5)

        # Add title
        plt.title(f"{method_name} Point Sampling")

        # Add center point if angular method
        if "Angular" in method_name:
            center = np.mean(contour, axis=0)
            plt.scatter([center[1]], [center[0]], c='cyan', s=100, marker='x')

            # Add a few radial lines to show the angular divisions
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                r = np.sqrt(image.shape[0]**2 + image.shape[1]**2) / 2  # Radius to edge of image
                x_end = center[1] + r * np.cos(angle)
                y_end = center[0] + r * np.sin(angle)
                plt.plot([center[1], x_end], [center[0], y_end], 'cyan', alpha=0.3, linestyle='--')

        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])

        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{method_name.lower().replace(' ', '_')}_sampling_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"DEBUG: Saved sampling visualization to {filename}")
