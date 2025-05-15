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

class CurvatureAnalyzer:
    """
    Class for analyzing cell curvature, intensity, and edge movement.
    Provides methods for both full stack processing and single frame analysis.
    """

    def __init__(self):
        """Initialize CurvatureAnalyzer"""
        pass

    def process_stack(self, image_path, mask_path, output_dir, n_points=100,
                     depth=20, width=5, min_cell_coverage=0.8, progress_callback=None):
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
            Number of equidistant points along the contour
        depth : int
            Depth of the sampling rectangle (how far into the cell)
        width : int
            Width of the sampling rectangle
        min_cell_coverage : float
            Minimum fraction of the rectangle that must be inside the cell
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
            "min_cell_coverage": min_cell_coverage
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
                image, mask, n_points, depth, width, min_cell_coverage)

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
                movement_score, movement_type, movement_map = self.detect_edge_movement(
                    contour, all_contours[frame_idx - 1], mask, previous_mask
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

    def analyze_frame(self, image, mask, n_points=100, depth=20, width=5, min_cell_coverage=0.8):
        """
        Analyze a single frame

        Parameters:
        -----------
        image : ndarray
            Microscope image
        mask : ndarray
            Binary mask
        n_points : int
            Number of equidistant points along the contour
        depth : int
            Depth of the sampling rectangle (how far into the cell)
        width : int
            Width of the sampling rectangle
        min_cell_coverage : float
            Minimum fraction of the rectangle that must be inside the cell

        Returns:
        --------
        dict
            Dictionary of analysis results for the frame
        """
        # Detect cell edge
        contour = self.detect_cell_edge(mask)

        if len(contour) == 0:
            return {"error": "No contour found"}

        # Sample equidistant points
        points = self.sample_equidistant_points(contour, n_points)

        # Measure curvature
        curvatures = self.measure_curvature(points)

        # Calculate inward normals
        normals = self.calculate_inward_normal(points, mask)

        # Measure intensity with cell coverage check
        intensities, sampling_regions, valid_points = self.measure_intensity(
            image, mask, points, normals, depth, width, min_cell_coverage
        )

        # Store results
        results = {
            "points": points,
            "curvatures": curvatures,
            "normals": normals,
            "intensities": intensities,
            "sampling_regions": sampling_regions,
            "valid_points": valid_points,
            "contour_data": contour  # Make sure contour data is included
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
        """
        Detect the cell edge from a binary mask.

        Parameters:
        -----------
        mask : ndarray
            Binary mask

        Returns:
        --------
        contour : ndarray
            Cell contour points
        """
        # Get contours from the binary mask
        contours = measure.find_contours(mask, 0.5)

        # Usually the longest contour corresponds to the cell edge
        if len(contours) > 0:
            contour = max(contours, key=len)
            return contour
        else:
            return np.array([])

    def sample_equidistant_points(self, contour, n_points=100):
        """
        Sample equidistant points along a contour.

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        sampled_points : ndarray
            Array of equidistant points
        """
        # Calculate the total contour length
        contour_length = 0
        for i in range(len(contour) - 1):
            contour_length += np.linalg.norm(contour[i+1] - contour[i])

        # Distance between equidistant points
        step_size = contour_length / n_points

        # Sample equidistant points
        sampled_points = []
        current_length = 0

        i = 0
        sampled_points.append(contour[0])

        while len(sampled_points) < n_points and i < len(contour) - 1:
            # Distance to the next point on the contour
            dist = np.linalg.norm(contour[i+1] - contour[i])

            if current_length + dist >= step_size:
                # Interpolate to find the next equidistant point
                t = (step_size - current_length) / dist
                next_point = contour[i] + t * (contour[i+1] - contour[i])
                sampled_points.append(next_point)
                current_length = 0
                # Do not increment i, stay on the same segment
            else:
                # Move to the next segment
                current_length += dist
                i += 1

        # If we couldn't sample enough points, just return what we have
        return np.array(sampled_points)

    def measure_curvature(self, points):
        """
        Measure curvature at each point along the contour.

        Parameters:
        -----------
        points : ndarray
            Array of contour points

        Returns:
        --------
        (sign_curvatures, magnitude_curvatures, normalized_curvatures) : tuple
            Tuple of curvature measurements
        """
        n_points = len(points)
        sign_curvatures = np.zeros(n_points)
        magnitude_curvatures = np.zeros(n_points)

        for i in range(n_points):
            # Get adjacent points (with wrapping for closed contours)
            left_idx = (i - 1) % n_points
            right_idx = (i + 1) % n_points

            p = points[i]
            p_left = points[left_idx]
            p_right = points[right_idx]

            # Vector from left to right point
            v = p_right - p_left
            v_length = np.linalg.norm(v)

            # Vector from left to current point
            w = p - p_left

            # Cross product to determine if point is left or right of the line
            cross_product = v[0] * w[1] - v[1] * w[0]

            # Normalized cross product as measure of curvature magnitude
            if v_length > 0:
                magnitude = abs(cross_product) / (v_length * v_length)
            else:
                magnitude = 0

            magnitude_curvatures[i] = magnitude

            # Sign of curvature
            if cross_product > 0:
                sign_curvatures[i] = 1  # Positive curvature (outside, wound side)
            else:
                sign_curvatures[i] = -1  # Negative curvature (inside, monolayer side)

        # Create normalized curvatures (preserve sign but normalize magnitude to [0, 1])
        max_magnitude = np.max(magnitude_curvatures)
        if max_magnitude > 0:
            normalized_curvatures = sign_curvatures * (magnitude_curvatures / max_magnitude)
        else:
            normalized_curvatures = sign_curvatures * 0

        return sign_curvatures, magnitude_curvatures, normalized_curvatures

    def calculate_inward_normal(self, points, mask):
        """
        Calculate the inward normal vector at each point.

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        mask : ndarray
            Binary mask

        Returns:
        --------
        normals : ndarray
            Array of inward normal vectors
        """
        n_points = len(points)
        normals = np.zeros((n_points, 2))
        mask_shape = mask.shape

        for i in range(n_points):
            # Get adjacent points (with wrapping for closed contours)
            left_idx = (i - 1) % n_points
            right_idx = (i + 1) % n_points

            p = points[i]
            p_left = points[left_idx]
            p_right = points[right_idx]

            # Calculate tangent vector
            tangent = p_right - p_left
            tangent = tangent / np.linalg.norm(tangent)

            # Calculate normal vectors (both possible directions)
            normal1 = np.array([-tangent[1], tangent[0]])
            normal2 = -normal1

            # Check which normal points into the cell
            test_point1 = p + 3 * normal1
            test_point2 = p + 3 * normal2

            # Ensure test points are within image bounds
            test_point1 = np.clip(test_point1, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])
            test_point2 = np.clip(test_point2, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])

            # Check mask values
            val1 = mask[int(test_point1[0]), int(test_point1[1])]
            val2 = mask[int(test_point2[0]), int(test_point2[1])]

            # Choose the normal that points into the cell (value 1 in mask)
            if val1 == 1:
                normals[i] = normal1
            elif val2 == 1:
                normals[i] = normal2
            else:
                # If neither point is inside the cell, use the curvature to guess
                normals[i] = normal2  # Default

        return normals

    def measure_intensity(self, image, mask, points, normals, depth=20, width=5, min_cell_coverage=0.8):
        """
        Measure mean intensity within rectangular regions extending from each point.

        Parameters:
        -----------
        image : ndarray
            The intensity image
        mask : ndarray
            Binary mask where 1 = cell, 0 = background
        points : ndarray
            Equidistant points along the contour
        normals : ndarray
            Normal vectors pointing into the cell at each point
        depth : int
            Length of the rectangle (how far into the cell)
        width : int
            Width of the rectangle
        min_cell_coverage : float
            Minimum fraction of the rectangle that must be inside the cell

        Returns:
        --------
        (intensities, sampling_regions, valid_points) : tuple
            Tuple of intensity measurements, sampling regions and validity flags
        """
        n_points = len(points)
        intensities = np.full(n_points, np.nan)  # Initialize with NaN
        sampling_regions = []
        valid_points = np.zeros(n_points, dtype=bool)

        image_shape = image.shape

        for i in range(n_points):
            p = points[i]
            normal = normals[i]

            # Calculate end point of the vector
            end_point = p + depth * normal

            # Calculate perpendicular direction (for width)
            perp = np.array([-normal[1], normal[0]])

            # Calculate corner points of the rectangle
            corner1 = p + width/2 * perp
            corner2 = p - width/2 * perp
            corner3 = end_point - width/2 * perp
            corner4 = end_point + width/2 * perp

            # Create a polygon for the rectangle
            vertices = np.array([corner1, corner4, corner3, corner2])

            # Get all pixels within the polygon
            rr, cc = draw.polygon(vertices[:, 0], vertices[:, 1], image_shape)

            # Check if any pixels are within bounds
            valid_pixels = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
            if np.any(valid_pixels):
                rr = rr[valid_pixels]
                cc = cc[valid_pixels]

                # Check cell coverage
                total_pixels = len(rr)
                cell_pixels = np.sum(mask[rr, cc])
                cell_coverage = cell_pixels / total_pixels

                # Only include points with sufficient cell coverage
                if cell_coverage >= min_cell_coverage:
                    # Calculate mean intensity
                    intensities[i] = np.mean(image[rr, cc])
                    valid_points[i] = True

            # Save sampling region for visualization (even if not valid)
            sampling_regions.append(vertices)

        return intensities, sampling_regions, valid_points

    def detect_edge_movement(self, current_contour, previous_contour, current_mask, previous_mask):
        """
        Analyze the movement of cell edge between two consecutive frames.

        Parameters:
        -----------
        current_contour : ndarray
            Contour points from current frame
        previous_contour : ndarray
            Contour points from previous frame
        current_mask : ndarray
            Binary mask from current frame
        previous_mask : ndarray
            Binary mask from previous frame

        Returns:
        --------
        (movement_score, movement_type, movement_map) : tuple
            Movement score, type and movement map
        """
        # Create a movement map (2D array with same shape as the masks)
        movement_map = np.zeros_like(current_mask, dtype=float)

        # Calculate the difference between current and previous masks
        diff_mask = current_mask.astype(int) - previous_mask.astype(int)

        # Identify extension and retraction regions
        extension_regions = (diff_mask == 1)
        retraction_regions = (diff_mask == -1)

        # Set values in the movement map
        movement_map[extension_regions] = 1.0  # Extension
        movement_map[retraction_regions] = -1.0  # Retraction

        # Count pixels in extension and retraction regions
        extension_pixels = np.sum(extension_regions)
        retraction_pixels = np.sum(retraction_regions)

        # Calculate net movement score (positive for extension, negative for retraction)
        movement_score = extension_pixels - retraction_pixels

        # Normalize by the perimeter length (approximated by total changed pixels)
        total_changed_pixels = extension_pixels + retraction_pixels
        if total_changed_pixels > 0:
            normalized_score = movement_score / total_changed_pixels
        else:
            normalized_score = 0.0

        # Classify movement type with a small threshold for stability
        if normalized_score > 0.1:
            movement_type = 'extending'
        elif normalized_score < -0.1:
            movement_type = 'retracting'
        else:
            movement_type = 'stable'

        return normalized_score, movement_type, movement_map

    def calculate_summary_correlation_stats(self, all_curvatures, all_intensities, all_valid_points):
        """
        Calculate summary correlation statistics for all valid data points.

        Parameters:
        -----------
        all_curvatures : list
            List of curvature arrays
        all_intensities : list
            List of intensity arrays
        all_valid_points : list
            List of validity masks

        Returns:
        --------
        stats : dict
            Dictionary of correlation statistics
        """
        # Combine data from all frames
        combined_curvatures = []
        combined_intensities = []

        for frame_idx in range(len(all_curvatures)):
            curvatures = all_curvatures[frame_idx]
            intensities = all_intensities[frame_idx]
            valid_points = all_valid_points[frame_idx]

            # Filter out invalid points
            valid_curvatures = curvatures[valid_points]
            valid_intensities = intensities[valid_points]

            combined_curvatures.extend(valid_curvatures)
            combined_intensities.extend(valid_intensities)

        # Convert to numpy arrays
        combined_curvatures = np.array(combined_curvatures)
        combined_intensities = np.array(combined_intensities)

        # Skip if no valid points
        if len(combined_curvatures) == 0:
            return {
                "r_squared": None,
                "p_value": None,
                "sample_size": 0
            }

        # Calculate correlation statistics
        if len(combined_curvatures) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_curvatures, combined_intensities)

            return {
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "sample_size": int(len(combined_curvatures)),
                "slope": float(slope),
                "intercept": float(intercept),
                "standard_error": float(std_err)
            }
        else:
            return {
                "r_squared": None,
                "p_value": None,
                "sample_size": int(len(combined_curvatures))
            }
