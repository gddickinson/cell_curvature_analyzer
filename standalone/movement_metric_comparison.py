#!/usr/bin/env python3
"""
Movement Metric Comparison Script

This script compares two methods of calculating membrane movement:
1. Binary mask difference (temporal comparison between frames)
2. Curvature gradient method (spatial derivative along contour)

Both methods are used to analyze the same points, and correlation with intensity
is calculated and visualized for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure, morphology, draw
from scipy import stats
import os
import tifffile
from pathlib import Path

#####################################################################
# CONFIGURATION
#####################################################################

# Paths
DATA_DIR = "data/test_1"  # Directory containing test data
IMAGE_FILE = "151_2019_07_29_TIRF_mKera_scratch_1_MMStack_Pos7_piezo1.tif"  # Intensity images
MASK_FILE = "151_2019_07_29_TIRF_mKera_scratch_Pos7_DIC_Mask_test.tif"     # Binary masks

# Output
OUTPUT_DIR = "results/movement_comparison"  # Output directory for results
SAVE_PLOTS = True                           # Whether to save plots to disk

# Analysis parameters
N_POINTS = 10          # Number of points to sample along contour
DEPTH = 150              # Depth of sampling rectangles
WIDTH = 75               # Width of sampling rectangles
MIN_CELL_COVERAGE = 0.8 # Minimum fraction of sampling rectangle inside cell
TRY_ROTATION = True     # Try rotating rejected sampling regions
EXCLUDE_ENDPOINTS = False # Exclude first and last points

# Visualization
COLORMAP_CURVATURE = 'coolwarm'  # Colormap for curvature
COLORMAP_INTENSITY = 'viridis'   # Colormap for intensity
COLORMAP_MOVEMENT = 'coolwarm'  # Colormap for movement (red=retraction, blue=extension)
FRAMES_TO_ANALYZE = [1, 2, 3]    # List of frames to analyze (0-based indexing)

#####################################################################
# UTILITY FUNCTIONS
#####################################################################

def load_tiff_stack(file_path):
    """Load a TIFF stack and handle single images"""
    images = tifffile.imread(file_path)
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    return images

def detect_cell_edge(mask):
    """Detect the cell edge from a binary mask"""
    contours = measure.find_contours(mask, 0.5)
    if len(contours) > 0:
        contour = max(contours, key=len)
        return contour
    else:
        return np.array([])

def sample_equidistant_points(contour, n_points=100):
    """Sample equidistant points along contour"""
    # Calculate total contour length
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
        # Distance to next point on contour
        dist = np.linalg.norm(contour[i+1] - contour[i])

        if current_length + dist >= step_size:
            # Interpolate to find next equidistant point
            t = (step_size - current_length) / dist
            next_point = contour[i] + t * (contour[i+1] - contour[i])
            sampled_points.append(next_point)
            current_length = 0
            # Don't increment i, stay on same segment
        else:
            # Move to next segment
            current_length += dist
            i += 1

    # If we couldn't sample enough points, return what we have
    return np.array(sampled_points)

def measure_curvature(points):
    """Measure curvature at each point along the contour"""
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

def calculate_inward_normal(points, mask):
    """Calculate the inward normal vector at each point"""
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
        tangent_norm = np.linalg.norm(tangent)

        # Check if tangent has non-zero length
        if tangent_norm > 1e-10:  # Only normalize if length is non-zero
            tangent = tangent / tangent_norm
        else:
            # If tangent is too small, try using more distant points
            left_idx2 = (i - 2) % n_points
            right_idx2 = (i + 2) % n_points
            p_left2 = points[left_idx2]
            p_right2 = points[right_idx2]
            tangent = p_right2 - p_left2
            tangent_norm = np.linalg.norm(tangent)

            if tangent_norm > 1e-10:
                tangent = tangent / tangent_norm
            else:
                # If still too small, use a default tangent along x-axis
                tangent = np.array([1.0, 0.0])

        # Calculate normal vectors (both possible directions)
        normal1 = np.array([-tangent[1], tangent[0]])
        normal2 = -normal1

        # Check which normal points into the cell
        test_point1 = p + 3 * normal1
        test_point2 = p + 3 * normal2

        # Ensure test points are within image bounds
        test_point1 = np.clip(test_point1, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])
        test_point2 = np.clip(test_point2, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])

        # Convert to integers
        tp1_y, tp1_x = int(round(test_point1[0])), int(round(test_point1[1]))
        tp2_y, tp2_x = int(round(test_point2[0])), int(round(test_point2[1]))

        try:
            # Check mask values
            val1 = mask[tp1_y, tp1_x]
            val2 = mask[tp2_y, tp2_x]

            # Choose the normal that points into the cell (value 1 in mask)
            if val1 == 1:
                normals[i] = normal1
            elif val2 == 1:
                normals[i] = normal2
            else:
                # If neither point is inside the cell, use a default
                normals[i] = normal2
        except IndexError:
            # Use a reasonable default if out of bounds
            normals[i] = normal1

    return normals

def measure_intensity(image, mask, points, normals, depth=20, width=5, min_cell_coverage=0.8, try_rotation=False, exclude_endpoints=False):
    """Measure mean intensity within rectangular regions extending from each point"""
    n_points = len(points)
    intensities = np.full(n_points, np.nan)  # Initialize with NaN
    sampling_regions = []
    valid_points = np.zeros(n_points, dtype=bool)
    recovered_by_rotation = np.zeros(n_points, dtype=bool)

    image_shape = image.shape

    for i in range(n_points):
        # Check if this is an endpoint that should be excluded
        if exclude_endpoints and (i == 0 or i == n_points - 1):
            # Add empty region for visualization consistency
            sampling_regions.append(np.zeros((4, 2)))
            continue

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

            # If coverage is insufficient and rotation is enabled, try rotating the region
            elif try_rotation:
                # Rotate normal vector 180 degrees
                rotated_normal = -normal

                # Recalculate end point with rotated normal
                rotated_end_point = p + depth * rotated_normal

                # Calculate corner points of the rotated rectangle
                rotated_corner1 = p + width/2 * perp
                rotated_corner2 = p - width/2 * perp
                rotated_corner3 = rotated_end_point - width/2 * perp
                rotated_corner4 = rotated_end_point + width/2 * perp

                # Create a polygon for the rotated rectangle
                rotated_vertices = np.array([rotated_corner1, rotated_corner4, rotated_corner3, rotated_corner2])

                # Get all pixels within the rotated polygon
                rotated_rr, rotated_cc = draw.polygon(rotated_vertices[:, 0], rotated_vertices[:, 1], image_shape)

                # Check if any pixels are within bounds
                rotated_valid_pixels = (rotated_rr >= 0) & (rotated_rr < image_shape[0]) & (rotated_cc >= 0) & (rotated_cc < image_shape[1])
                if np.any(rotated_valid_pixels):
                    rotated_rr = rotated_rr[rotated_valid_pixels]
                    rotated_cc = rotated_cc[rotated_valid_pixels]

                    # Check cell coverage of rotated region
                    rotated_total_pixels = len(rotated_rr)
                    rotated_cell_pixels = np.sum(mask[rotated_rr, rotated_cc])
                    rotated_cell_coverage = rotated_cell_pixels / rotated_total_pixels

                    # If rotated region has sufficient coverage, use it
                    if rotated_cell_coverage >= min_cell_coverage:
                        intensities[i] = np.mean(image[rotated_rr, rotated_cc])
                        valid_points[i] = True
                        recovered_by_rotation[i] = True

                        # Use the rotated vertices for visualization
                        vertices = rotated_vertices

        # Save sampling region for visualization (even if not valid)
        sampling_regions.append(vertices)

    return intensities, sampling_regions, valid_points, recovered_by_rotation

#####################################################################
# MOVEMENT CALCULATION METHODS
#####################################################################

def calculate_movement_from_masks(current_mask, previous_mask, points, valid_points):
    """
    Method 1: Calculate movement from binary mask difference

    This method calculates movement at each point by examining the local
    mask difference (current - previous) around each point.
    """
    n_points = len(points)
    movement_scores = np.zeros(n_points)
    movement_types = np.full(n_points, 'stable', dtype=object)

    # Calculate difference mask
    diff_mask = current_mask.astype(int) - previous_mask.astype(int)

    # For each valid point, calculate local movement
    for i in range(n_points):
        if not valid_points[i]:
            continue

        # Get a local region around the point
        point = points[i].astype(int)
        y, x = point

        # Define a circular region around the point (radius = 5 pixels)
        radius = 5
        y_min = max(0, y - radius)
        y_max = min(current_mask.shape[0] - 1, y + radius)
        x_min = max(0, x - radius)
        x_max = min(current_mask.shape[1] - 1, x + radius)

        # Extract local region of the difference mask
        local_diff = diff_mask[y_min:y_max+1, x_min:x_max+1]

        # Count extension and retraction pixels in the local region
        extension_pixels = np.sum(local_diff == 1)
        retraction_pixels = np.sum(local_diff == -1)

        # Calculate local movement score
        total_changed_pixels = extension_pixels + retraction_pixels
        if total_changed_pixels > 0:
            movement_scores[i] = (extension_pixels - retraction_pixels) / total_changed_pixels
        else:
            movement_scores[i] = 0

        # Classify movement type
        if movement_scores[i] > 0.1:
            movement_types[i] = 'extending'
        elif movement_scores[i] < -0.1:
            movement_types[i] = 'retracting'
        else:
            movement_types[i] = 'stable'

    return movement_scores, movement_types

def calculate_movement_from_curvature(points, curvatures, valid_points):
    """
    Method 2: Calculate movement from curvature gradient along contour

    This method uses the spatial gradient of curvature along the contour
    as a proxy for movement, as in the original implementation.
    """
    n_points = len(points)
    movement_scores = np.zeros(n_points)
    movement_types = np.full(n_points, 'stable', dtype=object)

    # Use normalized curvature for gradient calculation
    normalized_curvature = curvatures[2]

    # Calculate curvature gradient for each point
    for i in range(n_points):
        if not valid_points[i]:
            continue

        # Get neighboring points (with wrapping)
        prev_idx = (i - 1) % n_points
        next_idx = (i + 1) % n_points

        # Calculate curvature gradient (central difference)
        curvature_gradient = (normalized_curvature[next_idx] - normalized_curvature[prev_idx]) / 2
        movement_scores[i] = curvature_gradient

        # Classify movement type
        if curvature_gradient > 0.1:
            movement_types[i] = 'extending'
        elif curvature_gradient < -0.1:
            movement_types[i] = 'retracting'
        else:
            movement_types[i] = 'stable'

    return movement_scores, movement_types

#####################################################################
# VISUALIZATION FUNCTIONS
#####################################################################

def create_overlay(image, mask, points=None, valid_points=None, movement_scores=None,
                  mask_opacity=0.3, method_label=""):
    """Create an overlay with the original image, mask, and movement information"""
    # Normalize image to 0-1 range
    image_norm = image.astype(float)
    image_norm = (image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm) + 1e-10)

    # Create an RGB image with grayscale original
    overlay = np.stack([image_norm, image_norm, image_norm], axis=2)

    # Add red tint to masked areas
    overlay[mask > 0, 0] = np.maximum(overlay[mask > 0, 0], 0.8)  # Red channel
    overlay[mask > 0, 1] *= 0.8  # Reduce green
    overlay[mask > 0, 2] *= 0.8  # Reduce blue

    # Create a figure with the overlay
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlay)

    # If points are provided, add them to the overlay
    if points is not None and valid_points is not None and movement_scores is not None:
        # Color points by movement score
        cmap = plt.cm.get_cmap(COLORMAP_MOVEMENT)

        for i, point in enumerate(points):
            if valid_points[i]:
                # Map movement score to color (-1 to 1) -> (0 to 1)
                color = cmap((movement_scores[i] + 1) / 2)

                # Larger marker for more extreme movement
                size = 20 + 60 * abs(movement_scores[i])

                ax.scatter(point[1], point[0], color=color, s=size, alpha=0.8,
                          edgecolors='black', linewidths=0.5)

    # Add title
    ax.set_title(f"Movement Analysis - {method_label}")

    # Turn off axis
    ax.axis('off')

    return fig, ax

def create_correlation_plot(intensities, movement_scores, valid_points, method_label=""):
    """Create a scatter plot of movement vs intensity with regression line"""
    # Filter for valid points
    valid_intensities = intensities[valid_points]
    valid_movement = movement_scores[valid_points]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    scatter = ax.scatter(valid_movement, valid_intensities, alpha=0.7, c=valid_movement,
                        cmap=COLORMAP_MOVEMENT, vmin=-1, vmax=1)

    # Add regression line if enough points
    if len(valid_movement) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_movement, valid_intensities)

        # Create line data
        x_line = np.linspace(min(valid_movement), max(valid_movement), 100)
        y_line = slope * x_line + intercept

        # Plot line
        ax.plot(x_line, y_line, 'k--', linewidth=2,
               label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add stats to title
        correlation_stats = f"R² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_movement)}"
    else:
        correlation_stats = "Insufficient data for correlation"

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Movement Score (Blue=Extending, Red=Retracting)')

    # Set labels and title
    ax.set_xlabel('Movement Score')
    ax.set_ylabel('Intensity')
    ax.set_title(f"Movement-Intensity Correlation - {method_label}\n{correlation_stats}")

    if len(valid_movement) > 1:
        ax.legend()

    # Return correlation statistics and figure
    if len(valid_movement) > 1:
        stats_dict = {
            'r_squared': r_value**2,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'sample_size': len(valid_movement)
        }
    else:
        stats_dict = {
            'r_squared': None,
            'p_value': None,
            'slope': None,
            'intercept': None,
            'sample_size': len(valid_movement) if len(valid_movement) > 0 else 0
        }

    return fig, ax, stats_dict

def create_comparison_plot(mask_stats, curvature_stats, frames):
    """Create a bar plot comparing R² values between the two methods"""
    # Check if we have statistics to compare
    if not mask_stats or not curvature_stats:
        return None, None

    # Extract R² values
    mask_r2 = [stats.get('r_squared', 0) for stats in mask_stats]
    curvature_r2 = [stats.get('r_squared', 0) for stats in curvature_stats]

    # Replace None with 0
    mask_r2 = [0 if r2 is None else r2 for r2 in mask_r2]
    curvature_r2 = [0 if r2 is None else r2 for r2 in curvature_r2]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up bar positions
    bar_width = 0.35
    indices = np.arange(len(frames) - 1)  # One less than frames since we're comparing adjacent pairs

    # Create bars
    mask_bars = ax.bar(indices - bar_width/2, mask_r2, bar_width, alpha=0.7, color='royalblue',
                      label='Mask Difference Method')
    curve_bars = ax.bar(indices + bar_width/2, curvature_r2, bar_width, alpha=0.7, color='darkorange',
                      label='Curvature Gradient Method')

    # Add values above bars
    for i, v in enumerate(mask_r2):
        ax.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(curvature_r2):
        ax.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # Set labels and title
    ax.set_xlabel('Frame Pair')
    ax.set_ylabel('R² (Intensity-Movement Correlation)')
    ax.set_title('Comparison of Movement Calculation Methods')
    ax.set_xticks(indices)
    ax.set_xticklabels([f'{frames[i]}-{frames[i+1]}' for i in range(len(frames)-1)])
    ax.legend()

    # Set y-axis limit slightly above the maximum value
    max_val = max(max(mask_r2, default=0), max(curvature_r2, default=0))
    ax.set_ylim(0, max_val * 1.1 + 0.05)

    return fig, ax

#####################################################################
# MAIN ANALYSIS
#####################################################################

def analyze_frame_pair(current_frame_idx, previous_frame_idx, images, masks):
    """Analyze a pair of frames and return movement and correlation results for both methods"""
    print(f"Analyzing frames {previous_frame_idx+1} -> {current_frame_idx+1}")

    # Get the current frame data
    current_image = images[current_frame_idx]
    current_mask = masks[current_frame_idx]
    previous_mask = masks[previous_frame_idx]

    # Detect contour for current frame
    contour = detect_cell_edge(current_mask)
    if len(contour) == 0:
        print(f"  No contour found for frame {current_frame_idx+1}")
        return None

    # Sample points along the contour
    points = sample_equidistant_points(contour, N_POINTS)

    # Calculate curvature
    curvatures = measure_curvature(points)

    # Calculate inward normals
    normals = calculate_inward_normal(points, current_mask)

    # Measure intensity
    intensities, sampling_regions, valid_points, recovered = measure_intensity(
        current_image, current_mask, points, normals,
        DEPTH, WIDTH, MIN_CELL_COVERAGE, TRY_ROTATION, EXCLUDE_ENDPOINTS
    )

    # Method 1: Calculate movement from mask difference
    mask_movement, mask_types = calculate_movement_from_masks(
        current_mask, previous_mask, points, valid_points
    )

    # Method 2: Calculate movement from curvature gradient
    curve_movement, curve_types = calculate_movement_from_curvature(
        points, curvatures, valid_points
    )

    # Create overlays
    mask_overlay, _ = create_overlay(
        current_image, current_mask, points, valid_points, mask_movement,
        method_label="Mask Difference Method"
    )

    curve_overlay, _ = create_overlay(
        current_image, current_mask, points, valid_points, curve_movement,
        method_label="Curvature Gradient Method"
    )

    # Create correlation plots
    mask_corr, _, mask_stats = create_correlation_plot(
        intensities, mask_movement, valid_points, method_label="Mask Difference Method"
    )

    curve_corr, _, curve_stats = create_correlation_plot(
        intensities, curve_movement, valid_points, method_label="Curvature Gradient Method"
    )

    # Save figures if requested
    if SAVE_PLOTS:
        # Create output directory if it doesn't exist
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save overlay figures
        mask_overlay.savefig(output_dir / f"frame_{current_frame_idx+1}_mask_method_overlay.png", dpi=150, bbox_inches='tight')
        curve_overlay.savefig(output_dir / f"frame_{current_frame_idx+1}_curve_method_overlay.png", dpi=150, bbox_inches='tight')

        # Save correlation figures
        mask_corr.savefig(output_dir / f"frame_{current_frame_idx+1}_mask_method_correlation.png", dpi=150, bbox_inches='tight')
        curve_corr.savefig(output_dir / f"frame_{current_frame_idx+1}_curve_method_correlation.png", dpi=150, bbox_inches='tight')

    # Close figures to free memory
    plt.close(mask_overlay)
    plt.close(curve_overlay)
    plt.close(mask_corr)
    plt.close(curve_corr)

    # Return statistics for comparison
    return {
        'frame_pair': (previous_frame_idx, current_frame_idx),
        'mask_stats': mask_stats,
        'curve_stats': curve_stats,
        'points': len(points),
        'valid_points': np.sum(valid_points),
        'mask_movement_types': {
            'extending': np.sum(mask_types == 'extending'),
            'retracting': np.sum(mask_types == 'retracting'),
            'stable': np.sum(mask_types == 'stable')
        },
        'curve_movement_types': {
            'extending': np.sum(curve_types == 'extending'),
            'retracting': np.sum(curve_types == 'retracting'),
            'stable': np.sum(curve_types == 'stable')
        }
    }

def main():
    """Main function to run the analysis"""
    print("Movement Metric Comparison Script")
    print("=================================")

    # Load data
    image_path = os.path.join(DATA_DIR, IMAGE_FILE)
    mask_path = os.path.join(DATA_DIR, MASK_FILE)

    print(f"Loading images from: {image_path}")
    print(f"Loading masks from: {mask_path}")

    try:
        images = load_tiff_stack(image_path)
        masks = load_tiff_stack(mask_path)

        print(f"Loaded {images.shape[0]} frames")

        # Ensure binary masks
        masks = (masks > 0).astype(np.uint8)

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if frames to analyze are valid
    max_frame = images.shape[0] - 1
    valid_frames = [f for f in FRAMES_TO_ANALYZE if f <= max_frame]

    if len(valid_frames) < 2:
        print(f"Error: Need at least 2 valid frames for analysis. Max frame is {max_frame}.")
        return

    # Analyze frame pairs
    mask_stats_list = []
    curve_stats_list = []

    for i in range(len(valid_frames) - 1):
        current_idx = valid_frames[i+1]
        previous_idx = valid_frames[i]

        results = analyze_frame_pair(current_idx, previous_idx, images, masks)

        if results:
            mask_stats_list.append(results['mask_stats'])
            curve_stats_list.append(results['curve_stats'])

            # Print summary statistics
            print(f"\nResults for frames {previous_idx+1} -> {current_idx+1}:")
            print(f"  Valid points: {results['valid_points']}/{results['points']}")
            print(f"  Mask method R²: {results['mask_stats']['r_squared']:.4f}, p-value: {results['mask_stats']['p_value']:.4f}")
            print(f"  Curve method R²: {results['curve_stats']['r_squared']:.4f}, p-value: {results['curve_stats']['p_value']:.4f}")

            # Movement type distribution
            print(f"  Mask method movement types: {results['mask_movement_types']}")
            print(f"  Curve method movement types: {results['curve_movement_types']}")

    # Create comparison plot
    if mask_stats_list and curve_stats_list:
        comparison_fig, _ = create_comparison_plot(mask_stats_list, curve_stats_list, valid_frames)

        if SAVE_PLOTS and comparison_fig:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            comparison_fig.savefig(output_dir / "method_comparison.png", dpi=150, bbox_inches='tight')
            plt.close(comparison_fig)

        # Calculate and display average statistics
        avg_mask_r2 = np.mean([s['r_squared'] for s in mask_stats_list if s['r_squared'] is not None])
        avg_curve_r2 = np.mean([s['r_squared'] for s in curve_stats_list if s['r_squared'] is not None])

        print("\nSummary Statistics:")
        print(f"  Average Mask Method R²: {avg_mask_r2:.4f}")
        print(f"  Average Curvature Method R²: {avg_curve_r2:.4f}")

        if avg_mask_r2 > avg_curve_r2:
            print("  Mask difference method shows stronger correlation with intensity")
        elif avg_curve_r2 > avg_mask_r2:
            print("  Curvature gradient method shows stronger correlation with intensity")
        else:
            print("  Both methods show similar correlation with intensity")

    print("\nAnalysis complete.")
    if SAVE_PLOTS:
        print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
