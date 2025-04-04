import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tifffile
from skimage import measure, morphology, draw
import skimage
from pathlib import Path
import os
from scipy import stats
import scipy
import json
import datetime
import platform
import sys
import shutil
import random

import matplotlib.patches as mpatches



def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def load_tiff_stack(image_path, mask_path):
    """
    Load a TIFF stack of microscope images and binary masks.
    """
    images = tifffile.imread(image_path)
    masks = tifffile.imread(mask_path)

    # Ensure masks are binary (0 = background, 1 = cell)
    masks = (masks > 0).astype(np.uint8)

    # Handle single image case
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    print(f"Loaded {images.shape[0]} frames")
    return images, masks

def detect_cell_edge(mask):
    """
    Detect the cell edge from a binary mask.
    """
    # Get contours from the binary mask
    contours = measure.find_contours(mask, 0.5)

    # Usually the longest contour corresponds to the cell edge
    if len(contours) > 0:
        contour = max(contours, key=len)
        return contour
    else:
        return np.array([])



def detect_edge_movement(current_contour, previous_contour, current_mask, previous_mask):
    """
    Analyze the movement of cell edge between two consecutive frames.

    Parameters:
    - current_contour: Contour points from current frame
    - previous_contour: Contour points from previous frame
    - current_mask: Binary mask from current frame
    - previous_mask: Binary mask from previous frame

    Returns:
    - movement_score: Overall score indicating extension (positive) or retraction (negative)
    - movement_type: String classification ('extending', 'retracting', or 'stable')
    - movement_map: 2D array showing local movement (positive for extension, negative for retraction)
    """
    # Create a movement map (2D array with same shape as the masks)
    movement_map = np.zeros_like(current_mask, dtype=float)

    # Calculate the difference between current and previous masks
    # Positive values (1) indicate extension, negative values (-1) indicate retraction
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

def visualize_edge_movement(image, current_mask, previous_mask, movement_map, movement_type, frame_idx, output_dir):
    """
    Visualize the edge movement between consecutive frames.

    Parameters:
    - image: Current frame microscope image
    - current_mask: Binary mask for current frame
    - previous_mask: Binary mask for previous frame
    - movement_map: Map showing local movement (from detect_edge_movement)
    - movement_type: String classification ('extending', 'retracting', or 'stable')
    - frame_idx: Index of the current frame
    - output_dir: Directory to save the visualization
    """
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot mask comparison
    # Create an RGB image to visualize mask differences
    mask_vis = np.zeros((image.shape[0], image.shape[1], 3), dtype=float)
    mask_vis[..., 0] = previous_mask * 0.7  # Red channel - previous mask
    mask_vis[..., 1] = current_mask * 0.7   # Green channel - current mask
    # Overlapping regions will appear yellow

    axs[1].imshow(mask_vis)
    axs[1].set_title('Mask Comparison\nPrevious (Red), Current (Green), Overlap (Yellow)')
    axs[1].axis('off')

    # Plot movement map
    # Create a custom colormap for movement (red for retraction, blue for extension)
    movement_cmap = LinearSegmentedColormap.from_list('movement', ['red', 'white', 'blue'])

    # Only show movement at the cell edges
    edge_movement = np.zeros_like(movement_map)
    dilated_current = morphology.binary_dilation(current_mask)
    dilated_previous = morphology.binary_dilation(previous_mask)
    edge_regions = np.logical_or(
        np.logical_and(dilated_current, ~current_mask),  # Current edge
        np.logical_and(dilated_previous, ~previous_mask)  # Previous edge
    )
    edge_movement[edge_regions] = movement_map[edge_regions]

    # Plot the image with the movement map overlay
    axs[2].imshow(image, cmap='gray')
    movement_img = axs[2].imshow(edge_movement, cmap=movement_cmap, alpha=0.7, vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(movement_img, ax=axs[2])
    cbar.set_label('Edge Movement (Blue=Extending, Red=Retracting)')

    # Add movement type text with appropriate color
    if movement_type == 'extending':
        color = 'blue'
    elif movement_type == 'retracting':
        color = 'red'
    else:
        color = 'gray'

    axs[2].set_title(f'Edge Movement (Frame {frame_idx})\nOverall: {movement_type.upper()}',
                    color=color, fontweight='bold')
    axs[2].axis('off')

    # Add legend
    extension_patch = mpatches.Patch(color='blue', label='Extension')
    retraction_patch = mpatches.Patch(color='red', label='Retraction')
    axs[2].legend(handles=[extension_patch, retraction_patch], loc='lower right')

    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"edge_movement_frame_{frame_idx:04d}.png"), dpi=150)
    plt.close()

def create_movement_summary_visualization(all_movement_scores, all_movement_types, output_dir):
    """
    Create a summary visualization of edge movement over time.

    Parameters:
    - all_movement_scores: List of movement scores for each frame
    - all_movement_types: List of movement type strings for each frame
    - output_dir: Directory to save the visualization
    """
    if not all_movement_scores:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert movement types to numeric values for coloring
    movement_colors = []
    for movement_type in all_movement_types:
        if movement_type == 'extending':
            movement_colors.append('blue')
        elif movement_type == 'retracting':
            movement_colors.append('red')
        else:
            movement_colors.append('gray')

    # Plot movement scores
    frame_indices = range(len(all_movement_scores))
    ax.bar(frame_indices, all_movement_scores, color=movement_colors, alpha=0.7)

    # Add horizontal line at zero
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add thresholds for classification
    ax.axhline(y=0.1, color='blue', linestyle='--', alpha=0.3)
    ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.3)

    # Annotate bars with movement type
    for i, score in enumerate(all_movement_scores):
        offset = 0.05 if score >= 0 else -0.15
        ax.text(i, score + offset, all_movement_types[i],
               ha='center', rotation=90, fontsize=8, fontweight='bold',
               color=movement_colors[i])

    # Add legend
    extending_patch = mpatches.Patch(color='blue', label='Extending')
    retracting_patch = mpatches.Patch(color='red', label='Retracting')
    stable_patch = mpatches.Patch(color='gray', label='Stable')
    ax.legend(handles=[extending_patch, retracting_patch, stable_patch])

    # Labels and title
    ax.set_xlabel('Frame')
    ax.set_ylabel('Movement Score')
    ax.set_title('Cell Edge Movement Over Time')

    # Set x-ticks to frame indices
    ax.set_xticks(frame_indices)
    ax.set_xticklabels([str(i+1) for i in frame_indices])

    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "edge_movement_summary.png"), dpi=150)
    plt.close()

def save_movement_data_to_csv(all_movement_scores, all_movement_types, output_dir):
    """
    Save the edge movement data to a CSV file.

    Parameters:
    - all_movement_scores: List of movement scores for each frame
    - all_movement_types: List of movement type strings for each frame
    - output_dir: Directory to save the data
    """
    # Create data array
    data = np.column_stack((
        range(1, len(all_movement_scores) + 1),  # Frame indices (1-based)
        all_movement_scores,
        np.array(all_movement_types, dtype=str)
    ))

    # Save to CSV
    header = "frame,movement_score,movement_type"
    np.savetxt(
        os.path.join(output_dir, "edge_movement_data.csv"),
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%s"  # Use string format to handle the movement type text
    )

def sample_equidistant_points(contour, n_points=100):
    """
    Sample equidistant points along a contour.
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

def measure_curvature(points):
    """
    Measure curvature at each point along the contour.

    Returns:
    - sign_curvatures: Array of curvature signs (1 for positive/outside, -1 for negative/inside)
    - magnitude_curvatures: Array of curvature magnitudes (higher values = more curved)
    - normalized_curvatures: Array of curvature values normalized to [-1, 1] range
                            (preserves direction and relative magnitude)
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
        # (Normalize by the square of the distance between adjacent points to account for point spacing)
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
    """
    Calculate the inward normal vector at each point.
    The inward normal points into the cell (value 1 in the mask).

    Returns:
    - Array of normalized normal vectors
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
        # Take a small step in both directions and check mask value
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
            # Positive curvature (wound side) means inward is opposite to the normal
            # that would point to the right of the tangent
            normals[i] = normal2  # Default, might be refined based on curvature

    return normals

def measure_intensity(image, mask, points, normals, depth=20, width=5, min_cell_coverage=0.8):
    """
    Measure mean intensity within rectangular regions extending from each point.

    Parameters:
    - image: The intensity image
    - mask: Binary mask where 1 = cell, 0 = background
    - points: Equidistant points along the contour
    - normals: Normal vectors pointing into the cell at each point
    - depth: Length of the rectangle (how far into the cell)
    - width: Width of the rectangle
    - min_cell_coverage: Minimum fraction of the rectangle that must be inside the cell

    Returns:
    - Array of mean intensity values (NaN for points with insufficient cell coverage)
    - Array of sampling regions (for visualization)
    - Array of boolean values indicating which points passed the cell coverage check
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

def plot_curvature_correlation(curvature_type, curvatures, intensities, valid_points, frame_idx, output_dir, title_prefix=""):
    """
    Plot correlation between curvature (specified type) and intensity for a single frame.

    Parameters:
    - curvature_type: String describing the type of curvature ('sign', 'normalized')
    - curvatures: Array of curvature values
    - intensities: Array of intensity values (with NaNs for invalid points)
    - valid_points: Boolean array indicating which points are valid
    - frame_idx: Index of the current frame
    - output_dir: Directory to save the plot
    - title_prefix: Optional prefix for the plot title
    """
    # Filter out invalid points
    valid_curvatures = curvatures[valid_points]
    valid_intensities = intensities[valid_points]

    # Skip if no valid points
    if len(valid_curvatures) == 0:
        print(f"  No valid points for {curvature_type} correlation in frame {frame_idx}")
        return

    # Create figure
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(valid_curvatures, valid_intensities, alpha=0.7)

    # Add regression line if there are enough points
    if len(valid_curvatures) > 1:
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_curvatures, valid_intensities)
        x_line = np.linspace(min(valid_curvatures), max(valid_curvatures), 100)
        y_line = slope * x_line + intercept

        # Plot regression line
        plt.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add r-squared and p-value to the title
        title = f"{title_prefix}{curvature_type.capitalize()} Curvature vs. Intensity (Frame {frame_idx})"
        title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}"
    else:
        title = f"{title_prefix}{curvature_type.capitalize()} Curvature vs. Intensity (Frame {frame_idx})\nInsufficient data for regression"

    plt.title(title)
    plt.xlabel(f'{curvature_type.capitalize()} Curvature')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    if len(valid_curvatures) > 1:
        plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{curvature_type}_correlation_frame_{frame_idx:04d}.png"), dpi=150)
    plt.close()

def analyze_movement_grouped_correlations(all_curvatures, all_intensities, all_valid_points,
                                        all_movement_types, output_dir):
    """
    Perform correlation analyses on frames grouped by movement type.

    Parameters:
    - all_curvatures: List of curvature tuples for each frame
    - all_intensities: List of intensity arrays for each frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - all_movement_types: List of movement type strings for each frame
    - output_dir: Directory to save the results
    """
    # Create output directory for movement-grouped analyses
    movement_group_dir = os.path.join(output_dir, "movement_grouped_analysis")
    os.makedirs(movement_group_dir, exist_ok=True)

    # Group frames by movement type
    extending_frames = [i for i, movement_type in enumerate(all_movement_types) if movement_type == 'extending']
    retracting_frames = [i for i, movement_type in enumerate(all_movement_types) if movement_type == 'retracting']
    stable_frames = [i for i, movement_type in enumerate(all_movement_types) if movement_type == 'stable']

    # Dictionary to store grouped data
    movement_groups = {
        'extending': {
            'frames': extending_frames,
            'curvatures': [all_curvatures[i] for i in extending_frames],
            'intensities': [all_intensities[i] for i in extending_frames],
            'valid_points': [all_valid_points[i] for i in extending_frames],
            'stats': {}
        },
        'retracting': {
            'frames': retracting_frames,
            'curvatures': [all_curvatures[i] for i in retracting_frames],
            'intensities': [all_intensities[i] for i in retracting_frames],
            'valid_points': [all_valid_points[i] for i in retracting_frames],
            'stats': {}
        },
        'stable': {
            'frames': stable_frames,
            'curvatures': [all_curvatures[i] for i in stable_frames],
            'intensities': [all_intensities[i] for i in stable_frames],
            'valid_points': [all_valid_points[i] for i in stable_frames],
            'stats': {}
        }
    }

    # Initialize statistics dictionary for metadata
    movement_correlation_stats = {
        'extending': {
            'sign_curvature': {},
            'normalized_curvature': {},
            'frame_count': len(extending_frames)
        },
        'retracting': {
            'sign_curvature': {},
            'normalized_curvature': {},
            'frame_count': len(retracting_frames)
        },
        'stable': {
            'sign_curvature': {},
            'normalized_curvature': {},
            'frame_count': len(stable_frames)
        }
    }

    # Process each movement group
    for movement_type, group_data in movement_groups.items():
        print(f"Processing {movement_type} movement group ({len(group_data['frames'])} frames)")

        # Skip if no frames in this group
        if not group_data['frames']:
            print(f"  No frames in the {movement_type} group")
            continue

        # Extract sign and normalized curvatures
        sign_curvatures = [curvature_tuple[0] for curvature_tuple in group_data['curvatures']]
        normalized_curvatures = [curvature_tuple[2] for curvature_tuple in group_data['curvatures']]

        # Perform correlation analyses and generate plots for sign curvature
        sign_stats = plot_summary_curvature_correlation(
            'sign',
            sign_curvatures,
            group_data['intensities'],
            group_data['valid_points'],
            movement_group_dir,
            f"{movement_type}_sign"
        )
        movement_correlation_stats[movement_type]['sign_curvature'] = sign_stats

        # Perform correlation analyses and generate plots for normalized curvature
        normalized_stats = plot_summary_curvature_correlation(
            'normalized',
            normalized_curvatures,
            group_data['intensities'],
            group_data['valid_points'],
            movement_group_dir,
            f"{movement_type}_normalized"
        )
        movement_correlation_stats[movement_type]['normalized_curvature'] = normalized_stats

        # Save data to CSV
        save_movement_grouped_data_to_csv(
            group_data['frames'],
            group_data['curvatures'],
            group_data['intensities'],
            group_data['valid_points'],
            movement_type,
            movement_group_dir
        )

    # Create comparison plots for different movement types
    create_movement_group_comparison_plots(movement_correlation_stats, movement_group_dir)

    return movement_correlation_stats

def plot_summary_curvature_correlation(curvature_type, all_curvatures, all_intensities, all_valid_points,
                                     output_dir, prefix="summary"):
    """
    Plot combined correlation between curvature (specified type) and intensity across all frames.

    Parameters:
    - curvature_type: String describing the type of curvature ('sign', 'normalized')
    - all_curvatures: List of curvature arrays for each frame
    - all_intensities: List of intensity arrays for each frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - output_dir: Directory to save the plot
    - prefix: Prefix for the output file name

    Returns:
    - Dictionary of correlation statistics
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
        print(f"  No valid points for {prefix} {curvature_type} correlation")
        return {
            "r_squared": None,
            "p_value": None,
            "sample_size": 0
        }

    # Create figure
    plt.figure(figsize=(12, 10))

    # Scatter plot
    plt.scatter(combined_curvatures, combined_intensities, alpha=0.5)

    # Add regression line if there are enough points
    if len(combined_curvatures) > 1:
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(combined_curvatures, combined_intensities)
        x_line = np.linspace(min(combined_curvatures), max(combined_curvatures), 100)
        y_line = slope * x_line + intercept

        # Plot regression line
        plt.plot(x_line, y_line, 'r-', linewidth=2,
                 label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add r-squared and p-value to the title
        title = f"{prefix.replace('_', ' ').title()}: {curvature_type.capitalize()} Curvature vs. Intensity"
        title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_curvatures)}"
    else:
        title = f"{prefix.replace('_', ' ').title()}: {curvature_type.capitalize()} Curvature vs. Intensity\nInsufficient data for regression"

    plt.title(title)
    plt.xlabel(f'{curvature_type.capitalize()} Curvature')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    if len(combined_curvatures) > 1:
        plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{curvature_type}_correlation.png"), dpi=150)
    plt.close()

    # Return statistics for metadata
    if len(combined_curvatures) > 1:
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

def save_movement_grouped_data_to_csv(frame_indices, all_curvatures, all_intensities, all_valid_points,
                                    movement_type, output_dir):
    """
    Save the curvature and intensity data for a specific movement group to CSV files.

    Parameters:
    - frame_indices: List of original frame indices in this movement group
    - all_curvatures: List of curvature tuples for each frame in the group
    - all_intensities: List of intensity arrays for each frame in the group
    - all_valid_points: List of boolean arrays indicating which points are valid
    - movement_type: String indicating the movement type ('extending', 'retracting', 'stable')
    - output_dir: Directory to save the data
    """
    # Create combined CSV with all valid data points
    combined_data = []

    for group_idx, frame_idx in enumerate(frame_indices):
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_curvatures[group_idx]
        intensities = all_intensities[group_idx]
        valid_points = all_valid_points[group_idx]

        # Filter out invalid points
        valid_sign_curvatures = sign_curvatures[valid_points]
        valid_magnitude_curvatures = magnitude_curvatures[valid_points]
        valid_normalized_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        for i in range(len(valid_indices)):
            combined_data.append([
                str(frame_idx),
                str(valid_indices[i]),
                str(valid_sign_curvatures[i]),
                str(valid_magnitude_curvatures[i]),
                str(valid_normalized_curvatures[i]),
                str(valid_intensities[i]),
                movement_type
            ])

    # Save combined data
    if combined_data:
        # Save as a text file with manual formatting
        with open(os.path.join(output_dir, f"{movement_type}_combined_data.csv"), 'w') as f:
            # Write header
            f.write("frame,point_index,curvature_sign,curvature_magnitude,normalized_curvature,intensity,movement_type\n")

            # Write data rows
            for row in combined_data:
                f.write(','.join(row) + '\n')

def create_movement_group_comparison_plots(movement_correlation_stats, output_dir):
    """
    Create plots comparing correlation statistics across different movement groups.

    Parameters:
    - movement_correlation_stats: Dictionary containing correlation statistics for each movement group
    - output_dir: Directory to save the plots
    """
    # Check if we have valid statistics for all groups
    has_sign_data = all(stats['sign_curvature'].get('r_squared') is not None
                        for stats in movement_correlation_stats.values())
    has_normalized_data = all(stats['normalized_curvature'].get('r_squared') is not None
                             for stats in movement_correlation_stats.values())

    # Skip if any group is missing data
    if not has_sign_data and not has_normalized_data:
        print("  Not enough data to create movement group comparison plots")
        return

    # Create R² comparison plot
    plt.figure(figsize=(10, 8))

    # Set up bar positions
    movement_types = list(movement_correlation_stats.keys())
    x_pos = np.arange(len(movement_types))
    width = 0.35

    # Get R² values
    sign_r2_values = []
    normalized_r2_values = []

    for movement_type in movement_types:
        sign_stats = movement_correlation_stats[movement_type]['sign_curvature']
        normalized_stats = movement_correlation_stats[movement_type]['normalized_curvature']

        sign_r2_values.append(sign_stats.get('r_squared', 0))
        normalized_r2_values.append(normalized_stats.get('r_squared', 0))

    # Plot bars
    plt.bar(x_pos - width/2, sign_r2_values, width, label='Sign Curvature', color='blue', alpha=0.7)
    plt.bar(x_pos + width/2, normalized_r2_values, width, label='Normalized Curvature', color='red', alpha=0.7)

    # Add p-value annotations
    for i, movement_type in enumerate(movement_types):
        sign_stats = movement_correlation_stats[movement_type]['sign_curvature']
        normalized_stats = movement_correlation_stats[movement_type]['normalized_curvature']

        # Add p-value for sign curvature
        if sign_stats.get('p_value') is not None:
            p_value_sign = sign_stats['p_value']
            annotation = f"p={p_value_sign:.3f}"
            if p_value_sign < 0.001:
                annotation = "p<0.001"
            elif p_value_sign < 0.01:
                annotation = "p<0.01"
            elif p_value_sign < 0.05:
                annotation = "p<0.05"

            r2_value = sign_stats.get('r_squared', 0)
            plt.annotate(annotation, (i - width/2, r2_value + 0.02),
                        ha='center', va='bottom', fontsize=9, rotation=90)

        # Add p-value for normalized curvature
        if normalized_stats.get('p_value') is not None:
            p_value_norm = normalized_stats['p_value']
            annotation = f"p={p_value_norm:.3f}"
            if p_value_norm < 0.001:
                annotation = "p<0.001"
            elif p_value_norm < 0.01:
                annotation = "p<0.01"
            elif p_value_norm < 0.05:
                annotation = "p<0.05"

            r2_value = normalized_stats.get('r_squared', 0)
            plt.annotate(annotation, (i + width/2, r2_value + 0.02),
                        ha='center', va='bottom', fontsize=9, rotation=90)

    # Add sample size annotations
    for i, movement_type in enumerate(movement_types):
        sign_stats = movement_correlation_stats[movement_type]['sign_curvature']
        n = sign_stats.get('sample_size', 0)
        frame_count = movement_correlation_stats[movement_type]['frame_count']
        annotation = f"n={n}\n({frame_count} frames)"

        plt.annotate(annotation, (i, -0.05), ha='center', va='top', fontsize=8)

    # Customize plot
    plt.xlabel('Cell Movement Type')
    plt.ylabel('R² (Coefficient of Determination)')
    plt.title('Comparison of Curvature-Intensity Correlation by Movement Type')
    plt.xticks(x_pos, [t.capitalize() for t in movement_types])
    plt.ylim(0, max(max(sign_r2_values), max(normalized_r2_values)) * 1.2)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    # Add horizontal line at zero
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "movement_group_r2_comparison.png"), dpi=150)
    plt.close()

    # Create slope comparison plot
    plt.figure(figsize=(10, 8))

    # Get slope values
    sign_slope_values = []
    normalized_slope_values = []

    for movement_type in movement_types:
        sign_stats = movement_correlation_stats[movement_type]['sign_curvature']
        normalized_stats = movement_correlation_stats[movement_type]['normalized_curvature']

        sign_slope_values.append(sign_stats.get('slope', 0))
        normalized_slope_values.append(normalized_stats.get('slope', 0))

    # Plot bars
    plt.bar(x_pos - width/2, sign_slope_values, width, label='Sign Curvature', color='blue', alpha=0.7)
    plt.bar(x_pos + width/2, normalized_slope_values, width, label='Normalized Curvature', color='red', alpha=0.7)

    # Add standard error bars
    for i, movement_type in enumerate(movement_types):
        sign_stats = movement_correlation_stats[movement_type]['sign_curvature']
        normalized_stats = movement_correlation_stats[movement_type]['normalized_curvature']

        if sign_stats.get('standard_error') is not None:
            sign_std_err = sign_stats['standard_error']
            plt.errorbar(i - width/2, sign_stats.get('slope', 0), yerr=sign_std_err,
                        fmt='none', color='black', capsize=5)

        if normalized_stats.get('standard_error') is not None:
            norm_std_err = normalized_stats['standard_error']
            plt.errorbar(i + width/2, normalized_stats.get('slope', 0), yerr=norm_std_err,
                        fmt='none', color='black', capsize=5)

    # Customize plot
    plt.xlabel('Cell Movement Type')
    plt.ylabel('Regression Slope')
    plt.title('Comparison of Correlation Slope by Movement Type')
    plt.xticks(x_pos, [t.capitalize() for t in movement_types])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    # Add horizontal line at zero
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "movement_group_slope_comparison.png"), dpi=150)
    plt.close()

def calculate_summary_correlation_stats(all_curvatures, all_intensities, all_valid_points):
    """
    Calculate summary correlation statistics for all valid data points.

    Returns a dictionary with r_squared, p_value, and sample_size.
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
        slope, intercept, r_value, p_value, std_err = stats.linregress(combined_curvatures, combined_intensities)
        return {
            "r_squared": float(r_value**2),  # Convert to float for JSON serialization
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

def visualize_results(image, mask, points, curvatures, intensities, valid_points,
                     sampling_regions, frame_idx, output_dir):
    """
    Visualize the results, showing the original image, the mask, curvature and intensity measurements.

    Parameters:
    - image: Original microscope image
    - mask: Binary cell mask
    - points: Equidistant points along the contour
    - curvatures: Tuple of (sign_curvatures, magnitude_curvatures, normalized_curvatures)
    - intensities: Array of intensity values
    - valid_points: Boolean array indicating which points are valid
    - sampling_regions: List of rectangular sampling regions
    - frame_idx: Index of the current frame
    - output_dir: Directory to save the visualization
    """
    # Unpack curvatures
    sign_curvatures, magnitude_curvatures, normalized_curvatures = curvatures

    # Create custom colormap: red for positive curvature, blue for negative
    curvature_cmap = LinearSegmentedColormap.from_list('curvature', ['blue', 'white', 'red'])

    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Flatten axes for easy indexing
    axs = axs.flatten()

    # Plot original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot mask
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Cell Mask')
    axs[1].axis('off')

    # Plot image with curvature sign
    axs[2].imshow(image, cmap='gray', alpha=0.7)  # Reduced alpha to see points better

    # Add mask edge
    contour = measure.find_contours(mask, 0.5)
    if len(contour) > 0:
        contour = max(contour, key=len)
        axs[2].plot(contour[:, 1], contour[:, 0], 'y-', linewidth=0.5)

    # Add points colored by curvature sign
    scatter = axs[2].scatter(
        points[:, 1], points[:, 0],
        c=sign_curvatures, cmap=curvature_cmap, vmin=-1, vmax=1, s=50, edgecolor='k'  # Larger points
    )

    plt.colorbar(scatter, ax=axs[2], label='Curvature Sign')
    axs[2].set_title('Curvature Sign')
    axs[2].axis('off')

    # Plot image with curvature magnitude
    axs[3].imshow(image, cmap='gray', alpha=0.7)

    # Add mask edge
    if len(contour) > 0:
        axs[3].plot(contour[:, 1], contour[:, 0], 'y-', linewidth=0.5)

    # Add points colored by curvature magnitude
    scatter = axs[3].scatter(
        points[:, 1], points[:, 0],
        c=magnitude_curvatures, cmap='viridis', s=50, edgecolor='k'  # Larger points
    )

    plt.colorbar(scatter, ax=axs[3], label='Curvature Magnitude')
    axs[3].set_title('Curvature Magnitude')
    axs[3].axis('off')

    # Plot image with normalized curvature
    axs[4].imshow(image, cmap='gray', alpha=0.7)

    # Add mask edge
    if len(contour) > 0:
        axs[4].plot(contour[:, 1], contour[:, 0], 'y-', linewidth=0.5)

    # Add points colored by normalized curvature
    scatter = axs[4].scatter(
        points[:, 1], points[:, 0],
        c=normalized_curvatures, cmap=curvature_cmap, vmin=-1, vmax=1, s=50, edgecolor='k'  # Larger points
    )

    plt.colorbar(scatter, ax=axs[4], label='Normalized Curvature')
    axs[4].set_title('Normalized Curvature')
    axs[4].axis('off')

    # Plot image with intensity measurements
    axs[5].imshow(image, cmap='gray', alpha=0.7)

    # Add mask edge
    if len(contour) > 0:
        axs[5].plot(contour[:, 1], contour[:, 0], 'y-', linewidth=0.5)

    # Add sampling regions - color differently based on validity
    for i, region in enumerate(sampling_regions):
        # Convert to x, y coordinates for plotting
        polygon_y = region[:, 1]
        polygon_x = region[:, 0]
        if valid_points[i]:
            axs[5].fill(polygon_y, polygon_x, alpha=0.3, color='cyan')
        else:
            axs[5].fill(polygon_y, polygon_x, alpha=0.15, color='gray')  # Grayed out for invalid regions

    # Add points colored by intensity for valid points only
    valid_indices = np.where(valid_points)[0]

    if len(valid_indices) > 0:
        valid_intensities = intensities[valid_indices]

        # Calculate min and max for better color scaling
        # Using percentiles to avoid extreme outliers affecting the scale
        vmin = np.nanpercentile(valid_intensities, 5)
        vmax = np.nanpercentile(valid_intensities, 95)

        scatter = axs[5].scatter(
            points[valid_indices, 1], points[valid_indices, 0],
            c=valid_intensities, cmap='viridis', s=50, edgecolor='k',  # Larger points
            vmin=vmin, vmax=vmax
        )

        plt.colorbar(scatter, ax=axs[5], label='Intensity')

    axs[5].set_title('Intensity Measurement')
    axs[5].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"frame_{frame_idx:04d}.png"), dpi=150)
    plt.close()

def create_summary_visualization(all_curvatures, all_intensities, all_valid_points, output_dir):
    """
    Create a summary visualization for all frames showing how curvature and intensity change over time.

    Parameters:
    - all_curvatures: List of curvature tuples (sign, magnitude, normalized) for each frame
    - all_intensities: List of intensity arrays for each frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - output_dir: Directory to save the visualization
    """
    n_frames = len(all_curvatures)

    # Skip if no frames
    if n_frames == 0:
        return

    # Extract sign and normalized curvatures from tuples
    all_sign_curvatures = [curvature_tuple[0] for curvature_tuple in all_curvatures]
    all_normalized_curvatures = [curvature_tuple[2] for curvature_tuple in all_curvatures]

    # Create mask for invalid points
    masked_sign_curvatures = []
    masked_normalized_curvatures = []
    masked_intensities = []

    for i in range(n_frames):
        # Create masked arrays where invalid points are masked
        masked_sign = np.ma.masked_array(
            all_sign_curvatures[i],
            mask=~all_valid_points[i]
        )
        masked_normalized = np.ma.masked_array(
            all_normalized_curvatures[i],
            mask=~all_valid_points[i]
        )
        masked_intensity = np.ma.masked_array(
            all_intensities[i],
            mask=~all_valid_points[i]
        )

        masked_sign_curvatures.append(masked_sign)
        masked_normalized_curvatures.append(masked_normalized)
        masked_intensities.append(masked_intensity)

    # Convert to 2D arrays for visualization
    sign_array = np.ma.stack(masked_sign_curvatures)
    normalized_array = np.ma.stack(masked_normalized_curvatures)
    intensity_array = np.ma.stack(masked_intensities)

    # Calculate min and max intensity for better scaling
    # Using percentiles to avoid extreme outliers affecting the scale
    intensity_min = np.nanpercentile(intensity_array.compressed(), 5)
    intensity_max = np.nanpercentile(intensity_array.compressed(), 95)

    # Create figure for sign curvature and intensity
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Plot sign curvature heatmap
    sign_img = ax1.imshow(sign_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Frame')
    ax1.set_title('Curvature Sign Over Time')
    plt.colorbar(sign_img, ax=ax1, label='Curvature Sign')

    # Plot intensity heatmap
    intensity_img = ax2.imshow(intensity_array, cmap='viridis', aspect='auto',
                              vmin=intensity_min, vmax=intensity_max)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Frame')
    ax2.set_title('Intensity Over Time')
    plt.colorbar(intensity_img, ax=ax2, label='Intensity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_sign_visualization.png"), dpi=150)
    plt.close()

    # Create figure for normalized curvature and intensity
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Plot normalized curvature heatmap
    norm_img = ax1.imshow(normalized_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Frame')
    ax1.set_title('Normalized Curvature Over Time')
    plt.colorbar(norm_img, ax=ax1, label='Normalized Curvature')

    # Plot intensity heatmap (again)
    intensity_img = ax2.imshow(intensity_array, cmap='viridis', aspect='auto',
                              vmin=intensity_min, vmax=intensity_max)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Frame')
    ax2.set_title('Intensity Over Time')
    plt.colorbar(intensity_img, ax=ax2, label='Intensity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_normalized_visualization.png"), dpi=150)
    plt.close()

def save_data_to_csv(all_curvatures, all_intensities, all_valid_points, output_dir):
    """
    Save the curvature and intensity data to CSV files.
    Only include valid points in the data.

    Parameters:
    - all_curvatures: List of curvature tuples (sign, magnitude, normalized) for each frame
    - all_intensities: List of intensity arrays for each frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - output_dir: Directory to save the data
    """
    # Prepare data for each frame
    for frame_idx in range(len(all_curvatures)):
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_curvatures[frame_idx]
        intensities = all_intensities[frame_idx]
        valid_points = all_valid_points[frame_idx]

        # Filter out invalid points
        valid_sign_curvatures = sign_curvatures[valid_points]
        valid_magnitude_curvatures = magnitude_curvatures[valid_points]
        valid_normalized_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        # Skip if no valid points
        if len(valid_indices) == 0:
            continue

        # Create data array
        data = np.column_stack((
            valid_indices,
            valid_sign_curvatures,
            valid_magnitude_curvatures,
            valid_normalized_curvatures,
            valid_intensities
        ))

        # Save to CSV
        header = "point_index,curvature_sign,curvature_magnitude,normalized_curvature,intensity"
        np.savetxt(
            os.path.join(output_dir, f"data_frame_{frame_idx:04d}.csv"),
            data,
            delimiter=",",
            header=header,
            comments=""
        )

    # Create combined CSV with all valid data points
    combined_data = []

    for frame_idx in range(len(all_curvatures)):
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_curvatures[frame_idx]
        intensities = all_intensities[frame_idx]
        valid_points = all_valid_points[frame_idx]

        # Filter out invalid points
        valid_sign_curvatures = sign_curvatures[valid_points]
        valid_magnitude_curvatures = magnitude_curvatures[valid_points]
        valid_normalized_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        for i in range(len(valid_indices)):
            combined_data.append([
                frame_idx,
                valid_indices[i],
                valid_sign_curvatures[i],
                valid_magnitude_curvatures[i],
                valid_normalized_curvatures[i],
                valid_intensities[i]
            ])

    # Save combined data
    if combined_data:
        combined_data = np.array(combined_data)
        header = "frame,point_index,curvature_sign,curvature_magnitude,normalized_curvature,intensity"
        np.savetxt(
            os.path.join(output_dir, "combined_data.csv"),
            combined_data,
            delimiter=",",
            header=header,
            comments=""
        )

def generate_metadata(image_path, mask_path, output_dir, parameters, statistics, output_filename="metadata.json"):
    """
    Generate a metadata file with all the parameters and statistics of the analysis.

    Parameters:
    - image_path: Path to the microscope images
    - mask_path: Path to the binary masks
    - output_dir: Directory where results are saved
    - parameters: Dictionary of analysis parameters
    - statistics: Dictionary of analysis statistics
    - output_filename: Name of the metadata file
    """
    # Create metadata dictionary
    metadata = {
        "analysis_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "image_path": str(image_path),
            "mask_path": str(mask_path)
        },
        "output_directory": str(output_dir),
        "parameters": parameters,
        "statistics": statistics,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "matplotlib_version": plt.matplotlib.__version__,
            "scikit_image_version": skimage.__version__,
            "scipy_version": scipy.__version__
        },
        "analysis_description": {
            "summary": "Cell edge curvature and interior intensity analysis",
            "curvature_method": "Point position relative to line connecting adjacent points",
            "curvature_magnitude": "Normalized cross product as measure of curvature magnitude",
            "normalized_curvature": "Curvature sign [-1,1] multiplied by normalized magnitude [0,1]",
            "intensity_method": "Mean intensity within rectangular region extending inward from contour point",
            "cell_coverage_check": f"Minimum {parameters['min_cell_coverage']*100}% of sampling rectangle must be inside cell",
            "output_formats": ["PNG visualizations", "CSV data files", "Correlation plots", "JSON metadata"]
        }
    }

    # Convert NumPy types to Python native types for JSON serialization
    metadata = convert_numpy_types(metadata)

    # Write metadata to JSON file
    with open(os.path.join(output_dir, output_filename), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Also create a human-readable text version
    with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
        f.write("===== CELL CURVATURE AND INTENSITY ANALYSIS METADATA =====\n\n")
        f.write(f"Analysis performed: {metadata['analysis_timestamp']}\n\n")

        f.write("INPUT FILES:\n")
        f.write(f"- Image path: {metadata['input_files']['image_path']}\n")
        f.write(f"- Mask path: {metadata['input_files']['mask_path']}\n\n")

        f.write("OUTPUT:\n")
        f.write(f"- Directory: {metadata['output_directory']}\n\n")

        f.write("PARAMETERS:\n")
        for key, value in parameters.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        f.write("ANALYSIS STATISTICS:\n")
        for key, value in statistics.items():
            if isinstance(value, dict) and "r_squared" in value:
                f.write(f"- {key}:\n")
                for corr_key, corr_value in value.items():
                    f.write(f"  * {corr_key}: {corr_value}\n")
            else:
                f.write(f"- {key}: {value}\n")
        f.write("\n")

        f.write("ANALYSIS DESCRIPTION:\n")
        for key, value in metadata['analysis_description'].items():
            if key != "output_formats":
                f.write(f"- {value}\n")

        f.write("- Output formats:\n")
        for format in metadata['analysis_description']['output_formats']:
            f.write(f"  * {format}\n")
        f.write("\n")

        f.write("ENVIRONMENT:\n")
        f.write(f"- Python: {metadata['environment']['python_version'].split()[0]}\n")
        f.write(f"- Platform: {metadata['environment']['platform']}\n")
        f.write(f"- Key packages:\n")
        f.write(f"  * NumPy: {metadata['environment']['numpy_version']}\n")
        f.write(f"  * Matplotlib: {metadata['environment']['matplotlib_version']}\n")
        f.write(f"  * scikit-image: {metadata['environment']['scikit_image_version']}\n")
        f.write(f"  * SciPy: {metadata['environment']['scipy_version']}\n")

    print(f"Metadata saved to {os.path.join(output_dir, output_filename)} and {os.path.join(output_dir, 'metadata.txt')}")

def plot_temporal_curvature_correlation(curvature_type, current_curvatures, prev_intensities, valid_points,
                                       frame_idx, output_dir, title_prefix=""):
    """
    Plot correlation between current frame curvature (specified type) and previous frame intensity.

    Parameters:
    - curvature_type: String describing the type of curvature ('sign', 'normalized')
    - current_curvatures: Array of curvature values from current frame
    - prev_intensities: Array of intensity values from previous frame
    - valid_points: Boolean array indicating which points are valid
    - frame_idx: Index of the current frame
    - output_dir: Directory to save the plot
    - title_prefix: Optional prefix for the plot title
    """
    # Filter out invalid points
    valid_curvatures = current_curvatures[valid_points]
    valid_intensities = prev_intensities[valid_points]

    # Skip if no valid points
    if len(valid_curvatures) == 0:
        print(f"  No valid points for temporal {curvature_type} correlation in frame {frame_idx}")
        return

    # Create figure
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(valid_curvatures, valid_intensities, alpha=0.7)

    # Add regression line if there are enough points
    if len(valid_curvatures) > 1:
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_curvatures, valid_intensities)
        x_line = np.linspace(min(valid_curvatures), max(valid_curvatures), 100)
        y_line = slope * x_line + intercept

        # Plot regression line
        plt.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add r-squared and p-value to the title
        title = f"{title_prefix}Current {curvature_type.capitalize()} Curvature vs. Previous Intensity (Frame {frame_idx})"
        title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}"
    else:
        title = f"{title_prefix}Current {curvature_type.capitalize()} Curvature vs. Previous Intensity (Frame {frame_idx})\nInsufficient data for regression"

    plt.title(title)
    plt.xlabel(f'Current Frame {curvature_type.capitalize()} Curvature')
    plt.ylabel('Previous Frame Intensity')
    plt.grid(True, alpha=0.3)
    if len(valid_curvatures) > 1:
        plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"temporal_{curvature_type}_correlation_frame_{frame_idx:04d}.png"), dpi=150)
    plt.close()

def plot_summary_temporal_curvature_correlation(curvature_type, all_current_curvatures, all_prev_intensities,
                                              all_valid_points, output_dir):
    """
    Plot combined temporal correlation between current frame curvature (specified type) and previous frame intensity
    across all frames.

    Parameters:
    - curvature_type: String describing the type of curvature ('sign', 'normalized')
    - all_current_curvatures: List of curvature arrays for each current frame
    - all_prev_intensities: List of intensity arrays for each previous frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - output_dir: Directory to save the plot
    """
    # Combine data from all frames
    combined_curvatures = []
    combined_intensities = []

    for frame_idx in range(len(all_current_curvatures)):
        curvatures = all_current_curvatures[frame_idx]
        intensities = all_prev_intensities[frame_idx]
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
        print(f"  No valid points for summary temporal {curvature_type} correlation")
        return

    # Create figure
    plt.figure(figsize=(12, 10))

    # Scatter plot
    plt.scatter(combined_curvatures, combined_intensities, alpha=0.5)

    # Add regression line if there are enough points
    if len(combined_curvatures) > 1:
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(combined_curvatures, combined_intensities)
        x_line = np.linspace(min(combined_curvatures), max(combined_curvatures), 100)
        y_line = slope * x_line + intercept

        # Plot regression line
        plt.plot(x_line, y_line, 'r-', linewidth=2,
                 label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add r-squared and p-value to the title
        title = f"Summary: Current {curvature_type.capitalize()} Curvature vs. Previous Intensity (All Frames)"
        title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_curvatures)}"
    else:
        title = f"Summary: Current {curvature_type.capitalize()} Curvature vs. Previous Intensity (All Frames)\nInsufficient data for regression"

    plt.title(title)
    plt.xlabel(f'Current Frame {curvature_type.capitalize()} Curvature')
    plt.ylabel('Previous Frame Intensity')
    plt.grid(True, alpha=0.3)
    if len(combined_curvatures) > 1:
        plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_temporal_{curvature_type}_correlation.png"), dpi=150)
    plt.close()

    # Return statistics for metadata
    if len(combined_curvatures) > 1:
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

def create_summary_temporal_visualization(all_current_curvatures, all_prev_intensities, all_valid_points, output_dir):
    """
    Create a summary visualization for temporal correlation analysis.

    Parameters:
    - all_current_curvatures: List of curvature tuples for each current frame
    - all_prev_intensities: List of intensity arrays for each previous frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - output_dir: Directory to save the visualization
    """
    n_frames = len(all_current_curvatures)

    # Skip if no frames
    if n_frames == 0:
        return

    # Extract sign and normalized curvatures from tuples
    all_sign_curvatures = [curvature_tuple[0] for curvature_tuple in all_current_curvatures]
    all_normalized_curvatures = [curvature_tuple[2] for curvature_tuple in all_current_curvatures]

    # Create mask for invalid points
    masked_sign_curvatures = []
    masked_normalized_curvatures = []
    masked_intensities = []

    for i in range(n_frames):
        # Create masked arrays where invalid points are masked
        masked_sign = np.ma.masked_array(
            all_sign_curvatures[i],
            mask=~all_valid_points[i]
        )
        masked_normalized = np.ma.masked_array(
            all_normalized_curvatures[i],
            mask=~all_valid_points[i]
        )
        masked_intensity = np.ma.masked_array(
            all_prev_intensities[i],
            mask=~all_valid_points[i]
        )

        masked_sign_curvatures.append(masked_sign)
        masked_normalized_curvatures.append(masked_normalized)
        masked_intensities.append(masked_intensity)

    # Convert to 2D arrays for visualization
    sign_array = np.ma.stack(masked_sign_curvatures)
    normalized_array = np.ma.stack(masked_normalized_curvatures)
    intensity_array = np.ma.stack(masked_intensities)

    # Calculate min and max intensity for better scaling
    # Using percentiles to avoid extreme outliers affecting the scale
    intensity_min = np.nanpercentile(intensity_array.compressed(), 5)
    intensity_max = np.nanpercentile(intensity_array.compressed(), 95)

    # Create figure for sign curvature and previous intensity
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Plot sign curvature heatmap
    sign_img = ax1.imshow(sign_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Frame')
    ax1.set_title('Current Frame Curvature Sign Over Time')
    plt.colorbar(sign_img, ax=ax1, label='Curvature Sign')

    # Plot previous intensity heatmap
    intensity_img = ax2.imshow(intensity_array, cmap='viridis', aspect='auto',
                             vmin=intensity_min, vmax=intensity_max)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Frame')
    ax2.set_title('Previous Frame Intensity Over Time')
    plt.colorbar(intensity_img, ax=ax2, label='Previous Frame Intensity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_temporal_sign_visualization.png"), dpi=150)
    plt.close()

    # Create figure for normalized curvature and previous intensity
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Plot normalized curvature heatmap
    norm_img = ax1.imshow(normalized_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Frame')
    ax1.set_title('Current Frame Normalized Curvature Over Time')
    plt.colorbar(norm_img, ax=ax1, label='Normalized Curvature')

    # Plot previous intensity heatmap (again)
    intensity_img = ax2.imshow(intensity_array, cmap='viridis', aspect='auto',
                             vmin=intensity_min, vmax=intensity_max)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Frame')
    ax2.set_title('Previous Frame Intensity Over Time')
    plt.colorbar(intensity_img, ax=ax2, label='Previous Frame Intensity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_temporal_normalized_visualization.png"), dpi=150)
    plt.close()

def save_temporal_data_to_csv(all_current_curvatures, all_prev_intensities, all_valid_points, output_dir):
    """
    Save the temporal correlation data to CSV files.
    Only include valid points in the data.
    """
    # Prepare data for each frame
    for frame_idx in range(len(all_current_curvatures)):
        # Extract all curvature components
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_current_curvatures[frame_idx]
        intensities = all_prev_intensities[frame_idx]
        valid_points = all_valid_points[frame_idx]

        # Filter out invalid points
        valid_sign_curvatures = sign_curvatures[valid_points]
        valid_magnitude_curvatures = magnitude_curvatures[valid_points]
        valid_normalized_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        # Skip if no valid points
        if len(valid_indices) == 0:
            continue

        # Create data array
        data = np.column_stack((
            valid_indices,
            valid_sign_curvatures,
            valid_magnitude_curvatures,
            valid_normalized_curvatures,
            valid_intensities
        ))

        # Save to CSV
        header = "point_index,current_curvature_sign,current_curvature_magnitude,current_normalized_curvature,previous_intensity"
        np.savetxt(
            os.path.join(output_dir, f"temporal_data_frame_{frame_idx:04d}.csv"),
            data,
            delimiter=",",
            header=header,
            comments=""
        )

    # Create combined CSV with all valid data points
    combined_data = []

    for frame_idx in range(len(all_current_curvatures)):
        # Extract all curvature components
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_current_curvatures[frame_idx]
        intensities = all_prev_intensities[frame_idx]
        valid_points = all_valid_points[frame_idx]

        # Filter out invalid points
        valid_sign_curvatures = sign_curvatures[valid_points]
        valid_magnitude_curvatures = magnitude_curvatures[valid_points]
        valid_normalized_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        for i in range(len(valid_indices)):
            combined_data.append([
                frame_idx,
                valid_indices[i],
                valid_sign_curvatures[i],
                valid_magnitude_curvatures[i],
                valid_normalized_curvatures[i],
                valid_intensities[i]
            ])

    # Save combined data
    if combined_data:
        combined_data = np.array(combined_data)
        header = "frame,point_index,current_curvature_sign,current_curvature_magnitude,current_normalized_curvature,previous_intensity"
        np.savetxt(
            os.path.join(output_dir, "combined_temporal_data.csv"),
            combined_data,
            delimiter=",",
            header=header,
            comments=""
        )

def process_temporal_analysis(all_curvatures, all_intensities, all_valid_points,
                            image_path, mask_path, output_dir, parameters):
    """
    Perform temporal correlation analysis between current frame curvature and previous frame intensity.

    Parameters:
    - all_curvatures: List of curvature tuples for each frame
    - all_intensities: List of intensity arrays for each frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - image_path: Path to the original image file (for metadata)
    - mask_path: Path to the original mask file (for metadata)
    - output_dir: Directory to save results
    - parameters: Dictionary of analysis parameters
    """
    # Create output directory
    temporal_dir = os.path.join(output_dir, "temporal_analysis")
    os.makedirs(temporal_dir, exist_ok=True)

    # Skip the first frame since it has no previous frame
    if len(all_curvatures) <= 1:
        print("Not enough frames for temporal analysis")
        return

    # Initialize lists for temporal correlation
    all_current_curvatures = []
    all_prev_intensities = []
    all_temporal_valid_points = []

    # Initialize statistics dictionary
    statistics = {
        "total_frames": len(all_curvatures) - 1,  # Skip first frame
        "processed_frames": 0,
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
        }
    }

    # Process each frame (starting from the second frame)
    for frame_idx in range(1, len(all_curvatures)):
        print(f"Processing temporal correlation for frame {frame_idx}")

        # Get current frame curvature and previous frame intensity
        current_curvatures = all_curvatures[frame_idx]
        prev_intensities = all_intensities[frame_idx - 1]

        # Use the valid points mask from both frames (logical AND)
        current_valid = all_valid_points[frame_idx]
        prev_valid = all_valid_points[frame_idx - 1]
        valid_points = np.logical_and(current_valid, prev_valid)

        # Update statistics
        statistics["processed_frames"] += 1
        statistics["total_points"] += len(valid_points)
        statistics["valid_points"] += np.sum(valid_points)
        statistics["invalid_points"] += len(valid_points) - np.sum(valid_points)

        # Store data for this frame
        all_current_curvatures.append(current_curvatures)
        all_prev_intensities.append(prev_intensities)
        all_temporal_valid_points.append(valid_points)

        # Extract sign and normalized curvatures
        sign_curvatures = current_curvatures[0]
        normalized_curvatures = current_curvatures[2]

        # Plot correlations for this frame (both sign and normalized)
        plot_temporal_curvature_correlation('sign', sign_curvatures, prev_intensities,
                                          valid_points, frame_idx, temporal_dir, "Temporal: ")
        plot_temporal_curvature_correlation('normalized', normalized_curvatures, prev_intensities,
                                          valid_points, frame_idx, temporal_dir, "Temporal: ")

    # Calculate average valid points per frame
    if statistics["processed_frames"] > 0:
        statistics["average_valid_points_per_frame"] = statistics["valid_points"] / statistics["processed_frames"]

    # Create summary visualizations
    if all_current_curvatures and all_prev_intensities and all_temporal_valid_points:
        # Create summary visualizations
        create_summary_temporal_visualization(
            all_current_curvatures, all_prev_intensities, all_temporal_valid_points, temporal_dir
        )

        # Extract sign and normalized curvatures for all frames
        all_sign_curvatures = [curvature_tuple[0] for curvature_tuple in all_current_curvatures]
        all_normalized_curvatures = [curvature_tuple[2] for curvature_tuple in all_current_curvatures]

        # Plot summary correlations and get statistics
        sign_stats = plot_summary_temporal_curvature_correlation(
            'sign', all_sign_curvatures, all_prev_intensities, all_temporal_valid_points, temporal_dir
        )
        normalized_stats = plot_summary_temporal_curvature_correlation(
            'normalized', all_normalized_curvatures, all_prev_intensities, all_temporal_valid_points, temporal_dir
        )

        # Update statistics
        statistics["sign_curvature_correlation"] = sign_stats
        statistics["normalized_curvature_correlation"] = normalized_stats

    # Save data to CSV
    save_temporal_data_to_csv(
        all_current_curvatures, all_prev_intensities, all_temporal_valid_points, temporal_dir
    )

    # Generate metadata file
    generate_metadata(
        image_path, mask_path, temporal_dir, parameters, statistics, "temporal_metadata.json"
    )

    print(f"Temporal analysis results saved to {temporal_dir}")


def plot_random_curvature_correlation(curvature_type, current_curvatures, random_intensities, valid_points,
                                    frame_idx, random_frame_idx, output_dir, title_prefix=""):
    """
    Plot correlation between current frame curvature (specified type) and random frame intensity.

    Parameters:
    - curvature_type: String describing the type of curvature ('sign', 'normalized')
    - current_curvatures: Array of curvature values from current frame
    - random_intensities: Array of intensity values from random frame
    - valid_points: Boolean array indicating which points are valid
    - frame_idx: Index of the current frame
    - random_frame_idx: Index of the random frame
    - output_dir: Directory to save the plot
    - title_prefix: Optional prefix for the plot title
    """
    # Filter out invalid points
    valid_curvatures = current_curvatures[valid_points]
    valid_intensities = random_intensities[valid_points]

    # Skip if no valid points
    if len(valid_curvatures) == 0:
        print(f"  No valid points for random {curvature_type} correlation in frame {frame_idx} (random frame {random_frame_idx})")
        return

    # Create figure
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(valid_curvatures, valid_intensities, alpha=0.7)

    # Check if all curvature values are identical
    if len(np.unique(valid_curvatures)) > 1:
        # Add regression line if there are enough points and values are not all identical
        if len(valid_curvatures) > 1:
            # Calculate regression
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_curvatures, valid_intensities)
                x_line = np.linspace(min(valid_curvatures), max(valid_curvatures), 100)
                y_line = slope * x_line + intercept

                # Plot regression line
                plt.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

                # Add r-squared and p-value to the title
                title = f"{title_prefix}Frame {frame_idx} {curvature_type.capitalize()} Curvature vs. Random Frame {random_frame_idx} Intensity"
                title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}"
            except Exception as e:
                # Handle any other regression calculation errors
                title = f"{title_prefix}Frame {frame_idx} {curvature_type.capitalize()} Curvature vs. Random Frame {random_frame_idx} Intensity"
                title += f"\nRegression calculation failed: {str(e)}"
        else:
            title = f"{title_prefix}Frame {frame_idx} {curvature_type.capitalize()} Curvature vs. Random Frame {random_frame_idx} Intensity\nInsufficient data for regression"
    else:
        # If all curvature values are identical, note this in the title
        title = f"{title_prefix}Frame {frame_idx} {curvature_type.capitalize()} Curvature vs. Random Frame {random_frame_idx} Intensity"
        title += f"\nAll curvature values are identical ({valid_curvatures[0]:.2f}) - regression not possible"

    plt.title(title)
    plt.xlabel(f'Frame {frame_idx} {curvature_type.capitalize()} Curvature')
    plt.ylabel(f'Random Frame {random_frame_idx} Intensity')
    plt.grid(True, alpha=0.3)
    if 'r_value' in locals():
        plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"random_{curvature_type}_correlation_frame_{frame_idx}_random_{random_frame_idx}.png"), dpi=150)
    plt.close()

def plot_summary_random_curvature_correlation(curvature_type, all_curvatures, all_random_intensities,
                                            all_valid_points, output_dir):
    """
    Plot combined random correlation between curvature (specified type) and random frame intensity across all frames.

    Parameters:
    - curvature_type: String describing the type of curvature ('sign', 'normalized')
    - all_curvatures: List of curvature arrays for each frame
    - all_random_intensities: List of intensity arrays from random frames
    - all_valid_points: List of boolean arrays indicating which points are valid
    - output_dir: Directory to save the plot
    """
    # Combine data from all frames
    combined_curvatures = []
    combined_intensities = []

    for frame_idx in range(len(all_curvatures)):
        curvatures = all_curvatures[frame_idx]
        intensities = all_random_intensities[frame_idx]
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
        print(f"  No valid points for summary random {curvature_type} correlation")
        return

    # Create figure
    plt.figure(figsize=(12, 10))

    # Scatter plot
    plt.scatter(combined_curvatures, combined_intensities, alpha=0.5)

    # Add regression line if there are enough points
    if len(combined_curvatures) > 1:
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(combined_curvatures, combined_intensities)
        x_line = np.linspace(min(combined_curvatures), max(combined_curvatures), 100)
        y_line = slope * x_line + intercept

        # Plot regression line
        plt.plot(x_line, y_line, 'r-', linewidth=2,
                 label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add r-squared and p-value to the title
        title = f"Summary: {curvature_type.capitalize()} Curvature vs. Random Frame Intensity (Control)"
        title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_curvatures)}"
    else:
        title = f"Summary: {curvature_type.capitalize()} Curvature vs. Random Frame Intensity (Control)\nInsufficient data for regression"

    plt.title(title)
    plt.xlabel(f'{curvature_type.capitalize()} Curvature')
    plt.ylabel('Random Frame Intensity')
    plt.grid(True, alpha=0.3)
    if len(combined_curvatures) > 1:
        plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_random_{curvature_type}_correlation.png"), dpi=150)
    plt.close()

    # Return statistics for metadata
    if len(combined_curvatures) > 1:
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

def save_random_data_to_csv(all_curvatures, all_random_intensities, all_valid_points, all_random_frame_indices, output_dir):
    """
    Save the random correlation data to CSV files.
    Only include valid points in the data.
    """
    # Prepare data for each frame
    for frame_idx in range(len(all_curvatures)):
        # Extract all curvature components
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_curvatures[frame_idx]
        intensities = all_random_intensities[frame_idx]
        valid_points = all_valid_points[frame_idx]
        random_frame_idx = all_random_frame_indices[frame_idx]

        # Filter out invalid points
        valid_sign_curvatures = sign_curvatures[valid_points]
        valid_magnitude_curvatures = magnitude_curvatures[valid_points]
        valid_normalized_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        # Skip if no valid points
        if len(valid_indices) == 0:
            continue

        # Create data array
        data = np.column_stack((
            valid_indices,
            valid_sign_curvatures,
            valid_magnitude_curvatures,
            valid_normalized_curvatures,
            valid_intensities,
            np.full(len(valid_indices), random_frame_idx)
        ))

        # Save to CSV
        header = "point_index,curvature_sign,curvature_magnitude,normalized_curvature,random_intensity,random_frame_index"
        np.savetxt(
            os.path.join(output_dir, f"random_data_frame_{frame_idx:04d}.csv"),
            data,
            delimiter=",",
            header=header,
            comments=""
        )

    # Create combined CSV with all valid data points
    combined_data = []

    for frame_idx in range(len(all_curvatures)):
        # Extract all curvature components
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_curvatures[frame_idx]
        intensities = all_random_intensities[frame_idx]
        valid_points = all_valid_points[frame_idx]
        random_frame_idx = all_random_frame_indices[frame_idx]

        # Filter out invalid points
        valid_sign_curvatures = sign_curvatures[valid_points]
        valid_magnitude_curvatures = magnitude_curvatures[valid_points]
        valid_normalized_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        for i in range(len(valid_indices)):
            combined_data.append([
                frame_idx,
                valid_indices[i],
                valid_sign_curvatures[i],
                valid_magnitude_curvatures[i],
                valid_normalized_curvatures[i],
                valid_intensities[i],
                random_frame_idx
            ])

    # Save combined data
    if combined_data:
        combined_data = np.array(combined_data)
        header = "frame,point_index,curvature_sign,curvature_magnitude,normalized_curvature,random_intensity,random_frame_index"
        np.savetxt(
            os.path.join(output_dir, "combined_random_data.csv"),
            combined_data,
            delimiter=",",
            header=header,
            comments=""
        )

def create_summary_random_visualization(all_curvatures, all_random_intensities, all_valid_points, all_random_frame_indices, output_dir):
    """
    Create a summary visualization for random correlation analysis.

    Parameters:
    - all_curvatures: List of curvature tuples for each frame
    - all_random_intensities: List of intensity arrays from random frames
    - all_valid_points: List of boolean arrays indicating which points are valid
    - all_random_frame_indices: List of random frame indices used for each frame
    - output_dir: Directory to save the visualization
    """
    n_frames = len(all_curvatures)

    # Skip if no frames
    if n_frames == 0:
        return

    # Extract sign and normalized curvatures from tuples
    all_sign_curvatures = [curvature_tuple[0] for curvature_tuple in all_curvatures]
    all_normalized_curvatures = [curvature_tuple[2] for curvature_tuple in all_curvatures]

    # Create mask for invalid points
    masked_sign_curvatures = []
    masked_normalized_curvatures = []
    masked_intensities = []

    for i in range(n_frames):
        # Create masked arrays where invalid points are masked
        masked_sign = np.ma.masked_array(
            all_sign_curvatures[i],
            mask=~all_valid_points[i]
        )
        masked_normalized = np.ma.masked_array(
            all_normalized_curvatures[i],
            mask=~all_valid_points[i]
        )
        masked_intensity = np.ma.masked_array(
            all_random_intensities[i],
            mask=~all_valid_points[i]
        )

        masked_sign_curvatures.append(masked_sign)
        masked_normalized_curvatures.append(masked_normalized)
        masked_intensities.append(masked_intensity)

    # Convert to 2D arrays for visualization
    sign_array = np.ma.stack(masked_sign_curvatures)
    normalized_array = np.ma.stack(masked_normalized_curvatures)
    intensity_array = np.ma.stack(masked_intensities)

    # Calculate min and max intensity for better scaling
    # Using percentiles to avoid extreme outliers affecting the scale
    intensity_min = np.nanpercentile(intensity_array.compressed(), 5)
    intensity_max = np.nanpercentile(intensity_array.compressed(), 95)

    # Create figure for sign curvature and random intensity
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Plot sign curvature heatmap
    sign_img = ax1.imshow(sign_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Frame')
    ax1.set_title('Curvature Sign')
    plt.colorbar(sign_img, ax=ax1, label='Curvature Sign')

    # Plot random intensity heatmap
    intensity_img = ax2.imshow(intensity_array, cmap='viridis', aspect='auto',
                             vmin=intensity_min, vmax=intensity_max)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Frame')
    ax2.set_title('Random Frame Intensity (Control)')
    plt.colorbar(intensity_img, ax=ax2, label='Random Frame Intensity')

    # Add frame indices as y-tick labels
    ax1.set_yticks(range(n_frames))
    ax1.set_yticklabels([f"{i}" for i in range(n_frames)])
    ax2.set_yticks(range(n_frames))
    ax2.set_yticklabels([f"{i} (random: {all_random_frame_indices[i]})" for i in range(n_frames)])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_random_sign_visualization.png"), dpi=150)
    plt.close()

    # Create figure for normalized curvature and random intensity
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Plot normalized curvature heatmap
    norm_img = ax1.imshow(normalized_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Frame')
    ax1.set_title('Normalized Curvature')
    plt.colorbar(norm_img, ax=ax1, label='Normalized Curvature')

    # Plot random intensity heatmap (again)
    intensity_img = ax2.imshow(intensity_array, cmap='viridis', aspect='auto',
                             vmin=intensity_min, vmax=intensity_max)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Frame')
    ax2.set_title('Random Frame Intensity (Control)')
    plt.colorbar(intensity_img, ax=ax2, label='Random Frame Intensity')

    # Add frame indices as y-tick labels
    ax1.set_yticks(range(n_frames))
    ax1.set_yticklabels([f"{i}" for i in range(n_frames)])
    ax2.set_yticks(range(n_frames))
    ax2.set_yticklabels([f"{i} (random: {all_random_frame_indices[i]})" for i in range(n_frames)])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_random_normalized_visualization.png"), dpi=150)
    plt.close()

def process_random_analysis(all_curvatures, all_intensities, all_valid_points,
                          image_path, mask_path, output_dir, parameters):
    """
    Perform randomized control analysis between curvature and intensity from random frames.

    Parameters:
    - all_curvatures: List of curvature tuples for each frame
    - all_intensities: List of intensity arrays for each frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - image_path: Path to the original image file (for metadata)
    - mask_path: Path to the original mask file (for metadata)
    - output_dir: Directory to save results
    - parameters: Dictionary of analysis parameters
    """
    # Create output directory
    random_dir = os.path.join(output_dir, "random_analysis")
    os.makedirs(random_dir, exist_ok=True)

    # Skip if not enough frames
    if len(all_curvatures) <= 1:
        print("Not enough frames for random analysis")
        return

    # Initialize lists for random correlation
    all_random_intensities = []
    all_random_valid_points = []
    all_random_frame_indices = []

    # Initialize statistics dictionary
    statistics = {
        "total_frames": len(all_curvatures),
        "processed_frames": 0,
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
        }
    }

    # Process each frame
    for frame_idx in range(len(all_curvatures)):
        # Get current frame curvature
        current_curvatures = all_curvatures[frame_idx]
        current_valid = all_valid_points[frame_idx]

        # Choose a random frame (different from the current one)
        available_frames = list(range(len(all_curvatures)))
        if len(available_frames) > 1:  # If more than one frame available
            available_frames.remove(frame_idx)  # Remove the current frame
            random_frame_idx = random.choice(available_frames)  # Choose randomly from remaining frames
        else:
            # If only one frame available, use it (this should not happen with the earlier check)
            random_frame_idx = 0

        # Get intensity from the random frame
        random_intensities = all_intensities[random_frame_idx]
        random_valid = all_valid_points[random_frame_idx]

        # Use the valid points mask from both frames (logical AND)
        valid_points = np.logical_and(current_valid, random_valid)

        # Update statistics
        statistics["processed_frames"] += 1
        statistics["total_points"] += len(valid_points)
        statistics["valid_points"] += np.sum(valid_points)
        statistics["invalid_points"] += len(valid_points) - np.sum(valid_points)

        # Store data for this frame
        all_random_intensities.append(random_intensities)
        all_random_valid_points.append(valid_points)
        all_random_frame_indices.append(random_frame_idx)

        # Extract sign and normalized curvatures
        sign_curvatures = current_curvatures[0]
        normalized_curvatures = current_curvatures[2]

        # Plot correlations for this frame (both sign and normalized)
        plot_random_curvature_correlation('sign', sign_curvatures, random_intensities,
                                        valid_points, frame_idx, random_frame_idx, random_dir, "Random: ")
        plot_random_curvature_correlation('normalized', normalized_curvatures, random_intensities,
                                        valid_points, frame_idx, random_frame_idx, random_dir, "Random: ")

    # Calculate average valid points per frame
    if statistics["processed_frames"] > 0:
        statistics["average_valid_points_per_frame"] = statistics["valid_points"] / statistics["processed_frames"]

    # Create summary visualizations
    if all_curvatures and all_random_intensities and all_random_valid_points:
        # Create summary visualizations
        create_summary_random_visualization(
            all_curvatures, all_random_intensities, all_random_valid_points, all_random_frame_indices, random_dir
        )

        # Extract sign and normalized curvatures for all frames
        all_sign_curvatures = [curvature_tuple[0] for curvature_tuple in all_curvatures]
        all_normalized_curvatures = [curvature_tuple[2] for curvature_tuple in all_curvatures]

        # Plot summary correlations and get statistics
        sign_stats = plot_summary_random_curvature_correlation(
            'sign', all_sign_curvatures, all_random_intensities, all_random_valid_points, random_dir
        )
        normalized_stats = plot_summary_random_curvature_correlation(
            'normalized', all_normalized_curvatures, all_random_intensities, all_random_valid_points, random_dir
        )

        # Update statistics
        statistics["sign_curvature_correlation"] = sign_stats
        statistics["normalized_curvature_correlation"] = normalized_stats

    # Add random frame indices to statistics
    statistics["random_frame_indices"] = all_random_frame_indices

    # Save data to CSV
    save_random_data_to_csv(
        all_curvatures, all_random_intensities, all_random_valid_points, all_random_frame_indices, random_dir
    )

    # Generate metadata file
    generate_metadata(
        image_path, mask_path, random_dir, parameters, statistics, "random_metadata.json"
    )

    print(f"Random control analysis results saved to {random_dir}")

def process_stack(image_path, mask_path, output_dir, n_points=100, depth=20, width=5, min_cell_coverage=0.8):
    """
    Process a TIFF stack of microscope images and masks.
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
        images, masks = load_tiff_stack(image_path, mask_path)
        statistics["total_frames"] = images.shape[0]
    except Exception as e:
        statistics["error_loading_files"] = str(e)
        # Generate metadata with error information
        generate_metadata(image_path, mask_path, output_dir, parameters, statistics)
        print(f"Error loading files: {e}")
        return

    # Store data for all frames
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
        print(f"Processing frame {frame_idx+1}/{images.shape[0]}")

        image = images[frame_idx]
        mask = masks[frame_idx]

        # Detect cell edge
        contour = detect_cell_edge(mask)

        if len(contour) == 0:
            print(f"  No contour found in frame {frame_idx}")
            statistics["frames_with_no_contour"] += 1
            continue

        # Store mask and contour for region-specific analysis
        all_masks.append(mask)
        all_contours.append(contour)

        # Sample equidistant points
        points = sample_equidistant_points(contour, n_points)

        # Measure curvature (now returns a tuple of sign, magnitude, and normalized)
        curvatures = measure_curvature(points)

        # Calculate inward normals
        normals = calculate_inward_normal(points, mask)

        # Measure intensity with cell coverage check
        intensities, sampling_regions, valid_points = measure_intensity(
            image, mask, points, normals, depth, width, min_cell_coverage
        )

        # Update statistics
        statistics["processed_frames"] += 1
        statistics["total_points"] += len(points)
        statistics["valid_points"] += np.sum(valid_points)
        statistics["invalid_points"] += len(valid_points) - np.sum(valid_points)

        # Store data for this frame
        all_curvatures.append(curvatures)
        all_intensities.append(intensities)
        all_valid_points.append(valid_points)

        # Visualize results
        visualize_results(image, mask, points, curvatures, intensities,
                         valid_points, sampling_regions, frame_idx, output_dir)

        # Plot correlation for this frame - both sign and normalized curvature
        sign_curvatures = curvatures[0]
        normalized_curvatures = curvatures[2]

        plot_curvature_correlation('sign', sign_curvatures, intensities, valid_points, frame_idx, output_dir)
        plot_curvature_correlation('normalized', normalized_curvatures, intensities, valid_points, frame_idx, output_dir)

        # Analyze edge movement (for frame 1 and onward)
        if frame_idx > 0:
            previous_mask = masks[frame_idx - 1]
            movement_score, movement_type, movement_map = detect_edge_movement(
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

            # Visualize edge movement
            visualize_edge_movement(
                image, mask, previous_mask, movement_map, movement_type, frame_idx, output_dir
            )

    # Calculate average valid points per frame
    if statistics["processed_frames"] > 0:
        statistics["average_valid_points_per_frame"] = statistics["valid_points"] / statistics["processed_frames"]

    # Create summary visualizations
    if all_curvatures and all_intensities and all_valid_points:
        create_summary_visualization(all_curvatures, all_intensities, all_valid_points, output_dir)

        # Get summary correlation statistics using both sign and normalized curvature
        all_sign_curvatures = [curvature_tuple[0] for curvature_tuple in all_curvatures]
        all_normalized_curvatures = [curvature_tuple[2] for curvature_tuple in all_curvatures]

        # Plot and get stats for both curvature types
        sign_stats = plot_summary_curvature_correlation('sign', all_sign_curvatures, all_intensities, all_valid_points, output_dir)
        normalized_stats = plot_summary_curvature_correlation('normalized', all_normalized_curvatures, all_intensities, all_valid_points, output_dir)

        # Update statistics
        statistics["sign_curvature_correlation"] = sign_stats
        statistics["normalized_curvature_correlation"] = normalized_stats

    # Create edge movement summary visualization
    if all_movement_scores:
        create_movement_summary_visualization(all_movement_scores, all_movement_types, output_dir)
        save_movement_data_to_csv(all_movement_scores, all_movement_types, output_dir)

    # Add overall edge movement statistics to metadata
    if all_movement_types:
        statistics["edge_movement"]["predominant_behavior"] = max(
            ["extending", "retracting", "stable"],
            key=lambda x: statistics["edge_movement"][f"{x}_frames"]
        )

    # Save data to CSV
    if all_curvatures and all_intensities and all_valid_points:
        save_data_to_csv(all_curvatures, all_intensities, all_valid_points, output_dir)

    # Generate metadata file
    generate_metadata(image_path, mask_path, output_dir, parameters, statistics)

    print(f"Results saved to {output_dir}")

    # Perform movement-grouped correlation analysis
    # This needs to come after all frames are processed so we have all movement types
    if all_movement_types and len(all_movement_types) > 1:
        # Skip the first frame which has no movement type
        movement_correlation_stats = analyze_movement_grouped_correlations(
            all_curvatures[1:],  # Skip first frame
            all_intensities[1:],  # Skip first frame
            all_valid_points[1:],  # Skip first frame
            all_movement_types,   # Already skips first frame
            output_dir
        )

        # Update statistics with movement correlation stats
        statistics["movement_grouped_correlation"] = movement_correlation_stats

        # Regenerate metadata with updated statistics
        generate_metadata(image_path, mask_path, output_dir, parameters, statistics)


    # Perform temporal analysis (correlating current curvature with previous intensity)
    if len(all_curvatures) > 1:  # Need at least 2 frames for temporal analysis
        process_temporal_analysis(all_curvatures, all_intensities, all_valid_points,
                                image_path, mask_path, output_dir, parameters)

    # Perform random analysis (correlating curvature with intensity from random frames)
    if len(all_curvatures) > 1:  # Need at least 2 frames for random analysis
        process_random_analysis(all_curvatures, all_intensities, all_valid_points,
                              image_path, mask_path, output_dir, parameters)

    # Perform region-specific intensity analysis
    if len(all_masks) > 1:  # Need at least 2 frames
        region_statistics = analyze_region_specific_intensities(
            all_curvatures, all_intensities, all_valid_points,
            all_masks, all_contours, output_dir
        )

        # Update statistics with region-specific stats
        statistics["region_specific_analysis"] = region_statistics

        # Regenerate metadata with updated statistics
        generate_metadata(image_path, mask_path, output_dir, parameters, statistics)


def analyze_region_specific_intensities(all_curvatures, all_intensities, all_valid_points,
                                      all_masks, all_contours, output_dir):
    """
    Compare intensity measurements between extending and retracting regions of the cell edge.

    Parameters:
    - all_curvatures: List of curvature tuples for each frame
    - all_intensities: List of intensity arrays for each frame
    - all_valid_points: List of boolean arrays indicating which points are valid
    - all_masks: List of binary masks for each frame
    - all_contours: List of contours for each frame
    - output_dir: Directory to save the results
    """
    # Create output directory for region-specific analysis
    region_dir = os.path.join(output_dir, "region_specific_analysis")
    os.makedirs(region_dir, exist_ok=True)

    # Skip first frame since we need previous frame for movement detection
    if len(all_masks) <= 1:
        print("Not enough frames for region-specific analysis")
        return

    # Initialize lists to store intensity data by region type
    extending_intensities = []
    retracting_intensities = []
    stable_intensities = []

    # Initialize lists to store curvature data by region type
    extending_curvatures = []
    retracting_curvatures = []
    stable_curvatures = []

    # Process each frame (skipping the first)
    for frame_idx in range(1, len(all_masks)):
        print(f"Processing region-specific analysis for frame {frame_idx}")

        # Get current and previous masks
        current_mask = all_masks[frame_idx]
        previous_mask = all_masks[frame_idx - 1]

        # Get intensity measurements, curvatures, and valid points for this frame
        intensities = all_intensities[frame_idx]
        sign_curvatures, magnitude_curvatures, normalized_curvatures = all_curvatures[frame_idx]
        valid_points = all_valid_points[frame_idx]

        # Calculate movement map
        _, _, movement_map = detect_edge_movement(
            all_contours[frame_idx], all_contours[frame_idx - 1], current_mask, previous_mask
        )

        # Get sampled equidistant points (these match with our measurements)
        points = sample_equidistant_points(all_contours[frame_idx], len(valid_points))

        # For each valid point, determine if it's in an extending or retracting region
        for i in range(len(valid_points)):
            if not valid_points[i]:
                continue

            # Get coordinates of the point (rounded to integers for indexing)
            y, x = int(round(points[i][0])), int(round(points[i][1]))

            # Check bounds
            if y < 0 or y >= movement_map.shape[0] or x < 0 or x >= movement_map.shape[1]:
                continue

            # Check movement around this point (use a small window to ensure we capture edge movement)
            y_min = max(0, y - 2)
            y_max = min(movement_map.shape[0], y + 3)
            x_min = max(0, x - 2)
            x_max = min(movement_map.shape[1], x + 3)

            local_movement = movement_map[y_min:y_max, x_min:x_max]

            # If any pixel in the local window shows extension or retraction, use that value
            if np.any(local_movement > 0.1):  # Extending region
                extending_intensities.append(intensities[i])
                extending_curvatures.append(normalized_curvatures[i])
            elif np.any(local_movement < -0.1):  # Retracting region
                retracting_intensities.append(intensities[i])
                retracting_curvatures.append(normalized_curvatures[i])
            else:  # Stable region
                stable_intensities.append(intensities[i])
                stable_curvatures.append(normalized_curvatures[i])

    # Convert to numpy arrays
    extending_intensities = np.array(extending_intensities)
    retracting_intensities = np.array(retracting_intensities)
    stable_intensities = np.array(stable_intensities)

    extending_curvatures = np.array(extending_curvatures)
    retracting_curvatures = np.array(retracting_curvatures)
    stable_curvatures = np.array(stable_curvatures)

    # Calculate statistics
    region_statistics = {
        "extending": {
            "count": len(extending_intensities),
            "intensity_mean": float(np.mean(extending_intensities)) if len(extending_intensities) > 0 else None,
            "intensity_std": float(np.std(extending_intensities)) if len(extending_intensities) > 0 else None,
            "curvature_mean": float(np.mean(extending_curvatures)) if len(extending_curvatures) > 0 else None,
            "curvature_std": float(np.std(extending_curvatures)) if len(extending_curvatures) > 0 else None
        },
        "retracting": {
            "count": len(retracting_intensities),
            "intensity_mean": float(np.mean(retracting_intensities)) if len(retracting_intensities) > 0 else None,
            "intensity_std": float(np.std(retracting_intensities)) if len(retracting_intensities) > 0 else None,
            "curvature_mean": float(np.mean(retracting_curvatures)) if len(retracting_curvatures) > 0 else None,
            "curvature_std": float(np.std(retracting_curvatures)) if len(retracting_curvatures) > 0 else None
        },
        "stable": {
            "count": len(stable_intensities),
            "intensity_mean": float(np.mean(stable_intensities)) if len(stable_intensities) > 0 else None,
            "intensity_std": float(np.std(stable_intensities)) if len(stable_intensities) > 0 else None,
            "curvature_mean": float(np.mean(stable_curvatures)) if len(stable_curvatures) > 0 else None,
            "curvature_std": float(np.std(stable_curvatures)) if len(stable_curvatures) > 0 else None
        }
    }

    # Perform statistical tests if enough data
    # Compare extending vs retracting intensities
    if len(extending_intensities) > 0 and len(retracting_intensities) > 0:
        t_stat, p_value = stats.ttest_ind(extending_intensities, retracting_intensities, equal_var=False)
        region_statistics["intensity_comparison"] = {
            "extending_vs_retracting": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value)
            }
        }

    # Create visualizations

    # 1. Intensity distribution by region type
    plt.figure(figsize=(10, 8))

    # Use violin plots to show distributions
    positions = [1, 2, 3]
    violin_parts = plt.violinplot(
        [extending_intensities, stable_intensities, retracting_intensities],
        positions=positions,
        widths=0.8,
        showmeans=True,
        showextrema=True
    )

    # Customize violin colors
    violin_parts['bodies'][0].set_facecolor('blue')  # Extending
    violin_parts['bodies'][0].set_alpha(0.7)
    violin_parts['bodies'][1].set_facecolor('gray')  # Stable
    violin_parts['bodies'][1].set_alpha(0.7)
    violin_parts['bodies'][2].set_facecolor('red')   # Retracting
    violin_parts['bodies'][2].set_alpha(0.7)

    # Add box plots inside violins for additional statistics
    plt.boxplot(
        [extending_intensities, stable_intensities, retracting_intensities],
        positions=positions,
        widths=0.15,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='yellow', alpha=0.5)
    )

    # Add sample size annotations
    for i, (pos, data, label) in enumerate(zip(
        positions,
        [extending_intensities, stable_intensities, retracting_intensities],
        ['Extending', 'Stable', 'Retracting']
    )):
        plt.annotate(f"n={len(data)}", (pos, plt.ylim()[0] * 1.05), ha='center', va='top')

    # Add p-value annotation if available
    if "intensity_comparison" in region_statistics:
        p_value = region_statistics["intensity_comparison"]["extending_vs_retracting"]["p_value"]
        annotation = f"p={p_value:.3f}"
        if p_value < 0.001:
            annotation = "p<0.001"
        elif p_value < 0.01:
            annotation = "p<0.01"
        elif p_value < 0.05:
            annotation = "p<0.05"

        plt.annotate(
            annotation,
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-20, -20), textcoords='offset points',
            ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )

    # Customize plot
    plt.xticks(positions, ['Extending', 'Stable', 'Retracting'])
    plt.ylabel('Intensity')
    plt.title('Distribution of Intensity by Region Movement Type')
    plt.grid(True, axis='y', alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, "region_intensity_distribution.png"), dpi=150)
    plt.close()

    # 2. Curvature by region type
    plt.figure(figsize=(10, 8))

    # Use violin plots to show distributions
    violin_parts = plt.violinplot(
        [extending_curvatures, stable_curvatures, retracting_curvatures],
        positions=positions,
        widths=0.8,
        showmeans=True,
        showextrema=True
    )

    # Customize violin colors
    violin_parts['bodies'][0].set_facecolor('blue')  # Extending
    violin_parts['bodies'][0].set_alpha(0.7)
    violin_parts['bodies'][1].set_facecolor('gray')  # Stable
    violin_parts['bodies'][1].set_alpha(0.7)
    violin_parts['bodies'][2].set_facecolor('red')   # Retracting
    violin_parts['bodies'][2].set_alpha(0.7)

    # Add box plots inside violins for additional statistics
    plt.boxplot(
        [extending_curvatures, stable_curvatures, retracting_curvatures],
        positions=positions,
        widths=0.15,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='yellow', alpha=0.5)
    )

    # Customize plot
    plt.xticks(positions, ['Extending', 'Stable', 'Retracting'])
    plt.ylabel('Normalized Curvature')
    plt.title('Distribution of Curvature by Region Movement Type')
    plt.grid(True, axis='y', alpha=0.3)

    # Add horizontal line at zero
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, "region_curvature_distribution.png"), dpi=150)
    plt.close()

    # 3. Curvature vs Intensity scatter plot by region type
    plt.figure(figsize=(10, 8))

    # Scatter plot for each region type
    if len(extending_curvatures) > 0:
        plt.scatter(extending_curvatures, extending_intensities,
                  color='blue', alpha=0.5, label='Extending')
    if len(stable_curvatures) > 0:
        plt.scatter(stable_curvatures, stable_intensities,
                  color='gray', alpha=0.5, label='Stable')
    if len(retracting_curvatures) > 0:
        plt.scatter(retracting_curvatures, retracting_intensities,
                  color='red', alpha=0.5, label='Retracting')

    # Add regression lines for each region type
    for curvatures, intensities, color, label in [
        (extending_curvatures, extending_intensities, 'blue', 'Extending'),
        (retracting_curvatures, retracting_intensities, 'red', 'Retracting'),
        (stable_curvatures, stable_intensities, 'gray', 'Stable')
    ]:
        if len(curvatures) > 1:
            slope, intercept, r_value, p_value, _ = stats.linregress(curvatures, intensities)
            x_line = np.linspace(min(curvatures), max(curvatures), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color=color, linestyle='-',
                   label=f'{label}: R²={r_value**2:.3f}, p={p_value:.3f}')

    # Customize plot
    plt.xlabel('Normalized Curvature')
    plt.ylabel('Intensity')
    plt.title('Curvature vs Intensity by Region Movement Type')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, "region_curvature_intensity_correlation.png"), dpi=150)
    plt.close()

    # Save data to CSV
    region_data = []

    # Extending regions
    for i in range(len(extending_curvatures)):
        region_data.append([
            extending_curvatures[i],
            extending_intensities[i],
            "extending"
        ])

    # Retracting regions
    for i in range(len(retracting_curvatures)):
        region_data.append([
            retracting_curvatures[i],
            retracting_intensities[i],
            "retracting"
        ])

    # Stable regions
    for i in range(len(stable_curvatures)):
        region_data.append([
            stable_curvatures[i],
            stable_intensities[i],
            "stable"
        ])

    # Save to CSV
    if region_data:
        with open(os.path.join(region_dir, "region_specific_data.csv"), 'w') as f:
            f.write("curvature,intensity,region_type\n")
            for row in region_data:
                f.write(f"{row[0]},{row[1]},{row[2]}\n")

    return region_statistics

def main():
    """
    Main function to run the script.
    """
    # Define default variables separately for easy modification
    # These are the defaults that will be used when running from a script editor
    default_image_path = "/Users/george/Documents/python_projects/cell_edge_analysis/151_2019_07_29_TIRF_mKera_scratch_1_MMStack_Pos7_piezo1.tif"
    default_mask_path = "/Users/george/Documents/python_projects/cell_edge_analysis/151_2019_07_29_TIRF_mKera_scratch_Pos7_DIC_Mask_test.tif"
    default_output_dir = "results_n25"
    default_n_points = 25     # Number of equidistant points along the contour
    default_depth = 50         # Depth of the sampling rectangle (how far into the cell)
    default_width = 20          # Width of the sampling rectangle
    default_min_cell_coverage = 0.8  # Minimum fraction of the rectangle that must be inside the cell

    # Check if the script is being run directly (not imported)
    if __name__ == "__main__":
        # Create a command-line interface for running from terminal
        import argparse

        parser = argparse.ArgumentParser(description='Analyze cell curvature and intensity in microscope images.')
        parser.add_argument('--image', default=default_image_path, help='Path to microscope image stack (TIFF)')
        parser.add_argument('--mask', default=default_mask_path, help='Path to binary mask stack (TIFF)')
        parser.add_argument('--output', default=default_output_dir, help='Output directory for results')
        parser.add_argument('--points', type=int, default=default_n_points, help='Number of equidistant points along the contour')
        parser.add_argument('--depth', type=int, default=default_depth, help='Depth of the sampling rectangle')
        parser.add_argument('--width', type=int, default=default_width, help='Width of the sampling rectangle')
        parser.add_argument('--coverage', type=float, default=default_min_cell_coverage, help='Minimum cell coverage (0.0-1.0)')

        args = parser.parse_args()

        # Process the stack with command line arguments
        process_stack(
            args.image,
            args.mask,
            args.output,
            args.points,
            args.depth,
            args.width,
            args.coverage
        )
    else:
        # When imported as a module, run with default parameters
        # This makes it easy to modify the parameters directly in code
        process_stack(
            default_image_path,
            default_mask_path,
            default_output_dir,
            default_n_points,
            default_depth,
            default_width,
            default_min_cell_coverage
        )

if __name__ == "__main__":
    main()
