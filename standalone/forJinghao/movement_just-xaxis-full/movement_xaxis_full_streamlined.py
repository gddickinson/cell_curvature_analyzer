#!/usr/bin/env python3
"""
Streamlined Cell Edge Movement Analysis Script - X-Axis Full Method

This script analyzes the relationship between cell edge movement and PIEZO1 protein
intensity using the vertical_displacement_xaxis_full method, which processes every
x-pixel position and tracks the uppermost edge.

Method: vertical_displacement_xaxis_full
- Processes EVERY x-pixel position across the cell
- Finds uppermost (minimum y) point at each x on both frames
- Calculates displacement between uppermost points
- Creates 2D movement map with displacement values per x-column
- Handles cells that curve back on themselves
- No reliance on binary mask differences - works purely with contours

Sampling: x_axis
- Points distributed at regular x-coordinate intervals

Temporal Directions:
- 'past': Intensity at frame N correlated with movement from frame N-1 to N
- 'future': Intensity at frame N correlated with movement from frame N to N+1

Author: George Dickinson
Date: October 29, 2025
Version: Streamlined (xaxis_full only)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from skimage import measure, draw
from scipy import stats, interpolate
import json
import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Input file paths
#IMAGE_STACK_PATH = "/path/to/your/piezo1_images.tif"  # TIRF microscopy images
#MASK_STACK_PATH = "/path/to/your/cell_masks.tif"     # Binary cell masks

IMAGE_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/test_data/151_2019_07_29_TIRF_mKera_scratch_1_MMStack_Pos7_piezo1.tif"  # TIRF microscopy images
MASK_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/test_data/151_2019_07_29_TIRF_mKera_scratch_Pos7_DIC_Mask_test.tif"    # Binary cell masks


# Output directory
OUTPUT_DIR = "xaxis_movement_analysis_results"

# Analysis parameters
CONFIG = {
    'n_points': 12,               # Number of points to sample along cell edge (x-axis)
    'depth': 200,                 # Depth of sampling rectangle (pixels into cell)
    'width': 75,                  # Width of sampling rectangle (pixels)
    'min_cell_coverage': 0.8,     # Minimum fraction of rectangle inside cell
    'try_rotation': True,         # Try rotating rejected sampling regions 180°
    'movement_threshold': 0.1,    # Threshold for classifying movement (pixels)
    'min_movement_pixels': 5,     # Minimum valid displacements for overall score
    'exclude_endpoints': True,    # Exclude first and last edge points
    'temporal_direction': 'past', # 'past' or 'future'
}

# Output control parameters
OUTPUT_CONFIG = {
    'save_detailed_csv': True,        # Save detailed_movement_results.csv
    'save_summary_json': True,        # Save movement_summary_statistics.json
    'save_correlation_plots': True,   # Save main correlation plots
    'save_frame_plots': True,         # Save frame-by-frame visualizations
}

# Visualization configuration
VIZ_CONFIG = {
    'plot_every_nth_frame': 10,                    # Generate plots for every Nth frame
    'plot_every_nth_sampling_point': 1,            # Show every Nth rectangle
    'save_sampling_plots': True,                   # Sampling region visualization
    'save_intensity_plots': True,                  # Intensity analysis plots
    'save_movement_plots': True,                   # Movement analysis plots
    'save_edge_transition_plots': True,            # Consecutive frame edge comparison
    'save_movement_overlay_plots': True,           # Movement type overlaid on images
    'save_intensity_overlay_plots': True,          # Intensity values overlaid on images
    'save_movement_type_correlation': True,        # Binary movement type correlation
    'movement_cmap': 'RdBu_r',                     # Colormap for movement
    'intensity_cmap': 'viridis',                   # Colormap for intensity
    'marker_size': 30,                             # Scatter plot marker size
    'figure_dpi': 150                              # Resolution for saved figures
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data(image_path, mask_path):
    """Load image and mask stacks from TIFF files."""
    print(f"Loading image stack: {image_path}")
    images = tifffile.imread(image_path)

    print(f"Loading mask stack: {mask_path}")
    masks = tifffile.imread(mask_path)

    # Ensure masks are binary
    masks = (masks > 0).astype(np.uint8)

    # Handle single image case
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    print(f"Loaded {images.shape[0]} frames of size {images.shape[1]}x{images.shape[2]}")
    return images, masks

def detect_cell_edge(mask):
    """Detect the cell edge from a binary mask."""
    contours = measure.find_contours(mask, 0.5)

    if len(contours) > 0:
        contour = max(contours, key=len)
        return contour
    else:
        return np.array([])

def detect_edge_movement_vertical_xaxis(current_contour, previous_contour, current_mask,
                                       previous_mask, config):
    """
    Detect movement using vertical displacement at every x-position.
    Tracks uppermost (minimum y) point at each x on both frames.
    """
    # Get x-range spanning both frames
    x_min = int(min(np.min(current_contour[:, 1]), np.min(previous_contour[:, 1])))
    x_max = int(max(np.max(current_contour[:, 1]), np.max(previous_contour[:, 1])))

    # Create arrays for every x-position
    x_positions = np.arange(x_min, x_max + 1)
    displacements = np.full(len(x_positions), np.nan, dtype=float)

    # For each x position, find uppermost (minimum y) points
    for i, x in enumerate(x_positions):
        # Previous frame: find uppermost point at this x
        prev_mask = np.abs(previous_contour[:, 1] - x) < 0.5
        if np.any(prev_mask):
            prev_y = np.min(previous_contour[prev_mask, 0])
        else:
            prev_y = None

        # Current frame: find uppermost point at this x
        curr_mask = np.abs(current_contour[:, 1] - x) < 0.5
        if np.any(curr_mask):
            curr_y = np.min(current_contour[curr_mask, 0])
        else:
            curr_y = None

        # Calculate displacement if both exist
        if prev_y is not None and curr_y is not None:
            displacements[i] = curr_y - prev_y  # Positive = down, negative = up

    # Calculate overall movement score
    valid_displacements = displacements[~np.isnan(displacements)]

    if len(valid_displacements) > config['min_movement_pixels']:
        movement_score = np.mean(valid_displacements)
    else:
        movement_score = 0.0

    # Classify movement type
    # Since we track uppermost (minimum y) edge:
    # Negative displacement (moving up) = extending
    # Positive displacement (moving down) = retracting
    if movement_score < -config['movement_threshold']:
        movement_type = 'extending'
    elif movement_score > config['movement_threshold']:
        movement_type = 'retracting'
    else:
        movement_type = 'stable'

    # Create 2D movement map
    movement_map = create_movement_map_from_xaxis_displacement(
        displacements, x_positions, current_mask.shape)

    return movement_score, movement_type, movement_map, displacements, x_positions

def create_movement_map_from_xaxis_displacement(displacements, x_positions, image_shape):
    """Create 2D movement map where each x-column contains the displacement value."""
    movement_map = np.zeros(image_shape, dtype=float)

    for i, x in enumerate(x_positions):
        x_int = int(round(x))
        if 0 <= x_int < image_shape[1] and not np.isnan(displacements[i]):
            movement_map[:, x_int] = displacements[i]

    return movement_map

def sample_x_axis_points(contour, n_points=100):
    """Sample points along contour at regular x-coordinate intervals."""
    if len(contour) == 0:
        return np.array([])

    x_min, x_max = np.min(contour[:, 1]), np.max(contour[:, 1])

    if n_points <= 1:
        x_positions = [x_min + (x_max - x_min) / 2]
    else:
        x_positions = np.linspace(x_min, x_max, n_points)

    sampled_points = []
    tolerance = (x_max - x_min) / (n_points * 4)

    for x_pos in x_positions:
        x_distances = np.abs(contour[:, 1] - x_pos)
        x_mask = x_distances <= tolerance

        if np.any(x_mask):
            candidates = contour[x_mask]
            if len(candidates) == 1:
                sampled_points.append(candidates[0])
            else:
                best_idx = np.argmin(x_distances[x_mask])
                sampled_points.append(candidates[best_idx])
        else:
            closest_idx = np.argmin(x_distances)
            sampled_points.append(contour[closest_idx])

    sampled_points = np.array(sampled_points)

    # Interpolate if needed
    if len(sampled_points) < n_points and len(sampled_points) > 1:
        from scipy.interpolate import interp1d

        sort_idx = np.argsort(sampled_points[:, 1])
        sorted_points = sampled_points[sort_idx]

        x_coords = sorted_points[:, 1]
        y_coords = sorted_points[:, 0]

        if len(np.unique(x_coords)) > 1:
            interp_func = interp1d(x_coords, y_coords, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')

            new_x_positions = np.linspace(x_min, x_max, n_points)
            new_y_positions = interp_func(new_x_positions)

            sampled_points = np.column_stack([new_y_positions, new_x_positions])

    return sampled_points

def calculate_inward_normal(points, mask):
    """Calculate inward normal vectors at each point."""
    n_points = len(points)
    normals = np.zeros((n_points, 2))

    for i in range(n_points):
        # Use neighbors for tangent calculation
        prev_idx = max(0, i - 1)
        next_idx = min(n_points - 1, i + 1)

        tangent = points[next_idx] - points[prev_idx]
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm > 0:
            tangent = tangent / tangent_norm

            # Two possible normals
            normal1 = np.array([-tangent[1], tangent[0]])
            normal2 = np.array([tangent[1], -tangent[0]])

            # Choose the one pointing inward
            test_point1 = points[i] + normal1 * 5
            test_point2 = points[i] + normal2 * 5

            y1, x1 = int(round(test_point1[0])), int(round(test_point1[1]))
            y2, x2 = int(round(test_point2[0])), int(round(test_point2[1]))

            # Check if in bounds and inside mask
            in_bounds1 = 0 <= y1 < mask.shape[0] and 0 <= x1 < mask.shape[1]
            in_bounds2 = 0 <= y2 < mask.shape[0] and 0 <= x2 < mask.shape[1]

            if in_bounds1 and mask[y1, x1] > 0:
                normals[i] = normal1
            elif in_bounds2 and mask[y2, x2] > 0:
                normals[i] = normal2
            else:
                normals[i] = normal1
        else:
            normals[i] = np.array([0, 1])

    return normals

def measure_intensity(image, mask, points, normals, depth=200, width=75,
                     min_cell_coverage=0.8, try_rotation=True, exclude_endpoints=True):
    """Measure PIEZO1 intensity in rectangular regions extending into the cell."""
    n_points = len(points)
    intensities = np.zeros(n_points)
    valid_points = np.zeros(n_points, dtype=bool)
    point_status = np.zeros(n_points, dtype=int)
    image_shape = image.shape

    # Exclude endpoints if requested
    if exclude_endpoints and n_points > 2:
        point_status[0] = 0
        point_status[-1] = 0

    for i in range(n_points):
        if exclude_endpoints and (i == 0 or i == n_points - 1):
            continue

        p = points[i]
        normal = normals[i]

        # Calculate rectangle corners
        end_point = p + depth * normal
        perp = np.array([-normal[1], normal[0]])

        corner1 = p + width/2 * perp
        corner2 = p - width/2 * perp
        corner3 = end_point - width/2 * perp
        corner4 = end_point + width/2 * perp

        vertices = np.array([corner1, corner4, corner3, corner2])

        # Get pixels within rectangle
        rr, cc = draw.polygon(vertices[:, 0], vertices[:, 1], image_shape)

        # Check bounds
        valid_pixels = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
        if np.any(valid_pixels):
            rr = rr[valid_pixels]
            cc = cc[valid_pixels]

            total_pixels = len(rr)
            if total_pixels > 0:
                cell_pixels = np.sum(mask[rr, cc])
                cell_coverage = cell_pixels / total_pixels

                if cell_coverage >= min_cell_coverage:
                    intensities[i] = np.mean(image[rr, cc])
                    valid_points[i] = True
                    point_status[i] = 1
                elif try_rotation:
                    # Try 180° rotation
                    rotated_normal = -normal
                    rotated_end = p + depth * rotated_normal

                    rot_corner1 = p + width/2 * perp
                    rot_corner2 = p - width/2 * perp
                    rot_corner3 = rotated_end - width/2 * perp
                    rot_corner4 = rotated_end + width/2 * perp

                    rot_vertices = np.array([rot_corner1, rot_corner4, rot_corner3, rot_corner2])
                    rot_rr, rot_cc = draw.polygon(rot_vertices[:, 0], rot_vertices[:, 1], image_shape)

                    rot_valid = (rot_rr >= 0) & (rot_rr < image_shape[0]) & (rot_cc >= 0) & (rot_cc < image_shape[1])
                    if np.any(rot_valid):
                        rot_rr = rot_rr[rot_valid]
                        rot_cc = rot_cc[rot_valid]

                        rot_total = len(rot_rr)
                        if rot_total > 0:
                            rot_cell_pixels = np.sum(mask[rot_rr, rot_cc])
                            rot_coverage = rot_cell_pixels / rot_total

                            if rot_coverage >= min_cell_coverage:
                                intensities[i] = np.mean(image[rot_rr, rot_cc])
                                valid_points[i] = True
                                point_status[i] = 3
                            else:
                                point_status[i] = 2
                    else:
                        point_status[i] = 2
                else:
                    point_status[i] = 2
            else:
                point_status[i] = 2
        else:
            point_status[i] = 2

    return intensities, valid_points, point_status

def calculate_local_movement_scores(points, xaxis_displacements, xaxis_positions):
    """Calculate local movement scores by looking up displacement at each point's x-position."""
    local_scores = np.zeros(len(points))

    for i, point in enumerate(points):
        point_x = point[1]

        # Find closest x-position
        x_idx = np.argmin(np.abs(xaxis_positions - point_x))
        displacement = xaxis_displacements[x_idx]

        # Store displacement (0 if NaN)
        local_scores[i] = displacement if not np.isnan(displacement) else 0.0

    return local_scores

def analyze_frame_pair(current_image, current_mask, comparison_mask, config, temporal_direction='past'):
    """Analyze movement between two consecutive frames."""
    # Detect cell edges
    current_contour = detect_cell_edge(current_mask)
    comparison_contour = detect_cell_edge(comparison_mask)

    if len(current_contour) == 0 or len(comparison_contour) == 0:
        return {"error": "Could not detect contour in one or both frames"}

    # Detect edge movement using xaxis full method
    if temporal_direction == 'future':
        movement_score, movement_type, movement_map, xaxis_displacements, xaxis_positions = \
            detect_edge_movement_vertical_xaxis(
                comparison_contour, current_contour, comparison_mask, current_mask, config)
    else:  # 'past'
        movement_score, movement_type, movement_map, xaxis_displacements, xaxis_positions = \
            detect_edge_movement_vertical_xaxis(
                current_contour, comparison_contour, current_mask, comparison_mask, config)

    # Sample points along current frame edge using x-axis method
    points = sample_x_axis_points(current_contour, config['n_points'])

    if len(points) == 0:
        return {"error": "No sampling points generated"}

    # Calculate inward normals
    normals = calculate_inward_normal(points, current_mask)

    # Measure intensity
    intensities, valid_points, point_status = measure_intensity(
        current_image, current_mask, points, normals,
        depth=config['depth'],
        width=config['width'],
        min_cell_coverage=config['min_cell_coverage'],
        try_rotation=config['try_rotation'],
        exclude_endpoints=config['exclude_endpoints']
    )

    # Calculate local movement scores
    local_movement_scores = calculate_local_movement_scores(
        points, xaxis_displacements, xaxis_positions)

    # Package results
    results = {
        'points': points,
        'normals': normals,
        'intensities': intensities,
        'valid_points': valid_points,
        'point_status': point_status,
        'movement_score': movement_score,
        'movement_type': movement_type,
        'movement_map': movement_map,
        'local_movement_scores': local_movement_scores,
        'current_contour': current_contour,
        'comparison_contour': comparison_contour,
        'xaxis_displacements': xaxis_displacements,
        'xaxis_positions': xaxis_positions,
        'temporal_direction': temporal_direction
    }

    return results

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_movement_correlation_plot(all_movements, all_intensities, all_valid_points,
                                   movement_types, output_dir, viz_config, temporal_direction='past'):
    """Create and save movement-intensity correlation plot."""
    combined_movements = []
    combined_intensities = []
    combined_types = []

    for i in range(len(all_movements)):
        movements = all_movements[i]
        intensities = all_intensities[i]
        valid_points = all_valid_points[i]
        frame_type = movement_types[i]

        valid_mov = movements[valid_points]
        valid_int = intensities[valid_points]

        combined_movements.extend(valid_mov)
        combined_intensities.extend(valid_int)
        combined_types.extend([frame_type] * len(valid_mov))

    combined_movements = np.array(combined_movements)
    combined_intensities = np.array(combined_intensities)
    combined_types = np.array(combined_types)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: All points colored by movement score
    scatter = ax1.scatter(combined_movements, combined_intensities,
                         alpha=0.6, s=viz_config['marker_size'],
                         c=combined_movements, cmap=viz_config['movement_cmap'],
                         vmin=np.percentile(combined_movements, 5),
                         vmax=np.percentile(combined_movements, 95))

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Vertical Displacement (pixels)\n(Negative = Up, Positive = Down)', fontsize=10)

    # Calculate correlation
    correlation_stats = {}
    if len(combined_movements) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_movements, combined_intensities)

        x_line = np.linspace(min(combined_movements), max(combined_movements), 100)
        y_line = slope * x_line + intercept

        ax1.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        stats_text = f'R² = {r_value**2:.3f}\np-value = {p_value:.3e}\nn = {len(combined_movements)}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)

        correlation_stats = {
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'sample_size': int(len(combined_movements)),
            'slope': float(slope)
        }

    direction_label = "Future Movement" if temporal_direction == 'future' else "Past Movement"
    direction_subtitle = "(Intensity → Future Movement)" if temporal_direction == 'future' else "(Past Movement → Intensity)"

    ax1.set_xlabel('Vertical Displacement (pixels)', fontsize=12)
    ax1.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax1.set_title(f'Edge {direction_label} vs PIEZO1 Intensity\n{direction_subtitle}', fontsize=12)
    if len(combined_movements) > 1:
        ax1.legend()

    # Right plot: Points grouped by movement type
    colors = {'extending': 'red', 'retracting': 'blue', 'stable': 'gray'}

    for movement_type, color in colors.items():
        mask = combined_types == movement_type
        if np.any(mask):
            ax2.scatter(combined_movements[mask], combined_intensities[mask],
                       alpha=0.6, s=viz_config['marker_size'],
                       c=color, label=f'{movement_type.capitalize()} ({np.sum(mask)} points)')

    ax2.set_xlabel('Vertical Displacement (pixels)', fontsize=12)
    ax2.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax2.set_title(f'{direction_label} vs Intensity by Frame Type', fontsize=12)
    ax2.legend()

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'movement_intensity_correlation.png')
    plt.savefig(output_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

    return correlation_stats

def create_movement_summary_plot(movement_scores, movement_types, output_dir, viz_config):
    """Create summary plots showing movement over time and distribution."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Movement scores over time
    frames = np.arange(len(movement_scores))
    colors = []
    for mt in movement_types:
        if mt == 'extending':
            colors.append('red')
        elif mt == 'retracting':
            colors.append('blue')
        else:
            colors.append('gray')

    ax1.scatter(frames, movement_scores, c=colors, alpha=0.6, s=50)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Frame Transition', fontsize=12)
    ax1.set_ylabel('Movement Score (pixels)', fontsize=12)
    ax1.set_title('Movement Score Over Time\n(Positive = Down, Negative = Up)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Movement score distribution
    ax2.hist(movement_scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Movement Score (pixels)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Movement Scores', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Movement type counts
    type_counts = {'extending': 0, 'retracting': 0, 'stable': 0}
    for mt in movement_types:
        type_counts[mt] += 1

    ax3.bar(type_counts.keys(), type_counts.values(),
           color=['red', 'blue', 'gray'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Movement Type Distribution', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # Movement score box plot by type
    extending_scores = [s for s, t in zip(movement_scores, movement_types) if t == 'extending']
    retracting_scores = [s for s, t in zip(movement_scores, movement_types) if t == 'retracting']
    stable_scores = [s for s, t in zip(movement_scores, movement_types) if t == 'stable']

    data_to_plot = []
    labels = []
    if extending_scores:
        data_to_plot.append(extending_scores)
        labels.append('Extending')
    if retracting_scores:
        data_to_plot.append(retracting_scores)
        labels.append('Retracting')
    if stable_scores:
        data_to_plot.append(stable_scores)
        labels.append('Stable')

    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'blue', 'gray'][:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_ylabel('Movement Score (pixels)', fontsize=12)
    ax4.set_title('Movement Scores by Type', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'movement_summary.png')
    plt.savefig(output_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def save_frame_visualizations(frame_idx, results, current_image, comparison_image,
                              current_mask, comparison_mask, output_dir, viz_config, config):
    """Save frame-specific visualization plots."""
    if frame_idx % viz_config['plot_every_nth_frame'] != 0:
        return

    # Create subdirectories
    for subdir in ['frame_sampling_plots', 'frame_intensity_plots',
                   'frame_movement_plots', 'edge_transition_plots']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    points = results['points']
    normals = results['normals']
    intensities = results['intensities']
    valid_points = results['valid_points']
    point_status = results['point_status']
    local_movement_scores = results['local_movement_scores']
    movement_map = results['movement_map']
    current_contour = results['current_contour']
    comparison_contour = results['comparison_contour']
    movement_type = results['movement_type']
    movement_score = results['movement_score']

    # Edge transition plot
    if viz_config['save_edge_transition_plots']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Movement map with edges
        im1 = ax1.imshow(current_image, cmap='gray', alpha=0.7)

        movement_display = np.ma.masked_where(movement_map == 0, movement_map)
        vmin, vmax = np.percentile(movement_map[movement_map != 0], [5, 95]) if np.any(movement_map != 0) else (-1, 1)
        im_mov = ax1.imshow(movement_display, cmap=viz_config['movement_cmap'],
                           alpha=0.5, vmin=vmin, vmax=vmax)

        ax1.plot(comparison_contour[:, 1], comparison_contour[:, 0],
                'orange', linewidth=2, linestyle='--', label='Previous Edge')
        ax1.plot(current_contour[:, 1], current_contour[:, 0],
                'white', linewidth=2, label='Current Edge')
        ax1.scatter(points[:, 1], points[:, 0], c='lime', s=50, zorder=5,
                   label='Sampling Points')

        cbar = plt.colorbar(im_mov, ax=ax1)
        cbar.set_label('Displacement (pixels)', fontsize=10)

        ax1.set_title(f'Frame {frame_idx}: Edge Transition\n{movement_type.capitalize()} (score={movement_score:.2f})',
                     fontsize=12)
        ax1.legend(loc='upper right')
        ax1.axis('off')

        # Right: Mask comparison
        rgb_image = np.zeros((*current_mask.shape, 3))
        rgb_image[:, :, 0] = comparison_mask  # Red for previous
        rgb_image[:, :, 2] = current_mask     # Blue for current

        ax2.imshow(rgb_image)
        ax2.plot(comparison_contour[:, 1], comparison_contour[:, 0],
                'red', linewidth=1, alpha=0.7, label='Previous')
        ax2.plot(current_contour[:, 1], current_contour[:, 0],
                'cyan', linewidth=1, alpha=0.7, label='Current')
        ax2.set_title('Mask Comparison\n(Red=Previous, Blue=Current, Purple=Overlap)', fontsize=12)
        ax2.legend()
        ax2.axis('off')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'edge_transition_plots',
                                  f'frame_{frame_idx:04d}_edge_transition.png')
        plt.savefig(output_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
        plt.close()

    # Sampling regions plot
    if viz_config['save_sampling_plots']:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(current_image, cmap='gray')
        ax.plot(current_contour[:, 1], current_contour[:, 0], 'r-', linewidth=2, label='Cell Edge')

        depth = config['depth']
        width = config['width']
        sampling_freq = viz_config['plot_every_nth_sampling_point']

        for i in range(0, len(points), sampling_freq):
            p = points[i]
            normal = normals[i]
            status = point_status[i]

            # Determine color based on status
            if status == 0:
                continue  # Skip excluded endpoints
            elif status == 1:
                color = 'lime'
                label = 'Valid' if i == 0 else ''
            elif status == 2:
                color = 'blue'
                label = 'Rejected' if not any(point_status[:i] == 2) else ''
            else:  # status == 3
                color = 'yellow'
                label = 'Rotated' if not any(point_status[:i] == 3) else ''

            end_point = p + depth * normal
            perp = np.array([-normal[1], normal[0]])

            corner1 = p + width/2 * perp
            corner2 = p - width/2 * perp
            corner3 = end_point - width/2 * perp
            corner4 = end_point + width/2 * perp

            rect = plt.Polygon([corner1[::-1], corner4[::-1], corner3[::-1], corner2[::-1]],
                             fill=False, edgecolor=color, linewidth=1.5, label=label)
            ax.add_patch(rect)
            ax.plot(p[1], p[0], 'o', color=color, markersize=5)

        ax.set_title(f'Frame {frame_idx}: Sampling Regions', fontsize=12)
        ax.legend()
        ax.axis('off')

        output_path = os.path.join(output_dir, 'frame_sampling_plots',
                                  f'frame_{frame_idx:04d}_sampling.png')
        plt.savefig(output_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
        plt.close()

    # Intensity plots
    if viz_config['save_intensity_plots']:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Intensity profile
        valid_intensities = intensities[valid_points]
        valid_indices = np.where(valid_points)[0]

        if len(valid_intensities) > 0:
            ax1.plot(valid_indices, valid_intensities, 'o-', color='steelblue', linewidth=2)
            ax1.set_xlabel('Point Index', fontsize=10)
            ax1.set_ylabel('Intensity', fontsize=10)
            ax1.set_title('Intensity Profile', fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Intensity distribution
            ax2.hist(valid_intensities, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Intensity', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            ax2.set_title(f'Intensity Distribution (n={len(valid_intensities)})', fontsize=11)
            ax2.grid(True, alpha=0.3)

        # Intensity overlay
        if viz_config['save_intensity_overlay_plots']:
            ax3.imshow(current_image, cmap='gray', alpha=0.7)
            ax3.plot(current_contour[:, 1], current_contour[:, 0], 'w-', linewidth=1)

            if np.any(valid_points):
                scatter = ax3.scatter(points[valid_points, 1], points[valid_points, 0],
                                    c=intensities[valid_points], cmap=viz_config['intensity_cmap'],
                                    s=100, edgecolors='white', linewidth=1)
                plt.colorbar(scatter, ax=ax3, label='Intensity')

            # Show rejected points
            rejected = point_status == 2
            if np.any(rejected):
                ax3.scatter(points[rejected, 1], points[rejected, 0],
                          marker='x', c='gray', s=50, alpha=0.5)

            ax3.set_title('Intensity Overlay', fontsize=11)
            ax3.axis('off')

        # Summary statistics
        ax4.axis('off')
        if len(valid_intensities) > 0:
            stats_text = (
                f'Intensity Statistics\n\n'
                f'Valid Points: {len(valid_intensities)}\n'
                f'Mean: {np.mean(valid_intensities):.2f}\n'
                f'Std: {np.std(valid_intensities):.2f}\n'
                f'Min: {np.min(valid_intensities):.2f}\n'
                f'Max: {np.max(valid_intensities):.2f}\n'
                f'Median: {np.median(valid_intensities):.2f}'
            )
            ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    family='monospace')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'frame_intensity_plots',
                                  f'frame_{frame_idx:04d}_intensity.png')
        plt.savefig(output_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
        plt.close()

    # Movement plots
    if viz_config['save_movement_plots']:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Movement profile
        valid_movements = local_movement_scores[valid_points]
        valid_indices = np.where(valid_points)[0]

        if len(valid_movements) > 0:
            ax1.plot(valid_indices, valid_movements, 'o-', color='steelblue', linewidth=2)
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.set_xlabel('Point Index', fontsize=10)
            ax1.set_ylabel('Displacement (pixels)', fontsize=10)
            ax1.set_title('Displacement Profile', fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Movement distribution
            ax2.hist(valid_movements, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Displacement (pixels)', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            ax2.set_title(f'Displacement Distribution (n={len(valid_movements)})', fontsize=11)
            ax2.grid(True, alpha=0.3)

            # Movement vs intensity scatter
            valid_intensities = intensities[valid_points]
            ax3.scatter(valid_movements, valid_intensities, alpha=0.6, s=50, c='steelblue')
            ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax3.set_xlabel('Displacement (pixels)', fontsize=10)
            ax3.set_ylabel('Intensity', fontsize=10)
            ax3.set_title('Displacement vs Intensity', fontsize=11)
            ax3.grid(True, alpha=0.3)

        # Movement map
        ax4.imshow(current_image, cmap='gray', alpha=0.7)

        movement_display = np.ma.masked_where(movement_map == 0, movement_map)
        if np.any(movement_map != 0):
            vmin, vmax = np.percentile(movement_map[movement_map != 0], [5, 95])
            im = ax4.imshow(movement_display, cmap=viz_config['movement_cmap'],
                          alpha=0.5, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax4, label='Displacement (pixels)')

        ax4.plot(current_contour[:, 1], current_contour[:, 0], 'w-', linewidth=1)
        ax4.scatter(points[:, 1], points[:, 0], c='lime', s=30, alpha=0.7)
        ax4.set_title(f'Movement Map\n{movement_type.capitalize()}', fontsize=11)
        ax4.axis('off')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'frame_movement_plots',
                                  f'frame_{frame_idx:04d}_movement.png')
        plt.savefig(output_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
        plt.close()

        # Movement overlay
        if viz_config['save_movement_overlay_plots']:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(current_image, cmap='gray', alpha=0.7)
            ax.plot(current_contour[:, 1], current_contour[:, 0], 'w-', linewidth=1)

            # Color points by movement direction
            # Negative displacement (up) = extending, Positive (down) = retracting
            extending_mask = (local_movement_scores < 0) & valid_points
            retracting_mask = (local_movement_scores > 0) & valid_points
            stable_mask = (local_movement_scores == 0) & valid_points

            if np.any(extending_mask):
                ax.scatter(points[extending_mask, 1], points[extending_mask, 0],
                          c='red', s=100, label='Extending', edgecolors='white', linewidth=1)
            if np.any(retracting_mask):
                ax.scatter(points[retracting_mask, 1], points[retracting_mask, 0],
                          c='blue', s=100, label='Retracting', edgecolors='white', linewidth=1)
            if np.any(stable_mask):
                ax.scatter(points[stable_mask, 1], points[stable_mask, 0],
                          c='gray', s=100, label='Stable', edgecolors='white', linewidth=1)

            # Show rejected points
            rejected = point_status == 2
            if np.any(rejected):
                ax.scatter(points[rejected, 1], points[rejected, 0],
                          marker='x', c='gray', s=50, alpha=0.5)

            ax.set_title(f'Frame {frame_idx}: Movement Overlay', fontsize=12)
            ax.legend()
            ax.axis('off')

            output_path = os.path.join(output_dir, 'frame_movement_plots',
                                      f'frame_{frame_idx:04d}_movement_overlay.png')
            plt.savefig(output_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
            plt.close()

# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================

def save_results_csv(all_results, movement_scores, movement_types, output_dir):
    """Save detailed results to CSV file."""
    data = []

    for frame_idx, results in enumerate(all_results):
        if 'error' in results:
            continue

        points = results['points']
        intensities = results['intensities']
        valid_points = results['valid_points']
        point_status = results['point_status']
        local_movement_scores = results['local_movement_scores']
        movement_type = results['movement_type']
        movement_score = results['movement_score']

        for i in range(len(points)):
            status_names = {0: 'excluded', 1: 'valid', 2: 'rejected', 3: 'rotated_valid'}

            data.append({
                'frame_transition': frame_idx,
                'point_index': i,
                'x_position': points[i, 1],
                'y_position': points[i, 0],
                'intensity': intensities[i],
                'local_movement_score': local_movement_scores[i],
                'is_valid': valid_points[i],
                'point_status': status_names.get(point_status[i], 'unknown'),
                'frame_movement_type': movement_type,
                'frame_movement_score': movement_score
            })

    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, 'detailed_movement_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved detailed results to: {output_path}")

def save_summary_stats(summary_stats, output_dir):
    """Save summary statistics to JSON file."""
    # Add timestamp and configuration
    summary_stats['analysis_date'] = datetime.datetime.now().isoformat()
    summary_stats['configuration'] = CONFIG
    summary_stats['method'] = 'vertical_displacement_xaxis_full'
    summary_stats['sampling'] = 'x_axis'

    output_path = os.path.join(output_dir, 'movement_summary_statistics.json')
    with open(output_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved summary statistics to: {output_path}")

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def main():
    """Main analysis function."""
    print("\n" + "="*60)
    print("X-AXIS FULL VERTICAL DISPLACEMENT ANALYSIS")
    print("="*60)
    print(f"Method: vertical_displacement_xaxis_full")
    print(f"Sampling: x_axis")
    print(f"Temporal direction: {CONFIG['temporal_direction']}")
    print("="*60 + "\n")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    images, masks = load_data(IMAGE_STACK_PATH, MASK_STACK_PATH)

    # Initialize storage
    all_results = []
    all_movements = []
    all_intensities = []
    all_valid_points = []
    movement_scores = []
    movement_types = []

    # Process frame pairs
    print(f"\nProcessing {images.shape[0]-1} frame transitions...")

    temporal_direction = CONFIG['temporal_direction']

    for i in range(images.shape[0] - 1):
        if temporal_direction == 'past':
            # Compare frame i with i+1 (past movement affects current intensity)
            current_idx = i + 1
            comparison_idx = i
        else:  # 'future'
            # Compare frame i with i+1 (current intensity predicts future movement)
            current_idx = i
            comparison_idx = i + 1

        current_image = images[current_idx]
        current_mask = masks[current_idx]
        comparison_image = images[comparison_idx]
        comparison_mask = masks[comparison_idx]

        # Analyze frame pair
        results = analyze_frame_pair(current_image, current_mask, comparison_mask,
                                    CONFIG, temporal_direction)

        if 'error' not in results:
            all_results.append(results)
            all_movements.append(results['local_movement_scores'])
            all_intensities.append(results['intensities'])
            all_valid_points.append(results['valid_points'])
            movement_scores.append(results['movement_score'])
            movement_types.append(results['movement_type'])

            # Save frame visualizations
            if OUTPUT_CONFIG['save_frame_plots']:
                save_frame_visualizations(i, results, current_image, comparison_image,
                                        current_mask, comparison_mask,
                                        OUTPUT_DIR, VIZ_CONFIG, CONFIG)

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{images.shape[0]-1} frame transitions...")

    print(f"Completed processing {len(all_results)} frame transitions\n")

    # Calculate statistics
    processed_pairs = len(all_results)
    total_points = sum(len(m) for m in all_movements)
    valid_measurements = sum(np.sum(v) for v in all_valid_points)

    extending_count = sum(1 for t in movement_types if t == 'extending')
    retracting_count = sum(1 for t in movement_types if t == 'retracting')
    stable_count = sum(1 for t in movement_types if t == 'stable')

    # Create correlation plots
    if OUTPUT_CONFIG['save_correlation_plots'] and len(all_movements) > 0:
        print("Creating correlation plots...")
        correlation_stats = create_movement_correlation_plot(
            all_movements, all_intensities, all_valid_points,
            movement_types, OUTPUT_DIR, VIZ_CONFIG, temporal_direction)

        create_movement_summary_plot(movement_scores, movement_types,
                                    OUTPUT_DIR, VIZ_CONFIG)
    else:
        correlation_stats = {
            'r_squared': None,
            'p_value': None,
            'sample_size': 0,
            'slope': None
        }

    # Compile summary statistics
    summary_stats = {
        'total_frames': int(images.shape[0]),
        'total_transitions': int(images.shape[0] - 1),
        'processed_transitions': int(processed_pairs),
        'total_sampled_points': int(total_points),
        'valid_measurements': int(valid_measurements),
        'valid_measurement_percentage': float((valid_measurements / total_points * 100) if total_points > 0 else 0),
        'method': 'vertical_displacement_xaxis_full',
        'sampling': 'x_axis',
        'temporal_direction': temporal_direction,
        'movement_statistics': {
            'extending_transitions': int(extending_count),
            'retracting_transitions': int(retracting_count),
            'stable_transitions': int(stable_count),
            'average_movement_score': float(np.mean(movement_scores)) if movement_scores else 0.0,
            'movement_score_std': float(np.std(movement_scores)) if movement_scores else 0.0
        },
        'correlation_analysis': correlation_stats
    }

    # Save results
    print("\nSaving results...")

    if OUTPUT_CONFIG['save_detailed_csv']:
        save_results_csv(all_results, movement_scores, movement_types, OUTPUT_DIR)

    if OUTPUT_CONFIG['save_summary_json']:
        save_summary_stats(summary_stats, OUTPUT_DIR)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Method: vertical_displacement_xaxis_full (processes every x-pixel)")
    print(f"Sampling: x_axis ({CONFIG['n_points']} points)")
    print(f"Processed transitions: {processed_pairs}/{images.shape[0]-1}")
    print(f"Valid measurements: {valid_measurements}/{total_points} ({summary_stats['valid_measurement_percentage']:.1f}%)")
    print(f"\nMovement distribution:")
    total_transitions = len(movement_types) if movement_types else 1
    print(f"  - Extending: {extending_count} ({extending_count/total_transitions*100:.1f}%)")
    print(f"  - Retracting: {retracting_count} ({retracting_count/total_transitions*100:.1f}%)")
    print(f"  - Stable: {stable_count} ({stable_count/total_transitions*100:.1f}%)")

    if correlation_stats and correlation_stats.get('r_squared') is not None:
        print(f"\nMovement-Intensity Correlation R²: {correlation_stats['r_squared']:.3f}")
        print(f"P-value: {correlation_stats['p_value']:.3e}")
    else:
        print("\nInsufficient data for correlation analysis")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    if OUTPUT_CONFIG['save_summary_json']:
        print("- movement_summary_statistics.json: Analysis summary")
    if OUTPUT_CONFIG['save_detailed_csv']:
        print("- detailed_movement_results.csv: Point-by-point data")
    if OUTPUT_CONFIG['save_correlation_plots']:
        print("- movement_intensity_correlation.png: Main correlation plot")
        print("- movement_summary.png: Movement over time")
    if OUTPUT_CONFIG['save_frame_plots']:
        print("- edge_transition_plots/: Edge overlays with movement")
        print("- frame_sampling_plots/: Sampling region visualizations")
        print("- frame_intensity_plots/: Intensity analysis")
        print("- frame_movement_plots/: Movement analysis")

    print("\n" + "="*60)
    print("Method Details:")
    print("  → Processes every x-pixel position across cell")
    print("  → Tracks uppermost (minimum y) edge point at each x")
    print("  → Creates complete displacement profile")
    print("  → No mask differences - pure contour-based")
    print("  → Handles cells that curve back on themselves")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
