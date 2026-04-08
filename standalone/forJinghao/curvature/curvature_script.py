#!/usr/bin/env python3
"""
Standalone Cell Curvature Analysis Script

This script analyzes the relationship between cell edge curvature and PIEZO1 protein
intensity from TIRF microscopy recordings. It processes image and mask stacks to:
1. Detect cell edges
2. Sample points along the edge
3. Measure curvature at each point
4. Measure PIEZO1 intensity in regions extending into the cell
5. Generate correlation plots and export results

Features:
- Three sampling methods: standard (arc-length), x_axis, y_axis
- Comprehensive output control (summary-only, custom export options)
- Iterative parameter optimization mode for finding best analysis settings

Author: George Dickinson
Date: 05/26/2025
Enhanced: Added Y-axis sampling, output control, and iterative parameter optimization

ITERATIVE MODE USAGE:
====================
To enable parameter optimization, set ITERATIVE_MODE['enabled'] = True
The script will test all combinations of parameters defined in ITERATIVE_PARAMS
and export results to a CSV file for easy comparison.

Example iterative configuration:
ITERATIVE_PARAMS = {
    'n_points': [8, 12, 16],
    'depth': [150, 200, 250],
    'sampling_method': ['standard', 'x_axis', 'y_axis'],
}

This will test 3×3×3 = 27 parameter combinations and save results showing
which settings produce the best correlations and measurement success rates.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from skimage import measure, draw
from scipy import stats
import json
import datetime
from pathlib import Path

# =============================================================================
# ITERATIVE MODE CONFIGURATION
# =============================================================================

# Iterative mode settings
ITERATIVE_MODE = {
    'enabled': False,                    # Set to True to enable iterative parameter testing
    'output_file': 'parameter_optimization_results.csv',  # Output file for all iterations
    'suppress_individual_outputs': True,  # Disable individual file outputs during iteration
    'show_progress': True,               # Show progress during iterations
}

# Parameter ranges for iterative testing
# Each parameter can be a single value or a list of values to test
ITERATIVE_PARAMS = {
    'n_points': [8, 12, 16, 20],                    # Number of sampling points
    'depth': [150, 200, 250, 300],                  # Sampling rectangle depth
    'width': [50, 75, 100],                         # Sampling rectangle width
    'min_cell_coverage': [0.8],     # Minimum cell coverage
    'sampling_method': ['standard'],  # Sampling methods
    'try_rotation': [True],                   # Rotation fallback
    'exclude_endpoints': [True],              # Endpoint exclusion
}

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Input file paths
#IMAGE_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/test_data/151_2019_07_29_TIRF_mKera_scratch_1_MMStack_Pos7_piezo1.tif"  # TIRF microscopy images
#MASK_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/test_data/151_2019_07_29_TIRF_mKera_scratch_Pos7_DIC_Mask_test.tif"    # Binary cell masks

IMAGE_STACK_PATH = "//Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/simulated_data/tirf_simulation_output_convex/synthetic_cell_convex_clustered_images.tif"  # TIRF microscopy images
MASK_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/simulated_data/tirf_simulation_output_convex/synthetic_cell_convex_clustered_masks.tif"    # Binary cell masks


# Output directory
#OUTPUT_DIR = "curvature_analysis_results"
OUTPUT_DIR = "curvature_analysis_results_simulated-convex"

# Analysis parameters
CONFIG = {
    'n_points': 12,              # Number of points to sample along cell edge
    'depth': 200,                  # Depth of sampling rectangle (pixels into cell)
    'width': 75,                   # Width of sampling rectangle (pixels)
    'min_cell_coverage': 0.8,     # Minimum fraction of rectangle inside cell
    'try_rotation': True,        # Try rotating rejected sampling regions 180°
    'exclude_endpoints': True,   # Exclude first and last edge points
    'sampling_method': 'y_axis'  # 'standard' (arc-length), 'x_axis' (x-intervals), or 'y_axis' (y-intervals)
}

# Output control parameters
OUTPUT_CONFIG = {
    'save_detailed_csv': True,        # Save detailed_results.csv with all point data
    'save_summary_json': True,        # Save summary_statistics.json (always recommended)
    'save_correlation_plots': True,   # Save main correlation plots
    'save_frame_plots': True,         # Save individual frame analysis plots
    'summary_only': False,            # If True, only save summary_statistics.json (overrides other settings)
}

# Visualization parameters
VIZ_CONFIG = {
    'curvature_cmap': 'coolwarm',  # Colormap for curvature visualization
    'intensity_cmap': 'viridis',   # Colormap for intensity visualization
    'marker_size': 30,             # Size of scatter plot markers
    'figure_dpi': 150,            # DPI for saved figures

    # Frame-by-frame visualization options
    'save_frame_plots': True,     # Save individual frame analysis plots
    'save_sampling_plots': True,  # Save sampling region visualizations
    'save_intensity_plots': True, # Save intensity distribution plots
    'save_curvature_plots': True, # Save curvature distribution plots
    'plot_every_nth_frame': 10,    # Save plots for every Nth frame (1 = all frames)

    # Overlay visualization options
    'save_curvature_overlay_plots': True,  # Save curvature sign overlay on cell images
    'save_intensity_overlay_plots': True,  # Save intensity overlay on cell images

    # Sampling visualization options
    'plot_every_nth_sampling_point': 1,  # Show every Nth sampling rectangle (1 = all)

    # Additional correlation plots
    'save_sign_correlation': True # Save correlation plot for curvature sign
}

# =============================================================================
# ITERATIVE MODE FUNCTIONS
# =============================================================================

def generate_parameter_combinations():
    """
    Generate all combinations of parameters for iterative testing.

    Returns:
    --------
    combinations : list
        List of dictionaries, each containing one parameter combination
    """
    import itertools

    # Get parameter names and values
    param_names = []
    param_values = []

    for param_name, values in ITERATIVE_PARAMS.items():
        param_names.append(param_name)
        # Ensure values is a list
        if not isinstance(values, list):
            values = [values]
        param_values.append(values)

    # Generate all combinations
    combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)

    return combinations

def run_single_iteration(images, masks, base_config, iteration_params, iteration_num, total_iterations):
    """
    Run analysis for a single parameter combination.

    Parameters:
    -----------
    images : ndarray
        Image stack
    masks : ndarray
        Mask stack
    base_config : dict
        Base configuration parameters
    iteration_params : dict
        Parameters for this iteration
    iteration_num : int
        Current iteration number (1-indexed)
    total_iterations : int
        Total number of iterations

    Returns:
    --------
    results : dict
        Results from this iteration including parameters and statistics
    """
    if ITERATIVE_MODE['show_progress']:
        param_str = ", ".join([f"{k}={v}" for k, v in iteration_params.items()])
        print(f"\nIteration {iteration_num}/{total_iterations}: {param_str}")

    # Create configuration for this iteration
    config = base_config.copy()
    config.update(iteration_params)

    # Process frames
    all_results = []
    all_curvatures = []
    all_intensities = []
    all_valid_points = []
    processed_frames = 0

    for frame_idx in range(images.shape[0]):
        image = images[frame_idx]
        mask = masks[frame_idx]

        # Analyze frame
        frame_results = analyze_frame(image, mask, config)
        all_results.append(frame_results)

        if 'error' not in frame_results:
            all_curvatures.append(frame_results['curvatures'])
            all_intensities.append(frame_results['intensities'])
            all_valid_points.append(frame_results['valid_points'])
            processed_frames += 1

    # Calculate statistics without generating plots
    correlation_stats = None
    sign_correlation_stats = None

    if processed_frames > 0:
        # Combine data from all frames
        combined_curvatures = []
        combined_intensities = []
        combined_sign_curvatures = []

        for i in range(len(all_curvatures)):
            sign_curv, mag_curv, norm_curv = all_curvatures[i]
            intensities = all_intensities[i]
            valid_points = all_valid_points[i]

            # Use normalized curvature and filter for valid points
            valid_curv = norm_curv[valid_points]
            valid_int = intensities[valid_points]
            valid_sign_curv = sign_curv[valid_points]

            combined_curvatures.extend(valid_curv)
            combined_intensities.extend(valid_int)
            combined_sign_curvatures.extend(valid_sign_curv)

        # Convert to numpy arrays
        combined_curvatures = np.array(combined_curvatures)
        combined_intensities = np.array(combined_intensities)
        combined_sign_curvatures = np.array(combined_sign_curvatures)

        # Calculate correlation statistics
        if len(combined_curvatures) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_curvatures, combined_intensities)
            correlation_stats = {
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'sample_size': int(len(combined_curvatures)),
                'slope': float(slope)
            }
        else:
            correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': int(len(combined_curvatures)),
                'slope': None
            }

        # Calculate sign correlation statistics
        if len(combined_sign_curvatures) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_sign_curvatures, combined_intensities)
            sign_correlation_stats = {
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'sample_size': int(len(combined_sign_curvatures)),
                'slope': float(slope)
            }
        else:
            sign_correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': int(len(combined_sign_curvatures)),
                'slope': None
            }

    # Calculate summary statistics
    total_points = sum(len(valid) for valid in all_valid_points) if all_valid_points else 0
    valid_measurements = sum(np.sum(valid) for valid in all_valid_points) if all_valid_points else 0

    # Compile results
    iteration_results = {
        # Parameters used
        'iteration': iteration_num,
        **iteration_params,

        # Analysis results
        'total_frames': images.shape[0],
        'processed_frames': processed_frames,
        'total_sampled_points': total_points,
        'valid_measurements': valid_measurements,
        'valid_measurement_percentage': (valid_measurements / total_points * 100) if total_points > 0 else 0,

        # Correlation statistics
        'correlation_r_squared': correlation_stats['r_squared'] if correlation_stats else None,
        'correlation_p_value': correlation_stats['p_value'] if correlation_stats else None,
        'correlation_sample_size': correlation_stats['sample_size'] if correlation_stats else None,
        'correlation_slope': correlation_stats['slope'] if correlation_stats else None,

        # Sign correlation statistics
        'sign_correlation_r_squared': sign_correlation_stats['r_squared'] if sign_correlation_stats else None,
        'sign_correlation_p_value': sign_correlation_stats['p_value'] if sign_correlation_stats else None,
        'sign_correlation_sample_size': sign_correlation_stats['sample_size'] if sign_correlation_stats else None,
        'sign_correlation_slope': sign_correlation_stats['slope'] if sign_correlation_stats else None,

        # Analysis timestamp
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return iteration_results

def save_iterative_results(all_iteration_results, output_file):
    """
    Save results from all iterations to a CSV file.

    Parameters:
    -----------
    all_iteration_results : list
        List of dictionaries containing results from each iteration
    output_file : str
        Path to output CSV file
    """
    df = pd.DataFrame(all_iteration_results)

    # Reorder columns for better readability
    param_columns = list(ITERATIVE_PARAMS.keys())
    result_columns = [col for col in df.columns if col not in param_columns and col != 'iteration']

    column_order = ['iteration'] + param_columns + sorted(result_columns)
    df = df[column_order]

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nIterative results saved to: {output_file}")
    print(f"Total parameter combinations tested: {len(all_iteration_results)}")

    # Print best results summary
    if len(df) > 0:
        print("\n" + "="*60)
        print("BEST PARAMETER COMBINATIONS")
        print("="*60)

        # Best R-squared for correlation
        valid_correlations = df[df['correlation_r_squared'].notna()]
        if len(valid_correlations) > 0:
            best_corr_idx = valid_correlations['correlation_r_squared'].idxmax()
            best_corr = df.loc[best_corr_idx]
            print(f"\nBest Normalized Curvature Correlation (R² = {best_corr['correlation_r_squared']:.4f}):")
            for param in param_columns:
                print(f"  {param}: {best_corr[param]}")

        # Best R-squared for sign correlation
        valid_sign_correlations = df[df['sign_correlation_r_squared'].notna()]
        if len(valid_sign_correlations) > 0:
            best_sign_idx = valid_sign_correlations['sign_correlation_r_squared'].idxmax()
            best_sign = df.loc[best_sign_idx]
            print(f"\nBest Sign Curvature Correlation (R² = {best_sign['sign_correlation_r_squared']:.4f}):")
            for param in param_columns:
                print(f"  {param}: {best_sign[param]}")

        # Highest valid measurement percentage
        best_valid_idx = df['valid_measurement_percentage'].idxmax()
        best_valid = df.loc[best_valid_idx]
        print(f"\nHighest Valid Measurement Rate ({best_valid['valid_measurement_percentage']:.1f}%):")
        for param in param_columns:
            print(f"  {param}: {best_valid[param]}")

def run_iterative_analysis():
    """
    Run the complete iterative parameter optimization analysis.
    """
    print("="*60)
    print("ITERATIVE PARAMETER OPTIMIZATION")
    print("="*60)

    # Generate parameter combinations
    combinations = generate_parameter_combinations()
    total_combinations = len(combinations)

    print(f"Generated {total_combinations} parameter combinations to test")
    print(f"Parameters being varied: {list(ITERATIVE_PARAMS.keys())}")

    # Load image stacks once
    try:
        images, masks = load_tiff_stack(IMAGE_STACK_PATH, MASK_STACK_PATH)
        print(f"Loaded {images.shape[0]} frames for analysis")
    except Exception as e:
        print(f"Error loading image stacks: {e}")
        return

    # Run iterations
    all_results = []

    for i, params in enumerate(combinations):
        try:
            iteration_results = run_single_iteration(
                images, masks, CONFIG, params, i+1, total_combinations
            )
            all_results.append(iteration_results)

            if ITERATIVE_MODE['show_progress']:
                # Show brief results
                r2 = iteration_results.get('correlation_r_squared')
                valid_pct = iteration_results.get('valid_measurement_percentage', 0)
                r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
                print(f"  → R² = {r2_str}, Valid = {valid_pct:.1f}%")

        except Exception as e:
            print(f"  Error in iteration {i+1}: {e}")
            # Add failed iteration with error info
            error_result = {
                'iteration': i+1,
                **params,
                'error': str(e),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_results.append(error_result)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, ITERATIVE_MODE['output_file'])
    save_iterative_results(all_results, output_path)

# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def load_tiff_stack(image_path, mask_path):
    """
    Load TIFF stacks of microscope images and binary masks.

    Parameters:
    -----------
    image_path : str
        Path to microscope image stack
    mask_path : str
        Path to binary mask stack

    Returns:
    --------
    images : ndarray
        Array of microscope images
    masks : ndarray
        Array of binary masks
    """
    print(f"Loading image stack: {image_path}")
    images = tifffile.imread(image_path)

    print(f"Loading mask stack: {mask_path}")
    masks = tifffile.imread(mask_path)

    # Ensure masks are binary (0 = background, 1 = cell)
    masks = (masks > 0).astype(np.uint8)

    # Handle single image case
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    print(f"Loaded {images.shape[0]} frames of size {images.shape[1]}x{images.shape[2]}")
    return images, masks

def detect_cell_edge(mask):
    """
    Detect the cell edge from a binary mask.

    Parameters:
    -----------
    mask : ndarray
        Binary mask where 1 = cell, 0 = background

    Returns:
    --------
    contour : ndarray
        Array of (y, x) coordinates defining the cell edge
    """
    # Find contours at the 0.5 level (edge of binary mask)
    contours = measure.find_contours(mask, 0.5)

    # Return the longest contour (assumed to be the main cell edge)
    if len(contours) > 0:
        contour = max(contours, key=len)
        return contour
    else:
        return np.array([])

def sample_equidistant_points(contour, n_points=100):
    """
    Sample equidistant points along a contour using arc-length method.

    Parameters:
    -----------
    contour : ndarray
        Cell contour coordinates
    n_points : int
        Number of points to sample

    Returns:
    --------
    sampled_points : ndarray
        Array of equidistant points along the contour
    """
    # Calculate cumulative distance along contour
    distances = [0]
    for i in range(len(contour) - 1):
        dist = np.linalg.norm(contour[i+1] - contour[i])
        distances.append(distances[-1] + dist)

    total_length = distances[-1]
    step_size = total_length / n_points

    # Sample points at regular intervals
    sampled_points = []
    current_length = 0

    i = 0
    sampled_points.append(contour[0])

    while len(sampled_points) < n_points and i < len(contour) - 1:
        dist = np.linalg.norm(contour[i+1] - contour[i])

        if current_length + dist >= step_size:
            # Interpolate to find the exact position
            t = (step_size - current_length) / dist
            next_point = contour[i] + t * (contour[i+1] - contour[i])
            sampled_points.append(next_point)
            current_length = 0
        else:
            current_length += dist
            i += 1

    return np.array(sampled_points)

def sample_x_axis_points(contour, n_points=100):
    """
    Sample points along contour at regular x-coordinate intervals.

    Parameters:
    -----------
    contour : ndarray
        Cell contour coordinates (y, x)
    n_points : int
        Number of points to sample

    Returns:
    --------
    sampled_points : ndarray
        Array of points sampled at regular x intervals
    """
    if len(contour) == 0:
        return np.array([])

    # Find x-coordinate range
    x_min, x_max = np.min(contour[:, 1]), np.max(contour[:, 1])

    # Create regular x intervals
    if n_points <= 1:
        x_positions = [x_min + (x_max - x_min) / 2]
    else:
        x_positions = np.linspace(x_min, x_max, n_points)

    sampled_points = []
    tolerance = (x_max - x_min) / (n_points * 4)  # Adaptive tolerance based on cell width

    for x_pos in x_positions:
        # Find contour points near this x position
        x_distances = np.abs(contour[:, 1] - x_pos)
        x_mask = x_distances <= tolerance

        if np.any(x_mask):
            # Get all contour points at this x position
            candidates = contour[x_mask]

            if len(candidates) == 1:
                sampled_points.append(candidates[0])
            else:
                # If multiple points, take the one closest to the target x
                best_idx = np.argmin(x_distances[x_mask])
                sampled_points.append(candidates[best_idx])
        else:
            # If no points found within tolerance, find the closest point
            closest_idx = np.argmin(x_distances)
            sampled_points.append(contour[closest_idx])

    # Convert to array and ensure we have the right number of points
    sampled_points = np.array(sampled_points)

    # If we have fewer points than requested, try interpolation
    if len(sampled_points) < n_points and len(sampled_points) > 1:
        # Simple interpolation to reach target number
        from scipy.interpolate import interp1d

        # Sort by x coordinate for interpolation
        sort_idx = np.argsort(sampled_points[:, 1])
        sorted_points = sampled_points[sort_idx]

        # Interpolate both y and x coordinates
        x_coords = sorted_points[:, 1]
        y_coords = sorted_points[:, 0]

        # Create interpolation function
        if len(np.unique(x_coords)) > 1:  # Need at least 2 unique x values
            interp_func = interp1d(x_coords, y_coords, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')

            # Generate new x positions
            new_x_positions = np.linspace(x_min, x_max, n_points)
            new_y_positions = interp_func(new_x_positions)

            sampled_points = np.column_stack([new_y_positions, new_x_positions])

    return sampled_points

def sample_y_axis_points(contour, n_points=100):
    """
    Sample points along contour at regular y-coordinate intervals.

    Parameters:
    -----------
    contour : ndarray
        Cell contour coordinates (y, x)
    n_points : int
        Number of points to sample

    Returns:
    --------
    sampled_points : ndarray
        Array of points sampled at regular y intervals
    """
    if len(contour) == 0:
        return np.array([])

    # Find y-coordinate range
    y_min, y_max = np.min(contour[:, 0]), np.max(contour[:, 0])

    # Create regular y intervals
    if n_points <= 1:
        y_positions = [y_min + (y_max - y_min) / 2]
    else:
        y_positions = np.linspace(y_min, y_max, n_points)

    sampled_points = []
    tolerance = (y_max - y_min) / (n_points * 4)  # Adaptive tolerance based on cell height

    for y_pos in y_positions:
        # Find contour points near this y position
        y_distances = np.abs(contour[:, 0] - y_pos)
        y_mask = y_distances <= tolerance

        if np.any(y_mask):
            # Get all contour points at this y position
            candidates = contour[y_mask]

            if len(candidates) == 1:
                sampled_points.append(candidates[0])
            else:
                # If multiple points, take the one closest to the target y
                best_idx = np.argmin(y_distances[y_mask])
                sampled_points.append(candidates[best_idx])
        else:
            # If no points found within tolerance, find the closest point
            closest_idx = np.argmin(y_distances)
            sampled_points.append(contour[closest_idx])

    # Convert to array and ensure we have the right number of points
    sampled_points = np.array(sampled_points)

    # If we have fewer points than requested, try interpolation
    if len(sampled_points) < n_points and len(sampled_points) > 1:
        # Simple interpolation to reach target number
        from scipy.interpolate import interp1d

        # Sort by y coordinate for interpolation
        sort_idx = np.argsort(sampled_points[:, 0])
        sorted_points = sampled_points[sort_idx]

        # Interpolate both x and y coordinates
        y_coords = sorted_points[:, 0]
        x_coords = sorted_points[:, 1]

        # Create interpolation function
        if len(np.unique(y_coords)) > 1:  # Need at least 2 unique y values
            interp_func = interp1d(y_coords, x_coords, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')

            # Generate new y positions
            new_y_positions = np.linspace(y_min, y_max, n_points)
            new_x_positions = interp_func(new_y_positions)

            sampled_points = np.column_stack([new_y_positions, new_x_positions])

    return sampled_points

def measure_curvature(points):
    """
    Measure curvature at each point along the contour.

    Parameters:
    -----------
    points : ndarray
        Array of contour points

    Returns:
    --------
    sign_curvatures : ndarray
        Signed curvature values (+1 for convex, -1 for concave)
    magnitude_curvatures : ndarray
        Magnitude of curvature
    normalized_curvatures : ndarray
        Normalized curvature combining sign and magnitude
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

        # Cross product to determine curvature direction and magnitude
        cross_product = v[0] * w[1] - v[1] * w[0]

        # Normalized cross product as curvature magnitude
        if v_length > 0:
            magnitude = abs(cross_product) / (v_length * v_length)
        else:
            magnitude = 0

        magnitude_curvatures[i] = magnitude

        # Sign of curvature (positive = convex, negative = concave)
        sign_curvatures[i] = 1 if cross_product > 0 else -1

    # Create normalized curvatures (sign * normalized magnitude)
    max_magnitude = np.max(magnitude_curvatures)
    if max_magnitude > 0:
        normalized_curvatures = sign_curvatures * (magnitude_curvatures / max_magnitude)
    else:
        normalized_curvatures = sign_curvatures * 0

    return sign_curvatures, magnitude_curvatures, normalized_curvatures

def calculate_inward_normal(points, mask):
    """
    Calculate inward normal vectors at each point.

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
        # Get adjacent points
        left_idx = (i - 1) % n_points
        right_idx = (i + 1) % n_points

        p = points[i]
        p_left = points[left_idx]
        p_right = points[right_idx]

        # Calculate tangent vector
        tangent = p_right - p_left
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm
        else:
            tangent = np.array([1.0, 0.0])  # Default tangent

        # Calculate normal vectors (perpendicular to tangent)
        normal1 = np.array([-tangent[1], tangent[0]])
        normal2 = -normal1

        # Test which normal points into the cell
        test_point1 = p + 3 * normal1
        test_point2 = p + 3 * normal2

        # Clip to image bounds
        test_point1 = np.clip(test_point1, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])
        test_point2 = np.clip(test_point2, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])

        try:
            tp1_y, tp1_x = int(round(test_point1[0])), int(round(test_point1[1]))
            tp2_y, tp2_x = int(round(test_point2[0])), int(round(test_point2[1]))

            val1 = mask[tp1_y, tp1_x]
            val2 = mask[tp2_y, tp2_x]

            # Choose normal pointing into cell (mask value = 1)
            if val1 == 1:
                normals[i] = normal1
            elif val2 == 1:
                normals[i] = normal2
            else:
                normals[i] = normal2  # Default choice

        except (IndexError, ValueError):
            normals[i] = normal1  # Default fallback

    return normals

def measure_intensity(image, mask, points, normals, depth=20, width=5,
                     min_cell_coverage=0.8, try_rotation=False, exclude_endpoints=False):
    """
    Measure mean intensity within rectangular regions extending from each point.

    Parameters:
    -----------
    image : ndarray
        Intensity image
    mask : ndarray
        Binary mask
    points : ndarray
        Contour points
    normals : ndarray
        Inward normal vectors
    depth : int
        Depth of sampling rectangle
    width : int
        Width of sampling rectangle
    min_cell_coverage : float
        Minimum fraction of rectangle that must be inside cell
    try_rotation : bool
        Try rotating rejected regions 180°
    exclude_endpoints : bool
        Exclude first and last points

    Returns:
    --------
    intensities : ndarray
        Mean intensity values
    valid_points : ndarray
        Boolean array indicating valid measurements
    point_status : ndarray
        Status of each point: 0=excluded, 1=valid, 2=rejected, 3=rotated_valid
    """
    n_points = len(points)
    intensities = np.full(n_points, np.nan)
    valid_points = np.zeros(n_points, dtype=bool)
    point_status = np.zeros(n_points, dtype=int)  # 0=excluded, 1=valid, 2=rejected, 3=rotated_valid
    image_shape = image.shape

    for i in range(n_points):
        # Skip endpoints if requested
        if exclude_endpoints and (i == 0 or i == n_points - 1):
            point_status[i] = 0  # excluded
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

            # Check cell coverage
            total_pixels = len(rr)
            if total_pixels > 0:
                cell_pixels = np.sum(mask[rr, cc])
                cell_coverage = cell_pixels / total_pixels

                if cell_coverage >= min_cell_coverage:
                    intensities[i] = np.mean(image[rr, cc])
                    valid_points[i] = True
                    point_status[i] = 1  # valid
                elif try_rotation:
                    # Try rotating 180°
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
                                point_status[i] = 3  # rotated and valid
                            else:
                                point_status[i] = 2  # rejected even after rotation
                    else:
                        point_status[i] = 2  # rejected (rotation failed)
                else:
                    point_status[i] = 2  # rejected (no rotation attempted)
            else:
                point_status[i] = 2  # rejected (no pixels in rectangle)
        else:
            point_status[i] = 2  # rejected (rectangle out of bounds)

    return intensities, valid_points, point_status

def analyze_frame(image, mask, config):
    """
    Analyze a single frame for curvature-intensity relationships.

    Parameters:
    -----------
    image : ndarray
        Microscope image
    mask : ndarray
        Binary mask
    config : dict
        Analysis configuration parameters

    Returns:
    --------
    results : dict
        Analysis results for this frame
    """
    # Detect cell edge
    contour = detect_cell_edge(mask)
    if len(contour) == 0:
        return {"error": "No contour found"}

    # Sample points along edge using specified method
    sampling_method = config.get('sampling_method', 'standard')

    if sampling_method == 'x_axis':
        points = sample_x_axis_points(contour, config['n_points'])
    elif sampling_method == 'y_axis':
        points = sample_y_axis_points(contour, config['n_points'])
    else:  # default to 'standard'
        points = sample_equidistant_points(contour, config['n_points'])

    if len(points) == 0:
        return {"error": "No sampling points generated"}

    # Measure curvature
    curvatures = measure_curvature(points)

    # Calculate inward normals
    normals = calculate_inward_normal(points, mask)

    # Measure intensity (now returns status information)
    intensities, valid_points, point_status = measure_intensity(
        image, mask, points, normals,
        depth=config['depth'],
        width=config['width'],
        min_cell_coverage=config['min_cell_coverage'],
        try_rotation=config['try_rotation'],
        exclude_endpoints=config['exclude_endpoints']
    )

    # Package results
    results = {
        'points': points,
        'curvatures': curvatures,
        'normals': normals,
        'intensities': intensities,
        'valid_points': valid_points,
        'point_status': point_status,
        'contour': contour,
        'sampling_method': sampling_method
    }

    return results

def create_correlation_plot(all_curvatures, all_intensities, all_valid_points, output_dir, viz_config):
    """
    Create and save curvature-intensity correlation plot.

    Parameters:
    -----------
    all_curvatures : list
        List of curvature arrays from all frames
    all_intensities : list
        List of intensity arrays from all frames
    all_valid_points : list
        List of valid point masks from all frames
    output_dir : str
        Output directory for saving plot
    viz_config : dict
        Visualization configuration
    """
    # Combine data from all frames
    combined_curvatures = []
    combined_intensities = []

    for i in range(len(all_curvatures)):
        sign_curv, mag_curv, norm_curv = all_curvatures[i]
        intensities = all_intensities[i]
        valid_points = all_valid_points[i]

        # Use normalized curvature and filter for valid points
        valid_curv = norm_curv[valid_points]
        valid_int = intensities[valid_points]

        combined_curvatures.extend(valid_curv)
        combined_intensities.extend(valid_int)

    # Convert to numpy arrays
    combined_curvatures = np.array(combined_curvatures)
    combined_intensities = np.array(combined_intensities)

    # Create correlation plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    scatter = ax.scatter(combined_curvatures, combined_intensities,
                        alpha=0.6, s=viz_config['marker_size'],
                        c=combined_curvatures, cmap=viz_config['curvature_cmap'])

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Curvature', fontsize=12)

    # Calculate and plot regression line
    if len(combined_curvatures) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_curvatures, combined_intensities)

        # Create regression line
        x_line = np.linspace(min(combined_curvatures), max(combined_curvatures), 100)
        y_line = slope * x_line + intercept

        ax.plot(x_line, y_line, 'r-', linewidth=2,
                label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add statistics text
        stats_text = f'R² = {r_value**2:.3f}\np-value = {p_value:.3e}\nn = {len(combined_curvatures)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)

        correlation_stats = {
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'sample_size': int(len(combined_curvatures)),
            'slope': float(slope)
        }
    else:
        # Not enough data for correlation
        correlation_stats = {
            'r_squared': None,
            'p_value': None,
            'sample_size': int(len(combined_curvatures)),
            'slope': None
        }

    # Labels and title
    ax.set_xlabel('Normalized Curvature\n(Negative = Concave, Positive = Convex)', fontsize=12)
    ax.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax.set_title('Cell Edge Curvature vs PIEZO1 Intensity\n(All Frames Combined)', fontsize=14, pad=20)

    if len(combined_curvatures) > 1:
        ax.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'curvature_intensity_correlation.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

    print(f"Correlation plot saved to: {plot_path}")

    return correlation_stats

def create_sign_correlation_plot(all_curvatures, all_intensities, all_valid_points, output_dir, viz_config):
    """
    Create and save sign curvature-intensity correlation plot.

    Parameters:
    -----------
    all_curvatures : list
        List of curvature arrays from all frames
    all_intensities : list
        List of intensity arrays from all frames
    all_valid_points : list
        List of valid point masks from all frames
    output_dir : str
        Output directory for saving plot
    viz_config : dict
        Visualization configuration
    """
    # Combine data from all frames using sign curvature
    combined_sign_curvatures = []
    combined_intensities = []

    for i in range(len(all_curvatures)):
        sign_curv, mag_curv, norm_curv = all_curvatures[i]
        intensities = all_intensities[i]
        valid_points = all_valid_points[i]

        # Use sign curvature and filter for valid points
        valid_curv = sign_curv[valid_points]
        valid_int = intensities[valid_points]

        combined_sign_curvatures.extend(valid_curv)
        combined_intensities.extend(valid_int)

    # Convert to numpy arrays
    combined_sign_curvatures = np.array(combined_sign_curvatures)
    combined_intensities = np.array(combined_intensities)

    # Create correlation plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color points by sign: blue for concave (-1), red for convex (+1)
    colors = ['blue' if x < 0 else 'red' for x in combined_sign_curvatures]
    scatter = ax.scatter(combined_sign_curvatures, combined_intensities,
                        alpha=0.6, s=viz_config['marker_size'], c=colors)

    # Calculate and plot regression line
    sign_correlation_stats = {}
    if len(combined_sign_curvatures) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_sign_curvatures, combined_intensities)

        # Create regression line
        x_line = np.linspace(min(combined_sign_curvatures), max(combined_sign_curvatures), 100)
        y_line = slope * x_line + intercept

        ax.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add statistics text
        stats_text = f'R² = {r_value**2:.3f}\np-value = {p_value:.3e}\nn = {len(combined_sign_curvatures)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)

        sign_correlation_stats = {
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'sample_size': int(len(combined_sign_curvatures)),
            'slope': float(slope)
        }
    else:
        sign_correlation_stats = {
            'r_squared': None,
            'p_value': None,
            'sample_size': int(len(combined_sign_curvatures)),
            'slope': None
        }

    # Add custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Concave (negative)'),
                      Patch(facecolor='red', label='Convex (positive)')]
    if len(combined_sign_curvatures) > 1:
        legend_elements.append(plt.Line2D([0], [0], color='k', lw=2,
                                         label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}'))
    ax.legend(handles=legend_elements, loc='best')

    # Labels and title
    ax.set_xlabel('Curvature Sign\n(-1 = Concave, +1 = Convex)', fontsize=12)
    ax.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax.set_title('Cell Edge Curvature Sign vs PIEZO1 Intensity\n(All Frames Combined)', fontsize=14, pad=20)
    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['Concave\n(-1)', 'Convex\n(+1)'])

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'sign_curvature_intensity_correlation.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

    print(f"Sign curvature correlation plot saved to: {plot_path}")

    return sign_correlation_stats

def create_frame_sampling_plot(frame_idx, image, mask, results, output_dir, viz_config):
    """
    Create and save sampling region visualization for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    image : ndarray
        Original microscope image
    mask : ndarray
        Binary mask
    results : dict
        Analysis results for this frame
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create sampling plots directory
    sampling_dir = os.path.join(output_dir, 'frame_sampling_plots')
    os.makedirs(sampling_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Original image with contour and sampling points
    ax1.imshow(image, cmap='gray')

    # Plot contour
    contour = results['contour']
    if len(contour) > 0:
        ax1.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, label='Cell Edge')

    # Plot sampling points
    points = results['points']
    valid_points = results['valid_points']
    point_status = results['point_status']

    # Color code by status
    excluded_mask = (point_status == 0)
    valid_mask = (point_status == 1)
    rejected_mask = (point_status == 2)
    rotated_mask = (point_status == 3)

    if np.any(valid_mask):
        ax1.scatter(points[valid_mask, 1], points[valid_mask, 0],
                   c='lime', s=50, alpha=0.8, label='Valid Points', zorder=5)
    if np.any(rejected_mask):
        ax1.scatter(points[rejected_mask, 1], points[rejected_mask, 0],
                   c='blue', s=50, alpha=0.8, label='Rejected Points', zorder=5)
    if np.any(rotated_mask):
        ax1.scatter(points[rotated_mask, 1], points[rotated_mask, 0],
                   c='yellow', s=50, alpha=0.8, label='Rotated Valid Points', zorder=5)

    # Add sampling method to title
    sampling_method = results.get('sampling_method', 'standard')
    ax1.set_title(f'Frame {frame_idx+1}: Sampling Points ({sampling_method} method)', fontsize=12)
    ax1.legend()
    ax1.axis('off')

    # Right plot: Sampling regions overlay with cell edge
    ax2.imshow(image, cmap='gray', alpha=0.7)

    # Plot cell edge on sampling regions plot too
    if len(contour) > 0:
        ax2.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5, alpha=0.8, label='Cell Edge')

    # Draw sampling rectangles with status-based coloring
    normals = results['normals']
    depth = CONFIG['depth']
    width = CONFIG['width']

    # Use the configurable sampling point frequency
    plot_frequency = viz_config['plot_every_nth_sampling_point']

    for i, (point, normal, status) in enumerate(zip(points, normals, point_status)):
        # Only plot every Nth rectangle and skip excluded points
        if i % plot_frequency == 0 and status != 0:
            # Determine if this is a rotated region
            use_rotated = (status == 3)

            if use_rotated:
                # Use rotated normal for yellow rectangles
                effective_normal = -normal
                color = 'yellow'
                alpha = 0.7
                label_text = 'Rotated Valid'
            elif status == 1:
                # Regular valid rectangle
                effective_normal = normal
                color = 'lime'
                alpha = 0.6
                label_text = 'Valid'
            else:  # status == 2
                # Rejected rectangle
                effective_normal = normal
                color = 'blue'
                alpha = 0.4
                label_text = 'Rejected'

            # Calculate rectangle corners
            end_point = point + depth * effective_normal
            perp = np.array([-effective_normal[1], effective_normal[0]])

            corners = np.array([
                point + width/2 * perp,
                point - width/2 * perp,
                end_point - width/2 * perp,
                end_point + width/2 * perp,
                point + width/2 * perp  # Close the rectangle
            ])

            ax2.plot(corners[:, 1], corners[:, 0], color=color, linewidth=1.5, alpha=alpha)

    # Create custom legend for rectangle colors
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color='r', lw=1.5, label='Cell Edge')]

    if np.any(point_status == 1):
        legend_elements.append(Patch(facecolor='lime', alpha=0.6, label='Valid Regions'))
    if np.any(point_status == 2):
        legend_elements.append(Patch(facecolor='blue', alpha=0.4, label='Rejected Regions'))
    if np.any(point_status == 3):
        legend_elements.append(Patch(facecolor='yellow', alpha=0.7, label='Rotated Valid Regions'))

    ax2.legend(handles=legend_elements, loc='best')

    ax2.set_title(f'Frame {frame_idx+1}: Sampling Regions (every {plot_frequency} shown)', fontsize=12)
    ax2.axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(sampling_dir, f'frame_{frame_idx+1:03d}_sampling.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_intensity_plot(frame_idx, results, output_dir, viz_config):
    """
    Create and save intensity analysis plot for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    results : dict
        Analysis results for this frame
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create intensity plots directory
    intensity_dir = os.path.join(output_dir, 'frame_intensity_plots')
    os.makedirs(intensity_dir, exist_ok=True)

    intensities = results['intensities']
    valid_points = results['valid_points']
    points = results['points']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Intensity around the contour
    point_indices = np.arange(len(points))

    # Create masked array for plotting
    masked_intensities = np.ma.array(intensities, mask=~valid_points)

    ax1.plot(point_indices, masked_intensities, 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(point_indices[valid_points], intensities[valid_points],
               c='blue', s=30, alpha=0.8, label='Valid Measurements')
    ax1.scatter(point_indices[~valid_points], intensities[~valid_points],
               c='red', s=30, alpha=0.8, label='Invalid Measurements')

    ax1.set_xlabel('Point Index (around contour)')
    ax1.set_ylabel('PIEZO1 Intensity')
    ax1.set_title(f'Frame {frame_idx+1}: Intensity Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Intensity histogram
    valid_intensities = intensities[valid_points]
    if len(valid_intensities) > 0:
        ax2.hist(valid_intensities, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(valid_intensities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(valid_intensities):.1f}')
        ax2.axvline(np.median(valid_intensities), color='orange', linestyle='--',
                   label=f'Median: {np.median(valid_intensities):.1f}')

    ax2.set_xlabel('PIEZO1 Intensity')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Frame {frame_idx+1}: Intensity Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(intensity_dir, f'frame_{frame_idx+1:03d}_intensity.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_curvature_plot(frame_idx, results, output_dir, viz_config):
    """
    Create and save curvature analysis plot for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    results : dict
        Analysis results for this frame
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create curvature plots directory
    curvature_dir = os.path.join(output_dir, 'frame_curvature_plots')
    os.makedirs(curvature_dir, exist_ok=True)

    sign_curv, mag_curv, norm_curv = results['curvatures']
    valid_points = results['valid_points']
    points = results['points']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    point_indices = np.arange(len(points))

    # Top left: Sign curvature
    ax1.plot(point_indices, sign_curv, 'b-', linewidth=2)
    ax1.scatter(point_indices[sign_curv > 0], sign_curv[sign_curv > 0],
               c='red', s=30, alpha=0.8, label='Convex (+1)')
    ax1.scatter(point_indices[sign_curv < 0], sign_curv[sign_curv < 0],
               c='blue', s=30, alpha=0.8, label='Concave (-1)')
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Curvature Sign')
    ax1.set_title('Sign Curvature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.2, 1.2)

    # Top right: Magnitude curvature
    ax2.plot(point_indices, mag_curv, 'g-', linewidth=2)
    ax2.scatter(point_indices[valid_points], mag_curv[valid_points],
               c='green', s=30, alpha=0.8, label='Valid Points')
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Curvature Magnitude')
    ax2.set_title('Magnitude Curvature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom left: Normalized curvature
    colors = ['blue' if x < 0 else 'red' for x in norm_curv]
    ax3.plot(point_indices, norm_curv, 'k-', linewidth=2, alpha=0.7)
    ax3.scatter(point_indices, norm_curv, c=colors, s=30, alpha=0.8)
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('Normalized Curvature')
    ax3.set_title('Normalized Curvature')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Bottom right: Curvature distribution
    valid_norm_curv = norm_curv[valid_points]
    if len(valid_norm_curv) > 0:
        ax4.hist(valid_norm_curv, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(np.mean(valid_norm_curv), color='red', linestyle='--',
                   label=f'Mean: {np.mean(valid_norm_curv):.3f}')
        ax4.axvline(0, color='gray', linestyle='-', alpha=0.5, label='Zero')

    ax4.set_xlabel('Normalized Curvature')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Curvature Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Frame {frame_idx+1}: Curvature Analysis', fontsize=16)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(curvature_dir, f'frame_{frame_idx+1:03d}_curvature.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_curvature_overlay_plot(frame_idx, image, mask, results, output_dir, viz_config):
    """
    Create and save curvature sign overlay visualization for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    image : ndarray
        Original microscope image
    mask : ndarray
        Binary mask
    results : dict
        Analysis results for this frame
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create curvature plots directory
    curvature_dir = os.path.join(output_dir, 'frame_curvature_plots')
    os.makedirs(curvature_dir, exist_ok=True)

    sign_curv, mag_curv, norm_curv = results['curvatures']
    points = results['points']
    valid_points = results['valid_points']
    point_status = results['point_status']
    contour = results['contour']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the cell image as background
    ax.imshow(image, cmap='gray', alpha=0.8)

    # Plot cell edge outline
    if len(contour) > 0:
        ax.plot(contour[:, 1], contour[:, 0], 'white', linewidth=2, alpha=0.9, label='Cell Edge')

    # Separate points by curvature sign and validity
    convex_valid = (sign_curv > 0) & (point_status != 0)
    concave_valid = (sign_curv < 0) & (point_status != 0)
    ignored = (point_status == 0) | (point_status == 2)

    # Plot convex points (red)
    if np.any(convex_valid):
        ax.scatter(points[convex_valid, 1], points[convex_valid, 0],
                  c='red', s=80, alpha=0.9, marker='o',
                  label='Convex (+)', edgecolors='white', linewidth=1, zorder=5)

    # Plot concave points (blue)
    if np.any(concave_valid):
        ax.scatter(points[concave_valid, 1], points[concave_valid, 0],
                  c='blue', s=80, alpha=0.9, marker='o',
                  label='Concave (-)', edgecolors='white', linewidth=1, zorder=5)

    # Plot ignored points (fainter)
    if np.any(ignored):
        ax.scatter(points[ignored, 1], points[ignored, 0],
                  c='gray', s=40, alpha=0.4, marker='x',
                  label='Ignored', zorder=4)

    # Formatting
    ax.set_title(f'Frame {frame_idx+1}: Curvature Sign Overlay', fontsize=14, pad=20)
    ax.legend(loc='best', framealpha=0.8)
    ax.axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(curvature_dir, f'frame_{frame_idx+1:03d}_curvature_overlay.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_intensity_overlay_plot(frame_idx, image, mask, results, output_dir, viz_config):
    """
    Create and save intensity overlay visualization for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    image : ndarray
        Original microscope image
    mask : ndarray
        Binary mask
    results : dict
        Analysis results for this frame
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create intensity plots directory
    intensity_dir = os.path.join(output_dir, 'frame_intensity_plots')
    os.makedirs(intensity_dir, exist_ok=True)

    intensities = results['intensities']
    points = results['points']
    valid_points = results['valid_points']
    point_status = results['point_status']
    contour = results['contour']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the cell image as background
    ax.imshow(image, cmap='gray', alpha=0.8)

    # Plot cell edge outline
    if len(contour) > 0:
        ax.plot(contour[:, 1], contour[:, 0], 'white', linewidth=2, alpha=0.9, label='Cell Edge')

    # Separate valid and ignored points
    valid_mask = (point_status == 1) | (point_status == 3)  # valid or rotated_valid
    ignored_mask = (point_status == 0) | (point_status == 2)  # excluded or rejected

    # Plot valid points colored by intensity
    if np.any(valid_mask):
        valid_intensities = intensities[valid_mask]
        valid_coords = points[valid_mask]

        # Create scatter plot with intensity colormap
        scatter = ax.scatter(valid_coords[:, 1], valid_coords[:, 0],
                           c=valid_intensities, s=100, alpha=0.9,
                           cmap=viz_config['intensity_cmap'],
                           edgecolors='white', linewidth=1, zorder=5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('PIEZO1 Intensity', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Plot ignored points (fainter)
    if np.any(ignored_mask):
        ax.scatter(points[ignored_mask, 1], points[ignored_mask, 0],
                  c='gray', s=40, alpha=0.4, marker='x',
                  label='Ignored', zorder=4)

    # Formatting
    ax.set_title(f'Frame {frame_idx+1}: PIEZO1 Intensity Overlay', fontsize=14, pad=20)

    # Create custom legend
    legend_elements = [plt.Line2D([0], [0], color='white', lw=2, label='Cell Edge')]
    if np.any(valid_mask):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='gray', markersize=8,
                                        label='Valid Points (colored by intensity)'))
    if np.any(ignored_mask):
        legend_elements.append(plt.Line2D([0], [0], marker='x', color='gray',
                                        markersize=8, alpha=0.4, linestyle='None',
                                        label='Ignored Points'))

    ax.legend(handles=legend_elements, loc='best', framealpha=0.8)
    ax.axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(intensity_dir, f'frame_{frame_idx+1:03d}_intensity_overlay.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def save_results_csv(all_results, output_dir):
    """
    Save detailed results to CSV file.

    Parameters:
    -----------
    all_results : list
        List of frame analysis results
    output_dir : str
        Output directory
    """
    data_rows = []

    for frame_idx, results in enumerate(all_results):
        if 'error' in results:
            continue

        points = results['points']
        sign_curv, mag_curv, norm_curv = results['curvatures']
        intensities = results['intensities']
        valid_points = results['valid_points']
        point_status = results['point_status']
        sampling_method = results.get('sampling_method', 'standard')

        for i in range(len(points)):
            # Convert point status to descriptive text
            status_map = {0: 'excluded', 1: 'valid', 2: 'rejected', 3: 'rotated_valid'}
            status_text = status_map.get(point_status[i], 'unknown')

            data_rows.append({
                'frame': frame_idx,
                'point_index': i,
                'x_coord': points[i, 1],
                'y_coord': points[i, 0],
                'sign_curvature': sign_curv[i],
                'magnitude_curvature': mag_curv[i],
                'normalized_curvature': norm_curv[i],
                'intensity': intensities[i],
                'valid_measurement': valid_points[i],
                'point_status': status_text,
                'sampling_method': sampling_method
            })

    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"Detailed results saved to: {csv_path}")
    print(f"  - Point status meanings: excluded=endpoint excluded, valid=normal measurement,")
    print(f"    rejected=insufficient cell coverage, rotated_valid=successful after 180° rotation")

def convert_numpy_types(obj):
    """
    Convert NumPy types to Python native types for JSON serialization.

    Parameters:
    -----------
    obj : any
        Object that may contain NumPy types

    Returns:
    --------
    converted_obj : any
        Object with NumPy types converted to Python types
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

def save_summary_stats(summary_stats, output_dir):
    """
    Save summary statistics to JSON file.

    Parameters:
    -----------
    summary_stats : dict
        Summary statistics
    output_dir : str
        Output directory
    """
    # Add timestamp and configuration
    summary_stats['analysis_timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_stats['configuration'] = CONFIG
    summary_stats['output_configuration'] = OUTPUT_CONFIG
    summary_stats['iterative_mode'] = ITERATIVE_MODE

    # Convert NumPy types to Python types for JSON serialization
    serializable_stats = convert_numpy_types(summary_stats)

    json_path = os.path.join(output_dir, 'summary_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(serializable_stats, f, indent=4)

    print(f"Summary statistics saved to: {json_path}")

# =============================================================================
# MAIN ANALYSIS WORKFLOW
# =============================================================================

def main():
    """Main analysis workflow."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if iterative mode is enabled
    if ITERATIVE_MODE['enabled']:
        print(f"Output directory: {OUTPUT_DIR}")
        run_iterative_analysis()
        return

    # Standard single-run analysis
    print("="*60)
    print("CELL CURVATURE ANALYSIS")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load image stacks
    try:
        images, masks = load_tiff_stack(IMAGE_STACK_PATH, MASK_STACK_PATH)
    except Exception as e:
        print(f"Error loading image stacks: {e}")
        return

    # Print sampling method being used
    sampling_method = CONFIG.get('sampling_method', 'standard')
    print(f"Using sampling method: {sampling_method}")

    # Process each frame
    print(f"\nProcessing {images.shape[0]} frames...")
    all_results = []
    all_curvatures = []
    all_intensities = []
    all_valid_points = []

    processed_frames = 0

    for frame_idx in range(images.shape[0]):
        print(f"Processing frame {frame_idx + 1}/{images.shape[0]}...")

        image = images[frame_idx]
        mask = masks[frame_idx]

        # Analyze frame
        results = analyze_frame(image, mask, CONFIG)
        all_results.append(results)

        if 'error' not in results:
            all_curvatures.append(results['curvatures'])
            all_intensities.append(results['intensities'])
            all_valid_points.append(results['valid_points'])
            processed_frames += 1

            # Generate frame-by-frame plots if enabled
            should_plot_this_frame = (frame_idx % VIZ_CONFIG['plot_every_nth_frame'] == 0) or (VIZ_CONFIG['plot_every_nth_frame'] == 1)

            # Check if frame plots are enabled in OUTPUT_CONFIG
            frame_plots_enabled = OUTPUT_CONFIG['save_frame_plots'] and not OUTPUT_CONFIG['summary_only']

            if should_plot_this_frame and frame_plots_enabled:
                if VIZ_CONFIG['save_sampling_plots']:
                    print(f"  Creating sampling plot for frame {frame_idx + 1}...")
                    create_frame_sampling_plot(frame_idx, image, mask, results, OUTPUT_DIR, VIZ_CONFIG)

                if VIZ_CONFIG['save_intensity_plots']:
                    print(f"  Creating intensity plot for frame {frame_idx + 1}...")
                    create_frame_intensity_plot(frame_idx, results, OUTPUT_DIR, VIZ_CONFIG)

                    # Create intensity overlay plot if enabled
                    if VIZ_CONFIG['save_intensity_overlay_plots']:
                        print(f"  Creating intensity overlay plot for frame {frame_idx + 1}...")
                        create_frame_intensity_overlay_plot(frame_idx, image, mask, results, OUTPUT_DIR, VIZ_CONFIG)

                if VIZ_CONFIG['save_curvature_plots']:
                    print(f"  Creating curvature plot for frame {frame_idx + 1}...")
                    create_frame_curvature_plot(frame_idx, results, OUTPUT_DIR, VIZ_CONFIG)

                    # Create curvature overlay plot if enabled
                    if VIZ_CONFIG['save_curvature_overlay_plots']:
                        print(f"  Creating curvature overlay plot for frame {frame_idx + 1}...")
                        create_frame_curvature_overlay_plot(frame_idx, image, mask, results, OUTPUT_DIR, VIZ_CONFIG)

        else:
            print(f"  Warning: {results['error']}")

    print(f"\nSuccessfully processed {processed_frames}/{images.shape[0]} frames")

    if processed_frames == 0:
        print("No frames could be processed. Check your input files and parameters.")
        return

    # Check if summary_only mode is enabled (overrides other settings)
    if OUTPUT_CONFIG['summary_only']:
        print("\nSummary-only mode enabled. Skipping detailed outputs...")
        save_detailed = False
        save_plots = False
    else:
        save_detailed = OUTPUT_CONFIG['save_detailed_csv']
        save_plots = OUTPUT_CONFIG['save_correlation_plots']

    # Generate correlation plot and statistics
    correlation_stats = None
    sign_correlation_stats = None

    if save_plots:
        print("\nGenerating correlation analysis...")
        correlation_stats = create_correlation_plot(all_curvatures, all_intensities,
                                                  all_valid_points, OUTPUT_DIR, VIZ_CONFIG)

        # Generate sign correlation plot if enabled
        if VIZ_CONFIG['save_sign_correlation']:
            print("Generating sign correlation analysis...")
            sign_correlation_stats = create_sign_correlation_plot(all_curvatures, all_intensities,
                                                                all_valid_points, OUTPUT_DIR, VIZ_CONFIG)
    else:
        # Still calculate statistics for summary even if not plotting
        print("\nCalculating correlation statistics...")
        combined_curvatures = []
        combined_intensities = []
        combined_sign_curvatures = []

        for i in range(len(all_curvatures)):
            sign_curv, mag_curv, norm_curv = all_curvatures[i]
            intensities = all_intensities[i]
            valid_points = all_valid_points[i]

            # Use normalized curvature and filter for valid points
            valid_curv = norm_curv[valid_points]
            valid_int = intensities[valid_points]
            valid_sign_curv = sign_curv[valid_points]

            combined_curvatures.extend(valid_curv)
            combined_intensities.extend(valid_int)
            combined_sign_curvatures.extend(valid_sign_curv)

        # Convert to numpy arrays
        combined_curvatures = np.array(combined_curvatures)
        combined_intensities = np.array(combined_intensities)
        combined_sign_curvatures = np.array(combined_sign_curvatures)

        # Calculate correlation statistics without plotting
        if len(combined_curvatures) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_curvatures, combined_intensities)
            correlation_stats = {
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'sample_size': int(len(combined_curvatures)),
                'slope': float(slope)
            }
        else:
            correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': int(len(combined_curvatures)),
                'slope': None
            }

        # Calculate sign correlation statistics
        if len(combined_sign_curvatures) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_sign_curvatures, combined_intensities)
            sign_correlation_stats = {
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'sample_size': int(len(combined_sign_curvatures)),
                'slope': float(slope)
            }
        else:
            sign_correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': int(len(combined_sign_curvatures)),
                'slope': None
            }

    # Calculate summary statistics
    total_points = sum(len(valid) for valid in all_valid_points)
    valid_measurements = sum(np.sum(valid) for valid in all_valid_points)

    summary_stats = {
        'total_frames': int(images.shape[0]),
        'processed_frames': int(processed_frames),
        'total_sampled_points': int(total_points),
        'valid_measurements': int(valid_measurements),
        'valid_measurement_percentage': float((valid_measurements / total_points * 100) if total_points > 0 else 0),
        'sampling_method': sampling_method,
        'correlation_analysis': correlation_stats
    }

    # Add sign correlation stats if available
    if sign_correlation_stats is not None:
        summary_stats['sign_correlation_analysis'] = sign_correlation_stats

    # Save results based on configuration
    print("\nSaving results...")

    if save_detailed and OUTPUT_CONFIG['save_detailed_csv']:
        save_results_csv(all_results, OUTPUT_DIR)
    elif OUTPUT_CONFIG['summary_only']:
        print("Skipping detailed CSV export (summary-only mode)")
    elif not OUTPUT_CONFIG['save_detailed_csv']:
        print("Skipping detailed CSV export (disabled in OUTPUT_CONFIG)")

    if OUTPUT_CONFIG['save_summary_json']:
        save_summary_stats(summary_stats, OUTPUT_DIR)
    else:
        print("Skipping summary statistics export (disabled in OUTPUT_CONFIG)")

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Sampling method: {sampling_method}")
    print(f"Processed frames: {processed_frames}/{images.shape[0]}")
    print(f"Valid measurements: {valid_measurements}/{total_points} ({summary_stats['valid_measurement_percentage']:.1f}%)")

    if correlation_stats['r_squared'] is not None:
        print(f"Correlation R²: {correlation_stats['r_squared']:.3f}")
        print(f"P-value: {correlation_stats['p_value']:.3e}")
        print(f"Sample size: {correlation_stats['sample_size']}")
    else:
        print("Insufficient data for correlation analysis")

    if sign_correlation_stats is not None and sign_correlation_stats['r_squared'] is not None:
        print(f"Sign Correlation R²: {sign_correlation_stats['r_squared']:.3f}")
        print(f"Sign P-value: {sign_correlation_stats['p_value']:.3e}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    # Print what was actually saved based on configuration
    if OUTPUT_CONFIG['summary_only']:
        print("SUMMARY-ONLY MODE:")
        print("- summary_statistics.json: Summary statistics and parameters")
    else:
        if OUTPUT_CONFIG['save_correlation_plots']:
            print("- curvature_intensity_correlation.png: Main correlation plot")
            if VIZ_CONFIG['save_sign_correlation']:
                print("- sign_curvature_intensity_correlation.png: Sign correlation plot")

        if OUTPUT_CONFIG['save_frame_plots'] and any([VIZ_CONFIG['save_sampling_plots'], VIZ_CONFIG['save_intensity_plots'], VIZ_CONFIG['save_curvature_plots']]):
            if VIZ_CONFIG['save_sampling_plots']:
                print("- frame_sampling_plots/: Sampling region visualizations")
                print("  * Colors: lime=valid, blue=rejected, yellow=rotated valid, red line=cell edge")
            if VIZ_CONFIG['save_intensity_plots']:
                print("- frame_intensity_plots/: Intensity analysis plots")
                if VIZ_CONFIG['save_intensity_overlay_plots']:
                    print("  * Includes intensity overlay plots on cell images")
            if VIZ_CONFIG['save_curvature_plots']:
                print("- frame_curvature_plots/: Curvature analysis plots")
                if VIZ_CONFIG['save_curvature_overlay_plots']:
                    print("  * Includes curvature sign overlay plots on cell images")

        if OUTPUT_CONFIG['save_detailed_csv']:
            print("- detailed_results.csv: Point-by-point data (includes point status and sampling method)")

        if OUTPUT_CONFIG['save_summary_json']:
            print("- summary_statistics.json: Summary statistics and parameters")

    # Print frame plotting summary
    if OUTPUT_CONFIG['save_frame_plots'] and not OUTPUT_CONFIG['summary_only'] and any([VIZ_CONFIG['save_sampling_plots'], VIZ_CONFIG['save_intensity_plots'], VIZ_CONFIG['save_curvature_plots']]):
        frames_plotted = len([i for i in range(processed_frames) if i % VIZ_CONFIG['plot_every_nth_frame'] == 0])
        sampling_freq = VIZ_CONFIG['plot_every_nth_sampling_point']
        print(f"Frame plots generated for {frames_plotted} frames (every {VIZ_CONFIG['plot_every_nth_frame']} frames)")
        if VIZ_CONFIG['save_sampling_plots']:
            print(f"Sampling plots show every {sampling_freq} rectangle to avoid clutter")

if __name__ == "__main__":
    main()
