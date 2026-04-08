#!/usr/bin/env python3
"""
Standalone Cell Edge Movement Analysis Script

This script analyzes the relationship between cell edge movement and PIEZO1 protein
intensity from TIRF microscopy recordings. It processes image and mask stacks to:
1. Detect cell edges in consecutive frames
2. Calculate edge movement (extension/retraction)
3. Sample points along current frame edge
4. Measure PIEZO1 intensity in regions extending into the cell
5. Generate movement-intensity correlation plots and export results

Features:
- Three sampling methods: standard (arc-length), x_axis, y_axis
- Two movement sampling methods: neighbourhood (around point) or outward_region (rectangular region pointing out of cell)
- Two temporal directions: past (intensity vs past movement) or future (intensity vs future movement)
- Comprehensive output control (summary-only, custom export options)
- Iterative parameter optimization mode for finding best analysis settings
- Multiple movement analysis visualizations
- Point status tracking with rotation attempts for rejected regions
- Edge transition plots showing consecutive frame edges with movement areas

Author: George Dickinson
Date: 07/14/2025
Enhanced: Added Y-axis sampling, output control, iterative parameter optimization, outward region movement sampling, and temporal direction control

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
    'movement_sampling_method': ['neighbourhood', 'outward_region'],
    'temporal_direction': ['past', 'future'],
}

This will test multiple parameter combinations and save results showing
which settings produce the best movement-intensity correlations and measurement success rates.

MOVEMENT SAMPLING METHODS:
=========================
1. 'neighbourhood': Samples movement in a square neighborhood around each edge point
   - Controlled by: movement_sampling_neighbourhood_size

2. 'outward_region': Samples movement in a rectangular region pointing outward from the cell
   - Controlled by: movement_region_depth, movement_region_width
   - Uses outward normal vector (opposite direction from intensity sampling)
   - Useful for detecting movement patterns extending beyond the current cell boundary

TEMPORAL DIRECTIONS:
===================
1. 'past': Intensity at frame N is correlated with movement from frame N-1 to N
   - Tests if intensity responds to recent movement events
   - Traditional approach: movement causes intensity changes

2. 'future': Intensity at frame N is correlated with movement from frame N to N+1
   - Tests if intensity predicts upcoming movement events
   - Predictive approach: intensity patterns precede movement
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from skimage import measure, draw, morphology
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
    'output_file': 'movement_parameter_optimization_results.csv',  # Output file for all iterations
    'suppress_individual_outputs': True,  # Disable individual file outputs during iteration
    'show_progress': True,               # Show progress during iterations
}

# Parameter ranges for iterative testing
# Each parameter can be a single value or a list of values to test
ITERATIVE_PARAMS = {
    'n_points': [12, 22, 52],                    # Number of sampling points
    'depth': [150, 300],                  # Sampling rectangle depth
    'width': [50, 100],                         # Sampling rectangle width
    'min_cell_coverage': [0.8],     # Minimum cell coverage
    'sampling_method': ['standard'],  # Sampling methods
    'try_rotation': [True],                   # Rotation fallback
    'exclude_endpoints': [True],              # Endpoint exclusion
    'movement_threshold': [0.1],        # Movement classification threshold
    'min_movement_pixels': [5],              # Minimum pixels for movement detection

    # Movement sampling parameters
    'movement_sampling_method': ['outward_region'],  # Movement sampling methods
    'movement_sampling_neighbourhood_size': [20, 25, 30],     # Movement sampling neighborhood size
    'movement_region_depth': [50, 100],     # Movement region depth (outward from cell)
    'movement_region_width': [20, 75],      # Movement region width

    # Temporal direction parameters
    'temporal_direction': ['past', 'future'],       # Temporal direction for intensity-movement correlation
}

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Input file paths
IMAGE_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/test_data/151_2019_07_29_TIRF_mKera_scratch_1_MMStack_Pos7_piezo1.tif"  # TIRF microscopy images
MASK_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/test_data/151_2019_07_29_TIRF_mKera_scratch_Pos7_DIC_Mask_test.tif"    # Binary cell masks

#IMAGE_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/simulated_data/tirf_simulation_output_movement-predictive/synthetic_cell_movement_following_images.tif"  # TIRF microscopy images
#MASK_STACK_PATH = "/Users/george/Documents/python_projects/cell_curvature_analyzer/standalone/forJinghao/simulated_data/tirf_simulation_output_movement-predictive/synthetic_cell_movement_following_masks.tif"    # Binary cell masks

# Output directory
OUTPUT_DIR = "movement_analysis_results_TESTING-fixed"

# Analysis parameters
CONFIG = {
    'n_points': 12,               # Number of points to sample along cell edge
    'depth': 20,                 # Depth of sampling rectangle (pixels into cell)
    'width': 20,                  # Width of sampling rectangle (pixels)
    'min_cell_coverage': 0.8,     # Minimum fraction of rectangle inside cell
    'try_rotation': True,         # Try rotating rejected sampling regions 180°
    'movement_threshold': 0.1,    # Threshold for classifying movement (normalized)
    'min_movement_pixels': 5,     # Minimum pixels changed to register movement
    'exclude_endpoints': True,    # Exclude first and last edge points
    'sampling_method': 'y_axis',   # 'standard' (arc-length), 'x_axis', or 'y_axis'

    # Movement sampling parameters
    'movement_sampling_method': 'outward_region',  # 'neighbourhood' or 'outward_region'
    'movement_sampling_neighbourhood_size': 20,   # Size of neighborhood for 'neighbourhood' method
    'movement_region_depth': 10,                  # Depth of outward region for 'outward_region' method
    'movement_region_width': 10,                  # Width of outward region for 'outward_region' method

    # Temporal direction parameters
    'temporal_direction': 'past',                 # 'past' (intensity vs past movement) or 'future' (intensity vs future movement)
}

# Output control parameters
OUTPUT_CONFIG = {
    'save_detailed_csv': True,        # Save detailed_movement_results.csv with all point data
    'save_summary_json': True,        # Save movement_summary_statistics.json (always recommended)
    'save_correlation_plots': True,   # Save main correlation plots
    'save_frame_plots': True,         # Save individual frame analysis plots
    'summary_only': False,            # If True, only save summary_statistics.json (overrides other settings)
}

# Visualization parameters
VIZ_CONFIG = {
    'movement_cmap': 'RdBu_r',     # Colormap for movement visualization (Blue=retraction, Red=extension)
    'intensity_cmap': 'viridis',   # Colormap for intensity visualization
    'marker_size': 30,             # Size of scatter plot markers
    'figure_dpi': 150,            # DPI for saved figures

    # Frame-by-frame visualization options
    'save_frame_plots': True,     # Save individual frame analysis plots
    'save_sampling_plots': True,  # Save sampling region visualizations
    'save_intensity_plots': True, # Save intensity distribution plots
    'save_movement_plots': True,  # Save movement distribution plots
    'save_edge_transition_plots': True,  # Save edge transition plots showing consecutive frames
    'plot_every_nth_frame': 1,   # Save plots for every Nth frame (1 = all frames)

    # Overlay visualization options
    'save_movement_overlay_plots': True,  # Save movement overlay on cell images
    'save_intensity_overlay_plots': True, # Save intensity overlay on cell images

    # Sampling visualization options
    'plot_every_nth_sampling_point': 1,  # Show every Nth sampling rectangle (1 = all)

    # Additional correlation plots
    'save_movement_type_correlation': True  # Save correlation plot by movement type
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

    # Get temporal direction
    temporal_direction = config.get('temporal_direction', 'past')

    # Process frame pairs based on temporal direction
    all_results = []
    all_movements = []
    all_intensities = []
    all_valid_points = []
    movement_scores = []
    movement_types = []
    processed_pairs = 0

    if temporal_direction == 'future':
        # Future direction: intensity at frame N vs movement from N to N+1
        for frame_idx in range(0, images.shape[0] - 1):  # Stop before last frame
            current_image = images[frame_idx]
            current_mask = masks[frame_idx]
            next_mask = masks[frame_idx + 1]

            # Analyze frame pair (current frame intensity vs future movement)
            frame_results = analyze_frame_pair(current_image, current_mask, next_mask,
                                             config, temporal_direction='future')
            all_results.append(frame_results)

            if 'error' not in frame_results:
                all_movements.append(frame_results['local_movement_scores'])
                all_intensities.append(frame_results['intensities'])
                all_valid_points.append(frame_results['valid_points'])
                movement_scores.append(frame_results['movement_score'])
                movement_types.append(frame_results['movement_type'])
                processed_pairs += 1
            else:
                movement_scores.append(0.0)
                movement_types.append('stable')
    else:
        # Past direction: intensity at frame N vs movement from N-1 to N (original behavior)
        for frame_idx in range(1, images.shape[0]):  # Start from frame 1
            current_image = images[frame_idx]
            current_mask = masks[frame_idx]
            previous_mask = masks[frame_idx - 1]

            # Analyze frame pair (current frame intensity vs past movement)
            frame_results = analyze_frame_pair(current_image, current_mask, previous_mask,
                                             config, temporal_direction='past')
            all_results.append(frame_results)

            if 'error' not in frame_results:
                all_movements.append(frame_results['local_movement_scores'])
                all_intensities.append(frame_results['intensities'])
                all_valid_points.append(frame_results['valid_points'])
                movement_scores.append(frame_results['movement_score'])
                movement_types.append(frame_results['movement_type'])
                processed_pairs += 1
            else:
                movement_scores.append(0.0)
                movement_types.append('stable')

    # Calculate statistics without generating plots
    correlation_stats = None
    type_correlation_stats = None

    if processed_pairs > 0:
        # Combine data from all frame pairs
        combined_movements = []
        combined_intensities = []

        for i in range(len(all_movements)):
            movements = all_movements[i]
            intensities = all_intensities[i]
            valid_points = all_valid_points[i]

            # Filter for valid points
            valid_mov = movements[valid_points]
            valid_int = intensities[valid_points]

            combined_movements.extend(valid_mov)
            combined_intensities.extend(valid_int)

        # Convert to numpy arrays
        combined_movements = np.array(combined_movements)
        combined_intensities = np.array(combined_intensities)

        # Calculate correlation statistics
        if len(combined_movements) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_movements, combined_intensities)
            correlation_stats = {
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'sample_size': int(len(combined_movements)),
                'slope': float(slope)
            }
        else:
            correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': int(len(combined_movements)),
                'slope': None
            }

        # Calculate movement type statistics
        if len(combined_movements) > 1:
            # Create binary movement classification for type correlation
            movement_binary = np.array([1 if x > 0.1 else -1 if x < -0.1 else 0 for x in combined_movements])
            if len(np.unique(movement_binary)) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    movement_binary, combined_intensities)
                type_correlation_stats = {
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'sample_size': int(len(movement_binary)),
                    'slope': float(slope)
                }
            else:
                type_correlation_stats = {
                    'r_squared': None,
                    'p_value': None,
                    'sample_size': int(len(movement_binary)),
                    'slope': None
                }
        else:
            type_correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': int(len(combined_movements)),
                'slope': None
            }

    # Calculate summary statistics
    total_points = sum(len(valid) for valid in all_valid_points) if all_valid_points else 0
    valid_measurements = sum(np.sum(valid) for valid in all_valid_points) if all_valid_points else 0

    # Movement type statistics
    extending_count = movement_types.count('extending')
    retracting_count = movement_types.count('retracting')
    stable_count = movement_types.count('stable')

    # Compile results
    iteration_results = {
        # Parameters used
        'iteration': iteration_num,
        **iteration_params,

        # Analysis results
        'total_frames': images.shape[0],
        'total_transitions': images.shape[0] - 1,
        'processed_transitions': processed_pairs,
        'total_sampled_points': total_points,
        'valid_measurements': valid_measurements,
        'valid_measurement_percentage': (valid_measurements / total_points * 100) if total_points > 0 else 0,

        # Movement statistics
        'extending_transitions': extending_count,
        'retracting_transitions': retracting_count,
        'stable_transitions': stable_count,
        'average_movement_score': np.mean(movement_scores) if movement_scores else 0.0,
        'movement_score_std': np.std(movement_scores) if movement_scores else 0.0,

        # Correlation statistics
        'correlation_r_squared': correlation_stats['r_squared'] if correlation_stats else None,
        'correlation_p_value': correlation_stats['p_value'] if correlation_stats else None,
        'correlation_sample_size': correlation_stats['sample_size'] if correlation_stats else None,
        'correlation_slope': correlation_stats['slope'] if correlation_stats else None,

        # Movement type correlation statistics
        'type_correlation_r_squared': type_correlation_stats['r_squared'] if type_correlation_stats else None,
        'type_correlation_p_value': type_correlation_stats['p_value'] if type_correlation_stats else None,
        'type_correlation_sample_size': type_correlation_stats['sample_size'] if type_correlation_stats else None,
        'type_correlation_slope': type_correlation_stats['slope'] if type_correlation_stats else None,

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

        # Best R-squared for movement correlation
        valid_correlations = df[df['correlation_r_squared'].notna()]
        if len(valid_correlations) > 0:
            best_corr_idx = valid_correlations['correlation_r_squared'].idxmax()
            best_corr = df.loc[best_corr_idx]
            print(f"\nBest Movement-Intensity Correlation (R² = {best_corr['correlation_r_squared']:.4f}):")
            for param in param_columns:
                print(f"  {param}: {best_corr[param]}")

        # Best R-squared for movement type correlation
        valid_type_correlations = df[df['type_correlation_r_squared'].notna()]
        if len(valid_type_correlations) > 0:
            best_type_idx = valid_type_correlations['type_correlation_r_squared'].idxmax()
            best_type = df.loc[best_type_idx]
            print(f"\nBest Movement Type Correlation (R² = {best_type['type_correlation_r_squared']:.4f}):")
            for param in param_columns:
                print(f"  {param}: {best_type[param]}")

        # Highest valid measurement percentage
        best_valid_idx = df['valid_measurement_percentage'].idxmax()
        best_valid = df.loc[best_valid_idx]
        print(f"\nHighest Valid Measurement Rate ({best_valid['valid_measurement_percentage']:.1f}%):")
        for param in param_columns:
            print(f"  {param}: {best_valid[param]}")

        # Most dynamic movement (highest std of movement scores)
        best_dynamic_idx = df['movement_score_std'].idxmax()
        best_dynamic = df.loc[best_dynamic_idx]
        print(f"\nMost Dynamic Movement (StdDev = {best_dynamic['movement_score_std']:.4f}):")
        for param in param_columns:
            print(f"  {param}: {best_dynamic[param]}")

def run_iterative_analysis():
    """
    Run the complete iterative parameter optimization analysis.
    """
    print("="*60)
    print("ITERATIVE MOVEMENT PARAMETER OPTIMIZATION")
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

        if images.shape[0] < 2:
            print("Error: Need at least 2 frames for movement analysis")
            return
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
                extending = iteration_results.get('extending_transitions', 0)
                retracting = iteration_results.get('retracting_transitions', 0)
                r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
                print(f"  → R² = {r2_str}, Valid = {valid_pct:.1f}%, Ext/Ret = {extending}/{retracting}")

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

def detect_edge_movement(current_contour, previous_contour, current_mask, previous_mask, config):
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
    config : dict
        Configuration parameters

    Returns:
    --------
    movement_score : float
        Overall movement score (positive=extension, negative=retraction)
    movement_type : str
        Classification of movement ('extending', 'retracting', 'stable')
    movement_map : ndarray
        2D map showing local movement (+1=extension, -1=retraction, 0=no change)
    """
    # Create movement map by comparing masks
    movement_map = np.zeros_like(current_mask, dtype=float)

    # Calculate difference between current and previous masks
    diff_mask = current_mask.astype(int) - previous_mask.astype(int)

    # Identify extension and retraction regions
    extension_regions = (diff_mask == 1)  # New cell area
    retraction_regions = (diff_mask == -1)  # Lost cell area

    # Set values in movement map
    movement_map[extension_regions] = 1.0   # Extension
    movement_map[retraction_regions] = -1.0  # Retraction

    # Count pixels in each region
    extension_pixels = np.sum(extension_regions)
    retraction_pixels = np.sum(retraction_regions)

    # Calculate net movement score
    net_movement = extension_pixels - retraction_pixels
    total_changed = extension_pixels + retraction_pixels

    # Normalize movement score
    if total_changed > config['min_movement_pixels']:
        movement_score = net_movement / total_changed
    else:
        movement_score = 0.0

    # Classify movement type
    if movement_score > config['movement_threshold']:
        movement_type = 'extending'
    elif movement_score < -config['movement_threshold']:
        movement_type = 'retracting'
    else:
        movement_type = 'stable'

    return movement_score, movement_type, movement_map

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

def calculate_local_movement_scores(points, movement_map, config, normals=None):
    """
    Calculate local movement scores at each sampled point using the specified method.

    Parameters:
    -----------
    points : ndarray
        Sampled contour points
    movement_map : ndarray
        2D movement map (+1=extension, -1=retraction, 0=stable)
    config : dict
        Configuration parameters
    normals : ndarray, optional
        Inward normal vectors (required for 'outward_region' method)

    Returns:
    --------
    local_scores : ndarray
        Local movement score for each point
    """
    local_scores = np.zeros(len(points))

    method = config.get('movement_sampling_method', 'neighbourhood')

    if method == 'outward_region':
        if normals is None:
            raise ValueError("Normals are required for 'outward_region' movement sampling method")

        # Sample movement using outward rectangular regions
        local_scores = calculate_movement_scores_outward_region(
            points, normals, movement_map, config)

    else:  # default to 'neighbourhood' method
        # Sample movement map at each point location using neighborhood
        for i, point in enumerate(points):
            y, x = int(round(point[0])), int(round(point[1]))

            # Ensure coordinates are within bounds
            if 0 <= y < movement_map.shape[0] and 0 <= x < movement_map.shape[1]:
                # Sample in a small neighborhood around the point
                neighborhood_size = config['movement_sampling_neighbourhood_size']
                y_min = max(0, y - neighborhood_size)
                y_max = min(movement_map.shape[0], y + neighborhood_size + 1)
                x_min = max(0, x - neighborhood_size)
                x_max = min(movement_map.shape[1], x + neighborhood_size + 1)

                # Average movement in neighborhood
                neighborhood = movement_map[y_min:y_max, x_min:x_max]
                local_scores[i] = np.mean(neighborhood)

    return local_scores

def calculate_movement_scores_outward_region(points, normals, movement_map, config):
    """
    Calculate movement scores using outward rectangular regions.

    Parameters:
    -----------
    points : ndarray
        Sampled contour points
    normals : ndarray
        Inward normal vectors (will be inverted for outward sampling)
    movement_map : ndarray
        2D movement map (+1=extension, -1=retraction, 0=stable)
    config : dict
        Configuration parameters

    Returns:
    --------
    local_scores : ndarray
        Local movement score for each point
    """
    local_scores = np.zeros(len(points))
    image_shape = movement_map.shape

    # Get parameters for outward region sampling
    depth = config.get('movement_region_depth', 50)
    width = config.get('movement_region_width', 30)

    for i, (point, inward_normal) in enumerate(zip(points, normals)):
        # Use outward normal (opposite of inward normal)
        outward_normal = -inward_normal

        # Calculate rectangle corners extending outward from the cell
        end_point = point + depth * outward_normal
        perp = np.array([-outward_normal[1], outward_normal[0]])

        corner1 = point + width/2 * perp
        corner2 = point - width/2 * perp
        corner3 = end_point - width/2 * perp
        corner4 = end_point + width/2 * perp

        vertices = np.array([corner1, corner4, corner3, corner2])

        # Get pixels within rectangle
        try:
            rr, cc = draw.polygon(vertices[:, 0], vertices[:, 1], image_shape)

            # Check bounds
            valid_pixels = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
            if np.any(valid_pixels):
                rr = rr[valid_pixels]
                cc = cc[valid_pixels]

                # Calculate average movement in the outward region
                if len(rr) > 0:
                    local_scores[i] = np.mean(movement_map[rr, cc])
                else:
                    local_scores[i] = 0.0
            else:
                local_scores[i] = 0.0

        except (IndexError, ValueError):
            # Handle edge cases where polygon creation fails
            local_scores[i] = 0.0

    return local_scores

def analyze_frame_pair(current_image, current_mask, comparison_mask, config, temporal_direction='past'):
    """
    Analyze movement between two consecutive frames.

    Parameters:
    -----------
    current_image : ndarray
        Frame where intensity is measured
    current_mask : ndarray
        Binary mask for frame where intensity is measured
    comparison_mask : ndarray
        Binary mask for comparison frame (previous for 'past', next for 'future')
    config : dict
        Analysis configuration parameters
    temporal_direction : str
        'past' or 'future' - determines movement calculation direction

    Returns:
    --------
    results : dict
        Analysis results for this frame pair
    """
    # Detect cell edges
    current_contour = detect_cell_edge(current_mask)
    comparison_contour = detect_cell_edge(comparison_mask)

    if len(current_contour) == 0 or len(comparison_contour) == 0:
        return {"error": "Could not detect contour in one or both frames"}

    # Detect edge movement based on temporal direction
    if temporal_direction == 'future':
        # Movement from current to next frame (future movement)
        # Flip parameters to calculate movement from current to comparison (next)
        movement_score, movement_type, movement_map = detect_edge_movement(
            comparison_contour, current_contour, comparison_mask, current_mask, config)
    else:  # 'past'
        # Movement from previous to current frame (past movement) - original behavior
        movement_score, movement_type, movement_map = detect_edge_movement(
            current_contour, comparison_contour, current_mask, comparison_mask, config)

    # Sample points along current frame edge using specified method
    sampling_method = config.get('sampling_method', 'standard')

    if sampling_method == 'x_axis':
        points = sample_x_axis_points(current_contour, config['n_points'])
    elif sampling_method == 'y_axis':
        points = sample_y_axis_points(current_contour, config['n_points'])
    else:  # default to 'standard'
        points = sample_equidistant_points(current_contour, config['n_points'])

    if len(points) == 0:
        return {"error": "No sampling points generated"}

    # Calculate inward normals
    normals = calculate_inward_normal(points, current_mask)

    # Measure intensity (always in the current frame)
    intensities, valid_points, point_status = measure_intensity(
        current_image, current_mask, points, normals,
        depth=config['depth'],
        width=config['width'],
        min_cell_coverage=config['min_cell_coverage'],
        try_rotation=config['try_rotation'],
        exclude_endpoints=config['exclude_endpoints']
    )

    # Calculate local movement scores at each point
    local_movement_scores = calculate_local_movement_scores(points, movement_map, config, normals)

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
        'sampling_method': sampling_method,
        'temporal_direction': temporal_direction
    }

    return results

def create_movement_correlation_plot(all_movements, all_intensities, all_valid_points,
                                   movement_types, output_dir, viz_config, temporal_direction='past'):
    """
    Create and save movement-intensity correlation plot.

    Parameters:
    -----------
    all_movements : list
        List of local movement score arrays from all frame pairs
    all_intensities : list
        List of intensity arrays from all frame pairs
    all_valid_points : list
        List of valid point masks from all frame pairs
    movement_types : list
        List of overall movement types for each frame pair
    output_dir : str
        Output directory for saving plot
    viz_config : dict
        Visualization configuration
    temporal_direction : str
        Temporal direction ('past' or 'future')
    """
    # Combine data from all frame pairs
    combined_movements = []
    combined_intensities = []
    combined_types = []

    for i in range(len(all_movements)):
        movements = all_movements[i]
        intensities = all_intensities[i]
        valid_points = all_valid_points[i]
        frame_type = movement_types[i]

        # Filter for valid points
        valid_mov = movements[valid_points]
        valid_int = intensities[valid_points]

        combined_movements.extend(valid_mov)
        combined_intensities.extend(valid_int)
        combined_types.extend([frame_type] * len(valid_mov))

    # Convert to numpy arrays
    combined_movements = np.array(combined_movements)
    combined_intensities = np.array(combined_intensities)
    combined_types = np.array(combined_types)

    # Create movement correlation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: All points colored by movement score
    scatter = ax1.scatter(combined_movements, combined_intensities,
                         alpha=0.6, s=viz_config['marker_size'],
                         c=combined_movements, cmap=viz_config['movement_cmap'],
                         vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Local Movement Score\n(Blue = Retraction, Red = Extension)', fontsize=10)

    # Calculate and plot regression line
    correlation_stats = {}
    if len(combined_movements) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_movements, combined_intensities)

        # Create regression line
        x_line = np.linspace(min(combined_movements), max(combined_movements), 100)
        y_line = slope * x_line + intercept

        ax1.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add statistics text
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
    else:
        # Not enough data for correlation
        correlation_stats = {
            'r_squared': None,
            'p_value': None,
            'sample_size': int(len(combined_movements)),
            'slope': None
        }

    # Create temporal direction info for titles
    direction_label = "Future Movement" if temporal_direction == 'future' else "Past Movement"
    direction_subtitle = "(Intensity → Future Movement)" if temporal_direction == 'future' else "(Past Movement → Intensity)"

    ax1.set_xlabel('Local Movement Score', fontsize=12)
    ax1.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax1.set_title(f'Local Edge {direction_label} vs PIEZO1 Intensity\n{direction_subtitle} - All Frame Pairs', fontsize=12)
    if len(combined_movements) > 1:
        ax1.legend()

    # Right plot: Points grouped by overall movement type
    colors = {'extending': 'red', 'retracting': 'blue', 'stable': 'gray'}

    for movement_type, color in colors.items():
        mask = combined_types == movement_type
        if np.any(mask):
            ax2.scatter(combined_movements[mask], combined_intensities[mask],
                       alpha=0.6, s=viz_config['marker_size'],
                       c=color, label=f'{movement_type.capitalize()} ({np.sum(mask)} points)')

    ax2.set_xlabel('Local Movement Score', fontsize=12)
    ax2.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax2.set_title(f'{direction_label} vs Intensity by Frame Type', fontsize=12)
    ax2.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'movement_intensity_correlation.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

    print(f"Movement correlation plot saved to: {plot_path}")

    return correlation_stats

def create_movement_type_correlation_plot(all_movements, all_intensities, all_valid_points,
                                        movement_types, output_dir, viz_config, temporal_direction='past'):
    """
    Create and save movement type correlation plot.

    Parameters:
    -----------
    all_movements : list
        List of local movement score arrays from all frame pairs
    all_intensities : list
        List of intensity arrays from all frame pairs
    all_valid_points : list
        List of valid point masks from all frame pairs
    movement_types : list
        List of overall movement types for each frame pair
    output_dir : str
        Output directory for saving plot
    viz_config : dict
        Visualization configuration
    temporal_direction : str
        Temporal direction ('past' or 'future')
    """
    # Combine data from all frame pairs
    combined_movements = []
    combined_intensities = []

    for i in range(len(all_movements)):
        movements = all_movements[i]
        intensities = all_intensities[i]
        valid_points = all_valid_points[i]

        # Filter for valid points
        valid_mov = movements[valid_points]
        valid_int = intensities[valid_points]

        combined_movements.extend(valid_mov)
        combined_intensities.extend(valid_int)

    # Convert to numpy arrays
    combined_movements = np.array(combined_movements)
    combined_intensities = np.array(combined_intensities)

    # Create correlation plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color points by movement type: red for extension, blue for retraction, gray for stable
    colors = ['red' if x > 0.1 else 'blue' if x < -0.1 else 'gray' for x in combined_movements]
    scatter = ax.scatter(combined_movements, combined_intensities,
                        alpha=0.6, s=viz_config['marker_size'], c=colors)

    # Calculate and plot regression line
    type_correlation_stats = {}
    if len(combined_movements) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_movements, combined_intensities)

        # Create regression line
        x_line = np.linspace(min(combined_movements), max(combined_movements), 100)
        y_line = slope * x_line + intercept

        ax.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

        # Add statistics text
        stats_text = f'R² = {r_value**2:.3f}\np-value = {p_value:.3e}\nn = {len(combined_movements)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)

        type_correlation_stats = {
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'sample_size': int(len(combined_movements)),
            'slope': float(slope)
        }
    else:
        type_correlation_stats = {
            'r_squared': None,
            'p_value': None,
            'sample_size': int(len(combined_movements)),
            'slope': None
        }

    # Add custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Extension (>0.1)'),
                      Patch(facecolor='blue', label='Retraction (<-0.1)'),
                      Patch(facecolor='gray', label='Stable (-0.1 to 0.1)')]
    if len(combined_movements) > 1:
        legend_elements.append(plt.Line2D([0], [0], color='k', lw=2,
                                         label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}'))
    ax.legend(handles=legend_elements, loc='best')

    # Labels and title
    direction_label = "Future Movement" if temporal_direction == 'future' else "Past Movement"
    direction_subtitle = "(Intensity → Future Movement)" if temporal_direction == 'future' else "(Past Movement → Intensity)"

    ax.set_xlabel('Local Movement Score', fontsize=12)
    ax.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax.set_title(f'Cell Edge {direction_label} Type vs PIEZO1 Intensity\n{direction_subtitle} - All Frame Pairs Combined', fontsize=14, pad=20)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'movement_type_intensity_correlation.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

    print(f"Movement type correlation plot saved to: {plot_path}")

    return type_correlation_stats

def create_movement_summary_plot(movement_scores, movement_types, output_dir, viz_config):
    """
    Create movement summary plot showing movement over time.

    Parameters:
    -----------
    movement_scores : list
        Overall movement scores for each frame transition
    movement_types : list
        Movement types for each frame transition
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Movement scores over time
    frame_indices = range(len(movement_scores))
    colors = []
    for mtype in movement_types:
        if mtype == 'extending':
            colors.append('red')
        elif mtype == 'retracting':
            colors.append('blue')
        else:
            colors.append('gray')

    bars = ax1.bar(frame_indices, movement_scores, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Frame Transition')
    ax1.set_ylabel('Movement Score')
    ax1.set_title('Cell Edge Movement Over Time')

    # Add movement type labels
    for i, (score, mtype) in enumerate(zip(movement_scores, movement_types)):
        offset = 0.05 if score >= 0 else -0.15
        ax1.text(i, score + offset, mtype, ha='center', rotation=90,
                fontsize=8, fontweight='bold', color=colors[i])

    # Bottom plot: Movement type distribution
    type_counts = {
        'extending': movement_types.count('extending'),
        'retracting': movement_types.count('retracting'),
        'stable': movement_types.count('stable')
    }

    # Define colors dictionary for pie chart
    colors_dict = {'extending': 'red', 'retracting': 'blue', 'stable': 'gray'}

    # Filter out zero counts
    filtered_types = {k: v for k, v in type_counts.items() if v > 0}

    if filtered_types:
        wedges, texts, autotexts = ax2.pie(
            filtered_types.values(),
            labels=[f'{k.capitalize()}\n({v} frames)' for k, v in filtered_types.items()],
            colors=[colors_dict[k] for k in filtered_types.keys()],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Movement Type Distribution')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'movement_summary.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

    print(f"Movement summary plot saved to: {plot_path}")

def create_frame_edge_transition_plot(frame_idx, current_image, current_mask, comparison_mask, results, output_dir, viz_config):
    """
    Create and save edge transition plot showing consecutive frame edges with movement areas.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    current_image : ndarray
        Current frame microscope image (where intensity is measured)
    current_mask : ndarray
        Current frame binary mask (where intensity is measured)
    comparison_mask : ndarray
        Comparison frame binary mask (previous for 'past', next for 'future')
    results : dict
        Analysis results for this frame pair
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create edge transition plots directory
    edge_transition_dir = os.path.join(output_dir, 'edge_transition_plots')
    os.makedirs(edge_transition_dir, exist_ok=True)

    current_contour = results['current_contour']
    comparison_contour = results['comparison_contour']
    movement_map = results['movement_map']
    movement_score = results['movement_score']
    movement_type = results['movement_type']
    temporal_direction = results.get('temporal_direction', 'past')

    # Determine frame labels based on temporal direction
    if temporal_direction == 'future':
        # Current frame intensity vs future movement
        current_frame_label = f'Current Frame {frame_idx+1} (Intensity)'
        comparison_frame_label = f'Future Frame {frame_idx+2} (Movement)'
        title_direction = f'Frame {frame_idx+1} Intensity → Future Movement to {frame_idx+2}'
    else:
        # Past movement vs current frame intensity
        current_frame_label = f'Current Frame {frame_idx+2} (Intensity)'
        comparison_frame_label = f'Previous Frame {frame_idx+1} (Movement)'
        title_direction = f'Past Movement {frame_idx+1}→{frame_idx+2} vs Frame {frame_idx+2} Intensity'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left plot: Edge transition with movement map overlay
    ax1.imshow(current_image, cmap='gray', alpha=0.6)

    # Show movement map as colored overlay
    # Create custom colormap for movement visualization
    movement_display = np.zeros_like(movement_map)
    extension_mask = movement_map > 0
    retraction_mask = movement_map < 0

    # Create RGB overlay for movement areas
    if np.any(extension_mask) or np.any(retraction_mask):
        # Show movement regions with transparency
        masked_movement = np.ma.masked_where(movement_map == 0, movement_map)
        im = ax1.imshow(masked_movement, cmap=viz_config['movement_cmap'],
                       vmin=-1, vmax=1, alpha=0.7, interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.6)
        cbar.set_label('Movement\n(Blue=Retraction, Red=Extension)', fontsize=10)

    # Plot comparison frame edge (thinner, dashed)
    if len(comparison_contour) > 0:
        ax1.plot(comparison_contour[:, 1], comparison_contour[:, 0],
                color='orange', linewidth=3, linestyle='--', alpha=0.9,
                label=comparison_frame_label, zorder=10)

    # Plot current frame edge (thicker, solid)
    if len(current_contour) > 0:
        ax1.plot(current_contour[:, 1], current_contour[:, 0],
                color='white', linewidth=4, alpha=0.9,
                label=current_frame_label, zorder=11)

    # Add sampling points if available
    if 'points' in results:
        points = results['points']
        point_status = results['point_status']
        valid_mask = (point_status == 1) | (point_status == 3)

        if np.any(valid_mask):
            ax1.scatter(points[valid_mask, 1], points[valid_mask, 0],
                       c='lime', s=40, alpha=0.9, edgecolors='black', linewidth=1,
                       label='Sampling Points', zorder=12)

    # Add movement statistics text
    stats_text = f'Movement: {movement_type}\nScore: {movement_score:.3f}\nDirection: {temporal_direction}'
    if np.any(extension_mask):
        extension_pixels = np.sum(extension_mask)
        stats_text += f'\nExtension: {extension_pixels} pixels'
    if np.any(retraction_mask):
        retraction_pixels = np.sum(retraction_mask)
        stats_text += f'\nRetraction: {retraction_pixels} pixels'

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
             verticalalignment='top', fontsize=11, fontweight='bold')

    ax1.set_title(title_direction, fontsize=14, pad=20)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.axis('off')

    # Right plot: Side-by-side mask comparison
    # Create a composite image showing both masks
    composite = np.zeros((*current_mask.shape, 3))

    # Comparison mask in green channel
    composite[:, :, 1] = comparison_mask * 0.7

    # Current mask in yellow (red + green)
    composite[:, :, 0] = current_mask * 0.7  # Red component for yellow
    composite[:, :, 1] += current_mask * 0.7  # Green component for yellow

    # Overlap in lighter yellow/green mix
    overlap = (comparison_mask & current_mask).astype(float)
    composite[:, :, 0] += overlap * 0.3  # Add red for overlap
    composite[:, :, 1] += overlap * 0.3  # Add green for overlap

    ax2.imshow(composite)

    # Plot contours on mask comparison
    if len(comparison_contour) > 0:
        ax2.plot(comparison_contour[:, 1], comparison_contour[:, 0],
                color='green', linewidth=2, alpha=0.9,
                label=comparison_frame_label)

    if len(current_contour) > 0:
        ax2.plot(current_contour[:, 1], current_contour[:, 0],
                color='yellow', linewidth=2, alpha=0.9,
                label=current_frame_label)

    # Highlight extension and retraction areas with markers
    if np.any(extension_mask):
        ext_y, ext_x = np.where(extension_mask)
        # Sample some points to avoid overcrowding
        if len(ext_y) > 200:
            sample_indices = np.random.choice(len(ext_y), 200, replace=False)
            ext_y, ext_x = ext_y[sample_indices], ext_x[sample_indices]
        ax2.scatter(ext_x, ext_y, c='red', s=8, alpha=0.6, marker='s', label='Extension Areas')

    if np.any(retraction_mask):
        ret_y, ret_x = np.where(retraction_mask)
        # Sample some points to avoid overcrowding
        if len(ret_y) > 200:
            sample_indices = np.random.choice(len(ret_y), 200, replace=False)
            ret_y, ret_x = ret_y[sample_indices], ret_x[sample_indices]
        ax2.scatter(ret_x, ret_y, c='blue', s=8, alpha=0.6, marker='s', label='Retraction Areas')

    mask_title = f'Mask Comparison ({temporal_direction} direction)\n(Green=Comparison, Yellow=Current, Light Yellow/Green=Overlap)'
    ax2.set_title(mask_title, fontsize=14, pad=20)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.axis('off')

    plt.tight_layout()

    # Save plot with temporal direction in filename
    if temporal_direction == 'future':
        plot_path = os.path.join(edge_transition_dir, f'edge_transition_future_{frame_idx+1:03d}_to_{frame_idx+2:03d}.png')
    else:
        plot_path = os.path.join(edge_transition_dir, f'edge_transition_past_{frame_idx+1:03d}_to_{frame_idx+2:03d}.png')

    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_sampling_plot(frame_idx, current_image, current_mask, results, output_dir, viz_config):
    """
    Create and save sampling region visualization for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    current_image : ndarray
        Original microscope image
    current_mask : ndarray
        Binary mask
    results : dict
        Analysis results for this frame pair
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
    ax1.imshow(current_image, cmap='gray')

    # Plot contour
    contour = results['current_contour']
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
    ax1.set_title(f'Frame {frame_idx+2}: Sampling Points ({sampling_method} method)', fontsize=12)
    ax1.legend()
    ax1.axis('off')

    # Right plot: Sampling regions overlay with cell edge
    ax2.imshow(current_image, cmap='gray', alpha=0.7)

    # Plot cell edge on sampling regions plot too
    if len(contour) > 0:
        ax2.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5, alpha=0.8, label='Cell Edge')

    # Draw intensity sampling rectangles with status-based coloring
    normals = results['normals']
    depth = CONFIG['depth']
    width = CONFIG['width']

    # Use the configurable sampling point frequency
    plot_frequency = viz_config['plot_every_nth_sampling_point']

    # Track what we're plotting for legend
    intensity_regions_plotted = False
    movement_regions_plotted = False

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

            # Calculate intensity sampling rectangle corners (inward)
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
            intensity_regions_plotted = True

            # Also draw movement sampling regions if using outward_region method
            movement_method = CONFIG.get('movement_sampling_method', 'neighbourhood')
            if movement_method == 'outward_region':
                # Draw outward movement sampling rectangle
                movement_depth = CONFIG.get('movement_region_depth', 50)
                movement_width = CONFIG.get('movement_region_width', 30)

                # Use outward normal (opposite of inward normal)
                outward_normal = -effective_normal
                movement_end_point = point + movement_depth * outward_normal
                movement_perp = np.array([-outward_normal[1], outward_normal[0]])

                movement_corners = np.array([
                    point + movement_width/2 * movement_perp,
                    point - movement_width/2 * movement_perp,
                    movement_end_point - movement_width/2 * movement_perp,
                    movement_end_point + movement_width/2 * movement_perp,
                    point + movement_width/2 * movement_perp  # Close the rectangle
                ])

                # Use dashed line and different transparency for movement regions
                ax2.plot(movement_corners[:, 1], movement_corners[:, 0],
                        color='orange', linewidth=1.5, alpha=0.8, linestyle='--')
                movement_regions_plotted = True

    # Create custom legend for rectangle colors
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color='r', lw=1.5, label='Cell Edge')]

    if intensity_regions_plotted:
        if np.any(point_status == 1):
            legend_elements.append(Patch(facecolor='lime', alpha=0.6, label='Valid Intensity Regions'))
        if np.any(point_status == 2):
            legend_elements.append(Patch(facecolor='blue', alpha=0.4, label='Rejected Intensity Regions'))
        if np.any(point_status == 3):
            legend_elements.append(Patch(facecolor='yellow', alpha=0.7, label='Rotated Valid Intensity Regions'))

    if movement_regions_plotted:
        legend_elements.append(plt.Line2D([0], [0], color='orange', lw=1.5, linestyle='--',
                                        label='Movement Sampling Regions'))

    ax2.legend(handles=legend_elements, loc='best')

    # Update title to include movement sampling method if applicable
    movement_method = CONFIG.get('movement_sampling_method', 'neighbourhood')
    title_parts = [f'Frame {frame_idx+2}: Sampling Regions (every {plot_frequency} shown)']
    if movement_method == 'outward_region':
        title_parts.append('Solid=Intensity, Dashed=Movement')

    ax2.set_title('\n'.join(title_parts), fontsize=12)
    ax2.axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(sampling_dir, f'frame_{frame_idx+2:03d}_sampling.png')
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
        Analysis results for this frame pair
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
    ax1.set_title(f'Frame {frame_idx+2}: Intensity Profile')
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
    ax2.set_title(f'Frame {frame_idx+2}: Intensity Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(intensity_dir, f'frame_{frame_idx+2:03d}_intensity.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_movement_plot(frame_idx, results, output_dir, viz_config):
    """
    Create and save movement analysis plot for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    results : dict
        Analysis results for this frame pair
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create movement plots directory
    movement_dir = os.path.join(output_dir, 'frame_movement_plots')
    os.makedirs(movement_dir, exist_ok=True)

    local_movements = results['local_movement_scores']
    valid_points = results['valid_points']
    points = results['points']
    overall_score = results['movement_score']
    overall_type = results['movement_type']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    point_indices = np.arange(len(points))

    # Top left: Local movement scores
    colors = ['red' if x > 0.1 else 'blue' if x < -0.1 else 'gray' for x in local_movements]
    ax1.plot(point_indices, local_movements, 'k-', linewidth=2, alpha=0.7)
    ax1.scatter(point_indices, local_movements, c=colors, s=30, alpha=0.8)
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Local Movement Score')
    ax1.set_title('Local Movement Scores')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add overall movement info
    ax1.text(0.02, 0.98, f'Overall: {overall_type}\nScore: {overall_score:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Top right: Movement histogram
    valid_movements = local_movements[valid_points]
    if len(valid_movements) > 0:
        ax2.hist(valid_movements, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(np.mean(valid_movements), color='red', linestyle='--',
                   label=f'Mean: {np.mean(valid_movements):.3f}')
        ax2.axvline(0, color='gray', linestyle='-', alpha=0.5, label='Zero')

    ax2.set_xlabel('Local Movement Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Movement Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom left: Movement vs Intensity for this frame
    intensities = results['intensities']
    if np.any(valid_points):
        valid_mov = local_movements[valid_points]
        valid_int = intensities[valid_points]
        colors_valid = ['red' if x > 0.1 else 'blue' if x < -0.1 else 'gray' for x in valid_mov]

        ax3.scatter(valid_mov, valid_int, c=colors_valid, s=50, alpha=0.8)

        # Calculate correlation for this frame
        if len(valid_mov) > 1:
            corr_coef = np.corrcoef(valid_mov, valid_int)[0, 1]
            ax3.text(0.02, 0.98, f'Correlation: {corr_coef:.3f}',
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax3.set_xlabel('Local Movement Score')
    ax3.set_ylabel('PIEZO1 Intensity')
    ax3.set_title('Movement vs Intensity (This Frame)')
    ax3.grid(True, alpha=0.3)

    # Bottom right: Movement map visualization
    movement_map = results['movement_map']
    im = ax4.imshow(movement_map, cmap=viz_config['movement_cmap'], vmin=-1, vmax=1, alpha=0.8)

    # Overlay cell edge
    contour = results['current_contour']
    if len(contour) > 0:
        ax4.plot(contour[:, 1], contour[:, 0], 'k-', linewidth=2, alpha=0.8, label='Cell Edge')

    # Plot sampling points
    if len(points) > 0:
        ax4.scatter(points[:, 1], points[:, 0], c='white', s=20, alpha=0.9,
                   edgecolors='black', linewidth=0.5, zorder=5)

    ax4.set_title('Movement Map')
    ax4.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Movement\n(Blue=Retraction, Red=Extension)', fontsize=10)

    plt.suptitle(f'Frame {frame_idx+2}: Movement Analysis', fontsize=16)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(movement_dir, f'frame_{frame_idx+2:03d}_movement.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_movement_overlay_plot(frame_idx, current_image, current_mask, results, output_dir, viz_config):
    """
    Create and save movement overlay visualization for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    current_image : ndarray
        Original microscope image
    current_mask : ndarray
        Binary mask
    results : dict
        Analysis results for this frame pair
    output_dir : str
        Output directory
    viz_config : dict
        Visualization configuration
    """
    if 'error' in results:
        return

    # Create movement plots directory
    movement_dir = os.path.join(output_dir, 'frame_movement_plots')
    os.makedirs(movement_dir, exist_ok=True)

    local_movements = results['local_movement_scores']
    points = results['points']
    valid_points = results['valid_points']
    point_status = results['point_status']
    contour = results['current_contour']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the cell image as background
    ax.imshow(current_image, cmap='gray', alpha=0.8)

    # Plot cell edge outline
    if len(contour) > 0:
        ax.plot(contour[:, 1], contour[:, 0], 'white', linewidth=2, alpha=0.9, label='Cell Edge')

    # Separate points by movement type and validity
    extension_valid = (local_movements > 0.1) & (point_status != 0)
    retraction_valid = (local_movements < -0.1) & (point_status != 0)
    stable_valid = (np.abs(local_movements) <= 0.1) & (point_status != 0)
    ignored = (point_status == 0) | (point_status == 2)

    # Plot extension points (red)
    if np.any(extension_valid):
        ax.scatter(points[extension_valid, 1], points[extension_valid, 0],
                  c='red', s=80, alpha=0.9, marker='o',
                  label='Extension (+)', edgecolors='white', linewidth=1, zorder=5)

    # Plot retraction points (blue)
    if np.any(retraction_valid):
        ax.scatter(points[retraction_valid, 1], points[retraction_valid, 0],
                  c='blue', s=80, alpha=0.9, marker='o',
                  label='Retraction (-)', edgecolors='white', linewidth=1, zorder=5)

    # Plot stable points (gray)
    if np.any(stable_valid):
        ax.scatter(points[stable_valid, 1], points[stable_valid, 0],
                  c='gray', s=60, alpha=0.9, marker='o',
                  label='Stable', edgecolors='white', linewidth=1, zorder=5)

    # Plot ignored points (fainter)
    if np.any(ignored):
        ax.scatter(points[ignored, 1], points[ignored, 0],
                  c='gray', s=40, alpha=0.4, marker='x',
                  label='Ignored', zorder=4)

    # Formatting
    ax.set_title(f'Frame {frame_idx+2}: Movement Type Overlay', fontsize=14, pad=20)
    ax.legend(loc='best', framealpha=0.8)
    ax.axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(movement_dir, f'frame_{frame_idx+2:03d}_movement_overlay.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def create_frame_intensity_overlay_plot(frame_idx, current_image, current_mask, results, output_dir, viz_config):
    """
    Create and save intensity overlay visualization for a single frame.

    Parameters:
    -----------
    frame_idx : int
        Frame index
    current_image : ndarray
        Original microscope image
    current_mask : ndarray
        Binary mask
    results : dict
        Analysis results for this frame pair
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
    contour = results['current_contour']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the cell image as background
    ax.imshow(current_image, cmap='gray', alpha=0.8)

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
    ax.set_title(f'Frame {frame_idx+2}: PIEZO1 Intensity Overlay', fontsize=14, pad=20)

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
    plot_path = os.path.join(intensity_dir, f'frame_{frame_idx+2:03d}_intensity_overlay.png')
    plt.savefig(plot_path, dpi=viz_config['figure_dpi'], bbox_inches='tight')
    plt.close()

def save_results_csv(all_results, movement_scores, movement_types, output_dir):
    """
    Save detailed results to CSV file.

    Parameters:
    -----------
    all_results : list
        List of frame pair analysis results
    movement_scores : list
        Overall movement scores
    movement_types : list
        Overall movement types
    output_dir : str
        Output directory
    """
    data_rows = []

    for frame_idx, results in enumerate(all_results):
        if 'error' in results:
            continue

        points = results['points']
        intensities = results['intensities']
        valid_points = results['valid_points']
        point_status = results['point_status']
        local_movements = results['local_movement_scores']
        overall_score = movement_scores[frame_idx]
        overall_type = movement_types[frame_idx]
        sampling_method = results.get('sampling_method', 'standard')
        movement_sampling_method = CONFIG.get('movement_sampling_method', 'neighbourhood')
        temporal_direction = results.get('temporal_direction', 'past')

        for i in range(len(points)):
            # Convert point status to descriptive text
            status_map = {0: 'excluded', 1: 'valid', 2: 'rejected', 3: 'rotated_valid'}
            status_text = status_map.get(point_status[i], 'unknown')

            data_rows.append({
                'frame_transition': frame_idx + 1,  # Frame 1->2 is transition 1
                'point_index': i,
                'x_coord': points[i, 1],
                'y_coord': points[i, 0],
                'local_movement_score': local_movements[i],
                'intensity': intensities[i],
                'valid_measurement': valid_points[i],
                'point_status': status_text,
                'overall_movement_score': overall_score,
                'overall_movement_type': overall_type,
                'sampling_method': sampling_method,
                'movement_sampling_method': movement_sampling_method,
                'temporal_direction': temporal_direction
            })

    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(output_dir, 'detailed_movement_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"Detailed results saved to: {csv_path}")
    print(f"  - Point status meanings: excluded=endpoint excluded, valid=normal measurement,")
    print(f"    rejected=insufficient cell coverage, rotated_valid=successful after 180° rotation")
    print(f"  - Includes edge sampling method, movement sampling method, and temporal direction for each measurement")

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

    json_path = os.path.join(output_dir, 'movement_summary_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(serializable_stats, f, indent=4)

    print(f"Summary statistics saved to: {json_path}")

# =============================================================================
# MAIN ANALYSIS WORKFLOW
# =============================================================================

def main():
    """Main analysis workflow."""

    # Check if iterative mode is enabled
    if ITERATIVE_MODE['enabled']:
        # Create output directory for iterative results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        run_iterative_analysis()
        return

    print("="*60)
    print("CELL EDGE MOVEMENT ANALYSIS (ENHANCED)")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Check output configuration
    if OUTPUT_CONFIG['summary_only']:
        print("Running in summary-only mode: Only summary statistics will be saved")
        # Override other output settings when summary_only is True
        OUTPUT_CONFIG['save_detailed_csv'] = False
        OUTPUT_CONFIG['save_correlation_plots'] = False
        OUTPUT_CONFIG['save_frame_plots'] = False
        VIZ_CONFIG['save_frame_plots'] = False
        VIZ_CONFIG['save_sampling_plots'] = False
        VIZ_CONFIG['save_intensity_plots'] = False
        VIZ_CONFIG['save_movement_plots'] = False
        VIZ_CONFIG['save_edge_transition_plots'] = False
        VIZ_CONFIG['save_movement_overlay_plots'] = False
        VIZ_CONFIG['save_intensity_overlay_plots'] = False
        VIZ_CONFIG['save_movement_type_correlation'] = False

    # Load image stacks
    try:
        images, masks = load_tiff_stack(IMAGE_STACK_PATH, MASK_STACK_PATH)
    except Exception as e:
        print(f"Error loading image stacks: {e}")
        return

    if images.shape[0] < 2:
        print("Error: Need at least 2 frames for movement analysis")
        return

    # Print sampling method being used
    sampling_method = CONFIG.get('sampling_method', 'standard')
    movement_sampling_method = CONFIG.get('movement_sampling_method', 'neighbourhood')
    temporal_direction = CONFIG.get('temporal_direction', 'past')
    print(f"Using sampling method: {sampling_method}")
    print(f"Using movement sampling method: {movement_sampling_method}")
    print(f"Using temporal direction: {temporal_direction}")

    # Process frame pairs based on temporal direction
    if temporal_direction == 'future':
        print(f"Processing {images.shape[0]-1} frame transitions (intensity → future movement)...")
        frame_range_desc = "Future direction: Intensity at frame N vs movement from N to N+1"
    else:
        print(f"Processing {images.shape[0]-1} frame transitions (past movement → intensity)...")
        frame_range_desc = "Past direction: Intensity at frame N vs movement from N-1 to N"

    print(frame_range_desc)

    all_results = []
    all_movements = []
    all_intensities = []
    all_valid_points = []
    movement_scores = []
    movement_types = []
    processed_pairs = 0

    if temporal_direction == 'future':
        # Future direction: intensity at frame N vs movement from N to N+1
        for frame_idx in range(0, images.shape[0] - 1):  # Stop before last frame
            print(f"Processing frame {frame_idx+1} → future movement to frame {frame_idx+2}...")

            current_image = images[frame_idx]
            current_mask = masks[frame_idx]
            next_mask = masks[frame_idx + 1]

            # Analyze frame pair (current frame intensity vs future movement)
            results = analyze_frame_pair(current_image, current_mask, next_mask,
                                       CONFIG, temporal_direction='future')
            all_results.append(results)

            if 'error' not in results:
                all_movements.append(results['local_movement_scores'])
                all_intensities.append(results['intensities'])
                all_valid_points.append(results['valid_points'])
                movement_scores.append(results['movement_score'])
                movement_types.append(results['movement_type'])
                processed_pairs += 1

                # Generate frame-by-frame plots if enabled
                should_plot_this_frame = (frame_idx % VIZ_CONFIG['plot_every_nth_frame'] == 0) or (VIZ_CONFIG['plot_every_nth_frame'] == 1)

                if should_plot_this_frame and OUTPUT_CONFIG['save_frame_plots']:
                    if VIZ_CONFIG['save_edge_transition_plots']:
                        print(f"  Creating edge transition plot for frames {frame_idx+1} → {frame_idx+2}...")
                        create_frame_edge_transition_plot(frame_idx, current_image, current_mask, next_mask, results, OUTPUT_DIR, VIZ_CONFIG)

                    if VIZ_CONFIG['save_sampling_plots']:
                        print(f"  Creating sampling plot for frame {frame_idx+1}...")
                        create_frame_sampling_plot(frame_idx, current_image, current_mask, results, OUTPUT_DIR, VIZ_CONFIG)

                    if VIZ_CONFIG['save_intensity_plots']:
                        print(f"  Creating intensity plot for frame {frame_idx+1}...")
                        create_frame_intensity_plot(frame_idx, results, OUTPUT_DIR, VIZ_CONFIG)

                        # Create intensity overlay plot if enabled
                        if VIZ_CONFIG['save_intensity_overlay_plots']:
                            print(f"  Creating intensity overlay plot for frame {frame_idx+1}...")
                            create_frame_intensity_overlay_plot(frame_idx, current_image, current_mask, results, OUTPUT_DIR, VIZ_CONFIG)

                    if VIZ_CONFIG['save_movement_plots']:
                        print(f"  Creating movement plot for frame {frame_idx+1}...")
                        create_frame_movement_plot(frame_idx, results, OUTPUT_DIR, VIZ_CONFIG)

                        # Create movement overlay plot if enabled
                        if VIZ_CONFIG['save_movement_overlay_plots']:
                            print(f"  Creating movement overlay plot for frame {frame_idx+1}...")
                            create_frame_movement_overlay_plot(frame_idx, current_image, current_mask, results, OUTPUT_DIR, VIZ_CONFIG)

            else:
                print(f"  Warning: {results['error']}")
                # Add empty data to maintain indexing
                movement_scores.append(0.0)
                movement_types.append('stable')
    else:
        # Past direction: intensity at frame N vs movement from N-1 to N (original behavior)
        for frame_idx in range(1, images.shape[0]):  # Start from frame 1
            print(f"Processing past movement {frame_idx} → {frame_idx+1} vs intensity at frame {frame_idx+1}...")

            current_image = images[frame_idx]
            current_mask = masks[frame_idx]
            previous_mask = masks[frame_idx - 1]

            # Analyze frame pair (current frame intensity vs past movement)
            results = analyze_frame_pair(current_image, current_mask, previous_mask,
                                       CONFIG, temporal_direction='past')
            all_results.append(results)

            if 'error' not in results:
                all_movements.append(results['local_movement_scores'])
                all_intensities.append(results['intensities'])
                all_valid_points.append(results['valid_points'])
                movement_scores.append(results['movement_score'])
                movement_types.append(results['movement_type'])
                processed_pairs += 1

                # Generate frame-by-frame plots if enabled
                should_plot_this_frame = ((frame_idx-1) % VIZ_CONFIG['plot_every_nth_frame'] == 0) or (VIZ_CONFIG['plot_every_nth_frame'] == 1)

                if should_plot_this_frame and OUTPUT_CONFIG['save_frame_plots']:
                    if VIZ_CONFIG['save_edge_transition_plots']:
                        print(f"  Creating edge transition plot for frames {frame_idx} → {frame_idx + 1}...")
                        create_frame_edge_transition_plot(frame_idx-1, current_image, current_mask, previous_mask, results, OUTPUT_DIR, VIZ_CONFIG)

                    if VIZ_CONFIG['save_sampling_plots']:
                        print(f"  Creating sampling plot for frame {frame_idx + 1}...")
                        create_frame_sampling_plot(frame_idx-1, current_image, current_mask, results, OUTPUT_DIR, VIZ_CONFIG)

                    if VIZ_CONFIG['save_intensity_plots']:
                        print(f"  Creating intensity plot for frame {frame_idx + 1}...")
                        create_frame_intensity_plot(frame_idx-1, results, OUTPUT_DIR, VIZ_CONFIG)

                        # Create intensity overlay plot if enabled
                        if VIZ_CONFIG['save_intensity_overlay_plots']:
                            print(f"  Creating intensity overlay plot for frame {frame_idx + 1}...")
                            create_frame_intensity_overlay_plot(frame_idx-1, current_image, current_mask, results, OUTPUT_DIR, VIZ_CONFIG)

                    if VIZ_CONFIG['save_movement_plots']:
                        print(f"  Creating movement plot for frame {frame_idx + 1}...")
                        create_frame_movement_plot(frame_idx-1, results, OUTPUT_DIR, VIZ_CONFIG)

                        # Create movement overlay plot if enabled
                        if VIZ_CONFIG['save_movement_overlay_plots']:
                            print(f"  Creating movement overlay plot for frame {frame_idx + 1}...")
                            create_frame_movement_overlay_plot(frame_idx-1, current_image, current_mask, results, OUTPUT_DIR, VIZ_CONFIG)

            else:
                print(f"  Warning: {results['error']}")
                # Add empty data to maintain indexing
                movement_scores.append(0.0)
                movement_types.append('stable')

    print(f"\nSuccessfully processed {processed_pairs}/{images.shape[0]-1} frame transitions")

    if processed_pairs == 0:
        print("No frame pairs could be processed. Check your input files and parameters.")
        return

    # Generate correlation analysis if enabled
    correlation_stats = {}
    type_correlation_stats = {}

    if OUTPUT_CONFIG['save_correlation_plots']:
        print("\nGenerating movement-intensity correlation analysis...")
        correlation_stats = create_movement_correlation_plot(
            all_movements, all_intensities, all_valid_points, movement_types,
            OUTPUT_DIR, VIZ_CONFIG, temporal_direction) or {}

        # Generate movement type correlation plot if enabled
        if VIZ_CONFIG['save_movement_type_correlation']:
            print("Generating movement type correlation analysis...")
            type_correlation_stats = create_movement_type_correlation_plot(
                all_movements, all_intensities, all_valid_points, movement_types,
                OUTPUT_DIR, VIZ_CONFIG, temporal_direction) or {}

        # Generate movement summary
        print("Generating movement summary...")
        create_movement_summary_plot(movement_scores, movement_types, OUTPUT_DIR, VIZ_CONFIG)

    # Calculate summary statistics (always computed for summary JSON)
    total_points = sum(len(valid) for valid in all_valid_points)
    valid_measurements = sum(np.sum(valid) for valid in all_valid_points)

    # Movement type statistics
    extending_count = movement_types.count('extending')
    retracting_count = movement_types.count('retracting')
    stable_count = movement_types.count('stable')

    # Calculate correlation statistics for summary even if plots not saved
    if not correlation_stats and processed_pairs > 0:
        # Combine data from all frame pairs for correlation calculation
        combined_movements = []
        combined_intensities = []

        for i in range(len(all_movements)):
            movements = all_movements[i]
            intensities = all_intensities[i]
            valid_points = all_valid_points[i]

            # Filter for valid points
            valid_mov = movements[valid_points]
            valid_int = intensities[valid_points]

            combined_movements.extend(valid_mov)
            combined_intensities.extend(valid_int)

        # Convert to numpy arrays
        combined_movements = np.array(combined_movements)
        combined_intensities = np.array(combined_intensities)

        # Calculate correlation statistics
        if len(combined_movements) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_movements, combined_intensities)
            correlation_stats = {
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'sample_size': int(len(combined_movements)),
                'slope': float(slope)
            }
        else:
            correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': int(len(combined_movements)),
                'slope': None
            }

    summary_stats = {
        'total_frames': int(images.shape[0]),
        'total_transitions': int(images.shape[0] - 1),
        'processed_transitions': int(processed_pairs),
        'total_sampled_points': int(total_points),
        'valid_measurements': int(valid_measurements),
        'valid_measurement_percentage': float((valid_measurements / total_points * 100) if total_points > 0 else 0),
        'sampling_method': sampling_method,
        'movement_sampling_method': movement_sampling_method,
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

    # Add movement type correlation stats if available
    if type_correlation_stats:
        summary_stats['movement_type_correlation_analysis'] = type_correlation_stats

    # Save results based on output configuration
    print("\nSaving results...")

    if OUTPUT_CONFIG['save_detailed_csv']:
        save_results_csv(all_results, movement_scores, movement_types, OUTPUT_DIR)

    # Summary statistics are always saved (recommended)
    if OUTPUT_CONFIG['save_summary_json']:
        save_summary_stats(summary_stats, OUTPUT_DIR)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Sampling method: {sampling_method}")
    print(f"Processed transitions: {processed_pairs}/{images.shape[0]-1}")
    print(f"Valid measurements: {valid_measurements}/{total_points} ({summary_stats['valid_measurement_percentage']:.1f}%)")
    print(f"Movement distribution:")
    total_transitions = len(movement_types) if movement_types else 1  # Avoid division by zero
    print(f"  - Extending: {extending_count} ({extending_count/total_transitions*100:.1f}%)")
    print(f"  - Retracting: {retracting_count} ({retracting_count/total_transitions*100:.1f}%)")
    print(f"  - Stable: {stable_count} ({stable_count/total_transitions*100:.1f}%)")

    if correlation_stats and correlation_stats.get('r_squared') is not None:
        print(f"Movement-Intensity Correlation R²: {correlation_stats['r_squared']:.3f}")
        print(f"P-value: {correlation_stats['p_value']:.3e}")
    else:
        print("Insufficient data for movement-intensity correlation analysis")

    if type_correlation_stats and type_correlation_stats.get('r_squared') is not None:
        print(f"Movement Type Correlation R²: {type_correlation_stats['r_squared']:.3f}")
        print(f"Movement Type P-value: {type_correlation_stats['p_value']:.3e}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    # Print what was actually saved based on configuration
    saved_files = []

    if OUTPUT_CONFIG['save_summary_json']:
        saved_files.append("- movement_summary_statistics.json: Complete analysis summary and parameters")

    if OUTPUT_CONFIG['save_detailed_csv']:
        saved_files.append("- detailed_movement_results.csv: Point-by-point data with status and sampling method")

    if OUTPUT_CONFIG['save_correlation_plots']:
        saved_files.append("- movement_intensity_correlation.png: Main correlation plots")
        saved_files.append("- movement_summary.png: Movement over time and distribution")
        if VIZ_CONFIG['save_movement_type_correlation']:
            saved_files.append("- movement_type_intensity_correlation.png: Movement type correlation plot")

    if OUTPUT_CONFIG['save_frame_plots']:
        if VIZ_CONFIG['save_edge_transition_plots']:
            saved_files.append("- edge_transition_plots/: Edge transition visualizations showing consecutive frame edges")
            saved_files.append("  * Shows previous edge (orange, dashed), current edge (white, solid), movement areas (color overlay)")
        if VIZ_CONFIG['save_sampling_plots']:
            saved_files.append("- frame_sampling_plots/: Sampling region visualizations")
            saved_files.append("  * Colors: lime=valid, blue=rejected, yellow=rotated valid, red line=cell edge")
        if VIZ_CONFIG['save_intensity_plots']:
            saved_files.append("- frame_intensity_plots/: Intensity analysis plots")
            if VIZ_CONFIG['save_intensity_overlay_plots']:
                saved_files.append("  * Includes intensity overlay plots on cell images")
        if VIZ_CONFIG['save_movement_plots']:
            saved_files.append("- frame_movement_plots/: Movement analysis plots")
            if VIZ_CONFIG['save_movement_overlay_plots']:
                saved_files.append("  * Includes movement type overlay plots on cell images")

    if saved_files:
        print("\n".join(saved_files))
    else:
        print("- No additional files saved (summary-only mode or all outputs disabled)")

    # Print frame plotting summary if applicable
    if OUTPUT_CONFIG['save_frame_plots'] and any([VIZ_CONFIG['save_edge_transition_plots'], VIZ_CONFIG['save_sampling_plots'], VIZ_CONFIG['save_intensity_plots'], VIZ_CONFIG['save_movement_plots']]):
        frames_plotted = len([i for i in range(processed_pairs) if i % VIZ_CONFIG['plot_every_nth_frame'] == 0])
        sampling_freq = VIZ_CONFIG['plot_every_nth_sampling_point']
        print(f"\nFrame plots generated for {frames_plotted} frames (every {VIZ_CONFIG['plot_every_nth_frame']} frames)")
        if VIZ_CONFIG['save_sampling_plots']:
            print(f"Sampling plots show every {sampling_freq} rectangle to avoid clutter")

    # Print sampling method information
    method_info = {
        'standard': 'Arc-length based: Points distributed evenly along contour perimeter',
        'x_axis': 'X-coordinate based: Points distributed at regular X intervals across cell width',
        'y_axis': 'Y-coordinate based: Points distributed at regular Y intervals across cell height'
    }

    movement_method_info = {
        'neighbourhood': 'Neighborhood sampling: Movement sampled in square region around each edge point',
        'outward_region': 'Outward region sampling: Movement sampled in rectangular region pointing out from cell'
    }

    temporal_direction_info = {
        'past': 'Past direction: Intensity at frame N correlated with movement from frame N-1 to N',
        'future': 'Future direction: Intensity at frame N correlated with movement from frame N to N+1'
    }

    print(f"\nSampling method used: {sampling_method}")
    print(f"  → {method_info.get(sampling_method, 'Unknown method')}")
    print(f"Movement sampling method used: {movement_sampling_method}")
    print(f"  → {movement_method_info.get(movement_sampling_method, 'Unknown method')}")
    print(f"Temporal direction used: {temporal_direction}")
    print(f"  → {temporal_direction_info.get(temporal_direction, 'Unknown direction')}")

    if movement_sampling_method == 'neighbourhood':
        neighborhood_size = CONFIG.get('movement_sampling_neighbourhood_size', 20)
        print(f"  → Neighborhood size: {neighborhood_size} pixels")
    elif movement_sampling_method == 'outward_region':
        region_depth = CONFIG.get('movement_region_depth', 50)
        region_width = CONFIG.get('movement_region_width', 30)
        print(f"  → Region dimensions: {region_depth} pixels depth × {region_width} pixels width")

if __name__ == "__main__":
    main()
