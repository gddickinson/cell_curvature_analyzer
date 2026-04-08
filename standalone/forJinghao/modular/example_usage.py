#!/usr/bin/env python3
"""
Example Usage Script

Demonstrates how to use the modular pipeline components programmatically.

Author: George Dickinson  
Date: November 4, 2025
"""

import numpy as np
from utils.config import Config
from utils.edge_detection import detect_cell_edge, detect_edge_movement_vertical_xaxis
from utils.intensity_sampling import (get_edge_points_xaxis, calculate_tangent,
                                     create_sampling_rectangle, validate_sampling_rectangle,
                                     sample_intensity_at_rectangle)
from utils.visualization import plot_edge_overlay, plot_movement_map


def example_1_basic_edge_detection():
    """Example 1: Basic edge detection from a mask."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Edge Detection")
    print("="*60)
    
    # Create a synthetic circular mask
    y, x = np.ogrid[:100, :100]
    mask = ((x - 50)**2 + (y - 50)**2 <= 30**2).astype(np.uint8)
    
    # Detect edge
    contour = detect_cell_edge(mask)
    
    print(f"Detected edge with {len(contour)} points")
    print(f"Edge bounds: y=[{contour[:, 0].min():.1f}, {contour[:, 0].max():.1f}], "
          f"x=[{contour[:, 1].min():.1f}, {contour[:, 1].max():.1f}]")
    
    return mask, contour


def example_2_movement_detection():
    """Example 2: Detect movement between two frames."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Movement Detection")
    print("="*60)
    
    # Create two masks with vertical shift
    yy, xx = np.ogrid[:100, :100]
    mask1 = ((xx - 50)**2 + (yy - 50)**2 <= 25**2).astype(np.uint8)
    mask2 = ((xx - 50)**2 + (yy - 45)**2 <= 25**2).astype(np.uint8)  # Moved up (extending)
    
    # Detect edges
    contour1 = detect_cell_edge(mask1)
    contour2 = detect_cell_edge(mask2)
    
    # Create config
    config = Config()
    
    # Calculate movement
    movement_score, movement_type, movement_map, displacements, x_positions = \
        detect_edge_movement_vertical_xaxis(contour2, contour1, mask2, mask1, config.config)
    
    print(f"Movement score: {movement_score:.3f} pixels")
    print(f"Movement type: {movement_type}")
    print(f"Valid displacements: {np.sum(~np.isnan(displacements))}/{len(displacements)}")
    
    return movement_score, movement_type, movement_map


def example_3_intensity_sampling():
    """Example 3: Sample intensity at edge points."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Intensity Sampling")
    print("="*60)
    
    # Create synthetic image and mask
    image = np.random.rand(100, 100) * 255
    y, x = np.ogrid[:100, :100]
    mask = ((x - 50)**2 + (y - 50)**2 <= 30**2).astype(np.uint8)
    
    # Detect edge
    contour = detect_cell_edge(mask)
    
    # Get sampling points
    config = Config()
    points = get_edge_points_xaxis(contour, n_points=config.config['n_points'],
                                   exclude_endpoints=config.config['exclude_endpoints'])
    
    print(f"Selected {len(points)} sampling points")
    
    # Sample intensity at each point
    intensities = []
    for i, point in enumerate(points):
        # Calculate tangent
        idx = np.argmin(np.sum((contour - point)**2, axis=1))
        tangent = calculate_tangent(contour, idx)
        
        # Create sampling rectangle
        corners, center = create_sampling_rectangle(
            point, tangent, 
            width=config.config['width'],
            depth=config.config['depth']
        )
        
        # Validate
        if validate_sampling_rectangle(corners, mask, config.config['min_cell_coverage']):
            # Sample intensity
            intensity = sample_intensity_at_rectangle(image, corners)
            intensities.append(intensity)
        else:
            intensities.append(np.nan)
    
    valid_intensities = [i for i in intensities if not np.isnan(i)]
    print(f"Valid intensity samples: {len(valid_intensities)}/{len(points)}")
    print(f"Mean intensity: {np.mean(valid_intensities):.2f}")
    
    return intensities


def example_4_custom_configuration():
    """Example 4: Use custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    custom_config = {
        'analysis': {
            'n_points': 20,
            'depth': 300,
            'width': 100,
            'movement_threshold': 0.2,
            'temporal_direction': 'future',
        }
    }
    
    config = Config(custom_config)
    
    print(f"Configuration: {config}")
    print(f"  n_points: {config.config['n_points']}")
    print(f"  depth: {config.config['depth']}")
    print(f"  width: {config.config['width']}")
    print(f"  movement_threshold: {config.config['movement_threshold']}")
    print(f"  temporal_direction: {config.config['temporal_direction']}")
    
    # Validate
    is_valid = config.validate()
    print(f"  Configuration valid: {is_valid}")
    
    # Save to file
    config.save('example_config.json')
    print(f"  Saved to: example_config.json")
    
    # Load back
    loaded_config = Config.load('example_config.json')
    print(f"  Loaded config: {loaded_config}")
    
    return config


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MODULAR PIPELINE - USAGE EXAMPLES")
    print("="*70)
    print("\nThese examples show how to use the utility modules programmatically.")
    print("For command-line usage, see QUICK_START.txt")
    
    # Run examples
    example_1_basic_edge_detection()
    example_2_movement_detection()
    example_3_intensity_sampling()
    example_4_custom_configuration()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run the unit tests: python utils/config.py --test --visualize")
    print("  2. Try with your data: python run_single_analysis.py <images> <masks> <output>")
    print("  3. Customize parameters: Edit example_config.json")
    print("\nSee QUICK_START.txt and USAGE_GUIDE.txt for more information!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
