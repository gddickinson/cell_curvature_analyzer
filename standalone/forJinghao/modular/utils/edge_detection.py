#!/usr/bin/env python3
"""
Edge Detection and Movement Analysis Module

Functions for detecting cell edges from binary masks and calculating
movement between consecutive frames using vertical displacement method.

Author: George Dickinson
Date: November 4, 2025
"""

import numpy as np
from skimage import measure
from typing import Tuple, Dict, Any


def detect_cell_edge(mask: np.ndarray) -> np.ndarray:
    """
    Detect the cell edge from a binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask (2D array)
    
    Returns
    -------
    np.ndarray
        Contour points as (N, 2) array of (row, col) coordinates
    """
    contours = measure.find_contours(mask, 0.5)
    
    if len(contours) > 0:
        # Return the longest contour
        contour = max(contours, key=len)
        return contour
    else:
        return np.array([])


def detect_edge_movement_vertical_xaxis(current_contour: np.ndarray,
                                       previous_contour: np.ndarray,
                                       current_mask: np.ndarray,
                                       previous_mask: np.ndarray,
                                       config: Dict[str, Any]) -> Tuple:
    """
    Detect movement using vertical displacement at every x-position.
    Tracks uppermost (minimum y) point at each x on both frames.
    
    Parameters
    ----------
    current_contour : np.ndarray
        Contour from current frame
    previous_contour : np.ndarray
        Contour from previous frame
    current_mask : np.ndarray
        Binary mask from current frame
    previous_mask : np.ndarray
        Binary mask from previous frame
    config : dict
        Configuration parameters including movement_threshold and min_movement_pixels
    
    Returns
    -------
    tuple
        (movement_score, movement_type, movement_map, displacements, x_positions)
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


def create_movement_map_from_xaxis_displacement(displacements: np.ndarray,
                                               x_positions: np.ndarray,
                                               image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create 2D movement map where each x-column contains the displacement value.
    
    Parameters
    ----------
    displacements : np.ndarray
        Array of displacement values for each x position
    x_positions : np.ndarray
        Array of x-coordinate positions
    image_shape : tuple
        Shape of the output image (height, width)
    
    Returns
    -------
    np.ndarray
        2D movement map
    """
    movement_map = np.zeros(image_shape, dtype=float)
    
    for i, x in enumerate(x_positions):
        if 0 <= x < image_shape[1] and not np.isnan(displacements[i]):
            movement_map[:, int(x)] = displacements[i]
    
    return movement_map


# =============================================================================
# UNIT TEST
# =============================================================================

def run_unit_test(visualize: bool = False, test_data_path: str = None):
    """
    Unit test for edge detection module.
    
    Parameters
    ----------
    visualize : bool
        Whether to generate visualization plots
    test_data_path : str, optional
        Path to test data directory (should contain test TIFF files)
    """
    print("="*60)
    print("EDGE DETECTION MODULE UNIT TEST")
    print("="*60)
    
    # Test 1: Detect edge from synthetic mask
    print("\n[Test 1] Testing edge detection on synthetic mask...")
    
    # Create a simple circular mask
    y, x = np.ogrid[-50:50, -50:50]
    mask = (x**2 + y**2 <= 30**2).astype(np.uint8)
    
    contour = detect_cell_edge(mask)
    assert len(contour) > 0, "Contour should be detected"
    assert contour.shape[1] == 2, "Contour should have 2 columns (y, x)"
    print(f"✓ Detected contour with {len(contour)} points")
    
    # Test 2: Empty mask handling
    print("\n[Test 2] Testing empty mask handling...")
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    contour = detect_cell_edge(empty_mask)
    assert len(contour) == 0, "Empty mask should return empty contour"
    print("✓ Empty mask correctly returns empty contour")
    
    # Test 3: Movement detection
    print("\n[Test 3] Testing movement detection...")
    
    # Create two masks with slight vertical displacement
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    
    # Mask 1: circle at y=50
    yy, xx = np.ogrid[:100, :100]
    mask1 = ((xx - 50)**2 + (yy - 50)**2 <= 20**2).astype(np.uint8)
    
    # Mask 2: circle moved up to y=45 (extending)
    mask2 = ((xx - 50)**2 + (yy - 45)**2 <= 20**2).astype(np.uint8)
    
    contour1 = detect_cell_edge(mask1)
    contour2 = detect_cell_edge(mask2)
    
    config = {
        'movement_threshold': 0.1,
        'min_movement_pixels': 5,
    }
    
    movement_score, movement_type, movement_map, displacements, x_positions = \
        detect_edge_movement_vertical_xaxis(contour2, contour1, mask2, mask1, config)
    
    print(f"  Movement score: {movement_score:.3f}")
    print(f"  Movement type: {movement_type}")
    assert movement_type == 'extending', "Upward movement should be 'extending'"
    assert movement_score < 0, "Upward movement should have negative score"
    print("✓ Movement detection working correctly")
    
    # Test 4: Movement map creation
    print("\n[Test 4] Testing movement map creation...")
    assert movement_map.shape == mask1.shape, "Movement map should match mask shape"
    assert np.any(movement_map != 0), "Movement map should contain non-zero values"
    print(f"✓ Movement map created with shape {movement_map.shape}")
    
    if visualize:
        print("\n[Visualization] Generating test plots...")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot masks
        axes[0, 0].imshow(mask1, cmap='gray')
        axes[0, 0].plot(contour1[:, 1], contour1[:, 0], 'r-', linewidth=2)
        axes[0, 0].set_title('Frame 1: Mask & Contour')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask2, cmap='gray')
        axes[0, 1].plot(contour2[:, 1], contour2[:, 0], 'b-', linewidth=2)
        axes[0, 1].set_title('Frame 2: Mask & Contour')
        axes[0, 1].axis('off')
        
        # Plot overlay
        axes[0, 2].imshow(mask1, cmap='gray', alpha=0.5)
        axes[0, 2].plot(contour1[:, 1], contour1[:, 0], 'r-', linewidth=2, label='Frame 1')
        axes[0, 2].plot(contour2[:, 1], contour2[:, 0], 'b-', linewidth=2, label='Frame 2')
        axes[0, 2].set_title('Overlay')
        axes[0, 2].legend()
        axes[0, 2].axis('off')
        
        # Plot movement map
        im = axes[1, 0].imshow(movement_map, cmap='RdBu_r', vmin=-5, vmax=5)
        axes[1, 0].set_title('Movement Map')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], label='Displacement (pixels)')
        
        # Plot displacement profile
        axes[1, 1].plot(x_positions, displacements, 'o-')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].axhline(y=config['movement_threshold'], color='r', linestyle='--', alpha=0.3, label='Threshold')
        axes[1, 1].axhline(y=-config['movement_threshold'], color='r', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('X Position')
        axes[1, 1].set_ylabel('Displacement (pixels)')
        axes[1, 1].set_title('Displacement Profile')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot histogram
        valid_disp = displacements[~np.isnan(displacements)]
        axes[1, 2].hist(valid_disp, bins=20, edgecolor='black')
        axes[1, 2].axvline(x=movement_score, color='r', linestyle='--', linewidth=2, label=f'Mean: {movement_score:.2f}')
        axes[1, 2].set_xlabel('Displacement (pixels)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Displacement Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = 'edge_detection_unit_test.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")
        plt.close()
    
    # Test with real data if provided
    if test_data_path:
        print(f"\n[Test 5] Testing with real data from: {test_data_path}")
        try:
            import tifffile
            from pathlib import Path
            
            test_path = Path(test_data_path)
            if test_path.is_dir():
                # Look for mask files
                mask_files = list(test_path.glob('*mask*.tif'))
                if mask_files:
                    masks = tifffile.imread(str(mask_files[0]))
                    if masks.ndim == 3:
                        mask1 = (masks[0] > 0).astype(np.uint8)
                        mask2 = (masks[1] > 0).astype(np.uint8)
                        
                        contour1 = detect_cell_edge(mask1)
                        contour2 = detect_cell_edge(mask2)
                        
                        print(f"  Frame 1 contour: {len(contour1)} points")
                        print(f"  Frame 2 contour: {len(contour2)} points")
                        
                        if len(contour1) > 0 and len(contour2) > 0:
                            movement_score, movement_type, _, _, _ = \
                                detect_edge_movement_vertical_xaxis(contour2, contour1, 
                                                                   mask2, mask1, config)
                            print(f"  Movement score: {movement_score:.3f}")
                            print(f"  Movement type: {movement_type}")
                            print("✓ Real data processed successfully")
                else:
                    print("  No mask files found in test directory")
            else:
                print("  Test data path is not a directory")
        except Exception as e:
            print(f"  Could not process real data: {e}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    visualize = '--visualize' in sys.argv or '-v' in sys.argv
    test_data = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--test-data' and i + 1 < len(sys.argv):
            test_data = sys.argv[i + 1]
    
    run_unit_test(visualize=visualize, test_data_path=test_data)
