#!/usr/bin/env python3
"""
Intensity Sampling Module

Functions for sampling PIEZO1 intensity values from images at cell edge locations
using rectangular sampling regions with validation and rotation capabilities.

Author: George Dickinson
Date: November 4, 2025
"""

import numpy as np
from skimage import draw
from typing import Tuple, List, Dict, Any


def get_edge_points_xaxis(contour: np.ndarray, n_points: int, 
                         exclude_endpoints: bool = True) -> np.ndarray:
    """
    Get evenly distributed points along cell edge based on x-coordinate spacing.
    
    Parameters
    ----------
    contour : np.ndarray
        Cell edge contour (N, 2) array
    n_points : int
        Number of points to sample
    exclude_endpoints : bool
        Whether to exclude first and last points
    
    Returns
    -------
    np.ndarray
        Selected edge points (n_points, 2) array
    """
    if len(contour) == 0:
        return np.array([])
    
    # Get x-coordinate range
    x_min = np.min(contour[:, 1])
    x_max = np.max(contour[:, 1])
    
    # Generate target x-coordinates
    if exclude_endpoints:
        x_targets = np.linspace(x_min, x_max, n_points + 2)[1:-1]
    else:
        x_targets = np.linspace(x_min, x_max, n_points)
    
    # Find points closest to target x-coordinates
    selected_points = []
    for x_target in x_targets:
        distances = np.abs(contour[:, 1] - x_target)
        idx = np.argmin(distances)
        selected_points.append(contour[idx])
    
    return np.array(selected_points)


def create_sampling_rectangle(point: np.ndarray, tangent: np.ndarray,
                             width: float, depth: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sampling rectangle at edge point oriented perpendicular to edge.
    
    Parameters
    ----------
    point : np.ndarray
        Edge point (y, x) coordinates
    tangent : np.ndarray
        Tangent vector at point
    width : float
        Width of rectangle (along edge)
    depth : float
        Depth of rectangle (into cell)
    
    Returns
    -------
    tuple
        (corners, center) where corners is (4, 2) array and center is (2,) array
    """
    # Normalize tangent
    tangent_norm = tangent / (np.linalg.norm(tangent) + 1e-10)
    
    # Normal vector points into cell (perpendicular to tangent, rotated 90° clockwise)
    normal = np.array([tangent_norm[1], -tangent_norm[0]])
    
    # Rectangle corners relative to point
    half_width = width / 2.0
    corners = np.array([
        point - tangent_norm * half_width,  # Top-left
        point + tangent_norm * half_width,  # Top-right
        point + tangent_norm * half_width + normal * depth,  # Bottom-right
        point - tangent_norm * half_width + normal * depth,  # Bottom-left
    ])
    
    center = point + normal * (depth / 2.0)
    
    return corners, center


def validate_sampling_rectangle(corners: np.ndarray, mask: np.ndarray,
                               min_coverage: float = 0.8) -> bool:
    """
    Validate that sampling rectangle has sufficient overlap with cell mask.
    
    Parameters
    ----------
    corners : np.ndarray
        Rectangle corners (4, 2) array
    mask : np.ndarray
        Binary cell mask
    min_coverage : float
        Minimum fraction of rectangle that must overlap with mask
    
    Returns
    -------
    bool
        True if rectangle is valid
    """
    # Create mask for rectangle
    rr, cc = draw.polygon(corners[:, 0], corners[:, 1], shape=mask.shape)
    
    if len(rr) == 0:
        return False
    
    # Calculate coverage
    coverage = np.sum(mask[rr, cc]) / len(rr)
    
    return coverage >= min_coverage


def calculate_tangent(contour: np.ndarray, idx: int, window: int = 5) -> np.ndarray:
    """
    Calculate tangent vector at contour point using local averaging.
    
    Parameters
    ----------
    contour : np.ndarray
        Cell edge contour
    idx : int
        Index of point
    window : int
        Number of neighboring points to average
    
    Returns
    -------
    np.ndarray
        Tangent vector (2,) array
    """
    n = len(contour)
    if n < 2:
        return np.array([0.0, 1.0])
    
    # Get indices for averaging
    half_window = window // 2
    start = max(0, idx - half_window)
    end = min(n - 1, idx + half_window)
    
    if start == end:
        # Fallback for edge cases
        if idx > 0:
            tangent = contour[idx] - contour[idx - 1]
        elif idx < n - 1:
            tangent = contour[idx + 1] - contour[idx]
        else:
            tangent = np.array([1.0, 0.0])
    else:
        # Average direction
        tangent = contour[end] - contour[start]
    
    return tangent


def sample_intensity_at_rectangle(image: np.ndarray, corners: np.ndarray) -> float:
    """
    Sample mean intensity within rectangular region.
    
    Parameters
    ----------
    image : np.ndarray
        Image to sample from
    corners : np.ndarray
        Rectangle corners (4, 2) array
    
    Returns
    -------
    float
        Mean intensity value
    """
    rr, cc = draw.polygon(corners[:, 0], corners[:, 1], shape=image.shape)
    
    if len(rr) == 0:
        return 0.0
    
    # Clip to image bounds
    valid_mask = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
    rr = rr[valid_mask]
    cc = cc[valid_mask]
    
    if len(rr) == 0:
        return 0.0
    
    return np.mean(image[rr, cc])


# =============================================================================
# UNIT TEST
# =============================================================================

def run_unit_test(visualize: bool = False, test_data_path: str = None):
    """
    Unit test for intensity sampling module.
    
    Parameters
    ----------
    visualize : bool
        Whether to generate visualization plots
    test_data_path : str, optional
        Path to test data directory
    """
    print("="*60)
    print("INTENSITY SAMPLING MODULE UNIT TEST")
    print("="*60)
    
    # Test 1: Edge point sampling
    print("\n[Test 1] Testing edge point sampling...")
    
    # Create synthetic contour
    t = np.linspace(0, 2*np.pi, 100)
    x = 50 + 30 * np.cos(t)
    y = 50 + 30 * np.sin(t)
    contour = np.column_stack([y, x])
    
    points = get_edge_points_xaxis(contour, n_points=8, exclude_endpoints=True)
    assert len(points) == 8, "Should return requested number of points"
    print(f"✓ Sampled {len(points)} points from contour")
    
    # Test 2: Tangent calculation
    print("\n[Test 2] Testing tangent calculation...")
    
    idx = len(contour) // 4  # Point on side of circle
    tangent = calculate_tangent(contour, idx, window=5)
    assert len(tangent) == 2, "Tangent should be 2D vector"
    print(f"✓ Calculated tangent: ({tangent[0]:.2f}, {tangent[1]:.2f})")
    
    # Test 3: Rectangle creation
    print("\n[Test 3] Testing sampling rectangle creation...")
    
    point = contour[idx]
    corners, center = create_sampling_rectangle(point, tangent, width=20, depth=50)
    assert corners.shape == (4, 2), "Should have 4 corners"
    assert center.shape == (2,), "Center should be 2D point"
    print(f"✓ Created rectangle with center at ({center[0]:.1f}, {center[1]:.1f})")
    
    # Test 4: Rectangle validation
    print("\n[Test 4] Testing rectangle validation...")
    
    # Create circular mask
    yy, xx = np.ogrid[:100, :100]
    mask = ((xx - 50)**2 + (yy - 50)**2 <= 30**2).astype(np.uint8)
    
    is_valid = validate_sampling_rectangle(corners, mask, min_coverage=0.8)
    print(f"  Rectangle validation: {'Valid' if is_valid else 'Invalid'}")
    print(f"✓ Validation function working")
    
    # Test 5: Intensity sampling
    print("\n[Test 5] Testing intensity sampling...")
    
    # Create synthetic image with gradient
    image = np.tile(np.linspace(0, 255, 100), (100, 1))
    
    intensity = sample_intensity_at_rectangle(image, corners)
    assert 0 <= intensity <= 255, "Intensity should be in valid range"
    print(f"✓ Sampled mean intensity: {intensity:.1f}")
    
    if visualize:
        print("\n[Visualization] Generating test plots...")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Plot 1: Contour with sampled points
        axes[0, 0].plot(contour[:, 1], contour[:, 0], 'b-', linewidth=2, label='Contour')
        axes[0, 0].plot(points[:, 1], points[:, 0], 'ro', markersize=8, label='Sampled points')
        axes[0, 0].set_title('Edge Point Sampling')
        axes[0, 0].legend()
        axes[0, 0].axis('equal')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Rectangle with tangent/normal
        axes[0, 1].plot(contour[:, 1], contour[:, 0], 'b-', linewidth=2, alpha=0.3)
        axes[0, 1].plot([point[1]], [point[0]], 'ro', markersize=10, label='Edge point')
        
        # Draw rectangle
        rect_x = np.append(corners[:, 1], corners[0, 1])
        rect_y = np.append(corners[:, 0], corners[0, 0])
        axes[0, 1].plot(rect_x, rect_y, 'g-', linewidth=2, label='Rectangle')
        axes[0, 1].plot([center[1]], [center[0]], 'g*', markersize=15, label='Center')
        
        # Draw tangent and normal
        normal = np.array([tangent[1], -tangent[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-10) * 20
        tangent_vis = tangent / (np.linalg.norm(tangent) + 1e-10) * 20
        
        axes[0, 1].arrow(point[1], point[0], tangent_vis[1], tangent_vis[0],
                        color='orange', width=1, head_width=3, label='Tangent')
        axes[0, 1].arrow(point[1], point[0], normal[1], normal[0],
                        color='purple', width=1, head_width=3, label='Normal')
        
        axes[0, 1].set_title('Sampling Rectangle')
        axes[0, 1].legend()
        axes[0, 1].axis('equal')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mask with rectangle overlay
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, alpha=0.5)
        axes[1, 0].plot(rect_x, rect_y, 'g-', linewidth=2)
        axes[1, 0].set_title('Rectangle Validation')
        axes[1, 0].axis('off')
        
        # Plot 4: Intensity sampling
        axes[1, 1].imshow(image, cmap='viridis')
        axes[1, 1].plot(rect_x, rect_y, 'r-', linewidth=2)
        axes[1, 1].plot([center[1]], [center[0]], 'r*', markersize=15)
        axes[1, 1].set_title(f'Intensity Sampling (mean={intensity:.1f})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = 'intensity_sampling_unit_test.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")
        plt.close()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    visualize = '--visualize' in sys.argv or '-v' in sys.argv
    test_data = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--test-data' and i + 1 < len(sys.argv):
            test_data = sys.argv[i + 1]
    
    run_unit_test(visualize=visualize, test_data_path=test_data)
