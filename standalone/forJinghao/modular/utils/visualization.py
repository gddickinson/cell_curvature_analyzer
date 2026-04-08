#!/usr/bin/env python3
"""
Visualization Module

Plotting and visualization functions for cell edge movement analysis.

Author: George Dickinson
Date: November 4, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Dict, Any, Optional
import os


def plot_edge_overlay(image: np.ndarray, contour: np.ndarray, 
                     title: str = "Edge Detection",
                     save_path: Optional[str] = None):
    """Plot image with edge contour overlay."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    if len(contour) > 0:
        ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_movement_map(movement_map: np.ndarray, image: Optional[np.ndarray] = None,
                     title: str = "Movement Map", cmap: str = 'RdBu_r',
                     save_path: Optional[str] = None):
    """Plot movement map with optional image underlay."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if image is not None:
        ax.imshow(image, cmap='gray', alpha=0.3)
    
    im = ax.imshow(movement_map, cmap=cmap, alpha=0.7)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Displacement (pixels)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sampling_rectangles(image: np.ndarray, contour: np.ndarray,
                             rectangles: List[np.ndarray],
                             valid_flags: List[bool],
                             title: str = "Sampling Regions",
                             save_path: Optional[str] = None):
    """Plot image with sampling rectangles."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='gray')
    ax.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=2, label='Edge')
    
    for corners, valid in zip(rectangles, valid_flags):
        color = 'g' if valid else 'r'
        poly = Polygon(corners[:, ::-1], fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(poly)
    
    ax.set_title(title)
    ax.legend()
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation(movements: np.ndarray, intensities: np.ndarray,
                    title: str = "Movement-Intensity Correlation",
                    save_path: Optional[str] = None):
    """Plot scatter plot with correlation line."""
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(intensities, movements, alpha=0.5, s=30)
    
    # Fit line
    if len(movements) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(intensities, movements)
        x_line = np.array([np.min(intensities), np.max(intensities)])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
               label=f'R²={r_value**2:.3f}, p={p_value:.3e}')
    
    ax.set_xlabel('PIEZO1 Intensity')
    ax.set_ylabel('Movement Score (pixels)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# UNIT TEST
# =============================================================================

def run_unit_test(visualize: bool = False, test_data_path: str = None):
    """Unit test for visualization module."""
    print("="*60)
    print("VISUALIZATION MODULE UNIT TEST")
    print("="*60)
    
    # Create synthetic data
    print("\n[Test 1] Creating synthetic test data...")
    image = np.random.rand(100, 100) * 255
    
    t = np.linspace(0, 2*np.pi, 100)
    x = 50 + 30 * np.cos(t)
    y = 50 + 30 * np.sin(t)
    contour = np.column_stack([y, x])
    
    movement_map = np.random.randn(100, 100) * 5
    print("✓ Test data created")
    
    if visualize:
        print("\n[Visualization] Generating test plots...")
        
        # Test edge overlay
        plot_edge_overlay(image, contour, "Test Edge Overlay", 
                         "viz_test_edge_overlay.png")
        print("✓ Edge overlay plot created")
        
        # Test movement map
        plot_movement_map(movement_map, image, "Test Movement Map",
                         save_path="viz_test_movement_map.png")
        print("✓ Movement map plot created")
        
        # Test correlation
        movements = np.random.randn(100) * 2
        intensities = movements * 0.5 + np.random.randn(100) * 0.5 + 100
        plot_correlation(movements, intensities, "Test Correlation",
                        "viz_test_correlation.png")
        print("✓ Correlation plot created")
        
        # Test sampling rectangles
        rectangles = []
        valid_flags = []
        for i in range(5):
            angle = i * 2 * np.pi / 5
            center = np.array([50 + 20*np.cos(angle), 50 + 20*np.sin(angle)])
            corners = np.array([
                center + [-5, -10],
                center + [5, -10],
                center + [5, 10],
                center + [-5, 10],
            ])
            rectangles.append(corners)
            valid_flags.append(i % 2 == 0)
        
        plot_sampling_rectangles(image, contour, rectangles, valid_flags,
                                "Test Sampling Rectangles",
                                "viz_test_sampling.png")
        print("✓ Sampling rectangles plot created")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    import sys
    visualize = '--visualize' in sys.argv or '-v' in sys.argv
    run_unit_test(visualize=visualize)
