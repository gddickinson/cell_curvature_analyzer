#!/usr/bin/env python3
"""
Step 2: Detect Edges

Detects cell edges from binary masks for all frames using contour detection.
Saves contours for downstream movement analysis.

Author: George Dickinson
Date: November 4, 2025
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from utils.edge_detection import detect_cell_edge


def detect_edges_all_frames(masks: np.ndarray) -> dict:
    """
    Detect edges for all frames in mask stack.
    
    Parameters
    ----------
    masks : np.ndarray
        Stack of binary masks
    
    Returns
    -------
    dict
        Dictionary with frame indices as keys and contours as values
    """
    edges = {}
    n_frames = masks.shape[0]
    
    print(f"Detecting edges for {n_frames} frames...")
    
    for i in range(n_frames):
        contour = detect_cell_edge(masks[i])
        edges[i] = contour
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_frames} frames")
    
    return edges


def save_edges(edges: dict, output_dir: str):
    """
    Save detected edges to compressed numpy archive.
    
    Parameters
    ----------
    edges : dict
        Dictionary of edges
    output_dir : str
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to saveable format
    edge_arrays = {f'frame_{k}': v for k, v in edges.items()}
    
    np.savez_compressed(output_dir / 'edges.npz', **edge_arrays)
    print(f"Saved edges to: {output_dir / 'edges.npz'}")
    
    # Save summary statistics
    stats = {
        'n_frames': len(edges),
        'edge_lengths': {int(k): len(v) for k, v in edges.items()},
        'mean_edge_length': float(np.mean([len(v) for v in edges.values()])),
        'empty_frames': [int(k) for k, v in edges.items() if len(v) == 0],
    }
    
    with open(output_dir / 'edge_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved edge statistics to: {output_dir / 'edge_statistics.json'}")


def visualize_edges(edges: dict, masks: np.ndarray, output_dir: str,
                   n_examples: int = 3):
    """
    Create visualizations of detected edges.
    
    Parameters
    ----------
    edges : dict
        Dictionary of detected edges
    masks : np.ndarray
        Mask stack
    output_dir : str
        Output directory
    n_examples : int
        Number of example frames to visualize
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = len(edges)
    if n_frames <= n_examples:
        frame_indices = list(range(n_frames))
    else:
        frame_indices = np.linspace(0, n_frames-1, n_examples, dtype=int).tolist()
    
    fig, axes = plt.subplots(1, len(frame_indices), figsize=(5*len(frame_indices), 5))
    if len(frame_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(frame_indices):
        axes[i].imshow(masks[idx], cmap='gray')
        if len(edges[idx]) > 0:
            axes[i].plot(edges[idx][:, 1], edges[idx][:, 0], 'r-', linewidth=2)
        axes[i].set_title(f'Frame {idx}\n({len(edges[idx])} points)')
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / 'edge_detection.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")


def main(input_dir: str, output_dir: str, visualize: bool = False):
    """
    Main processing function.
    
    Parameters
    ----------
    input_dir : str
        Directory containing loaded_data.npz from step 1
    output_dir : str
        Output directory
    visualize : bool
        Whether to generate visualizations
    """
    print("="*60)
    print("STEP 2: DETECT EDGES")
    print("="*60)
    
    # Load data from step 1
    input_path = Path(input_dir) / 'loaded_data.npz'
    print(f"\nLoading data from: {input_path}")
    
    data = np.load(input_path)
    masks = data['masks']
    
    print(f"Loaded {masks.shape[0]} masks")
    
    # Detect edges
    edges = detect_edges_all_frames(masks)
    
    # Print summary
    empty_frames = [k for k, v in edges.items() if len(v) == 0]
    if empty_frames:
        print(f"\nWarning: Empty contours in frames: {empty_frames}")
    
    edge_lengths = [len(v) for v in edges.values() if len(v) > 0]
    if edge_lengths:
        print(f"\nEdge Statistics:")
        print(f"  Mean contour length: {np.mean(edge_lengths):.1f} points")
        print(f"  Min contour length: {np.min(edge_lengths)} points")
        print(f"  Max contour length: {np.max(edge_lengths)} points")
    
    # Save results
    print("\nSaving results...")
    save_edges(edges, output_dir)
    
    # Visualize
    if visualize:
        print("\nGenerating visualizations...")
        visualize_edges(edges, masks, output_dir)
    
    print("\n" + "="*60)
    print("STEP 2 COMPLETE")
    print("="*60)
    
    return True


def run_unit_test(visualize: bool = False, test_data_path: str = None):
    """Run unit test."""
    print("="*60)
    print("EDGE DETECTION SCRIPT UNIT TEST")
    print("="*60)
    
    # Create synthetic masks
    print("\n[Test] Creating synthetic test data...")
    n_frames = 5
    masks = np.zeros((n_frames, 100, 100), dtype=np.uint8)
    
    y, x = np.ogrid[:100, :100]
    for i in range(n_frames):
        radius = 30 + i * 2
        mask = (x - 50)**2 + (y - 50)**2 <= radius**2
        masks[i] = mask.astype(np.uint8)
    
    # Save as if from step 1
    os.makedirs('test_output_02_input', exist_ok=True)
    np.savez_compressed('test_output_02_input/loaded_data.npz',
                       images=np.random.rand(n_frames, 100, 100),
                       masks=masks)
    
    # Run main function
    success = main('test_output_02_input', 'test_output_02', visualize)
    
    assert success, "Edge detection should succeed"
    print("\n✓ Synthetic data test passed")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    if '--test' in sys.argv or len(sys.argv) == 1:
        visualize = '--visualize' in sys.argv or '--visualise' in sys.argv or '-v' in sys.argv
        test_data = None
        for i, arg in enumerate(sys.argv):
            if arg == '--test-data' and i + 1 < len(sys.argv):
                test_data = sys.argv[i + 1]
        run_unit_test(visualize=visualize, test_data_path=test_data)
    else:
        if len(sys.argv) < 3:
            print("Usage: python 02_detect_edges.py <input_dir> <output_dir> [--visualize]")
            sys.exit(1)
        
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        visualize = '--visualize' in sys.argv or '--visualise' in sys.argv or '-v' in sys.argv
        
        success = main(input_dir, output_dir, visualize)
        sys.exit(0 if success else 1)
