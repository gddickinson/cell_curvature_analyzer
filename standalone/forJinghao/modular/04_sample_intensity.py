#!/usr/bin/env python3
"""
Step 4: Sample Intensity

Samples PIEZO1 intensity values at cell edge locations using rectangular
sampling regions. Validates each sampling region for sufficient cell coverage.

Author: George Dickinson
Date: November 4, 2025
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from utils.config import Config
from utils.intensity_sampling import (get_edge_points_xaxis, calculate_tangent,
                                     create_sampling_rectangle, validate_sampling_rectangle,
                                     sample_intensity_at_rectangle)


def sample_intensity_all_frames(images, edges, masks, config):
    """
    Sample PIEZO1 intensity at edge points for all frames.
    
    Parameters
    ----------
    images : np.ndarray
        Image stack
    edges : dict
        Dictionary of edge contours
    masks : np.ndarray
        Mask stack
    config : Config
        Configuration object
    
    Returns
    -------
    dict
        Dictionary with intensity data per frame
    """
    intensity_data = {}
    n_frames = len(edges)
    
    print(f"Sampling intensity for {n_frames} frames...")
    
    for frame_idx in range(n_frames):
        contour = edges[frame_idx]
        
        if len(contour) == 0:
            print(f"  Warning: Frame {frame_idx} has no edge, skipping")
            continue
        
        # Get sampling points along edge
        sampling_points = get_edge_points_xaxis(
            contour, 
            n_points=config.config['n_points'],
            exclude_endpoints=config.config['exclude_endpoints']
        )
        
        if len(sampling_points) == 0:
            print(f"  Warning: Frame {frame_idx} has no valid sampling points")
            continue
        
        # Sample intensity at each point
        intensities = []
        rectangles = []
        valid_flags = []
        centers = []
        
        for point in sampling_points:
            # Find closest contour point to get tangent
            distances = np.sum((contour - point)**2, axis=1)
            idx = np.argmin(distances)
            
            # Calculate tangent at this point
            tangent = calculate_tangent(contour, idx, window=5)
            
            # Create sampling rectangle
            corners, center = create_sampling_rectangle(
                point, tangent,
                width=config.config['width'],
                depth=config.config['depth']
            )
            
            # Validate rectangle
            is_valid = validate_sampling_rectangle(
                corners, masks[frame_idx],
                min_coverage=config.config['min_cell_coverage']
            )
            
            # Try rotation if invalid and rotation is enabled
            if not is_valid and config.config['try_rotation']:
                # Rotate 180 degrees
                tangent_rotated = -tangent
                corners_rotated, center_rotated = create_sampling_rectangle(
                    point, tangent_rotated,
                    width=config.config['width'],
                    depth=config.config['depth']
                )
                
                is_valid_rotated = validate_sampling_rectangle(
                    corners_rotated, masks[frame_idx],
                    min_coverage=config.config['min_cell_coverage']
                )
                
                if is_valid_rotated:
                    corners = corners_rotated
                    center = center_rotated
                    is_valid = True
            
            # Sample intensity if valid
            if is_valid:
                intensity = sample_intensity_at_rectangle(images[frame_idx], corners)
            else:
                intensity = np.nan
            
            intensities.append(intensity)
            rectangles.append(corners)
            valid_flags.append(is_valid)
            centers.append(center)
        
        intensity_data[frame_idx] = {
            'intensities': np.array(intensities),
            'sampling_points': sampling_points,
            'rectangles': rectangles,
            'valid_flags': np.array(valid_flags),
            'centers': np.array(centers),
            'n_valid': np.sum(valid_flags),
            'n_total': len(sampling_points)
        }
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx+1}/{n_frames} frames")
    
    return intensity_data


def save_intensity_data(intensity_data, output_dir, config):
    """
    Save intensity sampling data.
    
    Parameters
    ----------
    intensity_data : dict
        Intensity data dictionary
    output_dir : str
        Output directory
    config : Config
        Configuration object
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_dict = {}
    for frame_idx, data in intensity_data.items():
        save_dict[f'frame_{frame_idx}_intensities'] = data['intensities']
        save_dict[f'frame_{frame_idx}_sampling_points'] = data['sampling_points']
        save_dict[f'frame_{frame_idx}_valid_flags'] = data['valid_flags']
        save_dict[f'frame_{frame_idx}_centers'] = data['centers']
        
        # Save rectangles separately (list of arrays)
        for i, rect in enumerate(data['rectangles']):
            save_dict[f'frame_{frame_idx}_rectangle_{i}'] = rect
    
    np.savez_compressed(output_dir / 'intensity_data.npz', **save_dict)
    print(f"Saved intensity data to: {output_dir / 'intensity_data.npz'}")
    
    # Save summary statistics
    total_valid = sum(d['n_valid'] for d in intensity_data.values())
    total_points = sum(d['n_total'] for d in intensity_data.values())
    
    all_valid_intensities = []
    for data in intensity_data.values():
        valid_intensities = data['intensities'][~np.isnan(data['intensities'])]
        all_valid_intensities.extend(valid_intensities)
    
    summary = {
        'n_frames': len(intensity_data),
        'total_sampling_points': int(total_points),
        'valid_sampling_points': int(total_valid),
        'validation_rate': float(total_valid / total_points * 100) if total_points > 0 else 0,
        'mean_intensity': float(np.mean(all_valid_intensities)) if all_valid_intensities else 0,
        'std_intensity': float(np.std(all_valid_intensities)) if all_valid_intensities else 0,
        'min_intensity': float(np.min(all_valid_intensities)) if all_valid_intensities else 0,
        'max_intensity': float(np.max(all_valid_intensities)) if all_valid_intensities else 0,
        'sampling_parameters': {
            'n_points': config.config['n_points'],
            'depth': config.config['depth'],
            'width': config.config['width'],
            'min_cell_coverage': config.config['min_cell_coverage'],
        }
    }
    
    with open(output_dir / 'intensity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {output_dir / 'intensity_summary.json'}")


def visualize_intensity_sampling(intensity_data, images, edges, output_dir, 
                                 n_examples=3):
    """
    Create visualizations of intensity sampling.
    
    Parameters
    ----------
    intensity_data : dict
        Intensity data
    images : np.ndarray
        Image stack
    edges : dict
        Edge contours
    output_dir : str
        Output directory
    n_examples : int
        Number of example frames to visualize
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_indices = list(intensity_data.keys())
    if len(frame_indices) > n_examples:
        step = len(frame_indices) // n_examples
        frame_indices = frame_indices[::step][:n_examples]
    
    for frame_idx in frame_indices:
        data = intensity_data[frame_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Image with sampling rectangles
        axes[0].imshow(images[frame_idx], cmap='gray')
        axes[0].plot(edges[frame_idx][:, 1], edges[frame_idx][:, 0], 
                    'b-', linewidth=2, alpha=0.5, label='Edge')
        
        for i, (rect, valid) in enumerate(zip(data['rectangles'], data['valid_flags'])):
            color = 'g' if valid else 'r'
            poly = Polygon(rect[:, ::-1], fill=False, edgecolor=color, 
                          linewidth=2, alpha=0.7)
            axes[0].add_patch(poly)
        
        axes[0].set_title(f'Frame {frame_idx}: Sampling Rectangles\n'
                         f'Valid: {data["n_valid"]}/{data["n_total"]}')
        axes[0].axis('off')
        
        # Plot 2: Image with intensity values
        axes[1].imshow(images[frame_idx], cmap='gray')
        axes[1].plot(edges[frame_idx][:, 1], edges[frame_idx][:, 0], 
                    'b-', linewidth=2, alpha=0.5)
        
        valid_mask = ~np.isnan(data['intensities'])
        if np.any(valid_mask):
            scatter = axes[1].scatter(
                data['centers'][valid_mask, 1],
                data['centers'][valid_mask, 0],
                c=data['intensities'][valid_mask],
                cmap='viridis', s=100, edgecolors='white', linewidth=2
            )
            plt.colorbar(scatter, ax=axes[1], label='Intensity')
        
        axes[1].set_title(f'Frame {frame_idx}: Intensity Values')
        axes[1].axis('off')
        
        # Plot 3: Intensity profile
        x_positions = np.arange(len(data['intensities']))
        axes[2].plot(x_positions, data['intensities'], 'o-', markersize=8)
        axes[2].axhline(y=np.nanmean(data['intensities']), color='r', 
                       linestyle='--', label=f'Mean: {np.nanmean(data["intensities"]):.1f}')
        axes[2].set_xlabel('Sampling Point Index')
        axes[2].set_ylabel('Intensity')
        axes[2].set_title(f'Frame {frame_idx}: Intensity Profile')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        save_path = output_dir / f'intensity_sampling_frame_{frame_idx:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization for frame {frame_idx}")


def main(input_dir, output_dir, config_path=None, visualize=False):
    """
    Main processing function.
    
    Parameters
    ----------
    input_dir : str
        Directory containing data from previous steps
    output_dir : str
        Output directory
    config_path : str, optional
        Path to configuration file
    visualize : bool
        Whether to generate visualizations
    """
    print("="*60)
    print("STEP 4: SAMPLE INTENSITY")
    print("="*60)
    
    # Load configuration
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config()
    
    print(f"\nConfiguration:")
    print(f"  n_points: {config.config['n_points']}")
    print(f"  depth: {config.config['depth']}")
    print(f"  width: {config.config['width']}")
    print(f"  min_cell_coverage: {config.config['min_cell_coverage']}")
    
    # Load data
    print("\nLoading data...")
    
    # Load images and masks from step1
    input_path = Path(input_dir)
    step1_dir = input_path.parent / 'step1_load'
    if not step1_dir.exists():
        step1_dir = input_path
    
    data_file = step1_dir / 'loaded_data.npz'
    if not data_file.exists():
        raise FileNotFoundError(f"Cannot find loaded_data.npz. Expected at: {data_file}")
    
    data = np.load(data_file)
    images = data['images']
    masks = data['masks']
    
    # Load edges from input_dir (step2)
    edges_data = np.load(input_path / 'edges.npz')
    edges = {int(k.split('_')[1]): v for k, v in edges_data.items()}
    
    print(f"Loaded {images.shape[0]} frames")
    print(f"Loaded {len(edges)} edge contours")
    
    # Sample intensity
    intensity_data = sample_intensity_all_frames(images, edges, masks, config)
    
    # Print summary
    print(f"\nIntensity Sampling Summary:")
    total_valid = sum(d['n_valid'] for d in intensity_data.values())
    total_points = sum(d['n_total'] for d in intensity_data.values())
    print(f"  Total sampling points: {total_points}")
    print(f"  Valid sampling points: {total_valid} ({total_valid/total_points*100:.1f}%)")
    
    all_valid_intensities = []
    for data in intensity_data.values():
        valid_intensities = data['intensities'][~np.isnan(data['intensities'])]
        all_valid_intensities.extend(valid_intensities)
    
    if all_valid_intensities:
        print(f"  Mean intensity: {np.mean(all_valid_intensities):.2f}")
        print(f"  Intensity range: [{np.min(all_valid_intensities):.2f}, "
              f"{np.max(all_valid_intensities):.2f}]")
    
    # Save results
    print("\nSaving results...")
    save_intensity_data(intensity_data, output_dir, config)
    
    # Visualize
    if visualize:
        print("\nGenerating visualizations...")
        visualize_intensity_sampling(intensity_data, images, edges, output_dir)
    
    print("\n" + "="*60)
    print("STEP 4 COMPLETE")
    print("="*60)
    
    return True


def run_unit_test(visualize=False, test_data_path=None):
    """Run unit test."""
    print("="*60)
    print("INTENSITY SAMPLING SCRIPT UNIT TEST")
    print("="*60)
    
    print("\n[Test] Creating synthetic test data...")
    
    # Create synthetic data
    n_frames = 5
    images = np.random.rand(n_frames, 100, 100) * 255
    
    masks = np.zeros((n_frames, 100, 100), dtype=np.uint8)
    yy, xx = np.ogrid[:100, :100]
    for i in range(n_frames):
        radius = 30 + i
        masks[i] = ((xx - 50)**2 + (yy - 50)**2 <= radius**2).astype(np.uint8)
    
    # Save as if from previous steps
    os.makedirs('test_output_04_input', exist_ok=True)
    np.savez_compressed('test_output_04_input/loaded_data.npz',
                       images=images, masks=masks)
    
    # Create edges
    from utils.edge_detection import detect_cell_edge
    edges = {}
    for i in range(n_frames):
        edges[i] = detect_cell_edge(masks[i])
    
    edge_arrays = {f'frame_{k}': v for k, v in edges.items()}
    np.savez_compressed('test_output_04_input/edges.npz', **edge_arrays)
    
    # Run main function
    success = main('test_output_04_input', 'test_output_04', 
                  config_path=None, visualize=visualize)
    
    assert success, "Intensity sampling should succeed"
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
            print("Usage: python 04_sample_intensity.py <input_dir> <output_dir> "
                  "[--config config.json] [--visualize]")
            sys.exit(1)
        
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        config_path = None
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
        visualize = '--visualize' in sys.argv or '--visualise' in sys.argv or '-v' in sys.argv
        
        success = main(input_dir, output_dir, config_path, visualize)
        sys.exit(0 if success else 1)
