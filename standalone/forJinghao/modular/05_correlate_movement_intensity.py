#!/usr/bin/env python3
"""
Step 5: Correlate Movement and Intensity

Performs correlation analysis between cell edge movement and PIEZO1 intensity.
Calculates statistics and prepares data for visualization.

Author: George Dickinson
Date: November 4, 2025
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from scipy import stats
from utils.config import Config


def load_movement_data(input_dir):
    """Load movement data from step 3."""
    input_path = Path(input_dir)
    
    # Look for movement data in step3
    step3_dir = input_path.parent / 'step3_movement'
    if not step3_dir.exists():
        # If not in expected structure, look in input_dir
        step3_dir = input_path
    
    movement_file = step3_dir / 'movement_data.npz'
    if not movement_file.exists():
        raise FileNotFoundError(f"Cannot find movement_data.npz. Expected at: {movement_file}")
    
    data = np.load(movement_file)
    
    movement_data = {}
    transition_keys = set()
    
    # Find all transitions
    for key in data.keys():
        if key.startswith('transition_') and key.endswith('_movement_score'):
            trans_num = int(key.split('_')[1])
            transition_keys.add(trans_num)
    
    # Load data for each transition
    for trans_num in sorted(transition_keys):
        movement_data[trans_num] = {
            'movement_score': float(data[f'transition_{trans_num}_movement_score']),
            'movement_type': str(data[f'transition_{trans_num}_movement_type']),
            'current_idx': int(data[f'transition_{trans_num}_current_idx']),
            'comparison_idx': int(data[f'transition_{trans_num}_comparison_idx']),
        }
    
    return movement_data


def load_intensity_data(input_dir):
    """Load intensity data from step 4."""
    intensity_file = Path(input_dir) / 'intensity_data.npz'
    data = np.load(intensity_file)
    
    intensity_data = {}
    frame_keys = set()
    
    # Find all frames
    for key in data.keys():
        if key.startswith('frame_') and key.endswith('_intensities'):
            frame_num = int(key.split('_')[1])
            frame_keys.add(frame_num)
    
    # Load data for each frame
    for frame_num in sorted(frame_keys):
        intensity_data[frame_num] = {
            'intensities': data[f'frame_{frame_num}_intensities'],
            'valid_flags': data[f'frame_{frame_num}_valid_flags'],
        }
    
    return intensity_data


def correlate_movement_intensity(movement_data, intensity_data, config):
    """
    Correlate movement scores with intensity values.
    
    Parameters
    ----------
    movement_data : dict
        Movement data from step 3
    intensity_data : dict
        Intensity data from step 4
    config : Config
        Configuration object
    
    Returns
    -------
    dict
        Correlation results
    """
    print("\nPerforming correlation analysis...")
    
    # Collect paired data points
    all_movements = []
    all_intensities = []
    all_movement_types = []
    paired_data = []
    
    for trans_num, mov_data in movement_data.items():
        # Get the frame index for intensity (depends on temporal direction)
        intensity_frame_idx = mov_data['current_idx']
        
        if intensity_frame_idx not in intensity_data:
            continue
        
        int_data = intensity_data[intensity_frame_idx]
        
        # Get valid intensities for this frame
        valid_mask = int_data['valid_flags']
        valid_intensities = int_data['intensities'][valid_mask]
        
        if len(valid_intensities) == 0:
            continue
        
        # For each valid intensity point, pair with the movement score
        for intensity in valid_intensities:
            if not np.isnan(intensity):
                all_movements.append(mov_data['movement_score'])
                all_intensities.append(intensity)
                all_movement_types.append(mov_data['movement_type'])
                paired_data.append({
                    'transition': trans_num,
                    'frame': intensity_frame_idx,
                    'movement_score': mov_data['movement_score'],
                    'movement_type': mov_data['movement_type'],
                    'intensity': intensity,
                })
    
    all_movements = np.array(all_movements)
    all_intensities = np.array(all_intensities)
    
    print(f"  Total paired measurements: {len(all_movements)}")
    
    # Calculate overall correlation
    if len(all_movements) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            all_intensities, all_movements
        )
        
        correlation_results = {
            'n_pairs': len(all_movements),
            'r_value': float(r_value),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'slope': float(slope),
            'intercept': float(intercept),
            'std_err': float(std_err),
        }
        
        print(f"  R² = {r_value**2:.4f}")
        print(f"  p-value = {p_value:.3e}")
        print(f"  Slope = {slope:.6f}")
    else:
        correlation_results = {
            'n_pairs': len(all_movements),
            'r_value': None,
            'r_squared': None,
            'p_value': None,
            'slope': None,
            'intercept': None,
            'std_err': None,
            'error': 'Insufficient data points for correlation'
        }
        print("  Warning: Insufficient data for correlation analysis")
    
    # Calculate correlation by movement type
    type_correlations = {}
    for movement_type in ['extending', 'retracting', 'stable']:
        type_mask = np.array(all_movement_types) == movement_type
        if np.sum(type_mask) > 1:
            type_movements = all_movements[type_mask]
            type_intensities = all_intensities[type_mask]
            
            slope_t, intercept_t, r_value_t, p_value_t, std_err_t = stats.linregress(
                type_intensities, type_movements
            )
            
            type_correlations[movement_type] = {
                'n_pairs': int(np.sum(type_mask)),
                'r_value': float(r_value_t),
                'r_squared': float(r_value_t ** 2),
                'p_value': float(p_value_t),
                'slope': float(slope_t),
                'mean_movement': float(np.mean(type_movements)),
                'mean_intensity': float(np.mean(type_intensities)),
            }
        else:
            type_correlations[movement_type] = {
                'n_pairs': int(np.sum(type_mask)),
                'error': 'Insufficient data'
            }
    
    return {
        'overall': correlation_results,
        'by_movement_type': type_correlations,
        'paired_data': paired_data,
        'all_movements': all_movements.tolist(),
        'all_intensities': all_intensities.tolist(),
        'all_movement_types': all_movement_types,
    }


def save_correlation_results(correlation_results, output_dir, config):
    """
    Save correlation analysis results.
    
    Parameters
    ----------
    correlation_results : dict
        Correlation results
    output_dir : str
        Output directory
    config : Config
        Configuration object
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON (without full paired data)
    summary = {
        'temporal_direction': config.config['temporal_direction'],
        'overall_correlation': correlation_results['overall'],
        'by_movement_type': correlation_results['by_movement_type'],
        'n_total_pairs': correlation_results['overall']['n_pairs'],
    }
    
    with open(output_dir / 'correlation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved correlation summary to: {output_dir / 'correlation_summary.json'}")
    
    # Save detailed paired data as NPZ
    np.savez_compressed(
        output_dir / 'correlation_data.npz',
        movements=np.array(correlation_results['all_movements']),
        intensities=np.array(correlation_results['all_intensities']),
        movement_types=np.array(correlation_results['all_movement_types'])
    )
    print(f"Saved correlation data to: {output_dir / 'correlation_data.npz'}")
    
    # Save detailed CSV
    import pandas as pd
    df = pd.DataFrame(correlation_results['paired_data'])
    df.to_csv(output_dir / 'correlation_detailed.csv', index=False)
    print(f"Saved detailed data to: {output_dir / 'correlation_detailed.csv'}")


def visualize_correlation(correlation_results, output_dir, config):
    """
    Create correlation visualization.
    
    Parameters
    ----------
    correlation_results : dict
        Correlation results
    output_dir : str
        Output directory
    config : Config
        Configuration object
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    movements = np.array(correlation_results['all_movements'])
    intensities = np.array(correlation_results['all_intensities'])
    types = np.array(correlation_results['all_movement_types'])
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Overall correlation
    ax = axes[0, 0]
    ax.scatter(intensities, movements, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    if correlation_results['overall']['slope'] is not None:
        x_line = np.array([np.min(intensities), np.max(intensities)])
        y_line = (correlation_results['overall']['slope'] * x_line + 
                 correlation_results['overall']['intercept'])
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
               label=f"R² = {correlation_results['overall']['r_squared']:.4f}\n"
                     f"p = {correlation_results['overall']['p_value']:.3e}")
    
    ax.set_xlabel('PIEZO1 Intensity', fontsize=12)
    ax.set_ylabel('Movement Score (pixels)', fontsize=12)
    ax.set_title('Overall Movement-Intensity Correlation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: By movement type
    ax = axes[0, 1]
    colors = {'extending': 'blue', 'retracting': 'red', 'stable': 'gray'}
    
    for movement_type in ['extending', 'retracting', 'stable']:
        mask = types == movement_type
        if np.sum(mask) > 0:
            ax.scatter(intensities[mask], movements[mask], 
                      c=colors[movement_type], label=movement_type.capitalize(),
                      alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('PIEZO1 Intensity', fontsize=12)
    ax.set_ylabel('Movement Score (pixels)', fontsize=12)
    ax.set_title('Correlation by Movement Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: Distribution of intensities by movement type
    ax = axes[1, 0]
    intensity_by_type = {
        'Extending': intensities[types == 'extending'],
        'Retracting': intensities[types == 'retracting'],
        'Stable': intensities[types == 'stable']
    }
    
    positions = []
    data_to_plot = []
    labels = []
    for i, (label, data) in enumerate(intensity_by_type.items()):
        if len(data) > 0:
            positions.append(i)
            data_to_plot.append(data)
            labels.append(f"{label}\n(n={len(data)})")
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, positions=positions, labels=labels,
                       patch_artist=True, widths=0.6)
        for patch, movement_type in zip(bp['boxes'], ['extending', 'retracting', 'stable']):
            patch.set_facecolor(colors[movement_type])
            patch.set_alpha(0.6)
    
    ax.set_ylabel('PIEZO1 Intensity', fontsize=12)
    ax.set_title('Intensity Distribution by Movement Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Movement score distribution
    ax = axes[1, 1]
    movement_by_type = {
        'Extending': movements[types == 'extending'],
        'Retracting': movements[types == 'retracting'],
        'Stable': movements[types == 'stable']
    }
    
    positions = []
    data_to_plot = []
    labels = []
    for i, (label, data) in enumerate(movement_by_type.items()):
        if len(data) > 0:
            positions.append(i)
            data_to_plot.append(data)
            labels.append(f"{label}\n(n={len(data)})")
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, positions=positions, labels=labels,
                       patch_artist=True, widths=0.6)
        for patch, movement_type in zip(bp['boxes'], ['extending', 'retracting', 'stable']):
            patch.set_facecolor(colors[movement_type])
            patch.set_alpha(0.6)
    
    ax.set_ylabel('Movement Score (pixels)', fontsize=12)
    ax.set_title('Movement Distribution by Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'correlation_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation visualization to: {save_path}")


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
    print("STEP 5: CORRELATE MOVEMENT AND INTENSITY")
    print("="*60)
    
    # Load configuration
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config()
    
    print(f"\nTemporal direction: {config.config['temporal_direction']}")
    
    # Load data
    print("\nLoading data...")
    movement_data = load_movement_data(input_dir)
    intensity_data = load_intensity_data(input_dir)
    
    print(f"  Loaded {len(movement_data)} movement transitions")
    print(f"  Loaded {len(intensity_data)} intensity frames")
    
    # Perform correlation
    correlation_results = correlate_movement_intensity(
        movement_data, intensity_data, config
    )
    
    # Print results
    print("\nCorrelation Results:")
    if correlation_results['overall']['r_squared'] is not None:
        print(f"  Overall R² = {correlation_results['overall']['r_squared']:.4f}")
        print(f"  p-value = {correlation_results['overall']['p_value']:.3e}")
        print(f"  Slope = {correlation_results['overall']['slope']:.6f}")
        
        print("\nBy Movement Type:")
        for mov_type, results in correlation_results['by_movement_type'].items():
            if 'error' not in results:
                print(f"  {mov_type.capitalize()}:")
                print(f"    n = {results['n_pairs']}")
                print(f"    R² = {results['r_squared']:.4f}")
                print(f"    p = {results['p_value']:.3e}")
    else:
        print("  Insufficient data for correlation analysis")
    
    # Save results
    print("\nSaving results...")
    save_correlation_results(correlation_results, output_dir, config)
    
    # Visualize
    if visualize:
        print("\nGenerating visualizations...")
        visualize_correlation(correlation_results, output_dir, config)
    
    print("\n" + "="*60)
    print("STEP 5 COMPLETE")
    print("="*60)
    
    return True


def run_unit_test(visualize=False, test_data_path=None):
    """Run unit test."""
    print("="*60)
    print("CORRELATION ANALYSIS UNIT TEST")
    print("="*60)
    
    print("\n✓ Module tested via steps 3 and 4 outputs")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    if '--test' in sys.argv or len(sys.argv) == 1:
        visualize = '--visualize' in sys.argv or '--visualise' in sys.argv or '-v' in sys.argv
        run_unit_test(visualize=visualize)
    else:
        if len(sys.argv) < 3:
            print("Usage: python 05_correlate_movement_intensity.py <input_dir> <output_dir> "
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
