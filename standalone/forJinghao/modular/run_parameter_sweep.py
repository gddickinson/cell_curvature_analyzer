#!/usr/bin/env python3
"""
Run Parameter Sweep

Tests multiple parameter combinations and compares results.
Useful for finding optimal analysis parameters.

Author: George Dickinson
Date: November 4, 2025
"""

import sys
import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from itertools import product
from utils.config import Config


def generate_parameter_combinations(param_grid):
    """
    Generate all combinations of parameters from a grid.
    
    Parameters
    ----------
    param_grid : dict
        Dictionary with parameter names as keys and lists of values
    
    Returns
    -------
    list
        List of parameter dictionaries
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    combinations = []
    for combo in product(*values):
        param_dict = {keys[i]: combo[i] for i in range(len(keys))}
        combinations.append(param_dict)
    
    return combinations


def run_analysis_with_params(image_path, mask_path, output_dir, params, param_id):
    """
    Run analysis with specific parameter set.
    
    Parameters
    ----------
    image_path : str
        Path to image stack
    mask_path : str
        Path to mask stack
    output_dir : str
        Base output directory
    params : dict
        Parameter dictionary
    param_id : int
        Parameter set ID
    
    Returns
    -------
    dict
        Results summary
    """
    print(f"\n{'='*60}")
    print(f"RUNNING PARAMETER SET {param_id}")
    print(f"{'='*60}")
    print(f"Parameters: {params}")
    
    # Create parameter-specific output directory
    param_output_dir = Path(output_dir) / f"params_{param_id:03d}"
    param_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration for this parameter set
    config = Config({
        'analysis': params
    })
    config_path = param_output_dir / 'config.json'
    config.save(config_path)
    
    # Run analysis
    command = (f"python run_single_analysis.py {image_path} {mask_path} "
              f"{param_output_dir} --config {config_path}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Analysis failed for parameter set {param_id}")
        return {
            'param_id': param_id,
            'params': params,
            'success': False,
            'error': result.stderr
        }
    
    # Load results
    try:
        summary_file = param_output_dir / 'step7_results' / 'movement_summary_statistics.json'
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Extract key metrics
        results = {
            'param_id': param_id,
            'params': params,
            'success': True,
        }
        
        # Add correlation metrics if available
        if 'correlation_analysis' in summary:
            corr = summary['correlation_analysis']
            if 'overall_correlation' in corr:
                overall = corr['overall_correlation']
                results['r_squared'] = overall.get('r_squared')
                results['p_value'] = overall.get('p_value')
                results['n_pairs'] = overall.get('n_pairs')
        
        # Add movement metrics
        if 'movement_analysis' in summary:
            mov = summary['movement_analysis']
            results['n_transitions'] = mov.get('n_transitions')
            if 'movement_types' in mov:
                results['extending_count'] = mov['movement_types'].get('extending', 0)
                results['retracting_count'] = mov['movement_types'].get('retracting', 0)
                results['stable_count'] = mov['movement_types'].get('stable', 0)
        
        # Add intensity metrics
        if 'intensity_sampling' in summary:
            intens = summary['intensity_sampling']
            results['validation_rate'] = intens.get('validation_rate')
            results['mean_intensity'] = intens.get('mean_intensity')
        
        print(f"\n✓ Parameter set {param_id} completed successfully")
        if results.get('r_squared') is not None:
            print(f"  R² = {results['r_squared']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"ERROR: Could not load results for parameter set {param_id}: {e}")
        return {
            'param_id': param_id,
            'params': params,
            'success': False,
            'error': str(e)
        }


def compare_results(all_results, output_dir):
    """
    Compare results across parameter sets.
    
    Parameters
    ----------
    all_results : list
        List of result dictionaries
    output_dir : str
        Output directory
    """
    print(f"\n{'='*60}")
    print("COMPARING RESULTS ACROSS PARAMETER SETS")
    print(f"{'='*60}")
    
    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        if result['success']:
            row = {'param_id': result['param_id']}
            row.update(result['params'])
            row.update({k: v for k, v in result.items() 
                       if k not in ['param_id', 'params', 'success']})
            comparison_data.append(row)
    
    if not comparison_data:
        print("No successful results to compare")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    output_dir = Path(output_dir)
    csv_path = output_dir / 'parameter_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to: {csv_path}")
    
    # Print summary
    print("\nParameter Comparison Summary:")
    print("-" * 60)
    
    if 'r_squared' in df.columns:
        # Find best R²
        best_idx = df['r_squared'].idxmax()
        best_result = df.loc[best_idx]
        
        print(f"\nBest R² = {best_result['r_squared']:.4f}")
        print(f"  Parameter set {int(best_result['param_id'])}")
        print(f"  n_points: {best_result.get('n_points', 'N/A')}")
        print(f"  depth: {best_result.get('depth', 'N/A')}")
        print(f"  width: {best_result.get('width', 'N/A')}")
        print(f"  movement_threshold: {best_result.get('movement_threshold', 'N/A')}")
        
        # Show range of R²
        print(f"\nR² range: [{df['r_squared'].min():.4f}, {df['r_squared'].max():.4f}]")
    
    if 'validation_rate' in df.columns:
        print(f"\nValidation rate range: "
              f"[{df['validation_rate'].min():.1f}%, {df['validation_rate'].max():.1f}%]")
    
    print("\n" + "="*60)


def main(image_path, mask_path, output_dir, param_grid_file=None):
    """
    Main function for parameter sweep.
    
    Parameters
    ----------
    image_path : str
        Path to image stack
    mask_path : str
        Path to mask stack
    output_dir : str
        Base output directory
    param_grid_file : str, optional
        Path to JSON file with parameter grid
    """
    print("="*70)
    print("PARAMETER SWEEP ANALYSIS")
    print("="*70)
    
    # Load or create parameter grid
    if param_grid_file:
        with open(param_grid_file, 'r') as f:
            param_grid = json.load(f)
        print(f"\nLoaded parameter grid from: {param_grid_file}")
    else:
        # Default parameter grid
        param_grid = {
            'n_points': [8, 12, 16],
            'depth': [150, 200, 250],
            'width': [50, 75, 100],
            'movement_threshold': [0.1, 0.15, 0.2],
        }
        print("\nUsing default parameter grid")
    
    # Generate combinations
    param_combinations = generate_parameter_combinations(param_grid)
    print(f"\nTotal parameter combinations: {len(param_combinations)}")
    
    if len(param_combinations) > 20:
        print(f"\nWARNING: This will run {len(param_combinations)} analyses!")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    # Run analyses
    all_results = []
    for i, params in enumerate(param_combinations):
        result = run_analysis_with_params(
            image_path, mask_path, output_dir, params, i
        )
        all_results.append(result)
    
    # Compare results
    compare_results(all_results, output_dir)
    
    print(f"\n{'='*70}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"Comparison table: {output_dir}/parameter_comparison.csv")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_parameter_sweep.py <image_path> <mask_path> <output_dir> "
              "[--param-grid grid.json]")
        print("\nExample parameter grid JSON:")
        print("""{
  "n_points": [8, 12, 16],
  "depth": [150, 200, 250],
  "width": [50, 75, 100],
  "movement_threshold": [0.1, 0.15, 0.2]
}""")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    param_grid_file = None
    for i, arg in enumerate(sys.argv):
        if arg == '--param-grid' and i + 1 < len(sys.argv):
            param_grid_file = sys.argv[i + 1]
    
    main(image_path, mask_path, output_dir, param_grid_file)
