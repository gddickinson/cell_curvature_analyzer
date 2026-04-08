#!/usr/bin/env python3
"""
Run Batch Files

Processes multiple image-mask file pairs with the same parameters.
Useful for analyzing multiple experiments or time series.

Author: George Dickinson
Date: November 4, 2025
"""

import sys
import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from utils.config import Config


def find_file_pairs(input_dir, image_pattern='*piezo*.tif', mask_pattern='*mask*.tif'):
    """
    Find matching image and mask file pairs in a directory.
    
    Parameters
    ----------
    input_dir : str
        Directory containing TIFF files
    image_pattern : str
        Glob pattern for image files
    mask_pattern : str
        Glob pattern for mask files
    
    Returns
    -------
    list
        List of (image_path, mask_path, name) tuples
    """
    input_path = Path(input_dir)
    
    image_files = sorted(input_path.glob(image_pattern))
    mask_files = sorted(input_path.glob(mask_pattern))
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(mask_files)} mask files")
    
    # Try to pair by name similarity
    pairs = []
    
    for img_file in image_files:
        # Extract base name (remove path and extension)
        img_base = img_file.stem
        
        # Try to find matching mask
        best_match = None
        best_score = 0
        
        for mask_file in mask_files:
            mask_base = mask_file.stem
            
            # Simple matching: count common words
            img_words = set(img_base.lower().split('_'))
            mask_words = set(mask_base.lower().split('_'))
            common = len(img_words & mask_words)
            
            if common > best_score:
                best_score = common
                best_match = mask_file
        
        if best_match:
            # Create a descriptive name
            name = img_base.replace('_piezo1', '').replace('_PIEZO1', '')
            pairs.append((str(img_file), str(best_match), name))
    
    return pairs


def run_analysis_on_pair(image_path, mask_path, name, output_dir, config_path=None):
    """
    Run analysis on a single file pair.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    mask_path : str
        Path to mask file
    name : str
        Descriptive name for this pair
    output_dir : str
        Output directory for this pair
    config_path : str, optional
        Path to configuration file
    
    Returns
    -------
    dict
        Results summary
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    print(f"  Image: {Path(image_path).name}")
    print(f"  Mask: {Path(mask_path).name}")
    
    # Run analysis
    config_flag = f'--config {config_path}' if config_path else ''
    command = (f"python run_single_analysis.py {image_path} {mask_path} "
              f"{output_dir} {config_flag}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Analysis failed for {name}")
        print(result.stderr)
        return {
            'name': name,
            'image_file': Path(image_path).name,
            'mask_file': Path(mask_path).name,
            'success': False,
            'error': 'Analysis failed'
        }
    
    # Load results
    try:
        summary_file = Path(output_dir) / 'step7_results' / 'movement_summary_statistics.json'
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Extract key metrics
        results = {
            'name': name,
            'image_file': Path(image_path).name,
            'mask_file': Path(mask_path).name,
            'success': True,
        }
        
        # Add data info
        if 'data_info' in summary:
            info = summary['data_info']
            results['n_frames'] = info.get('n_frames')
        
        # Add correlation metrics
        if 'correlation_analysis' in summary:
            corr = summary['correlation_analysis']
            if 'overall_correlation' in corr:
                overall = corr['overall_correlation']
                results['r_squared'] = overall.get('r_squared')
                results['p_value'] = overall.get('p_value')
                results['n_pairs'] = overall.get('n_pairs')
                results['slope'] = overall.get('slope')
        
        # Add movement metrics
        if 'movement_analysis' in summary:
            mov = summary['movement_analysis']
            results['n_transitions'] = mov.get('n_transitions')
            results['mean_movement_score'] = mov.get('mean_movement_score')
            if 'movement_types' in mov:
                results['extending'] = mov['movement_types'].get('extending', 0)
                results['retracting'] = mov['movement_types'].get('retracting', 0)
                results['stable'] = mov['movement_types'].get('stable', 0)
        
        # Add intensity metrics
        if 'intensity_sampling' in summary:
            intens = summary['intensity_sampling']
            results['validation_rate'] = intens.get('validation_rate')
            results['mean_intensity'] = intens.get('mean_intensity')
            results['total_sampling_points'] = intens.get('total_sampling_points')
        
        print(f"\n✓ Analysis completed successfully")
        if results.get('r_squared') is not None:
            print(f"  R² = {results['r_squared']:.4f}, p = {results.get('p_value', 'N/A'):.3e}")
        
        return results
        
    except Exception as e:
        print(f"ERROR: Could not load results for {name}: {e}")
        return {
            'name': name,
            'image_file': Path(image_path).name,
            'mask_file': Path(mask_path).name,
            'success': False,
            'error': str(e)
        }


def create_batch_summary(all_results, output_dir):
    """
    Create summary of batch analysis results.
    
    Parameters
    ----------
    all_results : list
        List of result dictionaries
    output_dir : str
        Output directory
    """
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Save summary table
    output_dir = Path(output_dir)
    csv_path = output_dir / 'batch_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved batch summary to: {csv_path}")
    
    # Print statistics
    successful = df['success'].sum()
    total = len(df)
    
    print(f"\nSuccess rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful > 0:
        successful_df = df[df['success'] == True]
        
        if 'r_squared' in successful_df.columns:
            r2_values = successful_df['r_squared'].dropna()
            if len(r2_values) > 0:
                print(f"\nCorrelation R² statistics:")
                print(f"  Mean: {r2_values.mean():.4f}")
                print(f"  Std: {r2_values.std():.4f}")
                print(f"  Range: [{r2_values.min():.4f}, {r2_values.max():.4f}]")
                
                # Best result
                best_idx = r2_values.idxmax()
                best = successful_df.loc[best_idx]
                print(f"\nBest correlation:")
                print(f"  File: {best['name']}")
                print(f"  R² = {best['r_squared']:.4f}")
        
        if 'validation_rate' in successful_df.columns:
            val_rates = successful_df['validation_rate'].dropna()
            if len(val_rates) > 0:
                print(f"\nValidation rate statistics:")
                print(f"  Mean: {val_rates.mean():.1f}%")
                print(f"  Range: [{val_rates.min():.1f}%, {val_rates.max():.1f}%]")
    
    print("\n" + "="*60)


def main(input_dir, output_dir, config_path=None, 
         image_pattern='*piezo*.tif', mask_pattern='*mask*.tif'):
    """
    Main function for batch processing.
    
    Parameters
    ----------
    input_dir : str
        Directory containing input files
    output_dir : str
        Base output directory
    config_path : str, optional
        Path to configuration file
    image_pattern : str
        Glob pattern for image files
    mask_pattern : str
        Glob pattern for mask files
    """
    print("="*70)
    print("BATCH FILE PROCESSING")
    print("="*70)
    
    # Find file pairs
    print(f"\nSearching for files in: {input_dir}")
    file_pairs = find_file_pairs(input_dir, image_pattern, mask_pattern)
    
    if not file_pairs:
        print("ERROR: No file pairs found!")
        print(f"  Image pattern: {image_pattern}")
        print(f"  Mask pattern: {mask_pattern}")
        return
    
    print(f"\nFound {len(file_pairs)} file pairs")
    
    # Show pairs
    print("\nFile pairs:")
    for i, (img, mask, name) in enumerate(file_pairs):
        print(f"  {i+1}. {name}")
        print(f"     Image: {Path(img).name}")
        print(f"     Mask: {Path(mask).name}")
    
    # Confirm
    if len(file_pairs) > 5:
        response = input(f"\nProcess all {len(file_pairs)} pairs? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    # Process each pair
    all_results = []
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    for img_path, mask_path, name in file_pairs:
        # Create output directory for this pair
        pair_output = output_base / name.replace(' ', '_')
        
        result = run_analysis_on_pair(
            img_path, mask_path, name, pair_output, config_path
        )
        all_results.append(result)
    
    # Create summary
    create_batch_summary(all_results, output_base)
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"Summary table: {output_dir}/batch_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_batch_files.py <input_dir> <output_dir> "
              "[--config config.json] [--image-pattern PATTERN] [--mask-pattern PATTERN]")
        print("\nExample:")
        print("  python run_batch_files.py ./data ./batch_results --config my_config.json")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    config_path = None
    image_pattern = '*piezo*.tif'
    mask_pattern = '*mask*.tif'
    
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        elif arg == '--image-pattern' and i + 1 < len(sys.argv):
            image_pattern = sys.argv[i + 1]
        elif arg == '--mask-pattern' and i + 1 < len(sys.argv):
            mask_pattern = sys.argv[i + 1]
    
    main(input_dir, output_dir, config_path, image_pattern, mask_pattern)
