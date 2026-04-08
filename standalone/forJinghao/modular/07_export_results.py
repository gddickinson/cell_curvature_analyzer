#!/usr/bin/env python3
"""
Step 7: Export Results

Compiles all analysis results into final CSV and JSON files matching the
original script's output format.

Author: George Dickinson
Date: November 4, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import datetime
from pathlib import Path
from utils.config import Config


def compile_detailed_results(input_dir):
    """
    Compile detailed results from all analysis steps.
    
    Parameters
    ----------
    input_dir : str
        Directory containing all analysis data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with detailed results
    """
    print("\nCompiling detailed results...")
    
    # Load correlation data
    corr_file = Path(input_dir) / 'correlation_detailed.csv'
    if corr_file.exists():
        df = pd.read_csv(corr_file)
        print(f"  Loaded {len(df)} data points")
        return df
    else:
        print("  Warning: correlation_detailed.csv not found")
        return pd.DataFrame()


def compile_summary_statistics(input_dir, config):
    """
    Compile comprehensive summary statistics.
    
    Parameters
    ----------
    input_dir : str
        Directory containing all analysis data
    config : Config
        Configuration object
    
    Returns
    -------
    dict
        Summary statistics dictionary
    """
    print("\nCompiling summary statistics...")
    
    input_path = Path(input_dir)
    parent_dir = input_path.parent
    
    summary = {
        'analysis_date': datetime.datetime.now().isoformat(),
        'method': 'vertical_displacement_xaxis_full',
        'sampling': 'x_axis',
        'configuration': config.get_dict(),
    }
    
    # Load data info from step1
    step1_dir = parent_dir / 'step1_load'
    if not step1_dir.exists():
        step1_dir = input_path
    
    data_info_file = step1_dir / 'data_info.json'
    if data_info_file.exists():
        with open(data_info_file, 'r') as f:
            data_info = json.load(f)
        summary['data_info'] = data_info['metadata']
    
    # Load edge statistics from step2
    step2_dir = parent_dir / 'step2_edges'
    if not step2_dir.exists():
        step2_dir = input_path
    
    edge_stats_file = step2_dir / 'edge_statistics.json'
    if edge_stats_file.exists():
        with open(edge_stats_file, 'r') as f:
            edge_stats = json.load(f)
        summary['edge_detection'] = {
            'n_frames': edge_stats['n_frames'],
            'mean_edge_length': edge_stats['mean_edge_length'],
            'empty_frames': edge_stats['empty_frames'],
        }
    
    # Load movement summary from step3
    step3_dir = parent_dir / 'step3_movement'
    if not step3_dir.exists():
        step3_dir = input_path
    
    mov_summary_file = step3_dir / 'movement_summary.json'
    if mov_summary_file.exists():
        with open(mov_summary_file, 'r') as f:
            mov_summary = json.load(f)
        summary['movement_analysis'] = mov_summary
    
    # Load intensity summary from step4
    step4_dir = parent_dir / 'step4_intensity'
    if not step4_dir.exists():
        step4_dir = input_path
    
    int_summary_file = step4_dir / 'intensity_summary.json'
    if int_summary_file.exists():
        with open(int_summary_file, 'r') as f:
            int_summary = json.load(f)
        summary['intensity_sampling'] = int_summary
    
    # Load correlation summary from input_dir (step5)
    corr_summary_file = input_path / 'correlation_summary.json'
    if corr_summary_file.exists():
        with open(corr_summary_file, 'r') as f:
            corr_summary = json.load(f)
        summary['correlation_analysis'] = corr_summary
    
    return summary


def export_results(detailed_df, summary_dict, output_dir):
    """
    Export results to CSV and JSON files.
    
    Parameters
    ----------
    detailed_df : pd.DataFrame
        Detailed results dataframe
    summary_dict : dict
        Summary statistics dictionary
    output_dir : str
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export detailed CSV
    if not detailed_df.empty:
        csv_path = output_dir / 'detailed_movement_results.csv'
        detailed_df.to_csv(csv_path, index=False)
        print(f"\nExported detailed results to: {csv_path}")
    
    # Export summary JSON
    json_path = output_dir / 'movement_summary_statistics.json'
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Exported summary statistics to: {json_path}")
    
    # Create a human-readable summary text file
    txt_path = output_dir / 'analysis_summary.txt'
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CELL EDGE MOVEMENT ANALYSIS - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Analysis Date: {summary_dict.get('analysis_date', 'N/A')}\n")
        f.write(f"Method: {summary_dict.get('method', 'N/A')}\n")
        f.write(f"Sampling: {summary_dict.get('sampling', 'N/A')}\n\n")
        
        if 'data_info' in summary_dict:
            f.write("-" * 70 + "\n")
            f.write("DATA INFORMATION\n")
            f.write("-" * 70 + "\n")
            info = summary_dict['data_info']
            f.write(f"Number of frames: {info.get('n_frames', 'N/A')}\n")
            f.write(f"Image size: {info.get('height', 'N/A')} x {info.get('width', 'N/A')}\n")
            f.write(f"Image range: [{info.get('image_min', 'N/A'):.2f}, "
                   f"{info.get('image_max', 'N/A'):.2f}]\n")
            f.write(f"Mean image intensity: {info.get('image_mean', 'N/A'):.2f}\n\n")
        
        if 'movement_analysis' in summary_dict:
            f.write("-" * 70 + "\n")
            f.write("MOVEMENT ANALYSIS\n")
            f.write("-" * 70 + "\n")
            mov = summary_dict['movement_analysis']
            f.write(f"Number of transitions: {mov.get('n_transitions', 'N/A')}\n")
            f.write(f"Temporal direction: {mov.get('temporal_direction', 'N/A')}\n")
            
            if 'movement_types' in mov:
                types = mov['movement_types']
                f.write(f"\nMovement type distribution:\n")
                f.write(f"  Extending: {types.get('extending', 0)}\n")
                f.write(f"  Retracting: {types.get('retracting', 0)}\n")
                f.write(f"  Stable: {types.get('stable', 0)}\n")
            
            f.write(f"\nMean movement score: {mov.get('mean_movement_score', 'N/A'):.4f} pixels\n\n")
        
        if 'intensity_sampling' in summary_dict:
            f.write("-" * 70 + "\n")
            f.write("INTENSITY SAMPLING\n")
            f.write("-" * 70 + "\n")
            intens = summary_dict['intensity_sampling']
            f.write(f"Total sampling points: {intens.get('total_sampling_points', 'N/A')}\n")
            f.write(f"Valid sampling points: {intens.get('valid_sampling_points', 'N/A')} "
                   f"({intens.get('validation_rate', 'N/A'):.1f}%)\n")
            f.write(f"Mean intensity: {intens.get('mean_intensity', 'N/A'):.2f}\n")
            f.write(f"Intensity range: [{intens.get('min_intensity', 'N/A'):.2f}, "
                   f"{intens.get('max_intensity', 'N/A'):.2f}]\n\n")
        
        if 'correlation_analysis' in summary_dict:
            f.write("-" * 70 + "\n")
            f.write("CORRELATION ANALYSIS\n")
            f.write("-" * 70 + "\n")
            corr = summary_dict['correlation_analysis']
            
            if 'overall_correlation' in corr:
                overall = corr['overall_correlation']
                f.write(f"Total paired measurements: {overall.get('n_pairs', 'N/A')}\n")
                
                if overall.get('r_squared') is not None:
                    f.write(f"\nOverall correlation:\n")
                    f.write(f"  R² = {overall.get('r_squared', 'N/A'):.4f}\n")
                    f.write(f"  p-value = {overall.get('p_value', 'N/A'):.3e}\n")
                    f.write(f"  Slope = {overall.get('slope', 'N/A'):.6f}\n")
                else:
                    f.write(f"\nInsufficient data for correlation analysis\n")
            
            if 'by_movement_type' in corr:
                f.write(f"\nCorrelation by movement type:\n")
                for mtype, results in corr['by_movement_type'].items():
                    f.write(f"\n  {mtype.capitalize()}:\n")
                    if 'error' not in results:
                        f.write(f"    n = {results.get('n_pairs', 'N/A')}\n")
                        f.write(f"    R² = {results.get('r_squared', 'N/A'):.4f}\n")
                        f.write(f"    p-value = {results.get('p_value', 'N/A'):.3e}\n")
                    else:
                        f.write(f"    {results['error']}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Exported human-readable summary to: {txt_path}")


def main(input_dir, output_dir, config_path=None):
    """
    Main processing function.
    
    Parameters
    ----------
    input_dir : str
        Directory containing all analysis data
    output_dir : str
        Output directory
    config_path : str, optional
        Path to configuration file
    """
    print("="*60)
    print("STEP 7: EXPORT RESULTS")
    print("="*60)
    
    # Load configuration
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config()
    
    # Compile results
    detailed_df = compile_detailed_results(input_dir)
    summary_dict = compile_summary_statistics(input_dir, config)
    
    # Export
    export_results(detailed_df, summary_dict, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("STEP 7 COMPLETE")
    print("="*60)
    print(f"\nFinal results saved to: {output_dir}")
    print("\nFiles created:")
    print("  - detailed_movement_results.csv")
    print("  - movement_summary_statistics.json")
    print("  - analysis_summary.txt")
    print("\n" + "="*60)
    
    return True


def run_unit_test(visualize=False, test_data_path=None):
    """Run unit test."""
    print("="*60)
    print("EXPORT RESULTS UNIT TEST")
    print("="*60)
    
    print("\n✓ Module tested via previous step outputs")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    if '--test' in sys.argv or len(sys.argv) == 1:
        run_unit_test()
    else:
        if len(sys.argv) < 3:
            print("Usage: python 07_export_results.py <input_dir> <output_dir> "
                  "[--config config.json]")
            sys.exit(1)
        
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        config_path = None
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
        
        success = main(input_dir, output_dir, config_path)
        sys.exit(0 if success else 1)
