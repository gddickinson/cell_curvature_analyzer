#!/usr/bin/env python3
"""
Run Single Analysis

Orchestrates the complete analysis pipeline for one file pair.

Author: George Dickinson
Date: November 4, 2025
"""

import sys
import os
import subprocess
from pathlib import Path
from utils.config import Config


def run_pipeline(image_path, mask_path, output_dir, config_path=None, visualize=False):
    """Run complete analysis pipeline."""
    print("="*60)
    print("RUNNING COMPLETE ANALYSIS PIPELINE")
    print("="*60)
    print(f"\nImage: {image_path}")
    print(f"Mask: {mask_path}")
    print(f"Output: {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step directories
    step_dirs = {
        'step1': output_dir / 'step1_load',
        'step2': output_dir / 'step2_edges',
        'step3': output_dir / 'step3_movement',
        'step4': output_dir / 'step4_intensity',
        'step5': output_dir / 'step5_correlation',
        'step6': output_dir / 'step6_visualization',
        'step7': output_dir / 'step7_results',
    }
    
    viz_flag = '--visualize' if visualize else ''
    config_flag = f'--config {config_path}' if config_path else ''
    
    steps = [
        ('Step 1: Load and Validate', 
         f'python 01_load_and_validate.py {image_path} {mask_path} {step_dirs["step1"]} {viz_flag}'),
        ('Step 2: Detect Edges',
         f'python 02_detect_edges.py {step_dirs["step1"]} {step_dirs["step2"]} {viz_flag}'),
        ('Step 3: Calculate Movement',
         f'python 03_calculate_movement.py {step_dirs["step2"]} {step_dirs["step3"]} {config_flag} {viz_flag}'),
        ('Step 4: Sample Intensity',
         f'python 04_sample_intensity.py {step_dirs["step2"]} {step_dirs["step4"]} {config_flag} {viz_flag}'),
        ('Step 5: Correlate Movement and Intensity',
         f'python 05_correlate_movement_intensity.py {step_dirs["step4"]} {step_dirs["step5"]} {config_flag} {viz_flag}'),
        ('Step 6: Generate Visualizations',
         f'python 06_generate_visualizations.py {step_dirs["step5"]} {step_dirs["step6"]} {config_flag}'),
        ('Step 7: Export Results',
         f'python 07_export_results.py {step_dirs["step5"]} {step_dirs["step7"]} {config_flag}'),
    ]
    
    for step_name, command in steps:
        print(f"\n{'='*60}")
        print(f"Running: {step_name}")
        print(f"{'='*60}")
        
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"\nERROR: {step_name} failed!")
            return False
        print(f"\n✓ {step_name} completed successfully")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_single_analysis.py <image_path> <mask_path> <output_dir> [--config config.json] [--visualize]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
    
    visualize = '--visualize' in sys.argv or '-v' in sys.argv
    
    success = run_pipeline(image_path, mask_path, output_dir, config_path, visualize)
    sys.exit(0 if success else 1)
