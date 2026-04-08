#!/usr/bin/env python3
"""
Step 1: Load and Validate Data

Loads image and mask stacks from TIFF files and performs validation checks.
Generates metadata about the data for downstream processing.

Author: George Dickinson
Date: November 4, 2025
"""

import os
import sys
import numpy as np
import tifffile
import json
from pathlib import Path
from typing import Tuple, Dict, Any


def load_data(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image and mask stacks from TIFF files.
    
    Parameters
    ----------
    image_path : str
        Path to image stack TIFF file
    mask_path : str
        Path to mask stack TIFF file
    
    Returns
    -------
    tuple
        (images, masks) as numpy arrays
    """
    print(f"Loading image stack: {image_path}")
    images = tifffile.imread(image_path)
    
    print(f"Loading mask stack: {mask_path}")
    masks = tifffile.imread(mask_path)
    
    # Ensure masks are binary
    masks = (masks > 0).astype(np.uint8)
    
    # Handle single image case
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    
    return images, masks


def validate_data(images: np.ndarray, masks: np.ndarray) -> Dict[str, Any]:
    """
    Validate loaded data and generate metadata.
    
    Parameters
    ----------
    images : np.ndarray
        Image stack
    masks : np.ndarray
        Mask stack
    
    Returns
    -------
    dict
        Validation report and metadata
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'metadata': {}
    }
    
    # Check dimensions
    if images.shape != masks.shape:
        report['errors'].append(
            f"Shape mismatch: images {images.shape} vs masks {masks.shape}"
        )
        report['valid'] = False
    
    # Check data types
    if not np.issubdtype(images.dtype, np.number):
        report['errors'].append(f"Invalid image dtype: {images.dtype}")
        report['valid'] = False
    
    if masks.dtype != np.uint8:
        report['warnings'].append(f"Mask dtype {masks.dtype} converted to uint8")
    
    # Check for NaN or Inf
    if np.any(np.isnan(images)) or np.any(np.isinf(images)):
        report['errors'].append("Images contain NaN or Inf values")
        report['valid'] = False
    
    # Check mask values
    unique_mask_vals = np.unique(masks)
    if not np.all(np.isin(unique_mask_vals, [0, 1])):
        report['warnings'].append(
            f"Masks contain non-binary values: {unique_mask_vals}"
        )
    
    # Generate metadata
    report['metadata'] = {
        'n_frames': int(images.shape[0]),
        'height': int(images.shape[1]),
        'width': int(images.shape[2]),
        'image_dtype': str(images.dtype),
        'mask_dtype': str(masks.dtype),
        'image_min': float(np.min(images)),
        'image_max': float(np.max(images)),
        'image_mean': float(np.mean(images)),
        'image_std': float(np.std(images)),
        'mask_coverage': float(np.mean(masks)),
    }
    
    # Check for empty masks
    empty_frames = []
    for i in range(masks.shape[0]):
        if np.sum(masks[i]) == 0:
            empty_frames.append(i)
    
    if empty_frames:
        report['warnings'].append(
            f"Empty masks in frames: {empty_frames}"
        )
    
    return report


def save_data(images: np.ndarray, masks: np.ndarray, 
             output_dir: str, validation_report: Dict[str, Any]):
    """
    Save data and metadata for downstream processing.
    
    Parameters
    ----------
    images : np.ndarray
        Image stack
    masks : np.ndarray
        Mask stack
    output_dir : str
        Output directory
    validation_report : dict
        Validation report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy data
    np.savez_compressed(
        output_dir / 'loaded_data.npz',
        images=images,
        masks=masks
    )
    print(f"Saved data to: {output_dir / 'loaded_data.npz'}")
    
    # Save metadata
    with open(output_dir / 'data_info.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    print(f"Saved metadata to: {output_dir / 'data_info.json'}")


def visualize_data(images: np.ndarray, masks: np.ndarray, output_dir: str,
                  n_examples: int = 3):
    """
    Create visualization of loaded data.
    
    Parameters
    ----------
    images : np.ndarray
        Image stack
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
    
    # Select frames to visualize
    n_frames = images.shape[0]
    if n_frames <= n_examples:
        frame_indices = list(range(n_frames))
    else:
        frame_indices = np.linspace(0, n_frames-1, n_examples, dtype=int)
    
    fig, axes = plt.subplots(len(frame_indices), 3, figsize=(15, 5*len(frame_indices)))
    if len(frame_indices) == 1:
        axes = axes[np.newaxis, :]
    
    for i, idx in enumerate(frame_indices):
        # Original image
        axes[i, 0].imshow(images[idx], cmap='gray')
        axes[i, 0].set_title(f'Frame {idx}: Image')
        axes[i, 0].axis('off')
        
        # Mask
        axes[i, 1].imshow(masks[idx], cmap='gray')
        axes[i, 1].set_title(f'Frame {idx}: Mask')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(images[idx], cmap='gray')
        axes[i, 2].imshow(masks[idx], cmap='Reds', alpha=0.3)
        axes[i, 2].set_title(f'Frame {idx}: Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / 'data_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(image_path: str, mask_path: str, output_dir: str, visualize: bool = False):
    """
    Main processing function.
    
    Parameters
    ----------
    image_path : str
        Path to image stack
    mask_path : str
        Path to mask stack
    output_dir : str
        Output directory
    visualize : bool
        Whether to generate visualizations
    """
    print("="*60)
    print("STEP 1: LOAD AND VALIDATE DATA")
    print("="*60)
    
    # Load data
    images, masks = load_data(image_path, mask_path)
    print(f"Loaded {images.shape[0]} frames of size {images.shape[1]}x{images.shape[2]}")
    
    # Validate
    print("\nValidating data...")
    report = validate_data(images, masks)
    
    if report['valid']:
        print("✓ Data validation passed")
    else:
        print("✗ Data validation failed:")
        for error in report['errors']:
            print(f"  ERROR: {error}")
    
    if report['warnings']:
        print("Warnings:")
        for warning in report['warnings']:
            print(f"  WARNING: {warning}")
    
    # Print metadata
    print("\nData Metadata:")
    for key, value in report['metadata'].items():
        print(f"  {key}: {value}")
    
    # Save results
    print("\nSaving results...")
    save_data(images, masks, output_dir, report)
    
    # Visualize
    if visualize:
        print("\nGenerating visualizations...")
        visualize_data(images, masks, output_dir)
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE")
    print("="*60)
    
    return report['valid']


# =============================================================================
# UNIT TEST
# =============================================================================

def run_unit_test(visualize: bool = False, test_data_path: str = None):
    """Run unit test with synthetic or real data."""
    print("="*60)
    print("LOAD AND VALIDATE MODULE UNIT TEST")
    print("="*60)
    
    if test_data_path and os.path.exists(test_data_path):
        print(f"\n[Test] Using real data from: {test_data_path}")
        from pathlib import Path
        test_path = Path(test_data_path)
        
        # Find image and mask files
        image_files = list(test_path.glob('*piezo*.tif'))
        mask_files = list(test_path.glob('*mask*.tif'))
        
        if image_files and mask_files:
            success = main(str(image_files[0]), str(mask_files[0]), 
                          'test_output_01', visualize)
            assert success, "Validation should pass for real data"
            print("\n✓ Real data test passed")
        else:
            print("No suitable files found in test directory")
    else:
        print("\n[Test] Using synthetic data...")
        
        # Create synthetic data
        n_frames = 5
        height, width = 100, 100
        
        images = np.random.rand(n_frames, height, width) * 255
        
        # Create circular masks
        masks = np.zeros((n_frames, height, width), dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        for i in range(n_frames):
            center_y, center_x = 50, 50
            radius = 30 + i * 2
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            masks[i] = mask.astype(np.uint8)
        
        # Save temp files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            img_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            mask_path = f.name
        
        tifffile.imwrite(img_path, images.astype(np.uint16))
        tifffile.imwrite(mask_path, masks)
        
        # Run main function
        success = main(img_path, mask_path, 'test_output_01', visualize)
        
        # Cleanup
        os.remove(img_path)
        os.remove(mask_path)
        
        assert success, "Validation should pass for synthetic data"
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
        # Parse command line arguments for actual use
        if len(sys.argv) < 4:
            print("Usage: python 01_load_and_validate.py <image_path> <mask_path> <output_dir> [--visualize]")
            sys.exit(1)
        
        image_path = sys.argv[1]
        mask_path = sys.argv[2]
        output_dir = sys.argv[3]
        visualize = '--visualize' in sys.argv or '--visualise' in sys.argv or '-v' in sys.argv
        
        success = main(image_path, mask_path, output_dir, visualize)
        sys.exit(0 if success else 1)
