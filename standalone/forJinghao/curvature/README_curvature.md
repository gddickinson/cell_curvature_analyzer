# Enhanced Cell Edge Curvature Analysis Script

## Overview

This script analyzes the relationship between cell edge curvature and PIEZO1 protein intensity from TIRF microscope recordings with comprehensive optimization and analysis capabilities.

**Author:** George Dickinson
**Date:** May 26, 2025
**Version:** Enhanced with Y-axis sampling, output control, and iterative parameter optimization

## 🚀 Key Features

### **Advanced Sampling Methods**
- **Standard (Arc-length)**: Equidistant spatial sampling along contour perimeter
- **X-axis**: Regular sampling at x-coordinate intervals (horizontal analysis)
- **Y-axis**: Regular sampling at y-coordinate intervals (vertical analysis)

### **Flexible Output Control**
- **Summary-only mode**: Export only key statistics for quick analysis
- **Custom output selection**: Choose which files to generate
- **Batch processing**: Optimized for high-throughput analysis

### **Iterative Parameter Optimization** ⭐ **NEW**
- **Systematic testing**: Automatically test all parameter combinations
- **Best parameter identification**: Find optimal settings for your data
- **Performance metrics**: Compare correlation strength and measurement success rates
- **Export optimization results**: Single CSV with all tested combinations

### **Comprehensive Analysis**
- **Three curvature measures**: Sign, magnitude, and normalized curvature
- **Quality control**: Automatic region validation and rotation fallback
- **Statistical analysis**: Pearson correlations with significance testing
- **Rich visualizations**: Overlay plots, sampling regions, and correlation graphs

## Biological Context

PIEZO1 channels are mechanosensitive proteins that detect mechanical forces in cell membranes. This script quantifies local membrane curvature and PIEZO1 intensity to test the hypothesis that curvature influences protein localization and activity.

## 🔧 Installation Requirements

```bash
pip install numpy matplotlib pandas tifffile scikit-image scipy
```

**Required Python packages:**
- `numpy` - Numerical computations and array operations
- `matplotlib` - Plotting, visualization, and figure generation
- `pandas` - Data handling, CSV export, and tabular analysis
- `tifffile` - TIFF image stack loading and processing
- `scikit-image` - Image processing (contour detection, polygon filling)
- `scipy` - Statistical analysis and interpolation functions

## 📊 Mathematical Approach

### 1. Edge Detection and Sampling

**Contour extraction:** Uses `skimage.measure.find_contours` at 0.5 level to detect cell boundaries.

**Three sampling strategies:**

#### Standard Method (Arc-Length Based)
```
L(i) = Σⱼ₌₀ᶦ ||cⱼ₊₁ - cⱼ||  (cumulative arc length)
Sample at: s = L(n) / n_points  (regular intervals)
```

#### X-Axis Method
```
x_positions = linspace(x_min, x_max, n_points)
For each x_i: find closest contour point(s)
```

#### Y-Axis Method
```
y_positions = linspace(y_min, y_max, n_points)
For each y_i: find closest contour point(s)
```

### 2. Curvature Calculation

**Three-point discrete approximation:**
```
κ_sign = sign(vₓ × wᵧ - vᵧ × wₓ)
κ_magnitude = |vₓ × wᵧ - vᵧ × wₓ| / ||v||²
κ_normalized = κ_sign × (κ_magnitude / max(κ_magnitude))
```

Where:
- **v** = p_{i+1} - p_{i-1} (chord vector)
- **w** = pᵢ - p_{i-1} (position vector)

### 3. Quality Control System

**Sampling regions:** Rectangular areas extending inward from edge points
**Validation criteria:**
- Cell coverage ≥ threshold (default 0.8)
- 180° rotation fallback if initial sampling fails
- Status tracking: excluded, valid, rejected, rotated_valid

## 🎯 Usage Guide

### Basic Single Analysis

1. **Configure input files:**
```python
IMAGE_STACK_PATH = "path/to/piezo1_images.tif"
MASK_STACK_PATH = "path/to/cell_masks.tif"
OUTPUT_DIR = "analysis_results"
```

2. **Set analysis parameters:**
```python
CONFIG = {
    'n_points': 12,                    # Sampling points around edge
    'depth': 200,                      # Rectangle depth (pixels)
    'width': 75,                       # Rectangle width (pixels)
    'min_cell_coverage': 0.8,          # Minimum cell fraction in rectangle
    'try_rotation': True,              # Enable 180° fallback
    'exclude_endpoints': True,         # Skip first/last points
    'sampling_method': 'standard'      # 'standard', 'x_axis', or 'y_axis'
}
```

3. **Choose output options:**
```python
OUTPUT_CONFIG = {
    'save_detailed_csv': True,         # Detailed point-by-point data
    'save_summary_json': True,         # Summary statistics
    'save_correlation_plots': True,    # Main correlation plots
    'save_frame_plots': True,          # Individual frame visualizations
    'summary_only': False,             # Quick analysis mode
}
```

4. **Run analysis:**
```bash
python curvature_script.py
```

### Summary-Only Mode (Fast Analysis)

For quick statistical analysis without detailed visualizations:

```python
OUTPUT_CONFIG = {
    'summary_only': True,  # Only export summary_statistics.json
}
```

### Custom Output Selection

```python
OUTPUT_CONFIG = {
    'save_detailed_csv': False,        # Skip large CSV file
    'save_summary_json': True,         # Keep summary stats
    'save_correlation_plots': True,    # Keep main plots
    'save_frame_plots': False,         # Skip frame-by-frame plots
    'summary_only': False,
}
```

## 🔍 Iterative Parameter Optimization

### Enabling Optimization Mode

```python
ITERATIVE_MODE = {
    'enabled': True,                   # Enable parameter testing
    'output_file': 'optimization_results.csv',
    'show_progress': True,             # Display progress
}
```

### Defining Parameter Ranges

```python
ITERATIVE_PARAMS = {
    'n_points': [8, 12, 16, 20],                    # Number of sampling points
    'depth': [150, 200, 250, 300],                  # Sampling rectangle depth
    'width': [50, 75, 100],                         # Rectangle width
    'min_cell_coverage': [0.6, 0.7, 0.8, 0.9],     # Coverage threshold
    'sampling_method': ['standard', 'x_axis', 'y_axis'],  # Sampling methods
    'try_rotation': [True, False],                   # Rotation fallback
    'exclude_endpoints': [True, False],              # Endpoint exclusion
}
```

### Optimization Examples

#### Quick Method Comparison (9 combinations):
```python
ITERATIVE_PARAMS = {
    'sampling_method': ['standard', 'x_axis', 'y_axis'],
    'n_points': [12, 16, 20],
}
```

#### Focused Geometry Optimization (80 combinations):
```python
ITERATIVE_PARAMS = {
    'depth': [150, 175, 200, 225, 250],
    'width': [60, 70, 80, 90],
    'min_cell_coverage': [0.7, 0.75, 0.8, 0.85],
}
```

#### Comprehensive Optimization (1,152 combinations):
```python
ITERATIVE_PARAMS = {
    'n_points': [8, 12, 16, 20],
    'depth': [150, 200, 250, 300],
    'width': [50, 75, 100],
    'min_cell_coverage': [0.6, 0.7, 0.8, 0.9],
    'sampling_method': ['standard', 'x_axis', 'y_axis'],
    'try_rotation': [True, False],
    'exclude_endpoints': [True, False],
}
```

## 📈 Understanding Optimization Results

### Output CSV Structure

| Column | Description |
|--------|-------------|
| `iteration` | Iteration number |
| `n_points`, `depth`, etc. | Parameter values tested |
| `correlation_r_squared` | R² for normalized curvature vs intensity |
| `correlation_p_value` | Statistical significance |
| `sign_correlation_r_squared` | R² for curvature sign vs intensity |
| `valid_measurement_percentage` | Success rate of measurements |
| `total_sampled_points` | Total points attempted |
| `valid_measurements` | Successfully measured points |

### Automatic Best Results Identification

The script automatically identifies:

```
BEST PARAMETER COMBINATIONS
============================================================

Best Normalized Curvature Correlation (R² = 0.4523):
  n_points: 16
  depth: 250
  width: 75
  sampling_method: y_axis

Best Sign Curvature Correlation (R² = 0.3891):
  n_points: 12
  sampling_method: standard

Highest Valid Measurement Rate (84.2%):
  n_points: 8
  min_cell_coverage: 0.6
```

## 📁 Output Structure

### Standard Analysis Mode

```
curvature_analysis_results/
├── curvature_intensity_correlation.png          # Main correlation plot
├── sign_curvature_intensity_correlation.png     # Binary correlation plot
├── detailed_results.csv                         # Point-by-point data
├── summary_statistics.json                      # Analysis summary
├── frame_sampling_plots/                        # Sampling visualizations
│   ├── frame_001_sampling.png
│   └── ...
├── frame_intensity_plots/                       # Intensity analysis
│   ├── frame_001_intensity.png
│   ├── frame_001_intensity_overlay.png
│   └── ...
└── frame_curvature_plots/                       # Curvature analysis
    ├── frame_001_curvature.png
    ├── frame_001_curvature_overlay.png
    └── ...
```

### Iterative Mode

```
curvature_analysis_results/
└── parameter_optimization_results.csv           # All tested combinations
```

### Summary-Only Mode

```
curvature_analysis_results/
└── summary_statistics.json                      # Statistics only
```

## 🎨 Visualization Guide

### Color Coding Systems

#### Sampling Regions
- 🟢 **Lime**: Valid measurements
- 🔵 **Blue**: Rejected (insufficient cell coverage)
- 🟡 **Yellow**: Valid after 180° rotation
- 🔴 **Red line**: Cell edge boundary

#### Curvature Overlay
- 🔴 **Red points**: Convex regions (membrane curves outward)
- 🔵 **Blue points**: Concave regions (membrane curves inward)
- ⚪ **White line**: Cell edge outline
- ❌ **Gray X**: Ignored points

#### Intensity Overlay
- **Viridis colormap**: Dark (low intensity) → Bright (high intensity)
- **Colorbar scale**: Quantitative intensity values
- ❌ **Gray X**: Invalid measurements

## 🔧 Parameter Optimization Guidelines

### Sampling Strategy Selection

| Cell Type | Recommended Method | Reason |
|-----------|-------------------|---------|
| **Irregular shapes** | `standard` | Arc-length preserves spatial relationships |
| **Horizontally oriented** | `x_axis` | Captures width variations |
| **Vertically oriented** | `y_axis` | Captures height variations |
| **Migration studies** | `x_axis` or `y_axis` | Directional analysis |

### Sampling Geometry Optimization

#### Number of Points (`n_points`)
- **8-12**: Fast processing, captures major features
- **16-20**: Higher resolution, better for complex shapes
- **>20**: May introduce noise, slower processing

#### Sampling Depth (`depth`)
- **150-200**: Conservative, near-membrane analysis
- **250-300**: Captures broader protein distribution
- **>300**: Risk of including irrelevant structures

#### Sampling Width (`width`)
- **50-75**: High spatial precision
- **75-100**: Better signal-to-noise ratio
- **>100**: Risk of averaging across features

#### Cell Coverage (`min_cell_coverage`)
- **0.6-0.7**: More measurements, potentially noisier
- **0.8-0.9**: Fewer but higher quality measurements
- **0.9+**: Very conservative, may miss edge regions

### Performance Considerations

#### Small Parameter Sweeps (≤50 combinations)
- Use for method comparison or focused optimization
- Complete within minutes
- Good for initial parameter exploration

#### Medium Sweeps (50-200 combinations)
- Comprehensive geometry optimization
- Complete within 30-60 minutes
- Ideal for most optimization needs

#### Large Sweeps (>200 combinations)
- Full parameter space exploration
- May take several hours
- Use for final optimization or publication-quality analysis

## 📊 Data Interpretation

### Statistical Significance
- **R² > 0.1**: Weak but potentially meaningful correlation
- **R² > 0.25**: Moderate correlation
- **R² > 0.5**: Strong correlation
- **p < 0.05**: Statistically significant
- **p < 0.001**: Highly significant

### Curvature Convention
- **Positive (+)**: Convex (membrane protrudes outward)
- **Negative (-)**: Concave (membrane indents inward)
- **Zero**: Straight edge segments

### Quality Metrics
- **Valid percentage >70%**: Good parameter settings
- **Valid percentage 50-70%**: Acceptable for complex cells
- **Valid percentage <50%**: Consider parameter adjustment

## 💡 Practical Tips

### For New Users
1. Start with **summary-only mode** for quick assessment
2. Use **default parameters** as baseline
3. Test different **sampling methods** first
4. Gradually refine **geometry parameters**

### For Parameter Optimization
1. Begin with **small parameter sweeps** (9-27 combinations)
2. Focus on **one parameter category** at a time
3. Use **correlation R²** as primary optimization metric
4. Consider **valid measurement percentage** for robustness

### For Publication
1. Run **comprehensive optimization** to justify parameter choices
2. Include **both correlation types** (normalized and sign)
3. Report **sample sizes** and **statistical significance**
4. Show **representative visualizations** from best parameters

### Troubleshooting Common Issues

#### Low Valid Measurement Percentage
- Decrease `min_cell_coverage` (try 0.6-0.7)
- Enable `try_rotation = True`
- Reduce `depth` or `width` parameters
- Check mask quality and cell edge detection

#### Poor Correlation Results
- Try different `sampling_method` options
- Increase `n_points` for better resolution
- Adjust `depth` to capture relevant protein distribution
- Verify image registration between channels

#### Slow Processing
- Enable `summary_only = True` for optimization
- Reduce parameter ranges in `ITERATIVE_PARAMS`
- Decrease `plot_every_nth_frame` frequency
- Consider analyzing subset of frames first

## 🔄 Workflow Examples

### Research Pipeline
```python
# Step 1: Quick assessment
OUTPUT_CONFIG = {'summary_only': True}
# Run to check data quality

# Step 2: Method comparison
ITERATIVE_MODE = {'enabled': True}
ITERATIVE_PARAMS = {'sampling_method': ['standard', 'x_axis', 'y_axis']}
# Run to find best sampling method

# Step 3: Parameter optimization
ITERATIVE_PARAMS = {
    'n_points': [12, 16, 20],
    'depth': [175, 200, 225],
    'width': [65, 75, 85],
}
# Run to optimize geometry

# Step 4: Final analysis with best parameters
ITERATIVE_MODE = {'enabled': False}
OUTPUT_CONFIG = {'summary_only': False}  # Full output
# Run final analysis with optimized settings
```

### High-Throughput Analysis
```python
# Configuration for batch processing
OUTPUT_CONFIG = {
    'save_detailed_csv': False,     # Skip large files
    'save_summary_json': True,      # Keep statistics
    'save_correlation_plots': False, # Skip plots
    'save_frame_plots': False,      # Skip visualizations
}
```

This enhanced script provides comprehensive tools for optimizing and analyzing cell edge curvature relationships with maximum flexibility and scientific rigor.
