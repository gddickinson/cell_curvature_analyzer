# Cell Edge Movement Analysis Script

## Overview

This script analyzes the relationship between cell edge movement and PIEZO1 protein intensity from TIRF microscope recordings by comparing consecutive frames.

**Author:** George Dickinson
**Date:** May 26, 2025

## Biological Context

PIEZO1 channels are mechanosensitive proteins that detect mechanical forces in cell membranes. The hypothesis is that local membrane dynamics (extension and retraction) influence PIEZO1 localization and/or activity. This script quantifies:

1. **Local movement** at points along the cell edge by comparing consecutive frames
2. **PIEZO1 intensity** in regions extending inward from each edge point
3. **Statistical correlation** between movement patterns and protein intensity
4. **Comprehensive visualizations** showing spatiotemporal relationships
5. **Edge transition dynamics** with frame-by-frame movement visualization

## Mathematical Approach

### 1. Edge Detection and Contour Extraction

The script uses `skimage.measure.find_contours` to detect cell edges from binary masks at the 0.5 contour level, creating discrete (y,x) coordinates defining the cell boundary for both current and previous frames.

### 2. Movement Detection via Mask Comparison

**Binary mask difference analysis:**
```
Given consecutive frames: previous_mask, current_mask
Difference mask: diff_mask = current_mask - previous_mask
Extension regions: diff_mask == +1 (new cell area)
Retraction regions: diff_mask == -1 (lost cell area)
Stable regions: diff_mask == 0 (no change)
```

**Movement map creation:**
```
movement_map[extension_regions] = +1.0    # Extension
movement_map[retraction_regions] = -1.0   # Retraction
movement_map[stable_regions] = 0.0        # No change
```

**Overall movement quantification:**
```
extension_pixels = sum(diff_mask == +1)
retraction_pixels = sum(diff_mask == -1)
net_movement = extension_pixels - retraction_pixels
total_changed = extension_pixels + retraction_pixels

movement_score = net_movement / total_changed  (if total_changed > threshold)
```

**Movement classification:**
- **Extending**: movement_score > movement_threshold
- **Retracting**: movement_score < -movement_threshold
- **Stable**: |movement_score| ≤ movement_threshold

### 3. Point Sampling Methods

Two sampling strategies are available (inherited from curvature analysis):

#### Standard Method (Arc-Length Based)
**Equidistant spatial sampling** along the current frame contour perimeter:

```
Given contour points C = {c₀, c₁, ..., cₙ}
Calculate cumulative arc length: L(i) = Σⱼ₌₀ᶦ ||cⱼ₊₁ - cⱼ||
Sample at intervals: s = L(n) / n_points
```

Linear interpolation finds exact positions at regular arc length intervals.

#### X-Axis Method (Coordinate Based)
**Regular x-coordinate sampling** across the cell width:

```
Find x-range: [x_min, x_max] from current contour
Create intervals: x_i = x_min + i(x_max - x_min)/(n_points - 1)
For each x_i, find closest contour point(s)
Apply adaptive tolerance and interpolation fallback
```

### 4. Local Movement Score Calculation

For each sampled point pᵢ on the current frame edge:

**Neighborhood sampling:**
```
Point coordinates: (y_i, x_i) = round(p_i)
Neighborhood window: [y_i ± 3, x_i ± 3]
Local movement score: mean(movement_map[neighborhood])
```

This provides a **local movement value** ∈ [-1, +1] for each edge point:
- **Positive values**: Local extension
- **Negative values**: Local retraction
- **Zero values**: Local stability

### 5. Inward Normal Vector Calculation

**Tangent vector:** t̂ = (p_{i+1} - p_{i-1}) / ||p_{i+1} - p_{i-1}||

**Normal candidates:** n₁ = (-tᵧ, tₓ), n₂ = (tᵧ, -tₓ)

**Selection:** Choose normal pointing into cell interior (mask value = 1).

### 6. Intensity Sampling with Quality Control

Rectangular sampling regions extend inward from current frame edge points:

**Geometry:**
```
Rectangle origin: Edge point pᵢ
Direction: Inward normal n̂ᵢ
Dimensions: depth × width pixels
End point: pᵢ + depth × n̂ᵢ
Corners: origin ± width/2 × perpendicular, end ± width/2 × perpendicular
```

**Quality Control System:**
- **Cell coverage requirement:** ≥ min_cell_coverage fraction inside cell
- **Rotation fallback:** Try 180° rotation (using -n̂ᵢ) if initial fails
- **Status tracking:**
  - `0` = excluded (endpoints if configured)
  - `1` = valid (successful measurement)
  - `2` = rejected (insufficient cell coverage)
  - `3` = rotated_valid (successful after rotation)

### 7. Statistical Analysis

**Pearson correlation analysis:**
```
r = Σ(Mᵢ - M̄)(Iᵢ - Ī) / √[Σ(Mᵢ - M̄)² × Σ(Iᵢ - Ī)²]
R² = r²
p-value from t-distribution with n-2 degrees of freedom
```

Where Mᵢ is local movement score and Iᵢ is intensity for valid measurements.

## Installation Requirements

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

## Usage

### 1. Configure Input Files

Edit the file paths in the script:

```python
IMAGE_STACK_PATH = "path/to/your/piezo1_images.tif"  # TIRF microscopy stack
MASK_STACK_PATH = "path/to/your/cell_masks.tif"      # Binary cell masks
OUTPUT_DIR = "movement_analysis_results"              # Output directory
```

### 2. Analysis Parameters

```python
CONFIG = {
    'n_points': 12,                                 # Number of sampling points around cell edge
    'depth': 200,                                   # Sampling rectangle depth (pixels into cell)
    'width': 75,                                    # Sampling rectangle width (pixels)
    'min_cell_coverage': 0.8,                       # Minimum fraction of rectangle in cell (0-1)
    'try_rotation': True,                           # Enable 180° rotation fallback
    'movement_threshold': 0.1,                      # Threshold for movement classification (-1 to 1)
    'min_movement_pixels': 5,                       # Minimum pixels changed to register movement
    'exclude_endpoints': True,                      # Exclude first/last points (avoids artifacts)
    'sampling_method': 'x_axis',                    # 'standard' (arc-length) or 'x_axis'
    'movement_sampling_Neighbourhood_size': 20      # Size region to sample for movement
}
```

### 3. Visualization Configuration

```python
VIZ_CONFIG = {
    # Frame plotting frequency
    'plot_every_nth_frame': 10,                    # Generate plots for every Nth frame
    'plot_every_nth_sampling_point': 1,            # Show every Nth rectangle (avoid clutter)

    # Main plot types
    'save_sampling_plots': True,                   # Sampling region visualization
    'save_intensity_plots': True,                  # Intensity profiles and distributions
    'save_movement_plots': True,                   # Movement analysis plots
    'save_edge_transition_plots': True,            # NEW: Consecutive frame edge comparison

    # Overlay visualizations
    'save_movement_overlay_plots': True,           # Movement type overlaid on cell images
    'save_intensity_overlay_plots': True,          # Intensity values overlaid on cell images

    # Correlation analysis
    'save_movement_type_correlation': True,        # Binary movement type correlation

    # Visual styling
    'movement_cmap': 'RdBu_r',                     # Colormap for movement (Red=retract, Blue=extend)
    'intensity_cmap': 'viridis',                   # Colormap for intensity plots
    'marker_size': 30,                             # Scatter plot marker size
    'figure_dpi': 150                              # Resolution for saved figures
}
```

### 4. Run Analysis

```bash
python movement_script.py
```

## Comprehensive Output Structure

### Main Analysis Results
- **`movement_intensity_correlation.png`** - Primary correlation plot (local movement vs intensity)
- **`movement_type_intensity_correlation.png`** - Binary movement analysis (extension/retraction vs intensity)
- **`movement_summary.png`** - Movement over time and distribution statistics
- **`detailed_movement_results.csv`** - Point-by-point data for all frame transitions with status tracking
- **`movement_summary_statistics.json`** - Analysis summary with correlation statistics and parameters

### Frame-by-Frame Visualizations

#### Edge Transition Analysis (`edge_transition_plots/`) **NEW**
- **Left panel**: Edge transition with movement map overlay
  - Background: Current frame grayscale image
  - **Orange dashed line**: Previous frame edge
  - **White solid line**: Current frame edge
  - **Color overlay**: Movement map (blue=extension, red=retraction)
  - **Lime dots**: Sampling points
  - **Statistics box**: Movement type and quantification
  - **Colorbar**: Movement interpretation scale

- **Right panel**: Mask comparison visualization
  - **Red channel**: Previous frame mask
  - **Blue channel**: Current frame mask
  - **Purple areas**: Overlap between frames
  - **Blue squares**: Extension areas (sampled to avoid clutter)
  - **Red squares**: Retraction areas (sampled to avoid clutter)
  - **Contour overlays**: Previous (red) and current (cyan) edges

#### Sampling Analysis (`frame_sampling_plots/`)
- Shows sampling regions overlaid on current frame microscopy images
- **Red line**: Current frame cell edge boundary
- **Rectangle colors**:
  - 🟢 **Lime**: Valid measurements
  - 🔵 **Blue**: Rejected (insufficient cell coverage)
  - 🟡 **Yellow**: Valid after 180° rotation
- **Point colors**: Match rectangle status

#### Intensity Analysis (`frame_intensity_plots/`)
- **Profile plots**: Intensity variation around cell perimeter
- **Distribution histograms**: Statistical summary of intensity values
- **Overlay plots**: Intensity values overlaid on cell images
  - Background: Grayscale cell image
  - Edge: White cell boundary outline
  - Points: Colored by intensity with colorbar scale
  - Ignored points: Gray X markers (faint)

#### Movement Analysis (`frame_movement_plots/`)
- **4-panel analysis**:
  - Local movement scores around perimeter
  - Movement score distribution histogram
  - Movement vs intensity scatter plot for frame
  - Movement map with edge overlay and sampling points
- **Overlay plots**: Movement type overlaid on cell images
  - Background: Grayscale cell image
  - Edge: White cell boundary outline
  - Points: 🔵 Blue (extension) / 🔴 Red (retraction) / ⚫ Gray (stable)
  - Ignored points: Gray X markers (faint)

## Data Interpretation Guide

### Movement Score Convention
- **Positive values:** Extension - cell edge moving outward
- **Negative values:** Retraction - cell edge moving inward
- **Zero values:** Stable - minimal edge movement

### Overall Movement Classification
- **Extending:** Net cell growth (movement_score > movement_threshold)
- **Retracting:** Net cell shrinkage (movement_score < -movement_threshold)
- **Stable:** Balanced or minimal movement (|movement_score| ≤ movement_threshold)

### Point Status Categories
- **excluded:** Endpoint excluded by configuration to avoid boundary artifacts
- **valid:** Successful measurement with sufficient cell coverage
- **rejected:** Insufficient cell coverage, measurement unreliable
- **rotated_valid:** Initial sampling failed, but 180° rotation succeeded

### Temporal Analysis
- **Frame transitions:** Analysis requires consecutive frames (N frames → N-1 transitions)
- **Movement maps:** Pixel-level change detection between frames
- **Local scores:** Neighborhood averaging provides smooth local movement values

### Statistical Significance
- **R² values:** Proportion of variance in intensity explained by movement (0-1 scale)
- **p-values:** Statistical significance (typically p < 0.05 considered significant)
- **Sample size:** Number of valid measurements across all processed frame transitions

### Visual Color Coding
- **Movement map**: RdBu_r colormap (red=retraction, blue=extension, white=stable)
- **Edge transitions**: Orange=previous, white=current edges
- **Movement overlay**: Blue=extension, red=retraction, gray=stable/ignored
- **Intensity overlay**: Viridis colormap (dark=low, bright=high intensity)

## Parameter Optimization Guidelines

### Movement Detection Sensitivity
- **movement_threshold (0.05-0.2):** Controls movement classification sensitivity
  - Lower (0.05): More sensitive, detects subtle movements
  - Higher (0.2): Less sensitive, only major movements classified
- **min_movement_pixels (3-10):** Minimum change to register movement
  - Lower: Detects smaller movements, potentially noisier
  - Higher: Only detects substantial movements

### Sampling Strategy Selection
- **Standard method**: Use for irregular cell shapes, general analysis
- **X-axis method**: Use for horizontally oriented cells, migration studies

### Sampling Geometry
- **n_points (8-20):** Balance spatial resolution vs noise
  - Fewer: Miss local variations, faster processing
  - More: Better resolution, potential noise increase
- **depth (100-300 pixels):** Should capture relevant PIEZO1 signal
  - Too shallow: May miss protein distribution
  - Too deep: May include irrelevant cellular structures
- **width (50-100 pixels):** Affects signal-to-noise ratio
  - Wider: Better signal averaging, less spatial precision
  - Narrower: Better spatial resolution, potentially noisier

### Quality Control
- **min_cell_coverage (0.6-0.9):** Balance data quantity vs quality
  - Higher (0.9): Fewer but more reliable measurements
  - Lower (0.6): More measurements but potentially noisier
- **try_rotation:** Recommended for complex cell shapes

### Visualization Control
- **plot_every_nth_frame:** Balance insight vs processing time
  - 1: Complete temporal analysis (slow for large datasets)
  - 10: Representative sampling (faster processing)
- **save_edge_transition_plots:** Essential for understanding movement detection
  - Shows exactly how extension/retraction areas are calculated
  - Validates movement detection algorithm performance



