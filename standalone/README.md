# Cell Edge Curvature and Intensity Analysis

A comprehensive Python tool for analyzing the relationship between cell edge curvature, fluorescence intensity, and edge movement in microscopy images. This tool processes time-lapse microscopy data to explore correlations between membrane geometry, protein localization, and cell motility.

## Features

- **Edge Detection & Curvature Analysis**: Detects cell edges from binary masks and measures three types of curvature:
  - Sign curvature (-1 or 1): Direction of curvature (inward or outward)
  - Magnitude curvature: Absolute strength of curvature
  - Normalized curvature (-1 to 1): Combined direction and magnitude

- **Intensity Sampling**: Measures intensity within rectangular regions extending inward from edge points

- **Edge Movement Analysis**: Quantifies and classifies cell edge dynamics between frames:
  - Extending: Cell growth/protrusion
  - Retracting: Cell shrinkage/retraction
  - Stable: Minimal net movement

- **Multiple Correlation Analyses**:
  - **Standard**: Correlates curvature with intensity in the same frame
  - **Temporal**: Correlates current frame curvature with previous frame intensity
  - **Random Control**: Correlates curvature with intensity from randomly selected frames

- **Comprehensive Visualizations**: Generates heatmaps, correlation plots, movement maps, and summary statistics

- **Data Export**: Saves all measurements to CSV files for further analysis

- **Detailed Metadata**: Records all analysis parameters and statistics

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib
- scikit-image
- tifffile
- SciPy

### Installation Steps

```bash
# Create a virtual environment (optional but recommended)
python -m venv cell_analysis_env
source cell_analysis_env/bin/activate  # On Windows: cell_analysis_env\Scripts\activate

# Install required packages
pip install numpy matplotlib scikit-image tifffile scipy
```

## Usage

### Command Line Interface

```bash
python cell_edge_analysis.py --image path/to/microscope_images.tif --mask path/to/binary_masks.tif --output results --points 100 --depth 20 --width 5 --coverage 0.8
```

### Parameters

- `--image`: Path to microscope image stack (TIFF)
- `--mask`: Path to binary mask stack (TIFF)
- `--output`: Output directory for results
- `--points`: Number of equidistant points along the contour (default: 100)
- `--depth`: Depth of the sampling rectangle (default: 20)
- `--width`: Width of the sampling rectangle (default: 5)
- `--coverage`: Minimum cell coverage required (0.0-1.0) (default: 0.8)

### Using as a Module

```python
from cell_edge_analysis import process_stack

# Set your parameters
image_path = "path/to/microscope_images.tif"
mask_path = "path/to/binary_masks.tif"
output_dir = "results"
n_points = 100
depth = 20
width = 5
min_cell_coverage = 0.8

# Run the analysis
process_stack(image_path, mask_path, output_dir, n_points, depth, width, min_cell_coverage)
```

## Analysis Methodology

### Curvature Measurement

The tool measures three types of curvature at equidistant points along the cell edge:

1. **Sign Curvature (-1 or 1)**: Indicates whether the curvature is inward (negative) or outward (positive)
   - A point has positive curvature if it is located outside the line connecting its adjacent points
   - A point has negative curvature if it is located inside this line

2. **Magnitude Curvature (≥0)**: The absolute strength of the curvature, normalized by the distance between adjacent points

3. **Normalized Curvature (-1 to 1)**: Combines direction and magnitude into a single value
   - Preserves the sign while normalizing the magnitude to the [0, 1] range

### Intensity Sampling

For each point along the cell edge:

1. An inward normal vector is calculated, pointing into the cell
2. A rectangular sampling region is defined, extending from the edge point into the cell
   - The rectangle has user-defined width and depth
   - The sampling region must have minimum cell coverage (default: 80%)
3. The mean fluorescence intensity within this region is measured

### Edge Movement Analysis

The tool analyzes cell edge movement between consecutive frames:

- **Movement Detection**:
  - Compares cell masks between consecutive frames
  - Identifies regions of extension (cell growth) and retraction (cell shrinkage)
  - Calculates a net movement score for each frame

- **Movement Classification**:
  - Each frame is classified as:
    - **Extending**: Overall cell growth (positive movement score)
    - **Retracting**: Overall cell shrinkage (negative movement score)
    - **Stable**: Minimal net movement (movement score near zero)

- **Visualizations**:
  - Color-coded maps showing local edge movement (blue for extension, red for retraction)
  - Comparative visualizations showing current and previous frame masks
  - Summary bar chart showing movement patterns over time

### Correlation Analyses

The tool performs three types of correlation analyses:

1. **Standard Correlation**:
   - Correlates curvature with intensity in the same frame
   - Both sign and normalized curvature are analyzed

2. **Temporal Correlation**:
   - Correlates current frame curvature with previous frame intensity
   - Helps identify temporal relationships (e.g., does curvature change precede or follow intensity changes?)

3. **Random Control Correlation**:
   - Correlates curvature with intensity from randomly selected frames
   - Serves as a statistical control to validate the significance of other correlations

## Output Files

The tool generates the following outputs:

### Visualizations

- **Frame Visualizations**: For each frame, shows:
  - Original image and cell mask
  - Curvature sign, magnitude, and normalized measurements
  - Intensity sampling regions and measurements

- **Correlation Plots**:
  - Per-frame correlation plots for both sign and normalized curvature
  - Summary correlation plots combining data from all frames
  - Both temporal and random control correlation plots

- **Edge Movement Visualizations**:
  - Per-frame edge movement maps
  - Color-coded mask comparisons between consecutive frames
  - Summary bar chart of movement over time

- **Summary Heatmaps**:
  - Heatmaps showing curvature and intensity patterns across all frames
  - Temporal sequence visualizations
  - Random control visualizations

### Data Files

- **Frame-Specific CSV Files**:
  - Data for each frame with point index, curvature measurements, and intensity

- **Combined CSV Files**:
  - Combined data from all frames for standard, temporal, and random analyses

- **Edge Movement Data**:
  - CSV file with movement scores and classifications for each frame

### Metadata

- **JSON Metadata**:
  - Analysis parameters
  - Statistics (correlation R², p-value, sample sizes, etc.)
  - Environment information

- **Human-Readable Text Metadata**:
  - Formatted version of the JSON data for easy reading

## Directory Structure

```
results/
├── frame_0000.png                  # Frame visualization
├── frame_0001.png
├── ...
├── sign_correlation_frame_0000.png # Sign curvature correlation
├── normalized_correlation_frame_0000.png # Normalized curvature correlation
├── ...
├── edge_movement_frame_0001.png    # Edge movement visualization
├── ...
├── summary_sign_correlation.png    # Summary sign correlation
├── summary_normalized_correlation.png # Summary normalized correlation
├── edge_movement_summary.png       # Summary of edge movement
├── summary_sign_visualization.png  # Summary heatmap for sign curvature
├── summary_normalized_visualization.png # Summary heatmap for normalized curvature
├── data_frame_0000.csv             # Frame-specific data
├── ...
├── combined_data.csv               # Combined data from all frames
├── edge_movement_data.csv          # Edge movement data
├── metadata.json                   # Detailed metadata in JSON format
├── metadata.txt                    # Human-readable metadata
│
├── temporal_analysis/              # Temporal correlation analysis
│   ├── temporal_sign_correlation_frame_0001.png
│   ├── ...
│   ├── summary_temporal_sign_correlation.png
│   ├── ...
│   ├── temporal_data_frame_0001.csv
│   ├── ...
│   ├── combined_temporal_data.csv
│   ├── temporal_metadata.json
│   └── metadata.txt
│
└── random_analysis/                # Random control analysis
    ├── random_sign_correlation_frame_0000_random_0002.png
    ├── ...
    ├── summary_random_sign_correlation.png
    ├── ...
    ├── random_data_frame_0000.csv
    ├── ...
    ├── combined_random_data.csv
    ├── random_metadata.json
    └── metadata.txt
```

## Interpreting Results

### Correlation Analysis

The correlation between curvature and intensity is quantified using:

- **R² (Coefficient of Determination)**:
  - Ranges from 0 to 1
  - Higher values indicate stronger correlation
  - Represents the proportion of variance in intensity explained by curvature

- **p-value**:
  - Statistical significance of the correlation
  - p < 0.05 is typically considered significant

### Curvature-Intensity Relationship

- **Positive Correlation**: Higher curvature associated with higher intensity
  - May indicate accumulation of fluorescent proteins at curved membrane regions

- **Negative Correlation**: Lower curvature associated with higher intensity
  - May indicate exclusion of fluorescent proteins from curved regions

- **Temporal Correlation**:
  - Stronger correlation with previous frame than current frame suggests intensity changes follow curvature changes

- **Random Control**:
  - Should show weak or no correlation
  - Similar correlation in random and temporal analyses suggests relationships are coincidental

### Edge Movement Patterns

- **Extending Phases**: Periods of cell protrusion/growth
  - Often associated with actin polymerization at the leading edge

- **Retracting Phases**: Periods of cell retraction/shrinkage
  - May be associated with myosin contractility and focal adhesion dynamics

- **Connection with Curvature**:
  - Positive curvature regions often correspond to extending edges
  - Negative curvature regions may precede retraction events

- **Connection with Intensity**:
  - Protein accumulation patterns may precede or follow edge movement
  - Temporal analysis helps establish causality in these relationships

## Examples of Use Cases

- Study of membrane curvature sensors
- Analysis of actin dynamics at cell edges
- Investigation of protein localization during cell migration
- Quantification of membrane deformation during phagocytosis or endocytosis
- Analysis of leading edge dynamics during wound healing
- Correlation of edge movement with cytoskeletal protein recruitment
- Study of cellular response to mechanical or chemical stimuli

## Tips for Best Results

1. Ensure binary masks accurately represent cell boundaries
2. Adjust sampling parameters based on cell size and image resolution
3. Use appropriate number of points for the cell perimeter length
4. Compare temporal and random correlations to establish causality
5. Consider multiple samples/timepoints for statistically robust conclusions
6. Examine edge movement in relation to curvature and intensity patterns
7. Use the random control analysis to distinguish real correlations from artifacts
8. For edge movement analysis, ensure consistent frame timing and minimal drift

## Acknowledgments

This tool combines techniques from computer vision, statistical analysis, and cell biology to provide a comprehensive framework for studying the relationship between cell edge geometry, protein localization, and cell motility.
