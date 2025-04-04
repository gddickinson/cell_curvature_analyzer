# Cell Curvature Analyzer

A comprehensive GUI application for analyzing the relationship between cell-membrane curvature and PIEZO1 protein locations from fluorescence microscope recordings.

![Application Screenshot](docs/images/screenshot.png)

## Features

- **Interactive Visualization**: View original images, binary masks, and analysis overlays side-by-side
- **Curvature Analysis**: Calculate three types of curvature (sign, magnitude, and normalized)
- **Intensity Correlation**: Analyze the relationship between curvature and fluorescence intensity
- **Temporal Analysis**: Track changes across frames and correlate with previous frames
- **Edge Movement Detection**: Classify cell edge regions as extending, retracting, or stable
- **Multi-tab Interface**: Dedicated views for different analysis aspects
- **Data Export**: Comprehensive export options for images, data, and reports

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy, Matplotlib, SciPy
- scikit-image
- pandas
- PyQt5
- tifffile

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/cell-curvature-analyzer.git
cd cell-curvature-analyzer

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Install from PyPI

```bash
pip install cell-curvature-analyzer
```

## Usage

### Starting the Application

```bash
# If installed from source with -e flag
cell-curvature-analyzer

# Or run directly from the source directory
python cell_curvature_analyzer.py
```

### Loading Data

1. Click **Open Images** to load a microscope image stack (TIFF format)
2. Click **Open Masks** to load corresponding binary masks (TIFF format)
3. Set analysis parameters in the right panel
4. Click **Run Analysis** to process all frames

### Navigating the Interface

- **Visualization Tab**: View images and overlay analysis results
- **Analysis Results Tab**: View frame-specific measurements and correlation plots
- **Correlation Analysis Tab**: Explore relationships between curvature and intensity
- **Temporal Analysis Tab**: Analyze changes over time
- **Edge Movement Tab**: Study cell edge dynamics
- **Export Results Tab**: Export data and visualizations
- **Settings Tab**: Configure analysis and visualization parameters
- **Log Tab**: View logging information and debug messages

### Exploring Results

- Use the frame navigation controls to move between frames
- Toggle different overlay components in the visualization tab
- View curvature and intensity profiles in the analysis tab
- Explore correlations by movement type in the correlation tab
- Compare current frame with previous/random frames in the temporal tab

### Exporting Data

1. Go to the **Export Results** tab
2. Select the types of data to export (images, raw data, statistics, figures, report)
3. Choose an export directory
4. Click **Export Data**

## Analysis Methodology

### Curvature Measurement

The application calculates three types of curvature at equidistant points along the cell edge:

1. **Sign Curvature (-1 or 1)**: Indicates whether the curvature is inward (negative) or outward (positive)
2. **Magnitude Curvature (≥0)**: The absolute strength of the curvature
3. **Normalized Curvature (-1 to 1)**: Combines direction and magnitude

### Intensity Sampling

For each point along the cell edge:
1. An inward normal vector is calculated
2. A rectangular sampling region extends into the cell
3. The mean fluorescence intensity within this region is measured

### Edge Movement Analysis

The application classifies cell edge movement into three categories:
- **Extending**: Cell growth/protrusion (blue in visualizations)
- **Retracting**: Cell shrinkage/retraction (red in visualizations)
- **Stable**: Minimal net movement (gray in visualizations)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool builds upon the curvature analysis approach described in [reference to relevant papers]
- Thanks to all contributors and the scientific community for feedback and suggestions
