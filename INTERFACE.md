# Cell Curvature Analyzer - Interface Map

## Project Structure

### Entry Point
- **`main.py`** - Creates PyQt5 `QApplication` and launches `MainWindow` from `modules/cell_curvature_analyzer.py`.

### Modules Package (`modules/`)
- **`cell_curvature_analyzer.py`** (4890 lines) - `MainWindow`: the monolithic main GUI window. Contains all tabs, data loading, visualization triggers, and analysis coordination. (NOTE: this file is very large and should be split in the future.)
- **`curvature_analysis.py`** (1329 lines) - `CurvatureAnalyzer`: core analysis engine. Key methods: `process_stack()`, `analyze_frame()`, `detect_cell_edge()`, `sample_equidistant_points()`, `measure_curvature()`, `measure_intensity()`, `detect_edge_movement()`.
- **`visualization.py`** (774 lines) - `Visualizer`: matplotlib-based plotting for curvature maps, intensity overlays, correlation plots.
- **`widgets.py`** (966 lines) - Custom PyQt5 widgets used by `MainWindow`.
- **`temporal_analysis.py`** (808 lines) - `TemporalAnalyzer`: frame-to-frame movement analysis and temporal statistics.
- **`correlation_analysis.py`** - `CorrelationAnalyzer`: curvature-intensity correlation statistics and plots.
- **`data_export.py`** (1425 lines) - `DataExporter`: exports results to CSV, JSON, TIFF, and figures.
- **`file_io.py`** - `FileManager`: TIFF stack loading/saving, JSON/CSV result I/O. Includes input validation.
- **`ml_analysis.py`** - `MLAnalyzer`: machine learning approaches for curvature prediction.
- **`settings.py`** - `SettingsManager`: persists window state and analysis parameters via QSettings.
- **`__init__.py`** - Package metadata.

### Standalone Scripts (`standalone/`)
- **`curvature_measurements.py`** - Standalone curvature analysis script.
- **`movement_metric_comparison.py`** - Movement metric comparison script.
- **`forJinghao/`** - Modular analysis pipeline with curvature and movement sub-analyses plus test data.

### Archive (`_archive/`)
- **`membrane_movement_analyzer.py`** - Former root-level script, duplicated functionality in `modules/`. Moved to archive.

### Tests (`tests/`)
- **`test_curvature_analysis.py`** - Tests for `CurvatureAnalyzer`: edge detection, point sampling, curvature measurement, frame analysis, TIFF I/O.

## Key Relationships
- `main.py` -> `modules/cell_curvature_analyzer.MainWindow`
- `MainWindow` uses `CurvatureAnalyzer`, `Visualizer`, `TemporalAnalyzer`, `CorrelationAnalyzer`, `DataExporter`, `FileManager`, `SettingsManager`
- `CurvatureAnalyzer` is the core engine; all other modules consume its output
- `FileManager` handles all file I/O with validation
- `SettingsManager` persists user preferences
