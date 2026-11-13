# Cell Curvature Analyzer - Interface Map

## Project Structure

### Entry Point
- **`main.py`** - Creates PyQt5 `QApplication` and launches `MainWindow` from `modules/cell_curvature_analyzer.py`.

### Modules Package (`modules/`)
- **`cell_curvature_analyzer.py`** - `MainWindow`: construction, toolbar, docks, log tab; its behaviour comes from the mixins in `modules/gui/`.
- **`gui/`** - `MainWindow` mixins, one concern per file: `tab_visualization.py` (Visualization + Analysis tabs), `tab_analysis.py` (Correlation, Temporal, Movement tabs), `tab_export_settings.py` (Export + Settings tabs), `frame_plots.py` (per-frame curvature / intensity / correlation plots and tables), `correlation_display.py` (Correlation tab dispatch, movement proxy, single-frame correlations), `correlation_pooled.py` (all-frames / by-movement correlations), `correlation_temporal.py` and `correlation_movement_temporal.py` (frame vs previous / reference frame), `navigation.py` (frame navigation, temporal / movement displays), `display.py` (image, overlay, results display), `data_actions.py` (loading stacks, running the analysis), `export_actions.py`, `settings_actions.py` (QSettings), `ml_tab.py` (ML tab; not currently added to the window), `popup_plots.py`, `workers.py` (QThread workers).
- **`curvature_analysis.py`** - `CurvatureAnalyzer`: stack/frame driver used by the GUI. `process_stack()`, `analyze_frame()`; edge detection, equidistant sampling, curvature, normals, intensity, movement and regression delegate to `edge_analysis`.
- **`sampling_methods.py`** - `AlternativeSamplingMixin` (of `CurvatureAnalyzer`): angular, axis-based and x-axis point sampling.
- **`sampling_debug.py`** - `SamplingDebugMixin`: matplotlib debug plots of the sampling methods (not used by the GUI).
- **`edge_analysis.py`** - Qt-free analysis core (also used by the web app): `detect_cell_edge`, `sample_equidistant_points`, `measure_curvature` (legacy three-point rule), `calculate_inward_normal` (open edges: no wrap-around), `measure_intensity`, `analyze_points`, `analyze_frame`, `detect_edge_movement`, `summary_correlation_stats`.
- **`edge_curvature.py`** - edge geometry and the curvature options (docstring = the sign convention: +1 convex, -1 concave, 0 straight): `is_closed`, `arc_length`, `sample_edge_points` (point_spacing 'equal' / 'legacy'), `sample_legacy`, `arc_positions`, `gaussian_smooth`, `smoothed_curvature` (smoothing_px), `three_point_curvature` (legacy rule, straight_threshold dead-band), `edge_curvature` (dispatch).
- **`analysis_defaults.py`** - `DEFAULTS`: the single source of analysis defaults (n_points, depth, width, min_cell_coverage, try_rotation, exclude_endpoints, movement_threshold, point_spacing, smoothing_px, straight_threshold); read by the widgets, settings, analyzer and web app.
- **`synthetic_cell.py`** - `make_demo_stack()`: seeded synthetic image + mask stack (wound-edge layout, enrichment at concave edge by construction) for the web demo and tests. Not measured data.
- **`visualization.py`** (774 lines) - `Visualizer`: matplotlib-based plotting for curvature maps, intensity overlays, correlation plots.
- **`widgets/`** - Custom PyQt5 widgets used by `MainWindow` (re-exported by `widgets/__init__.py`): `image_viewer.py` (`ImageViewer`), `overlay_canvas.py` (`OverlayCanvas`), `tables_log.py` (`ResultsTable`, `LogConsole`), `parameter_panel.py` (`ParameterPanel`).
- **`temporal_analysis.py`** (808 lines) - `TemporalAnalyzer`: frame-to-frame movement analysis and temporal statistics.
- **`correlation_analysis.py`** - `CorrelationAnalyzer`: curvature-intensity correlation statistics and plots.
- **`data_export.py`** (1425 lines) - `DataExporter`: exports results to CSV, JSON, TIFF, and figures.
- **`file_io.py`** - `FileManager`: TIFF stack loading/saving, JSON/CSV result I/O. Includes input validation.
- **`ml_analysis.py`** - `MLAnalyzer`: machine learning approaches for curvature prediction.
- **`settings.py`** - `SettingsManager`: persists window state and analysis parameters via QSettings.
- **`__init__.py`** - Package metadata.

### Web interface (`modules/web/`)
- **`app.py`** - FastAPI app: `GET /api/info`, `GET /api/demo`, `POST /api/analyze`; `STATIC_ENDPOINTS` exported by `scripts/build_static.py`.
- **`analysis.py`** - `analyze_stack()` (stack-level analysis on `edge_analysis`, same statistics as `process_stack`), `clean_params`, base64 `encode`/`decode`.
- **`static/`** - ES-module front end (no build step): `main.js` (router), `home.js`, `analyze.js` + `analyze_view.js` + `plots.js` (Analyze page), `convention.js` (curvature sign page), `about.js`; `stack_io.js` + `tiff_reader.js` (loading TIFF/PNG stacks); JS ports `contour_engine.js` (find_contours), `curvature_engine.js` (edge_curvature: spacing, smoothing, curvature), `edge_engine.js` (normals, draw.polygon, intensity, movement, stack), `stats_engine.js` (linregress); `edge_engine.test.mjs` (node test vs Python fixture); `static_api.js` (static-site data layer); `core.js`/`style.css` shared web kit.
- **`run_web.py`** (repo root) - starts the app on 127.0.0.1:8208.
- **`scripts/curvature_benchmark.py`** - seeded benchmark of the curvature options on masks of known curvature (chose the smoothing / straight-threshold defaults; also run in reduced form by `tests/test_edge_curvature.py`).
- **`scripts/desktop_demo_digest.py`** - offscreen digest of everything the desktop app computes and draws on the synthetic demo (pipeline, every GUI view, export); `--compare A B` to check that a refactor changes nothing.
- **`scripts/build_static.py`**, **`scripts/screenshots.py`**, **`scripts/make_edge_fixture.py`** - static export, Playwright smoke test, Python-vs-JS fixture (`tests/fixtures/`).

### Standalone Scripts (`standalone/`)
- **`curvature_measurements.py`** - Standalone curvature analysis script.
- **`movement_metric_comparison.py`** - Movement metric comparison script.
- **`forJinghao/`** - Modular analysis pipeline with curvature and movement sub-analyses plus test data.

### Archive (`_archive/`)
- **`membrane_movement_analyzer.py`** - Former root-level script, duplicated functionality in `modules/`. Moved to archive.

### Tests (`tests/`)
- **`test_curvature_analysis.py`** - Tests for `CurvatureAnalyzer`: edge detection, point sampling, curvature measurement, frame analysis, TIFF I/O.
- **`test_edge_analysis.py`** - The Qt-free core and the curvature sign convention (circle, notch, wound edge, straight run, spacing noise).
- **`test_edge_curvature.py`** - curvature options: legacy options reproduce the old code, equal spacing, open edges (no wrap-around), smoothed curvature, dead-band, defaults vs the benchmark.
- **`test_gui_offscreen.py`** - MainWindow offscreen: starts, parameter panel defaults, current-frame and stack analysis run in worker threads.
- **`test_process_stack.py`** - `process_stack` with frames that have no cell.
- **`test_packaging.py`** - pyproject build backend and console-script target exist.
- **`test_web.py`** - web API, static build, fixture currency, web vs desktop statistics, node test of the JS port.

## Key Relationships
- `main.py` -> `modules/cell_curvature_analyzer.MainWindow` (mixins from `modules/gui/`)
- `MainWindow` uses `CurvatureAnalyzer`, `Visualizer`, `TemporalAnalyzer`, `CorrelationAnalyzer`, `DataExporter`, `FileManager`, `SettingsManager`
- `CurvatureAnalyzer` drives the analysis; all other modules consume its output. Its per-frame maths lives in `edge_analysis` (shared with `modules/web`)
- Widgets, `SettingsManager` and `CurvatureAnalyzer` take their defaults from `analysis_defaults.DEFAULTS`
- `FileManager` handles all file I/O with validation
- `SettingsManager` persists user preferences
