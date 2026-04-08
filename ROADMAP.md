# Cell Curvature Analyzer -- Roadmap

## Current State
A large PyQt5 GUI application analyzing membrane curvature vs PIEZO1 protein localization. Main logic spread across ~12 modules in `modules/` plus `main.py`. Also contains `standalone/` scripts (including `forJinghao/` subdirectory with modular analysis pipeline and various one-off scripts). The standalone scripts suggest active research use but create organizational complexity. Has `setup-script.py` and data directories.

## Short-term Improvements
- [x] Add `requirements.txt` with pinned dependency versions
- [ ] Clean up `standalone/forJinghao/` -- archive or document purpose of each script variant
- [ ] Add type hints to core analysis functions in `modules/curvature_analysis.py`
- [x] Add error handling for missing/corrupt TIFF files in `modules/file_io.py`
- [x] Write tests for `modules/curvature_analysis.py` using synthetic edge data
- [x] Remove `__pycache__` from version control; add proper `.gitignore`
- [x] Document parameter meanings in `modules/settings.py`

## Feature Enhancements
- [ ] Add CLI mode to `main.py` for headless batch processing of multiple datasets
- [ ] Implement automated parameter optimization in `modules/curvature_analysis.py`
- [ ] Add statistical comparison between cell populations (e.g., treated vs control)
- [ ] Support loading ThunderSTORM CSV localization data alongside TIFF stacks
- [ ] Add kymograph generation for tracking curvature changes at specific edge positions
- [ ] Implement report generation combining all analysis tabs into a single PDF

## Long-term Vision
- [ ] Unify the `standalone/forJinghao/modular/` pipeline with the GUI application
- [ ] Add napari viewer integration as alternative visualization backend
- [ ] Support GPU-accelerated curvature calculation via CuPy for large datasets
- [ ] Publish as pip package with entry point scripts
- [ ] Create reproducible analysis workflows via configuration files (YAML/JSON)

## Technical Debt
- [ ] `modules/cell_curvature_analyzer.py` is likely the monolithic MainWindow class -- split into smaller components
- [x] `membrane_movement_analyzer.py` at project root duplicates functionality in `modules/` -- moved to `_archive/`
- [ ] `standalone/` contains 7+ script variants of similar analysis -- consolidate into parameterized versions
- [x] `setup-script.py` is non-standard naming; migrate to `pyproject.toml`
- [ ] `modules/ml_analysis.py` and `modules/correlation_analysis.py` may overlap -- audit
- [x] `data/` and `results/` directories should be in `.gitignore`, not tracked
