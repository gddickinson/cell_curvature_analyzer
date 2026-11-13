# Session log

## 2026-11-12 - Web interface and static site (branch `web-interface`)
- Fixed: `process_stack` crashed (IndexError) when two or more frames had no cell; regression test `tests/test_process_stack.py`.
- Fixed: `pip install -e .` failed (non-existent build backend) and the `cell-curvature-analyzer` console script pointed at a
  missing `main.main`; README run command corrected (`python main.py`), PyPI line removed. Test `tests/test_packaging.py`.
- Stopped tracking `__pycache__` / `.DS_Store` (ignored per ROADMAP but still in the index).
- Refactor (no change in results, bit-identical on 60 random masks): Qt-free core `modules/edge_analysis.py`;
  `CurvatureAnalyzer` delegates to it. Defaults in one place, `modules/analysis_defaults.py`, read by the widgets,
  settings and analyzer. The pooled regression returns None statistics when every curvature value is equal. Fixed the
  `angularr_distribution` key typo in `ParameterPanel`.
- Curvature sign convention documented (edge_analysis docstring, web Curvature sign page) and pinned by tests on a
  circular mask (+1 everywhere), a notch (-1), an open wound edge, and reversed traversal (flips).
- Web app (`modules/web/`, port 8208) and static GitHub Pages site: Analyze (own image + mask or synthetic demo),
  Curvature sign, About. JS port node-tested against Python (worst relative difference 1.8e-14 over 18,829 values).
- Synthetic demo `modules/synthetic_cell.py` (seeded; enrichment at concave edge by construction).
- Open questions for George: see "Needs decision" below.

### Needs decision
- With no smoothing, single-point signs on pixel masks are noise-dominated at small spacing: on a 40 px-radius circle
  only 44/100 points are +1 at the default n_points=100 (spacing 2.6 px); spacing >= ~8 px gives 100%. Smoothing the
  contour or a spacing-based default would change results.
- A straight run (cross product exactly 0, common on pixel masks) gets sign -1, i.e. is counted as concave. Mapping it
  to 0 would change the sign-curvature statistics.
- On open (border-touching) edges the first and last points are neighbours across the image, so their curvature is
  meaningless; `exclude_endpoints` only removes them from intensity sampling, not from the curvature values.

## 2026-11-12 (later) - Curvature fixes, GUI worker, module splits (branch `web-interface`)
George's decisions applied: fixes that change results are fine; situational choices are options with a legacy value.
- Refactors (no behaviour change, each verified with the new `scripts/desktop_demo_digest.py`: 0 differences in the
  pipeline with every sampling method, every GUI view, current-frame analysis, pop-up plots and export):
  MainWindow 4891 -> 358 lines + 15 mixins in `modules/gui/`; `widgets.py` -> `modules/widgets/` package;
  `curvature_analysis.py` 850 -> 452 lines (+ `sampling_methods.py`, `sampling_debug.py`).
- Curvature (`modules/edge_curvature.py`, JS `curvature_engine.js`), results change:
  - smoothing_px (default 6 px Gaussian along the edge; 0 = none, legacy three-point rule);
  - straight_threshold (default 0.01/px): sign 0 for straight; convention now +1 / -1 / 0;
  - open (border-touching) edges no longer wrap round; their end points get no curvature / intensity;
  - bug found: the "equidistant" sampler drifted (steps up to one segment short, one big gap at the end:
    2.0 px steps and a 42 px gap for 100 points on a 40 px circle). point_spacing "equal" (default) / "legacy".
  - Defaults chosen with `scripts/curvature_benchmark.py` (seeded shapes of known curvature, 0-0.6 px noise).
    Legacy options reproduce the old code exactly on closed edges (tested).
  - Synthetic demo: r^2 (intensity ~ normalized curvature) 0.03 -> 0.45 with the defaults (built-in effect now visible).
- GUI: "analyze current frame" runs in a `FunctionWorker` thread (was inline on the main thread); parameter panel has
  the three new options; `analysis_parameters()` builds the dict once with DEFAULTS fallbacks.
  Offscreen tests `tests/test_gui_offscreen.py` (pytest-qt, if installed, pinned to PyQt5 in pyproject).
- Incident: the first run of the digest script used the real QSettings and overwrote George's saved desktop settings
  (`~/Library/Preferences/com.cellcurvatureanalyzer.Application.plist`): n_points 10 -> 100, depth 150 -> 20, width 75
  -> 5, export_directory set to a deleted temp folder; other saved values (colormaps, marker size, export options,
  window geometry) may also have changed. The script and tests now redirect QSettings to a temporary file. Restoring
  the three known values needs George (`defaults write com.cellcurvatureanalyzer.Application n_points -int 10`, etc.).
- Still on the main thread: the Correlation tab's regressions (milliseconds) and the export (matplotlib pyplot is not
  thread-safe). The ML tab is never added to the window (`create_ml_analysis_tab` is not called; `ml_analyzer` is
  never created), so its training code is unreachable.
