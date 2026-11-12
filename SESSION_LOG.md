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
