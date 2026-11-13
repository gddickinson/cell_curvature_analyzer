"""Desktop GUI, offscreen: it starts, analysis runs in worker threads (never on the Qt main thread), and the
curvature options in the parameter panel reach the analysis. QSettings go to a temporary INI file."""
import os
import threading

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtCore import QSettings  # noqa: E402

from modules.analysis_defaults import DEFAULTS  # noqa: E402
from modules.synthetic_cell import make_demo_stack  # noqa: E402


@pytest.fixture
def window(tmp_path, monkeypatch):
    import modules.cell_curvature_analyzer as cca
    import modules.settings as st

    class TmpSettings(QSettings):
        def __init__(self, *a, **k):
            super().__init__(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    monkeypatch.setattr(cca, "QSettings", TmpSettings)
    monkeypatch.setattr(st, "QSettings", TmpSettings)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.chdir(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = cca.MainWindow()
    assert w.settings.fileName().startswith(str(tmp_path))
    images, masks = make_demo_stack()
    w.loaded_data.update(images=images, masks=masks)
    yield app, w
    w.close()


def _wait(app, worker):
    assert worker.wait(30000)
    for _ in range(10):
        app.processEvents()


def test_parameter_panel_starts_at_the_defaults(window):
    _, w = window
    p = w.analysis_parameters()
    for k in ("point_spacing", "smoothing_px", "straight_threshold", "try_rotation", "exclude_endpoints"):
        assert p[k] == DEFAULTS[k], k
    w.parameter_panel.smoothing_spin.setValue(0.0)
    w.parameter_panel.point_spacing_combo.setCurrentText("Legacy (pre-2026-11)")
    assert w.analysis_parameters()["smoothing_px"] == 0.0
    assert w.analysis_parameters()["point_spacing"] == "legacy"
    assert w.parameter_panel.smoothing_spin.text() == "none (legacy)"


def test_current_frame_analysis_runs_in_a_worker_thread(window):
    app, w = window
    seen = {}
    real = w.curvature_analyzer.analyze_frame

    def spy(*a, **k):
        seen["thread"] = threading.current_thread()
        seen["kwargs"] = k
        return real(*a, **k)
    w.curvature_analyzer.analyze_frame = spy
    w.set_current_frame(1)
    w.analyze_current_frame()
    _wait(app, w.frame_worker)
    assert seen["thread"] is not threading.main_thread()
    assert seen["kwargs"]["smoothing_px"] == DEFAULTS["smoothing_px"]
    r = w.loaded_data["results"][1]
    assert r["closed"] is False and len(r["points"]) == DEFAULTS["n_points"]
    assert w.statusBar().currentMessage() == "Frame 2 analysis completed"


def test_current_frame_error_is_reported_on_the_main_thread(window):
    app, w = window

    def boom(*a, **k):
        raise RuntimeError("no luck")
    w.curvature_analyzer.analyze_frame = boom
    w.analyze_current_frame()
    _wait(app, w.frame_worker)
    assert w.statusBar().currentMessage() == "Error analyzing frame"


def test_stack_analysis_runs_in_a_worker_thread(window, tmp_path):
    import tifffile
    app, w = window
    images, masks = w.loaded_data["images"], w.loaded_data["masks"]
    tifffile.imwrite(tmp_path / "i.tif", images)
    tifffile.imwrite(tmp_path / "m.tif", masks)
    w.loaded_data.update(image_path=str(tmp_path / "i.tif"), mask_path=str(tmp_path / "m.tif"))
    seen = {}
    real = w.curvature_analyzer.process_stack

    def spy(*a, **k):
        seen["thread"] = threading.current_thread()
        seen["kwargs"] = k
        return real(*a, **k)
    w.curvature_analyzer.process_stack = spy
    w.run_analysis()
    _wait(app, w.analysis_worker)
    assert seen["thread"] is not threading.main_thread()
    assert seen["kwargs"]["straight_threshold"] == DEFAULTS["straight_threshold"]
    res = w.loaded_data["results"]
    assert res["parameters"]["smoothing_px"] == DEFAULTS["smoothing_px"]
    assert set(np.unique(np.concatenate([res[t]["curvatures"][0] for t in range(4)]))) <= {-1.0, 0.0, 1.0}
