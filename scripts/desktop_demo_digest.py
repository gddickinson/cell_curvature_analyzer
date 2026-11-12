"""Digest of everything the desktop app computes and draws on the synthetic demo stack (offscreen Qt).

Used to check that refactors (module splits) do not change behaviour: run before and after, then compare.

    QT_QPA_PLATFORM=offscreen python scripts/desktop_demo_digest.py OUT.json
    python scripts/desktop_demo_digest.py --compare A.json B.json

It runs CurvatureAnalyzer.process_stack with every sampling method, then drives MainWindow (no dialogs: message
boxes and popup dialogs are patched, QSettings go to a temporary directory): runs the analysis through the GUI,
visits every frame with every correlation / temporal / movement view, analyses the current frame, trains and applies
the ML model, opens the popup plots and exports everything. It records rounded hashes of result arrays, the text of
every results table and the data of every matplotlib artist on every figure, and the exported files (CSV/JSON text
with timestamps removed; binary files by size only).
"""
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _h(a):
    a = np.asarray(a)
    if a.dtype == object:
        return hashlib.sha256(repr(a.tolist()).encode()).hexdigest()[:16]
    if a.dtype.kind in "fc":
        a = np.round(a.astype(float), 9) + 0.0
    return hashlib.sha256(np.ascontiguousarray(a).tobytes() + str(a.shape).encode()).hexdigest()[:16]


def _norm(v):
    """JSON-friendly digest of a results structure."""
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in v.items() if k != "timestamp"}
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(x, np.ndarray) for x in v) and len({x.shape for x in v}) == 1:
            return "arr:" + _h(np.stack(v))
        return [_norm(x) for x in v]
    if isinstance(v, np.ndarray):
        return "arr:" + _h(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 9)
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    return v


def _figure(fig):
    out = []
    for ax in fig.axes:
        a = {"title": ax.get_title(), "x": ax.get_xlabel(), "y": ax.get_ylabel(),
             "lines": [_h(l.get_xydata()) for l in ax.get_lines()],
             "colls": [_h(c.get_offsets()) + ("/" + _h(c.get_array()) if c.get_array() is not None else "")
                       for c in ax.collections],
             "images": [_h(im.get_array()) for im in ax.get_images()],
             "patches": len(ax.patches), "texts": [t.get_text() for t in ax.texts]}
        out.append(a)
    return out


def _table(t):
    return [[(t.item(r, c).text() if t.item(r, c) else None) for c in range(t.columnCount())]
            for r in range(t.rowCount())]


def isolate_settings(tmp):
    """Make every QSettings(...) in the app an INI file under tmp (must run before the app modules are imported)."""
    import PyQt5.QtCore as qc
    real = qc.QSettings

    class TmpSettings(real):
        def __init__(self, *a, **k):
            super().__init__(str(tmp / "settings.ini"), real.IniFormat)
    qc.QSettings = TmpSettings


def pipeline_digest(img_path, mask_path, out_dir):
    from modules.curvature_analysis import CurvatureAnalyzer
    an = CurvatureAnalyzer()
    res = {}
    for name, kw in [("standard", {}), ("angular", {"angular_distribution": True}),
                     ("axis", {"axis_based_distribution": True}), ("x_axis", {"x_axis_distribution": True}),
                     ("rot_excl", {"try_rotation": True, "exclude_endpoints": True, "n_points": 60})]:
        res[name] = _norm(an.process_stack(img_path, mask_path, out_dir, **kw))
    return res


def gui_digest(img_path, mask_path, images, masks, tmp):
    from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    app = QApplication.instance() or QApplication([])
    popups = []
    for name in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
    QDialog.exec_ = lambda self: popups.append(
        [_figure(c.figure) for c in self.findChildren(FigureCanvasQTAgg)]) or 0

    from modules.cell_curvature_analyzer import MainWindow
    w = MainWindow()
    for s in (w.settings, w.settings_manager.settings):  # never read or write the user's real settings
        assert s.fileName().startswith(str(tmp)), s.fileName()
    out = {}

    def snap(tag):
        figs = {k: _figure(v) for k, v in sorted(vars(w).items()) if isinstance(v, Figure)}
        tables = {k: _table(v) for k, v in sorted(vars(w).items())
                  if v.__class__.__name__ == "ResultsTable" or k.endswith("_table")}
        out[tag] = {"figs": figs, "tables": tables}

    def wait(attr):
        th = getattr(w, attr, None)
        while th is not None and th.isRunning():
            app.processEvents()
            th.wait(20)
        for _ in range(5):
            app.processEvents()

    w.loaded_data["images"], w.loaded_data["masks"] = images, masks
    w.loaded_data["image_path"], w.loaded_data["mask_path"] = str(img_path), str(mask_path)
    w.parameter_panel.update_frame_range(len(images) - 1)
    np.random.seed(0)
    w.run_analysis()
    wait("analysis_worker")
    out["results"] = _norm({k: v for k, v in w.loaded_data["results"].items()})
    for f in range(len(images)):
        w.set_current_frame(f)
        snap(f"frame{f}")
        for metric in ("Curvature", "Movement"):
            w.metric_type_combo.setCurrentText(metric)
            subs = w.correlation_type_combo if metric == "Curvature" else w.movement_type_combo
            for si in range(subs.count()):
                subs.setCurrentIndex(si)
                for mode in ("Current Frame", "All Frames", "By Movement Type"):
                    w.correlation_mode_combo.setCurrentText(mode)
                    for tm in ("Current Frame", "Current-Previous Frames", "Current-Random Frames"):
                        w.parameter_panel.temporal_mode_combo.setCurrentText(tm)
                        w.parameter_panel.reference_frame_spin.setValue(1)
                        np.random.seed(1)
                        w.update_correlation_display()
                        snap(f"corr{f}/{metric}/{si}/{mode}/{tm}")
        w.parameter_panel.temporal_mode_combo.setCurrentText("Current Frame")
        for i in range(w.temporal_measure_combo.count()):
            w.temporal_measure_combo.setCurrentIndex(i)
            w.update_temporal_display()
            snap(f"temporal{f}/{i}")
        for i in range(w.movement_display_combo.count()):
            w.movement_display_combo.setCurrentIndex(i)
            w.update_movement_display()
            snap(f"movement{f}/{i}")
        w.image_type_combo.setCurrentIndex(2)
        w.update_image_display()
        snap(f"image{f}")
    w.set_current_frame(1)
    w.analyze_current_frame()
    wait("frame_worker")
    out["current_frame"] = _norm(w.loaded_data["results"][1])
    snap("after_current_frame")
    w.train_ml_model()
    wait("ml_worker")
    snap("ml")
    for kind in ("sign", "magnitude", "normalized"):
        w.create_popup_curvature_plot(kind)
    w.create_popup_intensity_plot()
    w.create_popup_combined_plot()
    out["popups"] = popups
    exp = tmp / "export"
    exp.mkdir()
    w.export_directory_label.setText(str(exp))
    w.export_data()
    files = {}
    for p in sorted(exp.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(exp))
            if p.suffix in (".csv", ".json", ".txt"):
                txt = re.sub(r"\d{4}-\d\d-\d\d[ T_]\d\d:\d\d:\d\d(\.\d+)?", "<ts>", p.read_text(errors="replace"))
                files[rel] = hashlib.sha256(txt.encode()).hexdigest()[:16]
            else:
                files[rel] = "binary"
    out["export"] = files
    w.close()
    return out


def main(argv):
    if argv[:1] == ["--compare"]:
        a, b = (json.loads(Path(p).read_text()) for p in argv[1:3])
        diffs = []

        def walk(x, y, path):
            if type(x) is not type(y):
                diffs.append(path)
            elif isinstance(x, dict):
                for k in sorted(set(x) | set(y)):
                    walk(x.get(k), y.get(k), f"{path}/{k}")
            elif isinstance(x, list) and len(x) == len(y):
                for i, (u, v) in enumerate(zip(x, y)):
                    walk(u, v, f"{path}[{i}]")
            elif x != y:
                diffs.append(path)
        walk(a, b, "")
        print(f"{len(diffs)} differences" + "".join(f"\n  {d}" for d in diffs[:40]))
        return 1 if diffs else 0
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import matplotlib
    matplotlib.use("Agg") if os.environ.get("DIGEST_AGG") else None
    import tifffile
    from modules.synthetic_cell import make_demo_stack
    images, masks = make_demo_stack()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        isolate_settings(tmp)
        os.chdir(tmp)  # run_analysis writes ./results
        ip, mp = tmp / "img.tif", tmp / "mask.tif"
        tifffile.imwrite(ip, images)
        tifffile.imwrite(mp, masks)
        imgs, msks = tifffile.imread(ip), (tifffile.imread(mp) > 0).astype(np.uint8)
        digest = {"pipeline": pipeline_digest(ip, mp, tmp / "out"), "gui": gui_digest(ip, mp, imgs, msks, tmp)}
    Path(argv[0]).write_text(json.dumps(digest, indent=0, sort_keys=True))
    print(f"wrote {argv[0]}: {len(json.dumps(digest)) / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
