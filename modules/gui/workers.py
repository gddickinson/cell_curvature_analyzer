"""Background (QThread) workers used by MainWindow, so analysis never runs on the Qt main thread."""
from PyQt5.QtCore import QThread, pyqtSignal

from modules.analysis_defaults import DEFAULTS


class AnalysisWorker(QThread):
    """Worker thread for running analysis without freezing the UI"""
    progress_signal = pyqtSignal(int, str)
    complete_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, analyzer, params):
        super().__init__()
        self.analyzer = analyzer
        self.params = params
        print(f"DEBUG: AnalysisWorker initialized with try_rotation={params.get('try_rotation')}")


    def run(self):
        try:
            p, d = self.params, DEFAULTS
            results = self.analyzer.process_stack(
                p['image_path'], p['mask_path'], p['output_dir'], p['n_points'], p['depth'], p['width'],
                p['min_cell_coverage'],
                p.get('try_rotation', d['try_rotation']),
                p.get('exclude_endpoints', d['exclude_endpoints']),
                p.get('angular_distribution', False),
                p.get('axis_based_distribution', False),
                p.get('x_axis_distribution', False),
                self.progress_callback,
                point_spacing=p.get('point_spacing', d['point_spacing']),
                smoothing_px=p.get('smoothing_px', d['smoothing_px']),
                straight_threshold=p.get('straight_threshold', d['straight_threshold']),
            )
            self.complete_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))

    def progress_callback(self, percentage, message):
        self.progress_signal.emit(percentage, message)

