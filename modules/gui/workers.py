"""Background (QThread) workers used by MainWindow, so analysis never runs on the Qt main thread."""
from PyQt5.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    """Worker thread for running analysis without freezing the UI"""
    progress_signal = pyqtSignal(int, str)
    complete_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, analyzer, params):
        super().__init__()
        self.analyzer = analyzer
        self.params = params
        print(f"DEBUG: AnalysisWorker initialized with try_rotation={params.get('try_rotation', False)}")


    def run(self):
        try:
            results = self.analyzer.process_stack(
                self.params['image_path'],
                self.params['mask_path'],
                self.params['output_dir'],
                self.params['n_points'],
                self.params['depth'],
                self.params['width'],
                self.params['min_cell_coverage'],
                self.params.get('try_rotation', False),
                self.params.get('exclude_endpoints', False),
                self.params.get('angular_distribution', False),
                self.params.get('axis_based_distribution', False),
                self.params.get('x_axis_distribution', False),
                self.progress_callback
            )
            self.complete_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))

    def progress_callback(self, percentage, message):
        self.progress_signal.emit(percentage, message)
