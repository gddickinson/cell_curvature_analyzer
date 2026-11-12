"""Custom PyQt5 widgets used by MainWindow.

image_viewer.ImageViewer, overlay_canvas.OverlayCanvas, tables_log.ResultsTable / LogConsole,
parameter_panel.ParameterPanel (re-exported here, so `from modules.widgets import X` keeps working).
"""
from .image_viewer import ImageViewer
from .overlay_canvas import OverlayCanvas
from .tables_log import ResultsTable, LogConsole
from .parameter_panel import ParameterPanel

__all__ = ["ImageViewer", "OverlayCanvas", "ResultsTable", "LogConsole", "ParameterPanel"]
