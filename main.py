#!/usr/bin/env python3
"""
Cell Curvature Analyzer

A GUI application for analyzing cell-membrane curvature and PIEZO1 protein locations
from fluorescence microscope recordings.

Features:
- Interactive GUI with multiple visualization modes
- Curvature and intensity correlation analysis
- Temporal analysis across frames
- Edge movement detection and classification
- Comprehensive data export options
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Import main application window
from modules.cell_curvature_analyzer import MainWindow

def main():
    """Start the desktop application (also the `cell-curvature-analyzer` console script)."""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Cell Curvature Analyzer")

    # Set application style
    app.setStyle("Fusion")

    # Create main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
