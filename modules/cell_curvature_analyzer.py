import sys
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import tifffile
from skimage import measure, morphology, draw
import scipy.stats
import json
import datetime
import platform
import random
from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QApplication, QTabWidget, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit,
                            QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QGroupBox,
                            QSplitter, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
                            QMessageBox, QAction, QToolBar, QStatusBar, QProgressBar,
                            QDockWidget, QGridLayout, QFormLayout, QSlider, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QSize, QTimer
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QIcon, QFont, QPixmap

# Import custom modules
from modules.curvature_analysis import CurvatureAnalyzer
from modules.file_io import FileManager
from modules.visualization import VisualizationManager
from modules.data_export import ExportManager
from modules.settings import SettingsManager
from modules.correlation_analysis import CorrelationAnalyzer
from modules.temporal_analysis import TemporalAnalyzer
from modules.widgets import ImageViewer, OverlayCanvas, ResultsTable, LogConsole, ParameterPanel
from modules.ml_analysis import MLAnalyzer


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

class MainWindow(QMainWindow):
    """Main application window for CellCurvatureAnalyzer"""

    def __init__(self):
        super().__init__()

        # Initialize application components
        self.file_manager = FileManager()
        self.curvature_analyzer = CurvatureAnalyzer()
        self.visualization_manager = VisualizationManager()
        self.export_manager = ExportManager()
        self.settings_manager = SettingsManager()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()

        # Application state
        self.current_frame = 0
        self.loaded_data = {
            'images': None,
            'masks': None,
            'results': None
        }

        # Create parameter panel first (before connecting signals)
        self.parameter_panel = None

        # Setup UI
        self.init_ui()

        # Connect parameter panel signals AFTER UI is initialized
        if self.parameter_panel:
            self.parameter_panel.parameters_changed.connect(self.on_parameters_changed)

        # Restore previous settings
        self.settings = QSettings("CellCurvatureAnalyzer", "Application")
        self.restore_settings()

        # Status message
        self.statusBar().showMessage("Ready")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Cell Curvature & PIEZO1 Analyzer")
        self.setGeometry(100, 100, 1280, 800)

        # Create main toolbar
        self.create_toolbar()

        # Create dock widgets first (so parameter panel is available)
        self.create_parameter_dock()
        self.create_info_dock()

        # Create main layout with central tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add main tabs
        self.create_visualization_tab()
        self.create_analysis_tab()
        self.create_correlation_tab()
        self.create_temporal_tab()
        self.create_movement_tab()
        self.create_export_tab()
        self.create_settings_tab()
        self.create_log_tab()

        # Create status bar with progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        # Connect signals and slots
        self.connect_signals()

    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Open action
        open_action = QAction("Open Images", self)
        open_action.setStatusTip("Open microscope image stack")
        open_action.triggered.connect(self.open_image_stack)
        toolbar.addAction(open_action)

        # Open masks action
        masks_action = QAction("Open Masks", self)
        masks_action.setStatusTip("Open binary mask stack")
        masks_action.triggered.connect(self.open_mask_stack)
        toolbar.addAction(masks_action)

        toolbar.addSeparator()

        # Run analysis action
        run_action = QAction("Run Analysis", self)
        run_action.setStatusTip("Run full analysis")
        run_action.triggered.connect(self.run_analysis)
        toolbar.addAction(run_action)

        # Save results action
        save_action = QAction("Save Results", self)
        save_action.setStatusTip("Save analysis results")
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Frame control
        self.frame_label = QLabel("Frame: 0/0")
        toolbar.addWidget(self.frame_label)

        prev_frame_action = QAction("Previous", self)
        prev_frame_action.setStatusTip("Previous frame")
        prev_frame_action.triggered.connect(self.previous_frame)
        toolbar.addAction(prev_frame_action)

        next_frame_action = QAction("Next", self)
        next_frame_action.setStatusTip("Next frame")
        next_frame_action.triggered.connect(self.next_frame)
        toolbar.addAction(next_frame_action)

    def create_visualization_tab(self):
        """Create the visualization tab for viewing images and overlays"""
        viz_tab = QWidget()
        self.tabs.addTab(viz_tab, "Visualization")

        layout = QVBoxLayout(viz_tab)

        # Create splitter for flexible layout
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Images
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Image viewer
        self.image_viewer = ImageViewer()
        left_layout.addWidget(self.image_viewer)

        # Control panel
        image_control_panel = QGroupBox("Image Controls")
        image_control_layout = QVBoxLayout()
        image_control_panel.setLayout(image_control_layout)

        # Display type controls
        display_layout = QHBoxLayout()

        # Image type selector
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["Original Image", "Binary Mask", "Overlay"])
        self.image_type_combo.currentIndexChanged.connect(self.update_image_display)
        display_layout.addWidget(QLabel("Display:"))
        display_layout.addWidget(self.image_type_combo)

        image_control_layout.addLayout(display_layout)

        # Brightness controls
        brightness_layout = QGridLayout()

        # Min brightness
        self.min_brightness_label = QLabel("Min Brightness:")
        brightness_layout.addWidget(self.min_brightness_label, 0, 0)

        self.min_brightness_slider = QSlider(Qt.Horizontal)
        self.min_brightness_slider.setRange(0, 100)
        self.min_brightness_slider.setValue(0)
        self.min_brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.min_brightness_slider.setTickInterval(10)
        self.min_brightness_slider.valueChanged.connect(self.update_image_display)
        brightness_layout.addWidget(self.min_brightness_slider, 0, 1)

        self.min_brightness_spin = QSpinBox()
        self.min_brightness_spin.setRange(0, 100)
        self.min_brightness_spin.setValue(0)
        self.min_brightness_spin.valueChanged.connect(self.min_brightness_slider.setValue)
        brightness_layout.addWidget(self.min_brightness_spin, 0, 2)

        # Max brightness
        self.max_brightness_label = QLabel("Max Brightness:")
        brightness_layout.addWidget(self.max_brightness_label, 1, 0)

        self.max_brightness_slider = QSlider(Qt.Horizontal)
        self.max_brightness_slider.setRange(0, 100)
        self.max_brightness_slider.setValue(100)
        self.max_brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.max_brightness_slider.setTickInterval(10)
        self.max_brightness_slider.valueChanged.connect(self.update_image_display)
        brightness_layout.addWidget(self.max_brightness_slider, 1, 1)

        self.max_brightness_spin = QSpinBox()
        self.max_brightness_spin.setRange(0, 100)
        self.max_brightness_spin.setValue(100)
        self.max_brightness_spin.valueChanged.connect(self.max_brightness_slider.setValue)
        brightness_layout.addWidget(self.max_brightness_spin, 1, 2)

        # Mask opacity control
        self.mask_opacity_label = QLabel("Mask Opacity:")
        brightness_layout.addWidget(self.mask_opacity_label, 2, 0)

        self.mask_opacity_slider = QSlider(Qt.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(50)
        self.mask_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.mask_opacity_slider.setTickInterval(10)
        self.mask_opacity_slider.valueChanged.connect(self.update_image_display)
        brightness_layout.addWidget(self.mask_opacity_slider, 2, 1)

        self.mask_opacity_spin = QSpinBox()
        self.mask_opacity_spin.setRange(0, 100)
        self.mask_opacity_spin.setValue(50)
        self.mask_opacity_spin.valueChanged.connect(self.mask_opacity_slider.setValue)
        brightness_layout.addWidget(self.mask_opacity_spin, 2, 2)

        # Reset button
        self.reset_brightness_btn = QPushButton("Reset")
        self.reset_brightness_btn.clicked.connect(self.reset_brightness_controls)
        brightness_layout.addWidget(self.reset_brightness_btn, 3, 1)

        image_control_layout.addLayout(brightness_layout)

        # Connect sliders to spin boxes
        self.min_brightness_slider.valueChanged.connect(self.min_brightness_spin.setValue)
        self.max_brightness_slider.valueChanged.connect(self.max_brightness_spin.setValue)
        self.mask_opacity_slider.valueChanged.connect(self.mask_opacity_spin.setValue)

        left_layout.addWidget(image_control_panel)
        splitter.addWidget(left_widget)

        # Right panel - Overlays and results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Overlay canvas
        self.overlay_canvas = OverlayCanvas()
        right_layout.addWidget(self.overlay_canvas)

        # Overlay control panel
        overlay_control_panel = QGroupBox("Overlay Controls")
        overlay_control_layout = QGridLayout()
        overlay_control_panel.setLayout(overlay_control_layout)

        # Overlay checkboxes
        self.show_contour_cb = QCheckBox("Show Cell Contour")
        self.show_contour_cb.setChecked(True)
        self.show_contour_cb.toggled.connect(self.update_overlay_display)

        self.show_curvature_cb = QCheckBox("Show Curvature")
        self.show_curvature_cb.setChecked(True)
        self.show_curvature_cb.toggled.connect(self.update_overlay_display)

        # Add curvature sign checkbox
        self.show_curvature_sign_cb = QCheckBox("Show Curvature Sign")
        self.show_curvature_sign_cb.setChecked(False)
        self.show_curvature_sign_cb.toggled.connect(self.update_overlay_display)

        self.show_intensity_cb = QCheckBox("Show Intensity")
        self.show_intensity_cb.setChecked(True)
        self.show_intensity_cb.toggled.connect(self.update_overlay_display)

        self.show_sampling_cb = QCheckBox("Show Sampling Regions")
        self.show_sampling_cb.setChecked(True)
        self.show_sampling_cb.toggled.connect(self.update_overlay_display)

        overlay_control_layout.addWidget(self.show_contour_cb, 0, 0)
        overlay_control_layout.addWidget(self.show_curvature_cb, 0, 1)
        overlay_control_layout.addWidget(self.show_curvature_sign_cb, 1, 0)
        overlay_control_layout.addWidget(self.show_intensity_cb, 1, 1)
        overlay_control_layout.addWidget(self.show_sampling_cb, 2, 0)

        # Add movement overlay checkbox
        self.show_movement_cb = QCheckBox("Show Movement")
        self.show_movement_cb.setChecked(False)
        self.show_movement_cb.toggled.connect(self.update_overlay_display)

        overlay_control_layout.addWidget(self.show_movement_cb, 2, 1)


        # Add overlay brightness controls
        overlay_brightness_layout = QGridLayout()

        # Min brightness
        self.overlay_min_brightness_label = QLabel("Min Brightness:")
        overlay_brightness_layout.addWidget(self.overlay_min_brightness_label, 0, 0)

        self.overlay_min_brightness_slider = QSlider(Qt.Horizontal)
        self.overlay_min_brightness_slider.setRange(0, 100)
        self.overlay_min_brightness_slider.setValue(0)
        self.overlay_min_brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.overlay_min_brightness_slider.setTickInterval(10)
        self.overlay_min_brightness_slider.valueChanged.connect(self.update_overlay_display)
        overlay_brightness_layout.addWidget(self.overlay_min_brightness_slider, 0, 1)

        self.overlay_min_brightness_spin = QSpinBox()
        self.overlay_min_brightness_spin.setRange(0, 100)
        self.overlay_min_brightness_spin.setValue(0)
        self.overlay_min_brightness_spin.valueChanged.connect(self.overlay_min_brightness_slider.setValue)
        overlay_brightness_layout.addWidget(self.overlay_min_brightness_spin, 0, 2)

        # Max brightness
        self.overlay_max_brightness_label = QLabel("Max Brightness:")
        overlay_brightness_layout.addWidget(self.overlay_max_brightness_label, 1, 0)

        self.overlay_max_brightness_slider = QSlider(Qt.Horizontal)
        self.overlay_max_brightness_slider.setRange(0, 100)
        self.overlay_max_brightness_slider.setValue(100)
        self.overlay_max_brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.overlay_max_brightness_slider.setTickInterval(10)
        self.overlay_max_brightness_slider.valueChanged.connect(self.update_overlay_display)
        overlay_brightness_layout.addWidget(self.overlay_max_brightness_slider, 1, 1)

        self.overlay_max_brightness_spin = QSpinBox()
        self.overlay_max_brightness_spin.setRange(0, 100)
        self.overlay_max_brightness_spin.setValue(100)
        self.overlay_max_brightness_spin.valueChanged.connect(self.overlay_max_brightness_slider.setValue)
        overlay_brightness_layout.addWidget(self.overlay_max_brightness_spin, 1, 2)

        # Mask opacity for overlay
        self.overlay_mask_opacity_label = QLabel("Mask Opacity:")
        overlay_brightness_layout.addWidget(self.overlay_mask_opacity_label, 2, 0)

        self.overlay_mask_opacity_slider = QSlider(Qt.Horizontal)
        self.overlay_mask_opacity_slider.setRange(0, 100)
        self.overlay_mask_opacity_slider.setValue(50)
        self.overlay_mask_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.overlay_mask_opacity_slider.setTickInterval(10)
        self.overlay_mask_opacity_slider.valueChanged.connect(self.update_overlay_display)
        overlay_brightness_layout.addWidget(self.overlay_mask_opacity_slider, 2, 1)

        self.overlay_mask_opacity_spin = QSpinBox()
        self.overlay_mask_opacity_spin.setRange(0, 100)
        self.overlay_mask_opacity_spin.setValue(50)
        self.overlay_mask_opacity_spin.valueChanged.connect(self.overlay_mask_opacity_slider.setValue)
        overlay_brightness_layout.addWidget(self.overlay_mask_opacity_spin, 2, 2)

        # Reset button for overlay
        self.reset_overlay_brightness_btn = QPushButton("Reset")
        self.reset_overlay_brightness_btn.clicked.connect(self.reset_overlay_brightness_controls)
        overlay_brightness_layout.addWidget(self.reset_overlay_brightness_btn, 3, 1)

        # Connect sliders to spin boxes for overlay controls
        self.overlay_min_brightness_slider.valueChanged.connect(self.overlay_min_brightness_spin.setValue)
        self.overlay_max_brightness_slider.valueChanged.connect(self.overlay_max_brightness_spin.setValue)
        self.overlay_mask_opacity_slider.valueChanged.connect(self.overlay_mask_opacity_spin.setValue)

        # Add overlay brightness controls to the layout
        overlay_control_layout.addLayout(overlay_brightness_layout, 3, 0, 1, 2)

        right_layout.addWidget(overlay_control_panel)
        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([500, 500])


    def reset_overlay_brightness_controls(self):
        """Reset overlay brightness and opacity controls to their default values"""
        self.overlay_min_brightness_slider.setValue(0)
        self.overlay_max_brightness_slider.setValue(100)
        self.overlay_mask_opacity_slider.setValue(50)

    def reset_brightness_controls(self):
        """Reset brightness and opacity controls to their default values"""
        self.min_brightness_slider.setValue(0)
        self.max_brightness_slider.setValue(100)
        self.mask_opacity_slider.setValue(50)

    def create_analysis_tab(self):
        """Create the analysis tab with results tables and charts"""
        analysis_tab = QWidget()
        self.tabs.addTab(analysis_tab, "Analysis Results")

        layout = QVBoxLayout(analysis_tab)

        # Create splitter for flexible layout
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # Upper panel - Frame-specific results
        upper_widget = QWidget()
        upper_layout = QVBoxLayout(upper_widget)

        frame_results_group = QGroupBox("Current Frame Results")
        frame_results_layout = QVBoxLayout()
        frame_results_group.setLayout(frame_results_layout)

        # Frame results table
        self.frame_results_table = ResultsTable()
        frame_results_layout.addWidget(self.frame_results_table)

        upper_layout.addWidget(frame_results_group)
        splitter.addWidget(upper_widget)

        # Lower panel - Frame-specific charts
        lower_widget = QWidget()
        lower_layout = QGridLayout(lower_widget)

        # Create frame for curvature plot
        curvature_frame = QGroupBox("Curvature Profile")
        curvature_layout = QVBoxLayout()
        curvature_frame.setLayout(curvature_layout)

        self.curvature_figure = Figure(figsize=(5, 4), dpi=100)
        self.curvature_canvas = FigureCanvas(self.curvature_figure)
        self.curvature_toolbar = NavigationToolbar(self.curvature_canvas, self)
        curvature_layout.addWidget(self.curvature_toolbar)
        curvature_layout.addWidget(self.curvature_canvas)

        # Create frame for intensity plot
        intensity_frame = QGroupBox("Intensity Profile")
        intensity_layout = QVBoxLayout()
        intensity_frame.setLayout(intensity_layout)

        self.intensity_figure = Figure(figsize=(5, 4), dpi=100)
        self.intensity_canvas = FigureCanvas(self.intensity_figure)
        self.intensity_toolbar = NavigationToolbar(self.intensity_canvas, self)
        intensity_layout.addWidget(self.intensity_toolbar)
        intensity_layout.addWidget(self.intensity_canvas)

        # Create frame for correlation plot
        correlation_frame = QGroupBox("Curvature-Intensity Correlation")
        correlation_layout = QVBoxLayout()
        correlation_frame.setLayout(correlation_layout)

        self.correlation_figure = Figure(figsize=(5, 4), dpi=100)
        self.correlation_canvas = FigureCanvas(self.correlation_figure)
        self.correlation_toolbar = NavigationToolbar(self.correlation_canvas, self)
        correlation_layout.addWidget(self.correlation_toolbar)
        correlation_layout.addWidget(self.correlation_canvas)

        # Add frames to grid layout
        lower_layout.addWidget(curvature_frame, 0, 0)
        lower_layout.addWidget(intensity_frame, 0, 1)
        lower_layout.addWidget(correlation_frame, 1, 0, 1, 2)

        splitter.addWidget(lower_widget)

        # Set splitter proportions
        splitter.setSizes([200, 600])

    def create_correlation_tab(self):
        """Create the correlation analysis tab"""
        correlation_tab = QWidget()
        self.tabs.addTab(correlation_tab, "Correlation Analysis")

        layout = QVBoxLayout(correlation_tab)

        # Controls
        control_panel = QGroupBox("Correlation Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Add metric type selector (NEW)
        self.metric_type_combo = QComboBox()
        self.metric_type_combo.addItems(["Curvature", "Movement"])
        self.metric_type_combo.currentIndexChanged.connect(self.on_metric_type_changed)

        self.metric_type_label = QLabel("Metric Type:")
        control_layout.addWidget(self.metric_type_label)
        control_layout.addWidget(self.metric_type_combo)

        # Existing curvature type combo
        self.correlation_type_combo = QComboBox()
        self.correlation_type_combo.addItems(["Sign Curvature", "Magnitude Curvature", "Normalized Curvature"])
        self.correlation_type_combo.currentIndexChanged.connect(self.update_correlation_display)

        self.curvature_type_label = QLabel("Curvature Type:")
        control_layout.addWidget(self.curvature_type_label)
        control_layout.addWidget(self.correlation_type_combo)

        # NEW: Add movement type combo
        self.movement_type_combo = QComboBox()
        self.movement_type_combo.addItems(["Movement Score", "Extension/Retraction"])
        self.movement_type_combo.currentIndexChanged.connect(self.update_correlation_display)
        self.movement_type_combo.setVisible(False)  # Initially hidden

        self.movement_type_label = QLabel("Movement Type:")
        self.movement_type_label.setVisible(False)  # Initially hidden
        control_layout.addWidget(self.movement_type_label)
        control_layout.addWidget(self.movement_type_combo)

        self.correlation_mode_combo = QComboBox()
        self.correlation_mode_combo.addItems(["Current Frame", "All Frames", "By Movement Type"])
        self.correlation_mode_combo.currentIndexChanged.connect(self.update_correlation_display)

        control_layout.addWidget(QLabel("Analysis Mode:"))
        control_layout.addWidget(self.correlation_mode_combo)

        # Add comparison metric selector
        self.comparison_metric_combo = QComboBox()
        self.comparison_metric_combo.addItems(["R²", "p-value", "Slope", "Sample Size"])
        self.comparison_metric_combo.currentIndexChanged.connect(self.update_comparison_chart)

        control_layout.addWidget(QLabel("Comparison Metric:"))
        control_layout.addWidget(self.comparison_metric_combo)

        # Add a button to generate movement data for all frames
        self.generate_movement_btn = QPushButton("Generate Movement Data")
        self.generate_movement_btn.clicked.connect(self.generate_all_frames_movement_data)
        self.generate_movement_btn.setVisible(False)  # Initially hidden
        control_layout.addWidget(self.generate_movement_btn)

        layout.addWidget(control_panel)

        # Create correlation display with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Scatter plot
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        correlation_plot_group = QGroupBox("Correlation Plot")
        correlation_plot_layout = QVBoxLayout()
        correlation_plot_group.setLayout(correlation_plot_layout)

        self.correlation_plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.correlation_plot_canvas = FigureCanvas(self.correlation_plot_figure)
        self.correlation_plot_toolbar = NavigationToolbar(self.correlation_plot_canvas, self)
        correlation_plot_layout.addWidget(self.correlation_plot_toolbar)
        correlation_plot_layout.addWidget(self.correlation_plot_canvas)

        left_layout.addWidget(correlation_plot_group)
        splitter.addWidget(left_widget)

        # Right panel - Statistics and summary
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Statistics table
        stats_group = QGroupBox("Correlation Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.correlation_stats_table = QTableWidget()
        self.correlation_stats_table.setColumnCount(4)
        self.correlation_stats_table.setHorizontalHeaderLabels(["Parameter", "Value", "Significance", "Notes"])
        self.correlation_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.correlation_stats_table)

        right_layout.addWidget(stats_group)

        # Comparison chart (for movement types)
        comparison_group = QGroupBox("Comparison Chart")
        comparison_layout = QVBoxLayout()
        comparison_group.setLayout(comparison_layout)

        self.comparison_figure = Figure(figsize=(6, 4), dpi=100)
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        self.comparison_toolbar = NavigationToolbar(self.comparison_canvas, self)
        comparison_layout.addWidget(self.comparison_toolbar)
        comparison_layout.addWidget(self.comparison_canvas)

        right_layout.addWidget(comparison_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([600, 400])

    def create_temporal_tab(self):
        """Create the temporal analysis tab"""
        temporal_tab = QWidget()
        self.tabs.addTab(temporal_tab, "Temporal Analysis")

        layout = QVBoxLayout(temporal_tab)

        # Controls
        control_panel = QGroupBox("Temporal Analysis Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Keep only the temporal visualization type and measurement type
        self.temporal_type_combo = QComboBox()
        self.temporal_type_combo.addItems(["Heatmap Visualization"])
        self.temporal_type_combo.currentIndexChanged.connect(self.update_temporal_display)

        control_layout.addWidget(QLabel("Visualization Type:"))
        control_layout.addWidget(self.temporal_type_combo)

        self.temporal_measure_combo = QComboBox()
        self.temporal_measure_combo.addItems(["Sign Curvature", "Normalized Curvature", "Intensity"])
        self.temporal_measure_combo.currentIndexChanged.connect(self.update_temporal_display)

        control_layout.addWidget(QLabel("Measure:"))
        control_layout.addWidget(self.temporal_measure_combo)

        layout.addWidget(control_panel)

        # Create analysis display with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Visualization
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        temporal_plot_group = QGroupBox("Temporal Visualization")
        temporal_plot_layout = QVBoxLayout()
        temporal_plot_group.setLayout(temporal_plot_layout)

        self.temporal_plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.temporal_plot_canvas = FigureCanvas(self.temporal_plot_figure)
        self.temporal_plot_toolbar = NavigationToolbar(self.temporal_plot_canvas, self)
        temporal_plot_layout.addWidget(self.temporal_plot_toolbar)
        temporal_plot_layout.addWidget(self.temporal_plot_canvas)

        left_layout.addWidget(temporal_plot_group)
        splitter.addWidget(left_widget)

        # Right panel - Statistics and summary
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Statistics
        temporal_stats_group = QGroupBox("Temporal Statistics")
        temporal_stats_layout = QVBoxLayout()
        temporal_stats_group.setLayout(temporal_stats_layout)

        self.temporal_stats_table = QTableWidget()
        self.temporal_stats_table.setColumnCount(3)
        self.temporal_stats_table.setHorizontalHeaderLabels(["Parameter", "Value", "Notes"])
        self.temporal_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        temporal_stats_layout.addWidget(self.temporal_stats_table)

        right_layout.addWidget(temporal_stats_group)

        # Summary plot
        temporal_summary_group = QGroupBox("Temporal Summary")
        temporal_summary_layout = QVBoxLayout()
        temporal_summary_group.setLayout(temporal_summary_layout)

        self.temporal_summary_figure = Figure(figsize=(6, 4), dpi=100)
        self.temporal_summary_canvas = FigureCanvas(self.temporal_summary_figure)
        temporal_summary_layout.addWidget(self.temporal_summary_canvas)

        right_layout.addWidget(temporal_summary_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([600, 400])

    def update_comparison_chart(self):
        """Update the comparison chart based on the selected metric"""
        # Only update if in "By Movement Type" mode
        if self.correlation_mode_combo.currentText() != "By Movement Type":
            return

        # Get the current regression stats (this might be stored from the last analysis)
        # We'll need to add a class variable to store the regression stats
        if not hasattr(self, 'current_regression_stats') or not self.current_regression_stats:
            return

        # Get selected metric
        metric = self.comparison_metric_combo.currentText()

        # Map metric to data key
        metric_key = {
            "R²": 'r_squared',
            "p-value": 'p_value',
            "Slope": 'slope',
            "Sample Size": 'sample_size'
        }.get(metric)

        if not metric_key:
            return

        # Create bar chart
        self.comparison_figure.clear()
        ax = self.comparison_figure.add_subplot(111)

        # Get data for selected metric
        values = []
        labels = []
        colors_list = []

        # Colors for different movement types
        colors = {
            'extending': 'blue',
            'retracting': 'red',
            'stable': 'gray'
        }

        for movement in ['extending', 'retracting', 'stable']:
            if movement in self.current_regression_stats and metric_key in self.current_regression_stats[movement]:
                values.append(self.current_regression_stats[movement][metric_key])
                labels.append(movement.capitalize())
                colors_list.append(colors[movement])

        # Create bar chart
        bars = ax.bar(labels, values, color=colors_list, alpha=0.7)

        # Add value annotations
        for i, value in enumerate(values):
            # Format value based on metric
            if metric == "R²" or metric == "p-value":
                value_text = f"{value:.3f}"
            elif metric == "Slope":
                value_text = f"{value:.3f}"
            else:  # Sample Size
                value_text = f"{int(value)}"

            # Position annotation
            y_pos = value + (max(values) * 0.02)
            ax.annotate(value_text, (i, y_pos), ha='center', va='bottom', fontsize=9)

        # Add p-value annotations for certain metrics
        if metric == "R²":
            for i, movement in enumerate(['extending', 'retracting', 'stable']):
                if movement in self.current_regression_stats and i < len(values):
                    p_value = self.current_regression_stats[movement].get('p_value', 1.0)
                    annotation = f"p={p_value:.3f}"
                    if p_value < 0.001:
                        annotation = "p<0.001"
                    elif p_value < 0.01:
                        annotation = "p<0.01"
                    elif p_value < 0.05:
                        annotation = "p<0.05"

                    ax.annotate(annotation, (i, values[i] + max(values) * 0.05),
                                ha='center', va='bottom', fontsize=9, rotation=90)

        # Set labels and title based on metric
        ax.set_xlabel('Movement Type')
        ax.set_ylabel(metric)
        curvature_type = self.correlation_type_combo.currentText()
        ax.set_title(f'Comparison of {metric} by Movement Type for {curvature_type}')

        # Refresh canvas
        self.comparison_canvas.draw()

    def create_movement_tab(self):
        """Create the movement analysis tab"""
        movement_tab = QWidget()
        self.tabs.addTab(movement_tab, "Edge Movement")

        layout = QVBoxLayout(movement_tab)

        # Controls
        control_panel = QGroupBox("Movement Analysis Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        self.movement_display_combo = QComboBox()
        self.movement_display_combo.addItems(["Movement Map", "Edge Comparison", "Movement Over Time"])
        self.movement_display_combo.currentIndexChanged.connect(self.update_movement_display)

        control_layout.addWidget(QLabel("Display Type:"))
        control_layout.addWidget(self.movement_display_combo)

        self.movement_frame_spin = QSpinBox()
        self.movement_frame_spin.setMinimum(1)  # First frame has no movement data
        self.movement_frame_spin.setMaximum(1)
        self.movement_frame_spin.valueChanged.connect(self.on_movement_frame_changed)  # Use a separate handler

        control_layout.addWidget(QLabel("Frame:"))
        control_layout.addWidget(self.movement_frame_spin)

        layout.addWidget(control_panel)

        # Create analysis display with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Visualization
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        movement_plot_group = QGroupBox("Movement Visualization")
        movement_plot_layout = QVBoxLayout()
        movement_plot_group.setLayout(movement_plot_layout)

        self.movement_plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.movement_plot_canvas = FigureCanvas(self.movement_plot_figure)
        self.movement_plot_toolbar = NavigationToolbar(self.movement_plot_canvas, self)
        movement_plot_layout.addWidget(self.movement_plot_toolbar)
        movement_plot_layout.addWidget(self.movement_plot_canvas)

        left_layout.addWidget(movement_plot_group)
        splitter.addWidget(left_widget)

        # Right panel - Statistics and summary
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Movement statistics
        movement_stats_group = QGroupBox("Movement Statistics")
        movement_stats_layout = QVBoxLayout()
        movement_stats_group.setLayout(movement_stats_layout)

        self.movement_stats_table = QTableWidget()
        self.movement_stats_table.setColumnCount(3)
        self.movement_stats_table.setHorizontalHeaderLabels(["Parameter", "Value", "Notes"])
        self.movement_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        movement_stats_layout.addWidget(self.movement_stats_table)

        right_layout.addWidget(movement_stats_group)

        # Movement summary
        movement_summary_group = QGroupBox("Movement Summary")
        movement_summary_layout = QVBoxLayout()
        movement_summary_group.setLayout(movement_summary_layout)

        self.movement_summary_figure = Figure(figsize=(6, 4), dpi=100)
        self.movement_summary_canvas = FigureCanvas(self.movement_summary_figure)
        self.movement_summary_toolbar = NavigationToolbar(self.movement_summary_canvas, self)
        movement_summary_layout.addWidget(self.movement_summary_toolbar)
        movement_summary_layout.addWidget(self.movement_summary_canvas)

        right_layout.addWidget(movement_summary_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([600, 400])

    def create_export_tab(self):
        """Create the export tab for saving results"""
        export_tab = QWidget()
        self.tabs.addTab(export_tab, "Export Results")

        layout = QVBoxLayout(export_tab)

        # Data selection
        data_group = QGroupBox("Data Selection")
        data_layout = QGridLayout()
        data_group.setLayout(data_layout)

        # Checkboxes for different export options
        self.export_images_cb = QCheckBox("Visualization Images")
        self.export_images_cb.setChecked(True)

        self.export_raw_data_cb = QCheckBox("Raw Data (CSV)")
        self.export_raw_data_cb.setChecked(True)

        self.export_stats_cb = QCheckBox("Statistics (JSON)")
        self.export_stats_cb.setChecked(True)

        self.export_figures_cb = QCheckBox("Figures (PNG)")
        self.export_figures_cb.setChecked(True)

        self.export_report_cb = QCheckBox("Summary Report (PDF)")
        self.export_report_cb.setChecked(True)

        data_layout.addWidget(self.export_images_cb, 0, 0)
        data_layout.addWidget(self.export_raw_data_cb, 0, 1)
        data_layout.addWidget(self.export_stats_cb, 1, 0)
        data_layout.addWidget(self.export_figures_cb, 1, 1)
        data_layout.addWidget(self.export_report_cb, 2, 0, 1, 2)

        layout.addWidget(data_group)

        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QFormLayout()
        options_group.setLayout(options_layout)

        self.export_directory_label = QLabel("Not selected")
        self.export_directory_btn = QPushButton("Browse...")
        self.export_directory_btn.clicked.connect(self.select_export_directory)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.export_directory_label)
        dir_layout.addWidget(self.export_directory_btn)

        self.export_prefix_edit = QLineEdit("cell_analysis_")

        options_layout.addRow("Export Directory:", dir_layout)
        options_layout.addRow("Filename Prefix:", self.export_prefix_edit)

        layout.addWidget(options_group)

        # Preview and export buttons
        button_layout = QHBoxLayout()

        self.preview_export_btn = QPushButton("Preview Export")
        self.preview_export_btn.clicked.connect(self.preview_export)

        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)

        button_layout.addStretch()
        button_layout.addWidget(self.preview_export_btn)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

        # Preview area
        preview_group = QGroupBox("Export Preview")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)

        self.export_preview_text = QTextEdit()
        self.export_preview_text.setReadOnly(True)
        preview_layout.addWidget(self.export_preview_text)

        layout.addWidget(preview_group)

    def on_movement_frame_changed(self, frame_index):
        """Handle changes to the movement frame spinner"""
        # Update the current frame for movement analysis only
        # Note: This does not change the current_frame variable which affects other tabs
        self.update_movement_display()

        # Log frame change
        self.log_console.log(f"Movement frame changed to {frame_index}")

    def generate_all_frames_movement_data(self):
        """Generate movement data for all frames"""
        results = self.loaded_data.get('results')
        if results is None:
            return

        # Process each frame
        for frame_idx in list(results.keys()):  # Create a copy of keys
            if not isinstance(frame_idx, int):
                continue

            self.generate_frame_movement_data(frame_idx)

        # Log completion
        self.log_console.log(f"Generated movement data for all frames")

    def on_metric_type_changed(self):
        """Handle changes to the metric type selector"""
        metric_type = self.metric_type_combo.currentText()

        # Show/hide appropriate controls
        if metric_type == "Curvature":
            self.correlation_type_combo.setVisible(True)
            self.curvature_type_label.setVisible(True)
            self.movement_type_combo.setVisible(False)
            self.movement_type_label.setVisible(False)
            self.generate_movement_btn.setVisible(False)
        else:  # Movement
            self.correlation_type_combo.setVisible(False)
            self.curvature_type_label.setVisible(False)
            self.movement_type_combo.setVisible(True)
            self.movement_type_label.setVisible(True)
            self.generate_movement_btn.setVisible(True)

        # Update the display
        self.update_correlation_display()

    def create_settings_tab(self):
        """Create the settings tab"""
        settings_tab = QWidget()
        self.tabs.addTab(settings_tab, "Settings")

        layout = QVBoxLayout(settings_tab)

        # Analysis parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QGridLayout()
        analysis_group.setLayout(analysis_layout)

        # Number of points
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(10, 500)
        self.n_points_spin.setValue(100)
        self.n_points_spin.setSingleStep(5)
        analysis_layout.addWidget(QLabel("Number of Points:"), 0, 0)
        analysis_layout.addWidget(self.n_points_spin, 0, 1)

        # Sampling depth
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 200)
        self.depth_spin.setValue(20)
        analysis_layout.addWidget(QLabel("Sampling Depth:"), 1, 0)
        analysis_layout.addWidget(self.depth_spin, 1, 1)

        # Sampling width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100)
        self.width_spin.setValue(5)
        analysis_layout.addWidget(QLabel("Sampling Width:"), 2, 0)
        analysis_layout.addWidget(self.width_spin, 2, 1)

        # Minimum cell coverage
        self.min_coverage_spin = QDoubleSpinBox()
        self.min_coverage_spin.setRange(0.1, 1.0)
        self.min_coverage_spin.setValue(0.8)
        self.min_coverage_spin.setSingleStep(0.05)
        analysis_layout.addWidget(QLabel("Minimum Cell Coverage:"), 3, 0)
        analysis_layout.addWidget(self.min_coverage_spin, 3, 1)

        layout.addWidget(analysis_group)

        # Visualization settings
        visualization_group = QGroupBox("Visualization Settings")
        visualization_layout = QGridLayout()
        visualization_group.setLayout(visualization_layout)

        # Color map selection
        self.curvature_cmap_combo = QComboBox()
        self.curvature_cmap_combo.addItems(["coolwarm", "RdBu", "seismic", "bwr"])
        visualization_layout.addWidget(QLabel("Curvature Colormap:"), 0, 0)
        visualization_layout.addWidget(self.curvature_cmap_combo, 0, 1)

        self.intensity_cmap_combo = QComboBox()
        self.intensity_cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis"])
        visualization_layout.addWidget(QLabel("Intensity Colormap:"), 1, 0)
        visualization_layout.addWidget(self.intensity_cmap_combo, 1, 1)

        # Marker size
        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 50)
        self.marker_size_spin.setValue(10)
        visualization_layout.addWidget(QLabel("Marker Size:"), 2, 0)
        visualization_layout.addWidget(self.marker_size_spin, 2, 1)

        # Contour line width
        self.contour_width_spin = QDoubleSpinBox()
        self.contour_width_spin.setRange(0.5, 5.0)
        self.contour_width_spin.setValue(1.0)
        self.contour_width_spin.setSingleStep(0.5)
        visualization_layout.addWidget(QLabel("Contour Line Width:"), 3, 0)
        visualization_layout.addWidget(self.contour_width_spin, 3, 1)

        layout.addWidget(visualization_group)

        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout()
        advanced_group.setLayout(advanced_layout)

        # Enable temporal analysis
        self.enable_temporal_cb = QCheckBox("Enable Temporal Analysis")
        self.enable_temporal_cb.setChecked(True)
        advanced_layout.addWidget(self.enable_temporal_cb, 0, 0)

        # Enable random control
        self.enable_random_cb = QCheckBox("Enable Random Control Analysis")
        self.enable_random_cb.setChecked(True)
        self.enable_random_cb.toggled.connect(self.update_random_control_state)
        advanced_layout.addWidget(self.enable_random_cb, 0, 1)

        # Enable movement analysis
        self.enable_movement_cb = QCheckBox("Enable Movement Analysis")
        self.enable_movement_cb.setChecked(True)
        advanced_layout.addWidget(self.enable_movement_cb, 1, 0)

        # Enable region-specific analysis
        self.enable_region_cb = QCheckBox("Enable Region-Specific Analysis")
        self.enable_region_cb.setChecked(True)
        advanced_layout.addWidget(self.enable_region_cb, 1, 1)

        layout.addWidget(advanced_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.reset_settings_btn = QPushButton("Reset to Defaults")
        self.reset_settings_btn.clicked.connect(self.reset_settings)

        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)

        button_layout.addStretch()
        button_layout.addWidget(self.reset_settings_btn)
        button_layout.addWidget(self.save_settings_btn)

        layout.addLayout(button_layout)

    def update_curvature_plot(self, frame_results):
        """Update the curvature profile plot with frame data"""
        if 'points' not in frame_results or 'curvatures' not in frame_results:
            return

        # Get data
        points = frame_results['points']
        sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

        # Clear figure
        self.curvature_figure.clear()
        ax = self.curvature_figure.add_subplot(111)

        # Plot sign curvature
        ax.plot(sign_curvatures, 'b-', label='Sign')

        # Plot magnitude curvature on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(magnitude_curvatures, 'r-', label='Magnitude')

        # Plot normalized curvature
        ax.plot(normalized_curvatures, 'g-', label='Normalized')

        # Add labels and legend
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Curvature Sign / Normalized')
        ax2.set_ylabel('Curvature Magnitude')
        ax.set_title('Curvature Profile')

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Refresh canvas
        self.curvature_canvas.draw()

    def update_intensity_plot(self, frame_results):
        """Update the intensity profile plot with frame data"""
        if ('points' not in frame_results or
            'intensities' not in frame_results or
            'valid_points' not in frame_results):
            return

        # Get data
        points = frame_results['points']
        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Clear figure
        self.intensity_figure.clear()
        ax = self.intensity_figure.add_subplot(111)

        # Create x-axis (point indices)
        x = np.arange(len(points))

        # Create masked array for invalid points
        masked_intensities = np.ma.array(intensities, mask=~valid_points)

        # Plot intensity
        ax.plot(x, masked_intensities, 'b-', label='Intensity')
        ax.scatter(x[valid_points], intensities[valid_points], c='b', s=30, alpha=0.7)

        # Add labels and legend
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Intensity')
        ax.set_title('Intensity Profile')
        ax.legend(loc='best')

        # Refresh canvas
        self.intensity_canvas.draw()

    def update_correlation_plot(self, frame_results):
        """Update the correlation plot with frame data"""
        if ('curvatures' not in frame_results or
            'intensities' not in frame_results or
            'valid_points' not in frame_results):
            return

        # Get data
        sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']
        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Filter data for valid points
        valid_norm_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]

        # Clear figure
        self.correlation_figure.clear()
        ax = self.correlation_figure.add_subplot(111)

        # Plot scatter plot
        ax.scatter(valid_norm_curvatures, valid_intensities, alpha=0.7)

        # Add regression line if there are enough points
        if len(valid_norm_curvatures) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                valid_norm_curvatures, valid_intensities)

            # Create line data
            x_line = np.linspace(
                min(valid_norm_curvatures),
                max(valid_norm_curvatures),
                100
            )
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-',
                    label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = "Curvature-Intensity Correlation"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_norm_curvatures)}"
        else:
            title = "Curvature-Intensity Correlation\nInsufficient data for regression"

        # Add labels and title
        ax.set_xlabel('Normalized Curvature')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(valid_norm_curvatures) > 1:
            ax.legend(loc='best')

        # Refresh canvas
        self.correlation_canvas.draw()


    def update_comparative_curvature_plot(self, current_results, reference_results, reference_label):
        """Update curvature plot with data from both current and reference frames"""
        # Clear figure
        self.curvature_figure.clear()
        ax = self.curvature_figure.add_subplot(111)

        # Get current frame curvature data
        _, _, current_norm_curvatures = current_results['curvatures']

        # Get reference frame curvature data
        _, _, reference_norm_curvatures = reference_results['curvatures']

        # Plot both curvature profiles
        ax.plot(current_norm_curvatures, 'b-', label='Current Frame')
        ax.plot(reference_norm_curvatures, 'r-', label=reference_label)

        # Add labels and title
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Normalized Curvature')
        ax.set_title('Comparative Curvature Profile')
        ax.legend()

        # Refresh canvas
        self.curvature_canvas.draw()

    def update_comparative_intensity_plot(self, current_results, reference_results, reference_label):
        """Update intensity plot with data from both current and reference frames"""
        # Clear figure
        self.intensity_figure.clear()
        ax = self.intensity_figure.add_subplot(111)

        # Get current frame intensity data
        current_intensities = current_results['intensities']
        current_valid = current_results['valid_points']

        # Get reference frame intensity data
        reference_intensities = reference_results['intensities']
        reference_valid = reference_results['valid_points']

        # Create x-axis (point indices)
        x = np.arange(len(current_intensities))

        # Create masked arrays for invalid points
        current_masked = np.ma.array(current_intensities, mask=~current_valid)
        reference_masked = np.ma.array(reference_intensities, mask=~reference_valid)

        # Plot intensity profiles
        ax.plot(x, current_masked, 'b-', label='Current Frame')
        ax.scatter(x[current_valid], current_intensities[current_valid], c='b', s=30, alpha=0.7)

        ax.plot(x, reference_masked, 'r-', label=reference_label)
        ax.scatter(x[reference_valid], reference_intensities[reference_valid], c='r', s=30, alpha=0.7)

        # Add labels and title
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Intensity')
        ax.set_title('Comparative Intensity Profile')
        ax.legend()

        # Refresh canvas
        self.intensity_canvas.draw()


    def add_temporal_correlation_stats(self, current_results, reference_results, reference_label, table):
        """
        Add temporal correlation statistics to the given table

        Parameters:
        -----------
        current_results : dict
            Results for current frame
        reference_results : dict
            Results for reference frame
        reference_label : str
            Label for the reference frame
        table : ResultsTable
            Table to add statistics to
        """
        # Get current frame curvature data
        _, _, current_norm_curvatures = current_results['curvatures']

        # Get reference frame intensity data
        reference_intensities = reference_results['intensities']

        # Get valid points (both frames)
        current_valid = current_results['valid_points']
        reference_valid = reference_results['valid_points']

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, reference_valid)

        # Skip if no valid points
        if not np.any(valid_points):
            table.add_result("Status", "Insufficient valid points")
            return

        # Filter data for valid points
        valid_curvatures = current_norm_curvatures[valid_points]
        valid_intensities = reference_intensities[valid_points]

        # Calculate correlation if enough points
        if len(valid_curvatures) > 1:
            # Calculate regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_curvatures, valid_intensities)

            table.add_result("R-squared", r_value**2)
            table.add_result("p-value", p_value)
            table.add_result("Slope", slope)
            table.add_result("Sample Size", len(valid_curvatures))
        else:
            table.add_result("Status", "Insufficient data for correlation")



    def update_temporal_correlation_plot(self, current_results, reference_results, reference_label):
        """Update correlation plot with current curvature vs reference intensity"""
        # Clear figure
        self.correlation_figure.clear()
        ax = self.correlation_figure.add_subplot(111)

        # Get current frame curvature data
        _, _, current_norm_curvatures = current_results['curvatures']

        # Get reference frame intensity data
        reference_intensities = reference_results['intensities']

        # Get valid points (both frames)
        current_valid = current_results['valid_points']
        reference_valid = reference_results['valid_points']

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, reference_valid)

        # Skip if no valid points
        if not np.any(valid_points):
            ax.set_title("Insufficient data for temporal correlation")
            self.correlation_canvas.draw()
            return

        # Filter data for valid points
        valid_curvatures = current_norm_curvatures[valid_points]
        valid_intensities = reference_intensities[valid_points]

        # Create scatter plot
        ax.scatter(valid_curvatures, valid_intensities, alpha=0.7)

        # Add regression line if enough points
        if len(valid_curvatures) > 1:
            # Calculate regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_curvatures, valid_intensities)

            # Create line data
            x_line = np.linspace(min(valid_curvatures), max(valid_curvatures), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"Current Frame Curvature vs {reference_label} Intensity"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_curvatures)}"
        else:
            title = f"Current Frame Curvature vs {reference_label} Intensity\nInsufficient data for regression"

        # Set labels and title
        ax.set_xlabel('Current Frame Normalized Curvature')
        ax.set_ylabel(f'{reference_label} Intensity')
        ax.set_title(title)
        if len(valid_curvatures) > 1:
            ax.legend()

        # Refresh canvas
        self.correlation_canvas.draw()

    def update_frame_results_table(self, frame_results):
        """Update the frame results table with current frame data"""
        self.frame_results_table.setRowCount(0)

        # Add basic information
        if 'points' in frame_results:
            self.frame_results_table.add_result("Number of Points", len(frame_results['points']))

        if 'valid_points' in frame_results:
            valid_count = np.sum(frame_results['valid_points'])
            total_count = len(frame_results['valid_points'])
            self.frame_results_table.add_result("Valid Points", f"{valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")

        # Add curvature statistics
        if 'curvatures' in frame_results:
            sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

            self.frame_results_table.add_result("Mean Sign Curvature", np.mean(sign_curvatures))
            self.frame_results_table.add_result("Mean Magnitude Curvature", np.mean(magnitude_curvatures))
            self.frame_results_table.add_result("Mean Normalized Curvature", np.mean(normalized_curvatures))

        # Add intensity statistics
        if 'intensities' in frame_results and 'valid_points' in frame_results:
            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']

            if np.any(valid_points):
                valid_intensities = intensities[valid_points]
                self.frame_results_table.add_result("Mean Intensity", np.mean(valid_intensities))
                self.frame_results_table.add_result("Min Intensity", np.min(valid_intensities))
                self.frame_results_table.add_result("Max Intensity", np.max(valid_intensities))

        # Get temporal mode
        temporal_mode = self.parameter_panel.parameters['temporal_mode']

        # Add temporal correlation statistics if applicable
        if temporal_mode == "Current-Previous Frames" and self.current_frame > 0:
            previous_frame = self.current_frame - 1

            if previous_frame in self.loaded_data['results']:
                # Get previous frame results
                previous_results = self.loaded_data['results'][previous_frame]

                # Calculate correlation if both frames have valid data
                if ('curvatures' in frame_results and 'intensities' in previous_results and
                    'valid_points' in frame_results and 'valid_points' in previous_results):

                    # Add section header
                    self.frame_results_table.add_result("Temporal Correlation", "Current vs Previous Frame")

                    # Calculate correlation
                    self.add_temporal_correlation_stats(frame_results, previous_results,
                                                     "Previous Frame", self.frame_results_table)

        elif temporal_mode == "Current-Random Frames":
            reference_frame = self.parameter_panel.parameters['reference_frame']

            if reference_frame in self.loaded_data['results'] and reference_frame != self.current_frame:
                # Get reference frame results
                reference_results = self.loaded_data['results'][reference_frame]

                # Calculate correlation if both frames have valid data
                if ('curvatures' in frame_results and 'intensities' in reference_results and
                    'valid_points' in frame_results and 'valid_points' in reference_results):

                    # Add section header
                    self.frame_results_table.add_result("Temporal Correlation", f"Current vs Frame {reference_frame+1}")

                    # Calculate correlation
                    self.add_temporal_correlation_stats(frame_results, reference_results,
                                                     f"Frame {reference_frame+1}", self.frame_results_table)


    def update_frame_info(self, frame_results):
        """Update the frame information panel with current frame data"""
        # Update frame info
        image_path = self.loaded_data.get('image_path', 'N/A')
        mask_path = self.loaded_data.get('mask_path', 'N/A')
        frame = self.current_frame

        self.frame_info_label.setText(
            f"Frame: {frame + 1}\n"
            f"Image: {os.path.basename(image_path)}\n"
            f"Mask: {os.path.basename(mask_path)}"
        )

        # Update statistics
        if 'curvatures' in frame_results:
            _, _, normalized_curvatures = frame_results['curvatures']
            mean_curvature = np.mean(normalized_curvatures)
            std_curvature = np.std(normalized_curvatures)
            self.curvature_stats_label.setText(f"{mean_curvature:.3f} ± {std_curvature:.3f}")
        else:
            self.curvature_stats_label.setText("N/A")

        if 'intensities' in frame_results and 'valid_points' in frame_results:
            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']
            if np.any(valid_points):
                valid_intensities = intensities[valid_points]
                mean_intensity = np.mean(valid_intensities)
                std_intensity = np.std(valid_intensities)
                self.intensity_stats_label.setText(f"{mean_intensity:.3f} ± {std_intensity:.3f}")
            else:
                self.intensity_stats_label.setText("N/A")
        else:
            self.intensity_stats_label.setText("N/A")

        if 'valid_points' in frame_results:
            valid_count = np.sum(frame_results['valid_points'])
            total_count = len(frame_results['valid_points'])
            self.valid_points_label.setText(f"{valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        else:
            self.valid_points_label.setText("N/A")

        # Update movement type if available
        results = self.loaded_data.get('results')
        movement_type = "N/A"

        if results and 'movement_types' in results and frame > 0:
            movement_types = results.get('movement_types', [])
            if frame - 1 < len(movement_types):  # Subtract 1 because first frame has no movement
                movement_type = movement_types[frame - 1].capitalize()

        self.movement_type_label.setText(movement_type)


    def update_correlation_display(self):
        """Update the correlation analysis tab display"""
        results = self.loaded_data.get('results')

        if results is None:
            return

        # Get current selections
        metric_type = self.metric_type_combo.currentText()

        if metric_type == "Curvature":
            subtype = self.correlation_type_combo.currentText()
        else:  # Movement
            subtype = self.movement_type_combo.currentText()

        analysis_mode = self.correlation_mode_combo.currentText()

        # Get temporal mode from parameter panel
        temporal_mode = self.parameter_panel.parameters.get('temporal_mode', "Current Frame")
        reference_frame = self.parameter_panel.parameters.get('reference_frame', 0)

        # Update based on analysis mode
        if analysis_mode == "Current Frame":
            # Clear plot
            self.correlation_plot_figure.clear()
            ax = self.correlation_plot_figure.add_subplot(111)

            # Get current frame data
            frame = self.current_frame

            if frame in results:
                frame_results = results[frame]

                # For movement analysis, generate data if needed
                if metric_type == "Movement" and 'movement_data' not in frame_results:
                    self.generate_frame_movement_data(frame)

                if temporal_mode == "Current Frame":
                    # Standard single-frame correlation
                    self.analyze_current_frame_correlation(frame_results, metric_type, subtype, ax)

                elif temporal_mode == "Current-Previous Frames" and frame > 0:
                    previous_frame = frame - 1
                    if previous_frame in results:
                        previous_results = results[previous_frame]

                        # For movement analysis, generate data for previous frame if needed
                        if metric_type == "Movement" and 'movement_data' not in previous_results:
                            self.generate_frame_movement_data(previous_frame)

                        self.analyze_temporal_frame_correlation(frame_results, previous_results,
                                                              metric_type, subtype, "Previous Frame", ax)

                elif temporal_mode == "Current-Random Frames" and reference_frame != frame:
                    if reference_frame in results:
                        reference_results = results[reference_frame]

                        # For movement analysis, generate data for reference frame if needed
                        if metric_type == "Movement" and 'movement_data' not in reference_results:
                            self.generate_frame_movement_data(reference_frame)

                        self.analyze_temporal_frame_correlation(frame_results, reference_results,
                                                              metric_type, subtype, f"Frame {reference_frame+1}", ax)

                # Refresh canvas
                self.correlation_plot_canvas.draw()

                # Clear comparison chart
                self.comparison_figure.clear()
                self.comparison_canvas.draw()

        elif analysis_mode == "All Frames":
            # For movement analysis, generate data for all frames if needed
            if metric_type == "Movement":
                self.generate_all_frames_movement_data()

            # Apply temporal mode to "All Frames" analysis
            if temporal_mode == "Current Frame":
                # Standard all-frames analysis (within each frame)
                self.analyze_all_frames_standard_correlation(results, metric_type, subtype)

            elif temporal_mode == "Current-Previous Frames":
                # Analyze correlation between current frame curvature and previous frame intensity for all frames
                self.analyze_all_frames_temporal_correlation(results, metric_type, subtype, "previous")

            elif temporal_mode == "Current-Random Frames":
                # Analyze correlation between current frame curvature and random frame intensity
                self.analyze_all_frames_temporal_correlation(results, metric_type, subtype, "random", reference_frame)

        elif analysis_mode == "By Movement Type":
            # For movement analysis, generate data for all frames if needed
            if metric_type == "Movement":
                self.generate_all_frames_movement_data()

            # Apply temporal mode to "By Movement Type" analysis
            if temporal_mode == "Current Frame":
                # Standard movement-type analysis (within each frame)
                self.analyze_movement_standard_correlation(results, metric_type, subtype)

            elif temporal_mode == "Current-Previous Frames":
                # Analyze correlation between current frame curvature and previous frame intensity by movement type
                self.analyze_movement_temporal_correlation(results, metric_type, subtype, "previous")

            elif temporal_mode == "Current-Random Frames":
                # Analyze correlation between current frame curvature and random frame intensity by movement type
                self.analyze_movement_temporal_correlation(results, metric_type, subtype, "random", reference_frame)


    def generate_frame_movement_data(self, frame_idx):
        """
        Generate movement data for a single frame

        Parameters:
        -----------
        frame_idx : int
            Frame index
        """
        results = self.loaded_data.get('results')
        if results is None or frame_idx not in results:
            return

        frame_results = results[frame_idx]

        # Skip if we already have movement data
        if 'movement_data' in frame_results:
            return

        # We need points and contour data
        if 'points' not in frame_results or 'contour_data' not in frame_results:
            return

        points = frame_results['points']
        contour = frame_results['contour_data']

        # Initialize movement data with default values
        movement_scores = np.zeros(len(points))
        movement_types = np.array(['stable'] * len(points), dtype=object)

        # For each point, calculate local curvature change as a proxy for movement
        try:
            if 'curvatures' in frame_results:
                sign_curvature, magnitude_curvature, normalized_curvature = frame_results['curvatures']

                # Calculate the gradient of curvature along the contour
                for i in range(len(points)):
                    # Get neighboring points (with wrapping)
                    prev_idx = (i - 1) % len(points)
                    next_idx = (i + 1) % len(points)

                    # Calculate curvature gradient (central difference)
                    curvature_gradient = (normalized_curvature[next_idx] - normalized_curvature[prev_idx]) / 2
                    movement_scores[i] = curvature_gradient

                    # Classify as extending, retracting, or stable
                    if curvature_gradient > 0.1:
                        movement_types[i] = 'extending'
                    elif curvature_gradient < -0.1:
                        movement_types[i] = 'retracting'
                    else:
                        movement_types[i] = 'stable'
        except Exception as e:
            # Log error but continue
            print(f"Error generating movement data for frame {frame_idx}: {str(e)}")
            return

        # Store movement data in frame results
        frame_results['movement_data'] = {
            'scores': movement_scores,
            'types': movement_types
        }

    def analyze_all_frames_standard_correlation(self, results, metric_type, subtype):
        """Standard correlation analysis combining data from all frames (within each frame)"""
        # Combine data from all frames
        combined_metric_data = []
        combined_intensities = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            # Check if required data is available
            if ('intensities' not in frame_results or
                'valid_points' not in frame_results):
                continue

            # Get data based on metric type and subtype
            if metric_type == "Curvature":
                if 'curvatures' not in frame_results:
                    continue

                # Map subtype to curvature index
                if subtype == "Sign Curvature":
                    metric_data = frame_results['curvatures'][0]  # Sign curvature
                elif subtype == "Magnitude Curvature":
                    metric_data = frame_results['curvatures'][1]  # Magnitude curvature
                elif subtype == "Normalized Curvature":
                    metric_data = frame_results['curvatures'][2]  # Normalized curvature
                else:
                    continue

                metric_name = subtype

            elif metric_type == "Movement":
                if 'movement_data' not in frame_results:
                    # Try to generate movement data
                    self.generate_frame_movement_data(frame_idx)

                    if 'movement_data' not in frame_results:
                        continue

                if subtype == "Movement Score":
                    metric_data = frame_results['movement_data']['scores']
                    metric_name = "Movement Score"
                elif subtype == "Extension/Retraction":
                    # Convert categorical to numerical
                    movement_types = frame_results['movement_data']['types']
                    metric_data = np.zeros(len(movement_types))
                    for i, mtype in enumerate(movement_types):
                        if mtype == 'extending':
                            metric_data[i] = 1
                        elif mtype == 'retracting':
                            metric_data[i] = -1
                    metric_name = "Extension/Retraction"
                else:
                    continue
            else:
                continue

            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']

            # Filter valid points
            valid_metric = metric_data[valid_points]
            valid_intensities = intensities[valid_points]

            # Add to combined data
            combined_metric_data.extend(valid_metric)
            combined_intensities.extend(valid_intensities)

        # Convert to numpy arrays
        combined_metric_data = np.array(combined_metric_data)
        combined_intensities = np.array(combined_intensities)

        # Clear figure
        self.correlation_plot_figure.clear()
        ax = self.correlation_plot_figure.add_subplot(111)

        # Create scatter plot
        ax.scatter(combined_metric_data, combined_intensities, alpha=0.5)

        # Add regression line if enough points
        if len(combined_metric_data) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_metric_data, combined_intensities)

            # Create line data
            x_line = np.linspace(
                min(combined_metric_data),
                max(combined_metric_data),
                100
            )
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-',
                    label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"{metric_name}-Intensity Correlation (All Frames)"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_metric_data)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frames", "All")
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            title = f"{metric_name}-Intensity Correlation (All Frames)\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frames", "All")
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(combined_metric_data) > 1:
            ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Clear comparison chart
        self.comparison_figure.clear()
        self.comparison_canvas.draw()



    def analyze_movement_standard_correlation(self, results, metric_type, subtype):
        """Standard correlation analysis by movement type (within each frame)"""
        if 'movement_types' not in results:
            return

        movement_types = results.get('movement_types', [])

        # Group frames by movement type
        extending_frames = []
        retracting_frames = []
        stable_frames = []

        for i, movement_type in enumerate(movement_types):
            # Movement type is for transitions, so the frame index is i+1
            frame = i + 1
            if movement_type == 'extending':
                extending_frames.append(frame)
            elif movement_type == 'retracting':
                retracting_frames.append(frame)
            elif movement_type == 'stable':
                stable_frames.append(frame)

        # Collect data by movement type
        movement_data = {
            'extending': {'metric_data': [], 'intensities': []},
            'retracting': {'metric_data': [], 'intensities': []},
            'stable': {'metric_data': [], 'intensities': []}
        }

        # Process each movement type
        for movement_type_name, frames in [
            ('extending', extending_frames),
            ('retracting', retracting_frames),
            ('stable', stable_frames)
        ]:
            for frame in frames:
                if frame not in results:
                    continue

                frame_results = results[frame]

                # Check for required data
                if ('intensities' not in frame_results or
                    'valid_points' not in frame_results):
                    continue

                # Get data based on metric type and subtype
                if metric_type == "Curvature":
                    if 'curvatures' not in frame_results:
                        continue

                    # Map subtype to curvature index
                    if subtype == "Sign Curvature":
                        metric_data = frame_results['curvatures'][0]  # Sign curvature
                    elif subtype == "Magnitude Curvature":
                        metric_data = frame_results['curvatures'][1]  # Magnitude curvature
                    elif subtype == "Normalized Curvature":
                        metric_data = frame_results['curvatures'][2]  # Normalized curvature
                    else:
                        continue

                    metric_name = subtype

                elif metric_type == "Movement":
                    if 'movement_data' not in frame_results:
                        # Try to generate movement data
                        self.generate_frame_movement_data(frame)

                        if 'movement_data' not in frame_results:
                            continue

                    if subtype == "Movement Score":
                        metric_data = frame_results['movement_data']['scores']
                        metric_name = "Movement Score"
                    elif subtype == "Extension/Retraction":
                        # Convert categorical to numerical
                        frame_movement_types = frame_results['movement_data']['types']
                        metric_data = np.zeros(len(frame_movement_types))
                        for i, mtype in enumerate(frame_movement_types):
                            if mtype == 'extending':
                                metric_data[i] = 1
                            elif mtype == 'retracting':
                                metric_data[i] = -1
                        metric_name = "Extension/Retraction"
                    else:
                        continue
                else:
                    continue

                intensities = frame_results['intensities']
                valid_points = frame_results['valid_points']

                # Filter for valid points
                valid_metric = metric_data[valid_points]
                valid_intensities = intensities[valid_points]

                # Add to movement data
                movement_data[movement_type_name]['metric_data'].extend(valid_metric)
                movement_data[movement_type_name]['intensities'].extend(valid_intensities)

        # Create scatter plot with all movement types
        self.correlation_plot_figure.clear()
        ax = self.correlation_plot_figure.add_subplot(111)

        # Colors for different movement types
        colors = {
            'extending': 'blue',
            'retracting': 'red',
            'stable': 'gray'
        }

        # Plot each movement type
        for movement, color in colors.items():
            metric_values = np.array(movement_data[movement]['metric_data'])
            intensity_values = np.array(movement_data[movement]['intensities'])

            if len(metric_values) > 0:
                ax.scatter(metric_values, intensity_values,
                          color=color, alpha=0.5, label=movement.capitalize())

        # Add regression lines for each type
        regression_stats = {}

        for movement, color in colors.items():
            metric_values = np.array(movement_data[movement]['metric_data'])
            intensity_values = np.array(movement_data[movement]['intensities'])

            if len(metric_values) > 1:
                # Calculate regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    metric_values, intensity_values)

                # Create line data
                try:
                    x_line = np.linspace(
                        min(metric_values),
                        max(metric_values),
                        100
                    )
                    y_line = slope * x_line + intercept

                    # Plot line
                    ax.plot(x_line, y_line, color=color, linestyle='-',
                            label=f'{movement.capitalize()}: R²={r_value**2:.3f}')
                except:
                    # Handle case where metric values have no range
                    pass

                # Store stats
                regression_stats[movement] = {
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'slope': slope,
                    'intercept': intercept,
                    'sample_size': len(metric_values)
                }

        # Set labels and title
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Intensity')
        ax.set_title(f"{metric_name}-Intensity Correlation by Movement Type")
        ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Update stats table
        self.correlation_stats_table.setRowCount(0)
        self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
        self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)

        for movement in ['extending', 'retracting', 'stable']:
            if movement in regression_stats:
                stats_dict = regression_stats[movement]
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} R²", stats_dict['r_squared'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} p-value", stats_dict['p_value'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} Slope", stats_dict['slope'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} Sample Size", stats_dict['sample_size'])

        # Store regression stats for comparison chart updates
        self.current_regression_stats = regression_stats

        # Create comparison bar chart
        self.update_comparison_chart()

    def analyze_all_frames_temporal_correlation(self, results, metric_type, subtype, mode, reference_frame=None):
        """
        Temporal correlation analysis combining data from all frames

        Parameters:
        -----------
        results : dict
            Analysis results
        metric_type : str
            Type of metric to analyze ("Curvature" or "Movement")
        subtype : str
            Subtype of the metric (e.g., "Normalized Curvature" or "Movement Score")
        mode : str
            "previous" or "random"
        reference_frame : int, optional
            Reference frame for random mode
        """
        # Combine data from all frames
        combined_metric_data = []
        combined_intensities = []

        # Find valid frames (with results)
        valid_frames = []
        for frame_idx in results.keys():
            if isinstance(frame_idx, int):
                valid_frames.append(frame_idx)

        # Sort the valid frames
        valid_frames.sort()

        # Process each frame with its temporal comparison
        for frame_idx in valid_frames:
            if frame_idx == 0 and mode == "previous":
                # Skip first frame for previous mode
                continue

            # Get current frame data
            frame_results = results[frame_idx]

            # Check for required data
            if ('valid_points' not in frame_results):
                continue

            # Get data based on metric type and subtype
            if metric_type == "Curvature":
                if 'curvatures' not in frame_results:
                    continue

                # Map subtype to curvature index
                if subtype == "Sign Curvature":
                    metric_data = frame_results['curvatures'][0]  # Sign curvature
                elif subtype == "Magnitude Curvature":
                    metric_data = frame_results['curvatures'][1]  # Magnitude curvature
                elif subtype == "Normalized Curvature":
                    metric_data = frame_results['curvatures'][2]  # Normalized curvature
                else:
                    continue

                metric_name = subtype

            elif metric_type == "Movement":
                if 'movement_data' not in frame_results:
                    # Try to generate movement data
                    self.generate_frame_movement_data(frame_idx)

                    if 'movement_data' not in frame_results:
                        continue

                if subtype == "Movement Score":
                    metric_data = frame_results['movement_data']['scores']
                    metric_name = "Movement Score"
                elif subtype == "Extension/Retraction":
                    # Convert categorical to numerical
                    movement_types = frame_results['movement_data']['types']
                    metric_data = np.zeros(len(movement_types))
                    for i, mtype in enumerate(movement_types):
                        if mtype == 'extending':
                            metric_data[i] = 1
                        elif mtype == 'retracting':
                            metric_data[i] = -1
                    metric_name = "Extension/Retraction"
                else:
                    continue
            else:
                continue

            current_valid = frame_results['valid_points']

            # Determine reference frame based on mode
            if mode == "previous":
                ref_frame = frame_idx - 1
            elif mode == "random":
                if reference_frame is not None and reference_frame in valid_frames and reference_frame != frame_idx:
                    ref_frame = reference_frame
                else:
                    # Pick a random frame that's not the current one
                    available_frames = [f for f in valid_frames if f != frame_idx]
                    if not available_frames:
                        continue
                    ref_frame = np.random.choice(available_frames)
            else:
                continue

            # Get intensity data from reference frame
            if ref_frame not in results:
                continue

            ref_results = results[ref_frame]
            if ('intensities' not in ref_results or
                'valid_points' not in ref_results):
                continue

            ref_intensities = ref_results['intensities']
            ref_valid = ref_results['valid_points']

            # Use logical AND of valid points
            valid_points = np.logical_and(current_valid, ref_valid)

            # Skip if no valid points
            if not np.any(valid_points):
                continue

            # Filter data for valid points
            valid_metric_data = metric_data[valid_points]
            valid_intensities = ref_intensities[valid_points]

            # Add to combined data
            combined_metric_data.extend(valid_metric_data)
            combined_intensities.extend(valid_intensities)

        # Convert to numpy arrays
        combined_metric_data = np.array(combined_metric_data)
        combined_intensities = np.array(combined_intensities)

        # Clear plot
        self.correlation_plot_figure.clear()
        ax = self.correlation_plot_figure.add_subplot(111)

        # Create scatter plot
        ax.scatter(combined_metric_data, combined_intensities, alpha=0.5)

        # Add regression line if enough points
        if len(combined_metric_data) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_metric_data, combined_intensities)

            # Create line data
            x_line = np.linspace(
                min(combined_metric_data),
                max(combined_metric_data),
                100
            )
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-',
                    label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Create title based on mode
            if mode == "previous":
                title = f"Current Frame {metric_name} vs Previous Frame Intensity (All Frames)"
            else:
                if reference_frame is not None:
                    title = f"Current Frame {metric_name} vs Frame {reference_frame+1} Intensity (All Frames)"
                else:
                    title = f"Current Frame {metric_name} vs Random Frame Intensity (All Frames)"

            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_metric_data)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Temporal ({mode})")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            if mode == "random" and reference_frame is not None:
                self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            # Create title based on mode
            if mode == "previous":
                title = f"Current Frame {metric_name} vs Previous Frame Intensity (All Frames)"
            else:
                if reference_frame is not None:
                    title = f"Current Frame {metric_name} vs Frame {reference_frame+1} Intensity (All Frames)"
                else:
                    title = f"Current Frame {metric_name} vs Random Frame Intensity (All Frames)"

            title += "\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Temporal ({mode})")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            if mode == "random" and reference_frame is not None:
                self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(f"Current Frame {metric_name}")
        if mode == "previous":
            ax.set_ylabel('Previous Frame Intensity')
        else:
            if reference_frame is not None:
                ax.set_ylabel(f'Frame {reference_frame+1} Intensity')
            else:
                ax.set_ylabel('Random Frame Intensity')
        ax.set_title(title)
        if len(combined_metric_data) > 1:
            ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Clear comparison chart
        self.comparison_figure.clear()
        self.comparison_canvas.draw()


    def analyze_movement_temporal_correlation(self, results, metric_type, subtype, mode, reference_frame=None):
        """
        Temporal correlation analysis by movement type

        Parameters:
        -----------
        results : dict
            Analysis results
        metric_type : str
            Type of metric to analyze ("Curvature" or "Movement")
        subtype : str
            Subtype of the metric (e.g., "Normalized Curvature" or "Movement Score")
        mode : str
            "previous" or "random"
        reference_frame : int, optional
            Reference frame for random mode
        """
        # Check if using "Random Each Time" mode
        random_each_time = (mode == "random" and
                          self.parameter_panel.parameters.get('random_mode') == "Random Each Time")

        if 'movement_types' not in results:
            return

        movement_types = results.get('movement_types', [])

        # Group frames by movement type
        extending_frames = []
        retracting_frames = []
        stable_frames = []

        for i, movement_type in enumerate(movement_types):
            # Movement type is for transitions, so the frame index is i+1
            frame = i + 1
            if movement_type == 'extending':
                extending_frames.append(frame)
            elif movement_type == 'retracting':
                retracting_frames.append(frame)
            elif movement_type == 'stable':
                stable_frames.append(frame)

        # Find valid frames (with results)
        valid_frames = []
        for frame_idx in results.keys():
            if isinstance(frame_idx, int):
                valid_frames.append(frame_idx)

        # Sort the valid frames
        valid_frames.sort()

        # Collect data by movement type using temporal relationship
        movement_data = {
            'extending': {'metric_data': [], 'intensities': []},
            'retracting': {'metric_data': [], 'intensities': []},
            'stable': {'metric_data': [], 'intensities': []}
        }

        # Process each frame by movement type, using temporal comparison
        for movement_type_name, frames in [
            ('extending', extending_frames),
            ('retracting', retracting_frames),
            ('stable', stable_frames)
        ]:
            for frame in frames:
                if frame == 0 and mode == "previous":
                    # Skip first frame for previous mode
                    continue

                # Get current frame data
                if frame not in results:
                    continue

                frame_results = results[frame]

                # Check for required data
                if ('valid_points' not in frame_results):
                    continue

                # Get data based on metric type and subtype
                if metric_type == "Curvature":
                    if 'curvatures' not in frame_results:
                        continue

                    # Map subtype to curvature index
                    if subtype == "Sign Curvature":
                        metric_data = frame_results['curvatures'][0]  # Sign curvature
                    elif subtype == "Magnitude Curvature":
                        metric_data = frame_results['curvatures'][1]  # Magnitude curvature
                    elif subtype == "Normalized Curvature":
                        metric_data = frame_results['curvatures'][2]  # Normalized curvature
                    else:
                        continue

                    metric_name = subtype

                elif metric_type == "Movement":
                    if 'movement_data' not in frame_results:
                        # Try to generate movement data
                        self.generate_frame_movement_data(frame)

                        if 'movement_data' not in frame_results:
                            continue

                    if subtype == "Movement Score":
                        metric_data = frame_results['movement_data']['scores']
                        metric_name = "Movement Score"
                    elif subtype == "Extension/Retraction":
                        # Convert categorical to numerical
                        frame_movement_types = frame_results['movement_data']['types']
                        metric_data = np.zeros(len(frame_movement_types))
                        for i, mtype in enumerate(frame_movement_types):
                            if mtype == 'extending':
                                metric_data[i] = 1
                            elif mtype == 'retracting':
                                metric_data[i] = -1
                        metric_name = "Extension/Retraction"
                    else:
                        continue
                else:
                    continue

                current_valid = frame_results['valid_points']

                # Determine reference frame based on mode
                if mode == "previous":
                    ref_frame = frame - 1
                elif mode == "random":
                    if random_each_time:
                        # Pick a random frame for each comparison
                        available_frames = [f for f in valid_frames if f != frame]
                        if not available_frames:
                            continue
                        ref_frame = np.random.choice(available_frames)
                    elif reference_frame is not None and reference_frame in valid_frames and reference_frame != frame:
                        ref_frame = reference_frame
                    else:
                        # Pick a random frame that's not the current one
                        available_frames = [f for f in valid_frames if f != frame]
                        if not available_frames:
                            continue
                        ref_frame = np.random.choice(available_frames)
                else:
                    continue

                # Get reference frame data
                if ref_frame not in results:
                    continue

                ref_results = results[ref_frame]
                if ('intensities' not in ref_results or
                    'valid_points' not in ref_results):
                    continue

                ref_intensities = ref_results['intensities']
                ref_valid = ref_results['valid_points']

                # Use logical AND of valid points
                valid_points = np.logical_and(current_valid, ref_valid)

                # Skip if no valid points
                if not np.any(valid_points):
                    continue

                # Filter data for valid points
                valid_metric_data = metric_data[valid_points]
                valid_intensities = ref_intensities[valid_points]

                # Add to movement data
                movement_data[movement_type_name]['metric_data'].extend(valid_metric_data)
                movement_data[movement_type_name]['intensities'].extend(valid_intensities)

        # Create scatter plot with all movement types
        self.correlation_plot_figure.clear()
        ax = self.correlation_plot_figure.add_subplot(111)

        # Colors for different movement types
        colors = {
            'extending': 'blue',
            'retracting': 'red',
            'stable': 'gray'
        }

        # Plot each movement type
        for movement, color in colors.items():
            metric_values = np.array(movement_data[movement]['metric_data'])
            intensity_values = np.array(movement_data[movement]['intensities'])

            if len(metric_values) > 0:
                ax.scatter(metric_values, intensity_values,
                          color=color, alpha=0.5, label=movement.capitalize())

        # Add regression lines for each type
        regression_stats = {}

        for movement, color in colors.items():
            metric_values = np.array(movement_data[movement]['metric_data'])
            intensity_values = np.array(movement_data[movement]['intensities'])

            if len(metric_values) > 1:
                # Calculate regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    metric_values, intensity_values)

                # Create line data
                try:
                    x_line = np.linspace(
                        min(metric_values),
                        max(metric_values),
                        100
                    )
                    y_line = slope * x_line + intercept

                    # Plot line
                    ax.plot(x_line, y_line, color=color, linestyle='-',
                            label=f'{movement.capitalize()}: R²={r_value**2:.3f}')
                except:
                    # Handle case where metric values have no range
                    pass

                # Store stats
                regression_stats[movement] = {
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'slope': slope,
                    'intercept': intercept,
                    'sample_size': len(metric_values)
                }

        # Set labels and title based on mode
        ax.set_xlabel(f"Current Frame {metric_name}")
        if mode == "previous":
            ax.set_ylabel('Previous Frame Intensity')
            title_mode = "Previous Frame"
        else:
            if random_each_time:
                ax.set_ylabel('Random Frames Intensity')
                title_mode = "Random Frames"
            elif reference_frame is not None:
                ax.set_ylabel(f'Frame {reference_frame+1} Intensity')
                title_mode = f"Frame {reference_frame+1}"
            else:
                ax.set_ylabel('Random Frame Intensity')
                title_mode = "Random Frame"

        ax.set_title(f"Current Frame {metric_name} vs {title_mode} Intensity by Movement Type")
        ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Update stats table
        self.correlation_stats_table.setRowCount(0)
        self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Temporal ({mode}) by Movement")
        self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
        self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
        if mode == "random":
            if random_each_time:
                self.add_table_row(self.correlation_stats_table, "Reference Frames", "Different for each point")
            elif reference_frame is not None:
                self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_frame+1)

        for movement in ['extending', 'retracting', 'stable']:
            if movement in regression_stats:
                stats_dict = regression_stats[movement]
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} R²", stats_dict['r_squared'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} p-value", stats_dict['p_value'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} Slope", stats_dict['slope'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} Sample Size", stats_dict['sample_size'])

        # Store regression stats for comparison chart updates
        self.current_regression_stats = regression_stats

        # Create comparison bar chart
        self.update_comparison_chart()


    def analyze_current_frame_correlation(self, frame_results, metric_type, subtype, ax):
        """Analyze correlation for a single frame"""
        # Check if required data is available
        if ('intensities' not in frame_results or
            'valid_points' not in frame_results):
            return

        # Get data based on metric type and subtype
        if metric_type == "Curvature":
            if 'curvatures' not in frame_results:
                return

            # Map subtype to curvature index
            curvature_index = {
                "Sign Curvature": 0,
                "Magnitude Curvature": 1,
                "Normalized Curvature": 2
            }.get(subtype, 2)

            metric_data = frame_results['curvatures'][curvature_index]
            metric_name = subtype

        elif metric_type == "Movement":
            if 'movement_data' not in frame_results:
                # Try to generate movement data
                self.generate_frame_movement_data(self.current_frame)

                if 'movement_data' not in frame_results:
                    return

            if subtype == "Movement Score":
                metric_data = frame_results['movement_data']['scores']
                metric_name = "Movement Score"
            elif subtype == "Extension/Retraction":
                # Convert categorical to numerical
                movement_types = frame_results['movement_data']['types']
                metric_data = np.zeros(len(movement_types))
                for i, mtype in enumerate(movement_types):
                    if mtype == 'extending':
                        metric_data[i] = 1
                    elif mtype == 'retracting':
                        metric_data[i] = -1
                metric_name = "Extension/Retraction"
            else:
                return
        else:
            return

        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Filter data for valid points
        valid_metric = metric_data[valid_points]
        valid_intensities = intensities[valid_points]

        # Create scatter plot
        ax.scatter(valid_metric, valid_intensities, alpha=0.7)

        # Add regression line if enough points
        if len(valid_metric) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_metric, valid_intensities)

            # Create line data
            x_line = np.linspace(min(valid_metric), max(valid_metric), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"{metric_name}-Intensity Correlation (Current Frame)"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_metric)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            title = f"{metric_name}-Intensity Correlation (Current Frame)\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(valid_metric) > 1:
            ax.legend(loc='best')

    def analyze_temporal_frame_correlation(self, current_results, reference_results, metric_type, subtype,
                                         reference_label, ax):
        """Analyze correlation between current frame metric and reference frame intensity"""
        # Check if required data is available
        if ('intensities' not in reference_results or
            'valid_points' not in current_results or
            'valid_points' not in reference_results):
            return

        # Get data based on metric type and subtype
        if metric_type == "Curvature":
            if 'curvatures' not in current_results:
                return

            # Map subtype to curvature index
            curvature_index = {
                "Sign Curvature": 0,
                "Magnitude Curvature": 1,
                "Normalized Curvature": 2
            }.get(subtype, 2)

            metric_data = current_results['curvatures'][curvature_index]
            metric_name = subtype

        elif metric_type == "Movement":
            if 'movement_data' not in current_results:
                # Try to generate movement data
                self.generate_frame_movement_data(self.current_frame)

                if 'movement_data' not in current_results:
                    return

            if subtype == "Movement Score":
                metric_data = current_results['movement_data']['scores']
                metric_name = "Movement Score"
            elif subtype == "Extension/Retraction":
                # Convert categorical to numerical
                movement_types = current_results['movement_data']['types']
                metric_data = np.zeros(len(movement_types))
                for i, mtype in enumerate(movement_types):
                    if mtype == 'extending':
                        metric_data[i] = 1
                    elif mtype == 'retracting':
                        metric_data[i] = -1
                metric_name = "Extension/Retraction"
            else:
                return
        else:
            return

        current_valid = current_results['valid_points']
        reference_intensities = reference_results['intensities']
        reference_valid = reference_results['valid_points']

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, reference_valid)

        # Filter data for valid points
        valid_metric = metric_data[valid_points]
        valid_intensities = reference_intensities[valid_points]

        # Create scatter plot
        ax.scatter(valid_metric, valid_intensities, alpha=0.7)

        # Add regression line if enough points
        if len(valid_metric) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_metric, valid_intensities)

            # Create line data
            x_line = np.linspace(min(valid_metric), max(valid_metric), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"Current Frame {metric_name} vs {reference_label} Intensity"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_metric)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Current vs {reference_label}")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Current Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_label)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            title = f"Current Frame {metric_name} vs {reference_label} Intensity\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Current vs {reference_label}")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Current Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_label)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(f"Current Frame {metric_name}")
        ax.set_ylabel(f'{reference_label} Intensity')
        ax.set_title(title)
        if len(valid_metric) > 1:
            ax.legend(loc='best')


    def update_temporal_comparison_plots(self, current_results, reference_results, reference_label):
        """
        Update plots with temporal comparison between current and reference frames
        """
        # Check if we're using "Random Each Time" mode and need a different reference frame for each plot
        random_each_time = (self.parameter_panel.parameters.get('temporal_mode') == "Current-Random Frames" and
                          self.parameter_panel.parameters.get('random_mode') == "Random Each Time")

        # Update curvature plot (showing both frames)
        if 'curvatures' in current_results and 'curvatures' in reference_results:
            if not random_each_time:
                self.update_comparative_curvature_plot(current_results, reference_results, reference_label)
            else:
                # Get a random reference frame for this plot
                new_ref_frame = self.get_random_reference_frame()
                if new_ref_frame is not None and new_ref_frame in self.loaded_data['results']:
                    new_ref_results = self.loaded_data['results'][new_ref_frame]
                    self.update_comparative_curvature_plot(current_results, new_ref_results, f"Random Frame {new_ref_frame+1}")

        # Update intensity plot (showing both frames)
        if ('intensities' in current_results and 'valid_points' in current_results and
            'intensities' in reference_results and 'valid_points' in reference_results):
            if not random_each_time:
                self.update_comparative_intensity_plot(current_results, reference_results, reference_label)
            else:
                # Get a random reference frame for this plot
                new_ref_frame = self.get_random_reference_frame()
                if new_ref_frame is not None and new_ref_frame in self.loaded_data['results']:
                    new_ref_results = self.loaded_data['results'][new_ref_frame]
                    self.update_comparative_intensity_plot(current_results, new_ref_results, f"Random Frame {new_ref_frame+1}")

        # Update correlation plot between curvature and reference intensity
        if ('curvatures' in current_results and
            'intensities' in reference_results and
            'valid_points' in current_results and
            'valid_points' in reference_results):
            if not random_each_time:
                self.update_temporal_correlation_plot(current_results, reference_results, reference_label)
            else:
                # Get a random reference frame for this plot
                new_ref_frame = self.get_random_reference_frame()
                if new_ref_frame is not None and new_ref_frame in self.loaded_data['results']:
                    new_ref_results = self.loaded_data['results'][new_ref_frame]
                    self.update_temporal_correlation_plot(current_results, new_ref_results, f"Random Frame {new_ref_frame+1}")

    # Add a helper method to get a random reference frame
    def get_random_reference_frame(self):
        """Get a random reference frame that's different from the current frame"""
        # Find all valid frame indices
        valid_frames = []
        for frame_idx in self.loaded_data['results'].keys():
            if isinstance(frame_idx, int) and frame_idx != self.current_frame:
                valid_frames.append(frame_idx)

        if not valid_frames:
            return None

        # Return a random frame
        return np.random.choice(valid_frames)



    def update_temporal_display(self):
        """Update the temporal analysis tab display"""
        results = self.loaded_data.get('results')

        if results is None:
            return

        # Get current selections
        visualization_type = self.temporal_type_combo.currentText()
        measure_type = self.temporal_measure_combo.currentText()

        # Update display based on visualization type
        if visualization_type == "Heatmap Visualization":
            self.temporal_analyzer.plot_temporal_heatmap(
                results, measure_type,
                self.temporal_plot_figure, self.temporal_plot_canvas,
                self.temporal_summary_figure, self.temporal_summary_canvas
            )

    def update_movement_display(self):
        """Update the movement analysis tab display"""
        results = self.loaded_data.get('results')

        if results is None or 'movement_types' not in results:
            return

        # Get current selections
        display_type = self.movement_display_combo.currentText()
        frame = self.movement_frame_spin.value()

        # Update display based on display type
        if display_type == "Movement Map":
            self.visualization_manager.plot_movement_map(
                results, frame,
                self.loaded_data.get('images'),
                self.loaded_data.get('masks'),
                self.movement_plot_figure,
                self.movement_plot_canvas,
                self.movement_stats_table
            )

        elif display_type == "Edge Comparison":
            self.visualization_manager.plot_edge_comparison(
                results, frame,
                self.loaded_data.get('images'),
                self.loaded_data.get('masks'),
                self.movement_plot_figure,
                self.movement_plot_canvas,
                self.movement_stats_table
            )

        elif display_type == "Movement Over Time":
            self.visualization_manager.plot_movement_over_time(
                results,
                self.movement_plot_figure,
                self.movement_plot_canvas,
                self.movement_summary_figure,
                self.movement_summary_canvas,
                self.movement_stats_table
            )


    def set_current_frame(self, frame_index):
        """Set the current frame index and update displays"""
        # Ensure valid index
        images = self.loaded_data.get('images')
        if images is not None:
            max_index = images.shape[0] - 1
            frame_index = max(0, min(frame_index, max_index))

        # Update current frame
        self.current_frame = frame_index

        # Update UI
        if images is not None:
            self.frame_label.setText(f"Frame: {frame_index+1}/{images.shape[0]}")

            # Update reference frame spinner max value
            self.parameter_panel.update_frame_range(max_index)

            # Update movement frame spinner max value (minimum is always 1)
            self.movement_frame_spin.setMaximum(max(1, max_index))

            # Set a reasonable default reference frame that's different from current
            if max_index > 0:
                ref_frame = (frame_index + max_index // 2) % (max_index + 1)
                self.parameter_panel.reference_frame_spin.setValue(ref_frame)

        # Update displays
        self.update_image_display()
        self.update_overlay_display()
        self.update_results_display()

    def next_frame(self):
        """Go to next frame"""
        images = self.loaded_data.get('images')
        if images is not None:
            max_index = images.shape[0] - 1
            new_frame = min(self.current_frame + 1, max_index)
            self.set_current_frame(new_frame)

    def previous_frame(self):
        """Go to previous frame"""
        new_frame = max(self.current_frame - 1, 0)
        self.set_current_frame(new_frame)

    def on_parameters_changed(self):
        """Handle changes to analysis parameters"""
        # Update parameter panel values in main settings if needed
        if 'n_points' in self.parameter_panel.parameters:
            self.n_points_spin.setValue(self.parameter_panel.parameters['n_points'])
        if 'depth' in self.parameter_panel.parameters:
            self.depth_spin.setValue(self.parameter_panel.parameters['depth'])
        if 'width' in self.parameter_panel.parameters:
            self.width_spin.setValue(self.parameter_panel.parameters['width'])
        if 'min_cell_coverage' in self.parameter_panel.parameters:
            self.min_coverage_spin.setValue(self.parameter_panel.parameters['min_cell_coverage'])

        # Get the temporal mode
        temporal_mode = self.parameter_panel.parameters.get('temporal_mode', "Current Frame")
        reference_frame = self.parameter_panel.parameters.get('reference_frame', 0)

        # Log parameter changes
        self.log_console.log(f"Analysis parameters updated: n_points={self.n_points_spin.value()}, "
                           f"depth={self.depth_spin.value()}, width={self.width_spin.value()}, "
                           f"min_cell_coverage={self.min_coverage_spin.value()}, "
                           f"temporal_mode={temporal_mode}, reference_frame={reference_frame}")

        # Force an update of the display
        self.update_results_display()


    def update_visualization_settings(self):
        """Update visualization settings"""
        # Update displays
        self.update_overlay_display()

        # Log settings changes
        self.log_console.log(f"Visualization settings updated: curvature_cmap={self.curvature_cmap_combo.currentText()}, "
                           f"intensity_cmap={self.intensity_cmap_combo.currentText()}, "
                           f"marker_size={self.marker_size_spin.value()}, "
                           f"contour_width={self.contour_width_spin.value()}")

    def analyze_current_frame(self):
        """Analyze only the current frame"""
        # Check if data is loaded
        if (self.loaded_data.get('images') is None or
            self.loaded_data.get('masks') is None):
            QMessageBox.warning(self, "Warning", "Please load image and mask stacks first.")
            return

        # Get parameters from UI
        try_rotation = self.parameter_panel.parameters.get('try_rotation', False)
        exclude_endpoints = self.parameter_panel.parameters.get('exclude_endpoints', False)
        angular_distribution = self.parameter_panel.parameters.get('angular_distribution', False)
        axis_based_distribution = self.parameter_panel.parameters.get('axis_based_distribution', False)
        x_axis_distribution = self.parameter_panel.parameters.get('x_axis_distribution', False)

        print(f"DEBUG: Analyzing current frame with try_rotation={try_rotation}, "
              f"exclude_endpoints={exclude_endpoints}, angular_distribution={angular_distribution}, "
              f"axis_based_distribution={axis_based_distribution}, "
              f"x_axis_distribution={x_axis_distribution}")

        params = {
            'n_points': self.n_points_spin.value(),
            'depth': self.depth_spin.value(),
            'width': self.width_spin.value(),
            'min_cell_coverage': self.min_coverage_spin.value(),
            'try_rotation': try_rotation,
            'exclude_endpoints': exclude_endpoints,
            'angular_distribution': angular_distribution,
            'axis_based_distribution': axis_based_distribution,
            'x_axis_distribution': x_axis_distribution
        }


        # Get current frame data
        frame = self.current_frame
        image = self.loaded_data['images'][frame]
        mask = self.loaded_data['masks'][frame]

        # Log
        self.log_console.log(f"Analyzing frame {frame+1} with parameters: {params}")

        # Run analysis
        try:
            frame_results = self.curvature_analyzer.analyze_frame(
                image, mask, **params)

            # Update results
            if self.loaded_data.get('results') is None:
                self.loaded_data['results'] = {}

            self.loaded_data['results'][frame] = frame_results

            # Update displays
            self.update_overlay_display()
            self.update_results_display()

            # Log
            self.log_console.log(f"Frame {frame+1} analysis completed successfully")
            self.statusBar().showMessage(f"Frame {frame+1} analysis completed")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze frame: {str(e)}")
            self.log_console.log(f"Error analyzing frame {frame+1}: {str(e)}", LogConsole.ERROR)
            self.statusBar().showMessage("Error analyzing frame")

    def export_current_frame(self):
        """Export current frame data and visualizations"""
        # Check if data is loaded
        if self.loaded_data.get('images') is None:
            QMessageBox.warning(self, "Warning", "Please load image data first.")
            return

        # Get current frame
        frame = self.current_frame

        # Check if results exist for this frame
        has_results = (self.loaded_data.get('results') is not None and
                      frame in self.loaded_data.get('results', {}))

        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", os.getcwd())

        if not output_dir:
            return

        # Create frame-specific directory
        frame_dir = os.path.join(output_dir, f"frame_{frame+1}")
        os.makedirs(frame_dir, exist_ok=True)

        try:
            # Export original image
            if self.loaded_data.get('images') is not None:
                image = self.loaded_data['images'][frame]
                plt.imsave(os.path.join(frame_dir, "original_image.png"), image, cmap='gray')

            # Export mask if available
            if self.loaded_data.get('masks') is not None:
                mask = self.loaded_data['masks'][frame]
                plt.imsave(os.path.join(frame_dir, "mask.png"), mask, cmap='gray')

            # Export overlay if both image and mask available
            if (self.loaded_data.get('images') is not None and
                self.loaded_data.get('masks') is not None):
                overlay = self.visualization_manager.create_overlay(
                    self.loaded_data['images'][frame],
                    self.loaded_data['masks'][frame])
                plt.imsave(os.path.join(frame_dir, "overlay.png"), overlay)

            # Export results if available
            if has_results:
                # Export visualizations
                frame_results = self.loaded_data['results'][frame]

                # Save curvature plot
                self.curvature_figure.savefig(
                    os.path.join(frame_dir, "curvature_profile.png"), dpi=150)

                # Save intensity plot
                self.intensity_figure.savefig(
                    os.path.join(frame_dir, "intensity_profile.png"), dpi=150)

                # Save correlation plot
                self.correlation_figure.savefig(
                    os.path.join(frame_dir, "correlation.png"), dpi=150)

                # Export data as CSV
                if 'points' in frame_results and 'curvatures' in frame_results and 'intensities' in frame_results:
                    points = frame_results['points']
                    sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']
                    intensities = frame_results['intensities']
                    valid_points = frame_results['valid_points']

                    # Create DataFrame
                    data = {
                        'point_index': np.arange(len(points)),
                        'x': points[:, 1],  # x coordinate
                        'y': points[:, 0],  # y coordinate
                        'curvature_sign': sign_curvatures,
                        'curvature_magnitude': magnitude_curvatures,
                        'curvature_normalized': normalized_curvatures,
                        'intensity': intensities,
                        'valid': valid_points
                    }

                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(frame_dir, "frame_data.csv"), index=False)

            # Export analysis parameters
            params = {
                'n_points': self.n_points_spin.value(),
                'depth': self.depth_spin.value(),
                'width': self.width_spin.value(),
                'min_cell_coverage': self.min_coverage_spin.value(),
                'frame': frame,
                'export_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(os.path.join(frame_dir, "parameters.json"), 'w') as f:
                json.dump(params, f, indent=4)

            # Show success message
            QMessageBox.information(
                self, "Export Successful",
                f"Frame {frame+1} data and visualizations exported to {frame_dir}")

            # Log
            self.log_console.log(f"Exported frame {frame+1} to {frame_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export frame: {str(e)}")
            self.log_console.log(f"Error exporting frame {frame+1}: {str(e)}", LogConsole.ERROR)

    def select_export_directory(self):
        """Select directory for exporting results"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", os.getcwd())

        if directory:
            self.export_directory_label.setText(directory)

    def preview_export(self):
        """Preview the export files"""
        # Check if data is loaded
        if self.loaded_data.get('results') is None:
            QMessageBox.warning(self, "Warning", "Please run analysis first.")
            return

        # Get export directory
        export_dir = self.export_directory_label.text()
        if export_dir == "Not selected":
            export_dir = os.path.join(os.getcwd(), "analysis_results")

        # Get prefix
        prefix = self.export_prefix_edit.text()

        # Get export options
        export_images = self.export_images_cb.isChecked()
        export_raw_data = self.export_raw_data_cb.isChecked()
        export_stats = self.export_stats_cb.isChecked()
        export_figures = self.export_figures_cb.isChecked()
        export_report = self.export_report_cb.isChecked()

        # Create preview
        preview_text = f"Export Directory: {export_dir}\n"
        preview_text += f"Filename Prefix: {prefix}\n\n"
        preview_text += "Files to be created:\n\n"

        if export_images:
            preview_text += "Images:\n"
            preview_text += f"  └─ {prefix}original_images/\n"
            preview_text += f"  └─ {prefix}masks/\n"
            preview_text += f"  └─ {prefix}overlays/\n"
            preview_text += "\n"

        if export_raw_data:
            preview_text += "Raw Data:\n"
            preview_text += f"  └─ {prefix}combined_data.csv\n"
            for frame in sorted(self.loaded_data['results'].keys()):
                if isinstance(frame, int):
                    preview_text += f"  └─ {prefix}frame_{frame+1:04d}_data.csv\n"
            preview_text += "\n"

        if export_stats:
            preview_text += "Statistics:\n"
            preview_text += f"  └─ {prefix}metadata.json\n"
            preview_text += f"  └─ {prefix}metadata.txt\n"
            preview_text += f"  └─ {prefix}correlation_stats.json\n"
            if 'movement_types' in self.loaded_data['results']:
                preview_text += f"  └─ {prefix}movement_data.csv\n"
            preview_text += "\n"

        if export_figures:
            preview_text += "Figures:\n"
            preview_text += f"  └─ {prefix}summary_correlation.png\n"
            preview_text += f"  └─ {prefix}summary_curvature.png\n"
            preview_text += f"  └─ {prefix}summary_intensity.png\n"
            if 'movement_types' in self.loaded_data['results']:
                preview_text += f"  └─ {prefix}movement_summary.png\n"
            preview_text += "\n"

        if export_report:
            preview_text += "Report:\n"
            preview_text += f"  └─ {prefix}summary_report.pdf\n"
            preview_text += "\n"

        # Calculate total size estimate
        frame_count = len([k for k in self.loaded_data['results'].keys() if isinstance(k, int)])
        size_estimate = 0

        if export_images:
            # Rough estimate of image sizes
            size_estimate += frame_count * 3 * 500  # ~500 KB per image

        if export_raw_data:
            # Rough estimate of data sizes
            size_estimate += frame_count * 50  # ~50 KB per frame data file
            size_estimate += 100  # ~100 KB for combined data

        if export_stats:
            size_estimate += 50  # ~50 KB for metadata and stats

        if export_figures:
            size_estimate += 1000  # ~1 MB for figures

        if export_report:
            size_estimate += 5000  # ~5 MB for PDF report

        # Convert to appropriate units
        if size_estimate < 1024:
            size_str = f"{size_estimate} KB"
        elif size_estimate < 1024 * 1024:
            size_str = f"{size_estimate / 1024:.1f} MB"
        else:
            size_str = f"{size_estimate / (1024 * 1024):.1f} GB"

        preview_text += f"Estimated Total Size: {size_str}"

        # Update preview text
        self.export_preview_text.setText(preview_text)

    def export_data(self):
        """Export all data and results"""
        # Check if data is loaded
        if self.loaded_data.get('results') is None:
            QMessageBox.warning(self, "Warning", "Please run analysis first.")
            return

        # Get export directory
        export_dir = self.export_directory_label.text()
        if export_dir == "Not selected":
            # Ask user to select directory
            directory = QFileDialog.getExistingDirectory(
                self, "Select Export Directory", os.getcwd())

            if not directory:
                return

            export_dir = directory
            self.export_directory_label.setText(export_dir)

        # Create directory if it doesn't exist
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        # Get prefix
        prefix = self.export_prefix_edit.text()

        # Get export options
        export_images = self.export_images_cb.isChecked()
        export_raw_data = self.export_raw_data_cb.isChecked()
        export_stats = self.export_stats_cb.isChecked()
        export_figures = self.export_figures_cb.isChecked()
        export_report = self.export_report_cb.isChecked()

        # Show progress
        self.statusBar().showMessage("Exporting data...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        try:
            # Use export manager to export data
            self.export_manager.export_data(
                self.loaded_data, export_dir, prefix,
                export_images, export_raw_data, export_stats,
                export_figures, export_report,
                self.update_export_progress
            )

            # Show success message
            QMessageBox.information(
                self, "Export Successful",
                f"Data and results exported to {export_dir}")

            # Log
            self.log_console.log(f"Exported all data to {export_dir}")

            # Hide progress
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage("Export completed")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
            self.log_console.log(f"Error exporting data: {str(e)}", LogConsole.ERROR)
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage("Export failed")

    def update_export_progress(self, percentage, message):
        """Update progress bar during export"""
        self.progress_bar.setValue(percentage)
        self.statusBar().showMessage(message)
        self.log_console.log(message)
        QApplication.processEvents()  # Keep UI responsive

    def save_log(self):
        """Save log to file"""
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", os.getcwd(), "Log Files (*.log);;Text Files (*.txt);;All Files (*.*)")

        if not file_path:
            return

        try:
            # Get log text
            log_text = self.log_console.toPlainText()

            # Save to file
            with open(file_path, 'w') as f:
                f.write(log_text)

            # Show success message
            self.statusBar().showMessage(f"Log saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save log: {str(e)}")

    def save_results(self):
        """Save analysis results to file"""
        # Check if results exist
        if self.loaded_data.get('results') is None:
            QMessageBox.warning(self, "Warning", "No analysis results to save.")
            return

        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", os.getcwd(), "JSON Files (*.json);;All Files (*.*)")

        if not file_path:
            return

        try:
            # Convert results to serializable format
            serializable_results = self.export_manager.convert_results_for_json(
                self.loaded_data['results'])

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)

            # Show success message
            self.statusBar().showMessage(f"Results saved to {file_path}")
            self.log_console.log(f"Saved results to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
            self.log_console.log(f"Error saving results: {str(e)}", LogConsole.ERROR)

    def save_settings(self):
        """Save current settings"""
        # Analysis parameters
        self.settings.setValue("n_points", self.n_points_spin.value())
        self.settings.setValue("depth", self.depth_spin.value())
        self.settings.setValue("width", self.width_spin.value())
        self.settings.setValue("min_cell_coverage", self.min_coverage_spin.value())

        # Visualization settings
        self.settings.setValue("curvature_cmap", self.curvature_cmap_combo.currentIndex())
        self.settings.setValue("intensity_cmap", self.intensity_cmap_combo.currentIndex())
        self.settings.setValue("marker_size", self.marker_size_spin.value())
        self.settings.setValue("contour_width", self.contour_width_spin.value())

        # Advanced settings
        self.settings.setValue("enable_temporal", self.enable_temporal_cb.isChecked())
        self.settings.setValue("enable_random", self.enable_random_cb.isChecked())
        self.settings.setValue("enable_movement", self.enable_movement_cb.isChecked())
        self.settings.setValue("enable_region", self.enable_region_cb.isChecked())

        # Export settings
        self.settings.setValue("export_directory", self.export_directory_label.text())
        self.settings.setValue("export_prefix", self.export_prefix_edit.text())
        self.settings.setValue("export_images", self.export_images_cb.isChecked())
        self.settings.setValue("export_raw_data", self.export_raw_data_cb.isChecked())
        self.settings.setValue("export_stats", self.export_stats_cb.isChecked())
        self.settings.setValue("export_figures", self.export_figures_cb.isChecked())
        self.settings.setValue("export_report", self.export_report_cb.isChecked())

        # Window geometry
        self.settings.setValue("window_geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())

        # Show message
        self.statusBar().showMessage("Settings saved")
        self.log_console.log("Settings saved")

    def restore_settings(self):
        """Restore saved settings"""
        # Analysis parameters
        if self.settings.contains("n_points"):
            self.n_points_spin.setValue(self.settings.value("n_points", type=int))
        if self.settings.contains("depth"):
            self.depth_spin.setValue(self.settings.value("depth", type=int))
        if self.settings.contains("width"):
            self.width_spin.setValue(self.settings.value("width", type=int))
        if self.settings.contains("min_cell_coverage"):
            self.min_coverage_spin.setValue(self.settings.value("min_cell_coverage", type=float))

        # Visualization settings
        if self.settings.contains("curvature_cmap"):
            self.curvature_cmap_combo.setCurrentIndex(self.settings.value("curvature_cmap", type=int))
        if self.settings.contains("intensity_cmap"):
            self.intensity_cmap_combo.setCurrentIndex(self.settings.value("intensity_cmap", type=int))
        if self.settings.contains("marker_size"):
            self.marker_size_spin.setValue(self.settings.value("marker_size", type=int))
        if self.settings.contains("contour_width"):
            self.contour_width_spin.setValue(self.settings.value("contour_width", type=float))

        # Advanced settings
        if self.settings.contains("enable_temporal"):
            self.enable_temporal_cb.setChecked(self.settings.value("enable_temporal", type=bool))
        if self.settings.contains("enable_random"):
            self.enable_random_cb.setChecked(self.settings.value("enable_random", type=bool))
        if self.settings.contains("enable_movement"):
            self.enable_movement_cb.setChecked(self.settings.value("enable_movement", type=bool))
        if self.settings.contains("enable_region"):
            self.enable_region_cb.setChecked(self.settings.value("enable_region", type=bool))

        # Export settings
        if self.settings.contains("export_directory"):
            directory = self.settings.value("export_directory")
            if os.path.exists(directory):
                self.export_directory_label.setText(directory)
        if self.settings.contains("export_prefix"):
            self.export_prefix_edit.setText(self.settings.value("export_prefix"))
        if self.settings.contains("export_images"):
            self.export_images_cb.setChecked(self.settings.value("export_images", type=bool))
        if self.settings.contains("export_raw_data"):
            self.export_raw_data_cb.setChecked(self.settings.value("export_raw_data", type=bool))
        if self.settings.contains("export_stats"):
            self.export_stats_cb.setChecked(self.settings.value("export_stats", type=bool))
        if self.settings.contains("export_figures"):
            self.export_figures_cb.setChecked(self.settings.value("export_figures", type=bool))
        if self.settings.contains("export_report"):
            self.export_report_cb.setChecked(self.settings.value("export_report", type=bool))

        # Window geometry
        if self.settings.contains("window_geometry"):
            self.restoreGeometry(self.settings.value("window_geometry"))
        if self.settings.contains("window_state"):
            self.restoreState(self.settings.value("window_state"))

    def reset_settings(self):
        """Reset settings to defaults"""
        # Analysis parameters
        self.n_points_spin.setValue(100)
        self.depth_spin.setValue(20)
        self.width_spin.setValue(5)
        self.min_coverage_spin.setValue(0.8)

        # Visualization settings
        self.curvature_cmap_combo.setCurrentIndex(0)
        self.intensity_cmap_combo.setCurrentIndex(0)
        self.marker_size_spin.setValue(10)
        self.contour_width_spin.setValue(1.0)

        # Advanced settings
        self.enable_temporal_cb.setChecked(True)
        self.enable_random_cb.setChecked(True)
        self.enable_movement_cb.setChecked(True)
        self.enable_region_cb.setChecked(True)

        # Export settings
        self.export_directory_label.setText("Not selected")
        self.export_prefix_edit.setText("cell_analysis_")
        self.export_images_cb.setChecked(True)
        self.export_raw_data_cb.setChecked(True)
        self.export_stats_cb.setChecked(True)
        self.export_figures_cb.setChecked(True)
        self.export_report_cb.setChecked(True)

        # Show message
        self.statusBar().showMessage("Settings reset to defaults")
        self.log_console.log("Settings reset to defaults")

    def closeEvent(self, event):
        """Handle application closing"""
        # Save settings
        self.save_settings()
        event.accept()


    def open_image_stack(self):
        """Open a microscope image stack"""
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Microscope Image Stack", "",
            "TIFF Files (*.tif *.tiff);;All Files (*.*)")

        if not file_path:
            return

        # Show loading indicator
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Loading image stack, please wait...")

        # Create a worker thread for file loading
        class LoadImageWorker(QThread):
            finished = pyqtSignal(object)
            error = pyqtSignal(str)

            def __init__(self, file_path, parent=None):
                super().__init__(parent)
                self.file_path = file_path

            def run(self):
                try:
                    # Load images with tifffile
                    images = tifffile.imread(self.file_path)

                    # Handle single image case
                    if images.ndim == 2:
                        images = images[np.newaxis, ...]

                    self.finished.emit(images)
                except Exception as e:
                    self.error.emit(str(e))

        # Create and start the worker
        self.image_loader = LoadImageWorker(file_path)
        self.image_loader.finished.connect(self.image_loading_finished)
        self.image_loader.error.connect(self.image_loading_error)
        self.image_loader.start()

    def image_loading_finished(self, images):
        """Handle successful image loading"""
        # Store loaded images
        self.loaded_data['images'] = images
        self.loaded_data['image_path'] = self.image_loader.file_path

        # Update UI
        self.image_type_combo.setCurrentIndex(0)  # Set to "Original Image"
        self.update_image_display()

        # Update frame controls
        self.current_frame = 0
        self.frame_label.setText(f"Frame: 1/{images.shape[0]}")

        # Update movement frame spinner
        self.movement_frame_spin.setMinimum(1)
        self.movement_frame_spin.setMaximum(max(1, images.shape[0] - 1))
        self.movement_frame_spin.setValue(1)

        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Loaded {images.shape[0]} frames")

        # Log
        self.log_console.log(f"Loaded image stack: {self.image_loader.file_path} ({images.shape[0]} frames)")

        # Clean up
        self.image_loader.deleteLater()
        self.image_loader = None

    def image_loading_error(self, error_message):
        """Handle image loading errors"""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Failed to load image stack")

        # Show error
        QMessageBox.critical(self, "Error", f"Failed to load image stack: {error_message}")
        self.log_console.log(f"Error loading image stack: {error_message}", self.log_console.ERROR)

        # Clean up
        self.image_loader.deleteLater()
        self.image_loader = None


    def open_mask_stack(self):
        """Open a binary mask stack"""
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Binary Mask Stack", "",
            "TIFF Files (*.tif *.tiff);;All Files (*.*)")

        if not file_path:
            return

        # Show loading indicator
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Loading mask stack, please wait...")

        # Create a worker thread for file loading
        class LoadMaskWorker(QThread):
            finished = pyqtSignal(object)
            error = pyqtSignal(str)

            def __init__(self, file_path, parent=None):
                super().__init__(parent)
                self.file_path = file_path

            def run(self):
                try:
                    # Load masks with tifffile
                    masks = tifffile.imread(self.file_path)

                    # Ensure masks are binary
                    masks = (masks > 0).astype(np.uint8)

                    # Handle single image case
                    if masks.ndim == 2:
                        masks = masks[np.newaxis, ...]

                    self.finished.emit(masks)
                except Exception as e:
                    self.error.emit(str(e))

        # Create and start the worker
        self.mask_loader = LoadMaskWorker(file_path)
        self.mask_loader.finished.connect(self.mask_loading_finished)
        self.mask_loader.error.connect(self.mask_loading_error)
        self.mask_loader.start()

    def mask_loading_finished(self, masks):
        """Handle successful mask loading"""
        # Check if masks match the dimensions of loaded images
        if 'images' in self.loaded_data and self.loaded_data['images'] is not None:
            images = self.loaded_data['images']
            if images.shape[0] != masks.shape[0]:
                QMessageBox.warning(self, "Warning",
                                    f"Mask stack contains {masks.shape[0]} frames, but image stack has {images.shape[0]} frames.")
            if images.shape[1:] != masks.shape[1:]:
                QMessageBox.warning(self, "Warning",
                                    f"Mask dimensions ({masks.shape[1:]}) do not match image dimensions ({images.shape[1:]}).")

        # Store loaded masks
        self.loaded_data['masks'] = masks
        self.loaded_data['mask_path'] = self.mask_loader.file_path

        # Update UI if set to show masks
        if self.image_type_combo.currentText() == "Binary Mask":
            self.update_image_display()

        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Loaded {masks.shape[0]} mask frames")

        # Log
        self.log_console.log(f"Loaded mask stack: {self.mask_loader.file_path} ({masks.shape[0]} frames)")

        # Clean up
        self.mask_loader.deleteLater()
        self.mask_loader = None

    def mask_loading_error(self, error_message):
        """Handle mask loading errors"""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Failed to load mask stack")

        # Show error
        QMessageBox.critical(self, "Error", f"Failed to load mask stack: {error_message}")
        self.log_console.log(f"Error loading mask stack: {error_message}", self.log_console.ERROR)

        # Clean up
        self.mask_loader.deleteLater()
        self.mask_loader = None



    def run_analysis(self):
        """Run the analysis on all frames"""
        # Check if data is loaded
        if self.loaded_data.get('images') is None or self.loaded_data.get('masks') is None:
            QMessageBox.warning(self, "Warning", "Please load image and mask stacks first.")
            return

        # Get parameters
        try_rotation = self.parameter_panel.parameters.get('try_rotation', False)
        exclude_endpoints = self.parameter_panel.parameters.get('exclude_endpoints', False)
        angular_distribution = self.parameter_panel.parameters.get('angular_distribution', False)
        axis_based_distribution = self.parameter_panel.parameters.get('axis_based_distribution', False)
        x_axis_distribution = self.parameter_panel.parameters.get('x_axis_distribution', False)

        print(f"DEBUG: Starting analysis with try_rotation={try_rotation}, "
              f"exclude_endpoints={exclude_endpoints}, angular_distribution={angular_distribution}, "
              f"axis_based_distribution={axis_based_distribution}, "
              f"x_axis_distribution={x_axis_distribution}")

        params = {
            'image_path': self.loaded_data.get('image_path', ''),
            'mask_path': self.loaded_data.get('mask_path', ''),
            'output_dir': os.path.join(os.getcwd(), "results"),
            'n_points': self.n_points_spin.value(),
            'depth': self.depth_spin.value(),
            'width': self.width_spin.value(),
            'min_cell_coverage': self.min_coverage_spin.value(),
            'try_rotation': try_rotation,
            'exclude_endpoints': exclude_endpoints,
            'angular_distribution': angular_distribution,
            'axis_based_distribution': axis_based_distribution,
            'x_axis_distribution': x_axis_distribution
        }

        # Log all parameters
        print(f"DEBUG: Analysis parameters:")
        for key, value in params.items():
            print(f"DEBUG:   {key}: {value}")

        # Create worker thread
        self.analysis_worker = AnalysisWorker(self.curvature_analyzer, params)
        self.analysis_worker.progress_signal.connect(self.update_analysis_progress)
        self.analysis_worker.complete_signal.connect(self.analysis_completed)
        self.analysis_worker.error_signal.connect(self.analysis_error)

        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Running analysis...")

        # Disable UI controls during analysis
        self.setEnabled(False)

        # Start worker thread
        self.analysis_worker.start()

    def update_analysis_progress(self, percentage, message):
        """Update progress bar during analysis"""
        self.progress_bar.setValue(percentage)
        self.statusBar().showMessage(message)
        self.log_console.log(message)
        QApplication.processEvents()  # Keep UI responsive

    def analysis_completed(self, results):
        """Handle analysis completion"""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Analysis completed")

        # Store results
        self.loaded_data['results'] = results

        # Re-enable UI
        self.setEnabled(True)

        # Update displays
        self.set_current_frame(self.current_frame)

        # Log
        self.log_console.log("Analysis completed successfully")

    def analysis_error(self, error_message):
        """Handle analysis error"""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Analysis failed")

        # Re-enable UI
        self.setEnabled(True)

        # Show error message
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.log_console.log(f"Analysis error: {error_message}", self.log_console.ERROR)


    def update_image_display(self):
        """Update the image display based on selected type and brightness settings"""
        # Check if data is loaded
        if not self.loaded_data:
            return

        # Get current selection
        image_type = self.image_type_combo.currentText()

        # Get current frame
        frame = self.current_frame

        # Get brightness settings
        min_brightness = self.min_brightness_slider.value() / 100.0
        max_brightness = self.max_brightness_slider.value() / 100.0
        mask_opacity = self.mask_opacity_slider.value() / 100.0

        # Update based on type
        if image_type == "Original Image" and 'images' in self.loaded_data and self.loaded_data['images'] is not None:
            images = self.loaded_data['images']
            if frame < images.shape[0]:
                # Copy the image but preserve original intensity values
                image = images[frame].copy().astype(float)

                # Skip normalization, but still apply brightness windowing
                # Calculate actual min/max values of the image
                actual_min = np.min(image)
                actual_max = np.max(image)

                # Calculate brightness window - scale between min and max based on slider settings
                window_min = actual_min + min_brightness * (actual_max - actual_min)
                window_max = actual_min + max_brightness * (actual_max - actual_min)

                # Apply brightness window without normalization
                image = np.clip(image, window_min, window_max)

                self.image_viewer.set_image(image)
                self.image_viewer.set_title(f"Original Image (Frame {frame+1}/{images.shape[0]})")

        elif image_type == "Binary Mask" and 'masks' in self.loaded_data and self.loaded_data['masks'] is not None:
            masks = self.loaded_data['masks']
            if frame < masks.shape[0]:
                self.image_viewer.set_image(masks[frame])
                self.image_viewer.set_title(f"Binary Mask (Frame {frame+1}/{masks.shape[0]})")

        elif image_type == "Overlay":
            if ('images' in self.loaded_data and self.loaded_data['images'] is not None and
                'masks' in self.loaded_data and self.loaded_data['masks'] is not None):

                images = self.loaded_data['images']
                masks = self.loaded_data['masks']

                if frame < images.shape[0] and frame < masks.shape[0]:
                    # Get raw image without normalization
                    image = images[frame].copy().astype(float)
                    mask = masks[frame]

                    # Calculate actual min/max values of the image
                    actual_min = np.min(image)
                    actual_max = np.max(image)

                    # Calculate brightness window
                    window_min = actual_min + min_brightness * (actual_max - actual_min)
                    window_max = actual_min + max_brightness * (actual_max - actual_min)

                    # Apply brightness window without normalization
                    image_windowed = np.clip(image, window_min, window_max)

                    # To create overlay, we need to normalize for the RGB array
                    # This is only for display purposes - raw values are preserved in the image_data property
                    image_range = window_max - window_min
                    if image_range > 0:
                        image_norm = (image_windowed - window_min) / image_range
                    else:
                        image_norm = np.zeros_like(image_windowed)

                    # Create RGB overlay
                    overlay = np.stack([image_norm, image_norm, image_norm], axis=2)

                    # Add red tint to masked areas with opacity control
                    red_mask = np.zeros_like(overlay)
                    red_mask[mask > 0, 0] = 1.0  # Red channel

                    # Blend using mask opacity
                    overlay = overlay * (1 - mask_opacity * mask[:, :, np.newaxis]) + red_mask * mask_opacity * mask[:, :, np.newaxis]

                    # Display the overlay, but store original data
                    self.image_viewer.set_image(overlay)
                    self.image_viewer.image_data = image  # Store original intensity data for mouse hover
                    self.image_viewer.set_title(f"Overlay (Frame {frame+1}/{images.shape[0]})")



    def update_overlay_display(self):
        """Update the overlay display based on selected options"""
        # Check if data is loaded
        if not self.loaded_data:
            return

        # Get current frame
        frame = self.current_frame

        # Check if we have results for this frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            frame not in self.loaded_data['results']):
            return

        # Get frame results
        frame_results = self.loaded_data['results'][frame]

        # Clear overlay canvas
        self.overlay_canvas.clear()

        # Add background image if available
        if 'images' in self.loaded_data and self.loaded_data['images'] is not None:
            images = self.loaded_data['images']
            if frame < images.shape[0]:
                # Pass the raw image data to the overlay canvas without normalization
                image = images[frame].copy().astype(float)
                self.overlay_canvas.add_background(image)

                # Store brightness settings for display purposes
                self.overlay_canvas.min_brightness = self.overlay_min_brightness_slider.value() / 100.0
                self.overlay_canvas.max_brightness = self.overlay_max_brightness_slider.value() / 100.0

        # Add mask if available
        if 'masks' in self.loaded_data and self.loaded_data['masks'] is not None:
            masks = self.loaded_data['masks']
            if frame < masks.shape[0]:
                mask = masks[frame]
                self.overlay_canvas.add_mask(mask, opacity=self.overlay_mask_opacity_slider.value() / 100.0)

        # Add contour if available and checkbox is checked
        if (self.show_contour_cb.isChecked() and
            'contour_data' in frame_results):
            self.overlay_canvas.add_contour(
                frame_results['contour_data'],
                color='r',  # Changed from 'y' to 'r' for red
                linewidth=2.0  # Increased from default value for better visibility
            )

        # Add curvature points if available and checkbox is checked
        if (self.show_curvature_cb.isChecked() and
            'points' in frame_results and
            'curvatures' in frame_results):
            # Determine which curvature to use
            if self.show_curvature_sign_cb.isChecked():
                # Use sign curvature
                curvature_index = 0
            else:
                # Use normalized curvature
                curvature_index = 2

            self.overlay_canvas.add_curvature_points(
                frame_results['points'],
                frame_results['curvatures'][curvature_index],  # Use selected curvature type
                cmap=self.curvature_cmap_combo.currentText(),
                marker_size=self.marker_size_spin.value()
            )

        # Add intensity points if available and checkbox is checked
        if (self.show_intensity_cb.isChecked() and
            'points' in frame_results and
            'intensities' in frame_results):
            self.overlay_canvas.add_intensity_points(
                frame_results['points'],
                frame_results['intensities'],
                cmap=self.intensity_cmap_combo.currentText(),
                marker_size=self.marker_size_spin.value()
            )

        # Add sampling regions if available and checkbox is checked
        if (self.show_sampling_cb.isChecked() and
            'sampling_regions' in frame_results and
            'valid_points' in frame_results):
            self.overlay_canvas.add_sampling_regions(
                frame_results['sampling_regions'],
                frame_results['valid_points'],
                frame_results.get('recovered_by_rotation'),
                frame_results.get('exclude_endpoints', False)
            )


        # Add movement visualization if checkbox is checked and data exists
        try:
            if (self.show_movement_cb.isChecked()):
                # Try to generate movement data if it doesn't exist
                if 'movement_data' not in frame_results:
                    self.generate_frame_movement_data(self.current_frame)

                # If we now have movement data, display it
                if 'movement_data' in frame_results and 'points' in frame_results:
                    movement_scores = frame_results['movement_data']['scores']
                    movement_types = frame_results['movement_data']['types']

                    # Create color-coded points based on movement types
                    colors = []
                    for mtype in movement_types:
                        if mtype == 'extending':
                            colors.append('blue')
                        elif mtype == 'retracting':
                            colors.append('red')
                        else:
                            colors.append('gray')

                    self.overlay_canvas.add_movement_visualization(
                        frame_results['points'],
                        movement_scores,
                        colors,
                        marker_size=self.marker_size_spin.value()
                    )
        except Exception as e:
            # Log error but continue
            print(f"Error updating movement overlay: {str(e)}")


        # Update canvas
        self.overlay_canvas.update_canvas()

        # Set title
        self.overlay_canvas.set_title(f"Analysis Overlay (Frame {frame+1})")


    def update_random_control_state(self):
        """Enable or disable the random control analysis option based on settings"""
        enabled = self.enable_random_cb.isChecked()

        # Get current temporal mode
        current_mode = self.parameter_panel.temporal_mode_combo.currentText()

        # Block signals to prevent unnecessary updates
        self.parameter_panel.temporal_mode_combo.blockSignals(True)

        # Clear and rebuild the combo items
        self.parameter_panel.temporal_mode_combo.clear()
        self.parameter_panel.temporal_mode_combo.addItem("Current Frame")
        self.parameter_panel.temporal_mode_combo.addItem("Current-Previous Frames")

        if enabled:
            self.parameter_panel.temporal_mode_combo.addItem("Current-Random Frames")

        # Restore current selection if possible
        index = self.parameter_panel.temporal_mode_combo.findText(current_mode)
        if index >= 0:
            self.parameter_panel.temporal_mode_combo.setCurrentIndex(index)

        # Re-enable signals
        self.parameter_panel.temporal_mode_combo.blockSignals(False)

        # Update reference frame visibility
        self.parameter_panel.update_reference_frame_visibility()



    def update_results_display(self):
        """Update the results display based on current frame and temporal mode"""
        # Check if results are available for current frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            self.current_frame not in self.loaded_data['results']):
            return

        # Get frame results for current frame
        frame_results = self.loaded_data['results'][self.current_frame]

        # Get temporal mode from parameter panel
        temporal_mode = self.parameter_panel.parameters.get('temporal_mode', "Current Frame")
        reference_frame = self.parameter_panel.parameters.get('reference_frame', 0)

        # Update frame info
        self.update_frame_info(frame_results)

        # Clear all plots first to ensure proper updates
        self.curvature_figure.clear()
        self.intensity_figure.clear()
        self.correlation_figure.clear()

        # Process based on temporal mode
        if temporal_mode == "Current Frame":
            # Standard single frame analysis
            if 'curvatures' in frame_results:
                self.update_curvature_plot(frame_results)

            if 'intensities' in frame_results and 'valid_points' in frame_results:
                self.update_intensity_plot(frame_results)

            if ('curvatures' in frame_results and
                'intensities' in frame_results and
                'valid_points' in frame_results):
                self.update_correlation_plot(frame_results)

        elif temporal_mode == "Current-Previous Frames" and self.current_frame > 0:
            # Get previous frame
            previous_frame = self.current_frame - 1

            if previous_frame in self.loaded_data['results']:
                previous_frame_results = self.loaded_data['results'][previous_frame]

                # Update plots with temporal comparison
                self.update_temporal_comparison_plots(frame_results, previous_frame_results,
                                                    "Previous Frame")

                # Log that we're using temporal comparison
                self.log_console.log(f"Using temporal comparison with previous frame {previous_frame+1}")

        elif temporal_mode == "Current-Random Frames" and reference_frame != self.current_frame:
            # Ensure reference frame exists in results
            if reference_frame in self.loaded_data['results']:
                reference_frame_results = self.loaded_data['results'][reference_frame]

                # Update plots with temporal comparison
                self.update_temporal_comparison_plots(frame_results, reference_frame_results,
                                                    f"Reference Frame {reference_frame+1}")

                # Log that we're using temporal comparison
                self.log_console.log(f"Using temporal comparison with reference frame {reference_frame+1}")
        else:
            # Fallback to standard display if temporal mode doesn't apply
            if 'curvatures' in frame_results:
                self.update_curvature_plot(frame_results)

            if 'intensities' in frame_results and 'valid_points' in frame_results:
                self.update_intensity_plot(frame_results)

            if ('curvatures' in frame_results and
                'intensities' in frame_results and
                'valid_points' in frame_results):
                self.update_correlation_plot(frame_results)

            # Log that we fell back to standard display
            self.log_console.log(f"Falling back to standard display (temporal mode {temporal_mode} not applicable)")

        # Update frame results table regardless of mode
        self.update_frame_results_table(frame_results)

        # Refresh all canvases
        self.curvature_canvas.draw()
        self.intensity_canvas.draw()
        self.correlation_canvas.draw()


    def create_log_tab(self):
        """Create the log tab with console output"""
        log_tab = QWidget()
        self.tabs.addTab(log_tab, "Log")

        layout = QVBoxLayout(log_tab)

        # Create log console
        self.log_console = LogConsole()
        layout.addWidget(self.log_console)

        # Add buttons
        button_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_console.clear)

        save_btn = QPushButton("Save Log")
        save_btn.clicked.connect(self.save_log)

        button_layout.addStretch()
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)


    def create_parameter_dock(self):
        """Create the parameter dock widget"""
        # Create dock widget
        self.parameter_dock = QDockWidget("Analysis Parameters", self)
        self.parameter_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create parameter panel
        self.parameter_panel = ParameterPanel()

        # Set widget
        self.parameter_dock.setWidget(self.parameter_panel)

        # Add dock to main window
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameter_dock)

    def create_info_dock(self):
        """Create the info dock widget"""
        # Create dock widget
        self.info_dock = QDockWidget("Frame Information", self)
        self.info_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create widget
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # Frame info
        frame_group = QGroupBox("Frame")
        frame_layout = QVBoxLayout()
        frame_group.setLayout(frame_layout)

        self.frame_info_label = QLabel("No frame loaded")
        frame_layout.addWidget(self.frame_info_label)

        info_layout.addWidget(frame_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        stats_group.setLayout(stats_layout)

        self.curvature_stats_label = QLabel("N/A")
        self.intensity_stats_label = QLabel("N/A")
        self.valid_points_label = QLabel("N/A")
        self.movement_type_label = QLabel("N/A")

        stats_layout.addRow("Curvature (mean ± std):", self.curvature_stats_label)
        stats_layout.addRow("Intensity (mean ± std):", self.intensity_stats_label)
        stats_layout.addRow("Valid Points:", self.valid_points_label)
        stats_layout.addRow("Movement Type:", self.movement_type_label)

        info_layout.addWidget(stats_group)

        # Add plot buttons panel
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)

        # Curvature buttons
        curvature_layout = QHBoxLayout()
        self.sign_curvature_btn = QPushButton("Sign Curvature")
        self.sign_curvature_btn.clicked.connect(lambda: self.create_popup_curvature_plot("sign"))

        self.magnitude_curvature_btn = QPushButton("Magnitude Curvature")
        self.magnitude_curvature_btn.clicked.connect(lambda: self.create_popup_curvature_plot("magnitude"))

        self.normalized_curvature_btn = QPushButton("Normalized Curvature")
        self.normalized_curvature_btn.clicked.connect(lambda: self.create_popup_curvature_plot("normalized"))

        curvature_layout.addWidget(self.sign_curvature_btn)
        curvature_layout.addWidget(self.magnitude_curvature_btn)
        curvature_layout.addWidget(self.normalized_curvature_btn)

        plot_layout.addLayout(curvature_layout)

        # Intensity and combined buttons
        intensity_layout = QHBoxLayout()
        self.intensity_btn = QPushButton("Intensity Profile")
        self.intensity_btn.clicked.connect(self.create_popup_intensity_plot)

        self.combined_btn = QPushButton("Combined Plot")
        self.combined_btn.clicked.connect(self.create_popup_combined_plot)

        intensity_layout.addWidget(self.intensity_btn)
        intensity_layout.addWidget(self.combined_btn)

        plot_layout.addLayout(intensity_layout)

        info_layout.addWidget(plot_group)

        # Add stretch to push everything to the top
        info_layout.addStretch()

        # Set widget
        self.info_dock.setWidget(info_widget)

        # Add dock to main window
        self.addDockWidget(Qt.RightDockWidgetArea, self.info_dock)

    def connect_signals(self):
        """Connect signals and slots"""
        # Movement frame spin
        self.movement_frame_spin.valueChanged.connect(self.update_movement_display)

        # Temporal frame spins
        #self.current_frame_spin.valueChanged.connect(self.update_temporal_display)
        #self.reference_frame_spin.valueChanged.connect(self.update_temporal_display)

        # Settings controls
        self.curvature_cmap_combo.currentIndexChanged.connect(self.update_visualization_settings)
        self.intensity_cmap_combo.currentIndexChanged.connect(self.update_visualization_settings)
        self.marker_size_spin.valueChanged.connect(self.update_visualization_settings)
        self.contour_width_spin.valueChanged.connect(self.update_visualization_settings)

    def add_table_row(self, table, parameter, value):
        """
        Add a row to a table widget (works with both ResultsTable and QTableWidget)

        Parameters:
        -----------
        table : QTableWidget or ResultsTable
            Table widget to add row to
        parameter : str
            Parameter name/label
        value : Any
            Parameter value
        """
        # Check if the table has the add_result method (ResultsTable)
        if hasattr(table, 'add_result'):
            table.add_result(parameter, value)
        else:
            # Regular QTableWidget - add row manually
            row = table.rowCount()
            table.insertRow(row)

            # Convert value to string representation
            if isinstance(value, float):
                value_str = f"{value:.6g}"
            else:
                value_str = str(value)

            # Create items
            parameter_item = QTableWidgetItem(parameter)
            value_item = QTableWidgetItem(value_str)

            # Add to table
            table.setItem(row, 0, parameter_item)
            table.setItem(row, 1, value_item)

            # Make sure the table has the right number of columns
            if table.columnCount() < 2:
                table.setColumnCount(2)
                table.setHorizontalHeaderLabels(["Parameter", "Value"])


    def create_ml_analysis_tab(self):
        """Create the machine learning analysis tab"""
        ml_tab = QWidget()
        self.tabs.addTab(ml_tab, "ML Analysis")

        layout = QVBoxLayout(ml_tab)

        # Controls
        control_panel = QGroupBox("ML Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Train button
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_ml_model)
        control_layout.addWidget(self.train_model_btn)

        # Apply to current frame button
        self.apply_ml_btn = QPushButton("Apply to Current Frame")
        self.apply_ml_btn.clicked.connect(self.apply_ml_model)
        control_layout.addWidget(self.apply_ml_btn)

        # Save/load model buttons
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_ml_model)
        control_layout.addWidget(self.save_model_btn)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_ml_model)
        control_layout.addWidget(self.load_model_btn)

        layout.addWidget(control_panel)

        # Create visualization with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - ML overlay
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        ml_overlay_group = QGroupBox("ML Classification Overlay")
        ml_overlay_layout = QVBoxLayout()
        ml_overlay_group.setLayout(ml_overlay_layout)

        self.ml_overlay_figure = Figure(figsize=(5, 5))
        self.ml_overlay_canvas = FigureCanvas(self.ml_overlay_figure)
        self.ml_overlay_toolbar = NavigationToolbar(self.ml_overlay_canvas, self)
        ml_overlay_layout.addWidget(self.ml_overlay_toolbar)
        ml_overlay_layout.addWidget(self.ml_overlay_canvas)

        left_layout.addWidget(ml_overlay_group)
        splitter.addWidget(left_widget)

        # Right panel - Feature importances and stats
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Feature importances
        feature_group = QGroupBox("Feature Importances")
        feature_layout = QVBoxLayout()
        feature_group.setLayout(feature_layout)

        self.feature_figure = Figure(figsize=(5, 4))
        self.feature_canvas = FigureCanvas(self.feature_figure)
        feature_layout.addWidget(self.feature_canvas)

        right_layout.addWidget(feature_group)

        # ML statistics
        stats_group = QGroupBox("ML Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.ml_stats_table = QTableWidget()
        self.ml_stats_table.setColumnCount(2)
        self.ml_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.ml_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.ml_stats_table)

        right_layout.addWidget(stats_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([500, 300])


    def train_ml_model(self):
        """Train the ML model on all analyzed frames"""
        if self.loaded_data.get('results') is None:
            QMessageBox.warning(self, "Warning", "Please run analysis first.")
            return

        try:
            # Prepare training data
            features, labels = self.ml_analyzer.prepare_training_data(self.loaded_data['results'])

            if len(features) == 0:
                QMessageBox.warning(self, "Warning", "No valid data points for training.")
                return

            # Train model
            accuracy = self.ml_analyzer.train_model(features, labels)

            # Update UI
            self.statusBar().showMessage(f"Model trained with accuracy: {accuracy:.2f}")
            self.log_console.log(f"Trained ML model with accuracy: {accuracy:.2f}")

            # Update feature importances visualization
            self.update_feature_importances()

            # Update ML stats table
            self.ml_stats_table.setRowCount(0)
            self.ml_stats_table.add_result("Training Accuracy", f"{accuracy:.4f}")
            self.ml_stats_table.add_result("Training Samples", len(features))
            self.ml_stats_table.add_result("Positive Class Samples", np.sum(labels))
            self.ml_stats_table.add_result("Negative Class Samples", len(labels) - np.sum(labels))

            # Apply to current frame
            self.apply_ml_model()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error training model: {str(e)}")
            self.log_console.log(f"Error training model: {str(e)}", LogConsole.ERROR)

    def apply_ml_model(self):
        """Apply the ML model to the current frame"""
        if not self.ml_analyzer.trained:
            QMessageBox.warning(self, "Warning", "Please train the model first.")
            return

        # Get current frame
        frame = self.current_frame

        # Check if results exist for this frame
        if (self.loaded_data.get('results') is None or
            frame not in self.loaded_data['results']):
            QMessageBox.warning(self, "Warning", f"No analysis results for frame {frame+1}.")
            return

        try:
            # Get frame results
            frame_results = self.loaded_data['results'][frame]

            # Classify points
            classifications, probabilities = self.ml_analyzer.classify_points(frame_results)

            # Add to frame results
            frame_results['ml_classifications'] = classifications
            frame_results['ml_probabilities'] = probabilities

            # Update visualization
            self.update_ml_overlay()

            # Update stats
            valid_classifications = classifications[classifications >= 0]
            if len(valid_classifications) > 0:
                self.ml_stats_table.add_result("High Interest Points", np.sum(valid_classifications == 1))
                self.ml_stats_table.add_result("Low Interest Points", np.sum(valid_classifications == 0))
                self.ml_stats_table.add_result("High Interest Percentage",
                                             f"{np.sum(valid_classifications == 1) / len(valid_classifications) * 100:.1f}%")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying model: {str(e)}")
            self.log_console.log(f"Error applying model: {str(e)}", LogConsole.ERROR)

    def update_ml_overlay(self):
        """Update the ML classification overlay visualization"""
        # Get current frame
        frame = self.current_frame

        # Check if ML results exist for this frame
        if (self.loaded_data.get('results') is None or
            frame not in self.loaded_data['results'] or
            'ml_classifications' not in self.loaded_data['results'][frame]):
            return

        # Get frame results
        frame_results = self.loaded_data['results'][frame]

        # Get data
        points = frame_results['points']
        valid_points = frame_results['valid_points']
        classifications = frame_results['ml_classifications']
        probabilities = frame_results['ml_probabilities']

        # Get image if available
        image = None
        if self.loaded_data.get('images') is not None:
            if frame < self.loaded_data['images'].shape[0]:
                image = self.loaded_data['images'][frame]

        # Clear figure
        self.ml_overlay_figure.clear()
        ax = self.ml_overlay_figure.add_subplot(111)

        # Display image if available
        if image is not None:
            ax.imshow(image, cmap='gray')

        # Create scatter points for classifications
        high_interest_mask = (classifications == 1)
        low_interest_mask = (classifications == 0)

        # Plot points
        if np.any(high_interest_mask):
            ax.scatter(points[high_interest_mask, 1], points[high_interest_mask, 0],
                     c='r', s=50, alpha=0.7, label='High Interest')

        if np.any(low_interest_mask):
            ax.scatter(points[low_interest_mask, 1], points[low_interest_mask, 0],
                     c='b', s=20, alpha=0.5, label='Low Interest')

        # Add title and legend
        ax.set_title(f"ML Classification (Frame {frame+1})")
        ax.legend()

        # Hide axes
        ax.axis('off')

        # Refresh canvas
        self.ml_overlay_canvas.draw()

    def update_feature_importances(self):
        """Update the feature importances visualization"""
        if not self.ml_analyzer.trained:
            return

        # Get feature importances
        importances = self.ml_analyzer.get_feature_importances()

        # Clear figure
        self.feature_figure.clear()
        ax = self.feature_figure.add_subplot(111)

        # Create bar plot
        features = list(importances.keys())
        values = list(importances.values())

        # Sort by importance
        sorted_indices = np.argsort(values)
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # Create horizontal bar plot
        bars = ax.barh(features, values, color='skyblue')

        # Add values as text
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')

        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')

        # Refresh canvas
        self.feature_canvas.draw()

    def save_ml_model(self):
        """Save the trained ML model to a file"""
        if not self.ml_analyzer.trained:
            QMessageBox.warning(self, "Warning", "Please train the model first.")
            return

        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ML Model", os.getcwd(), "Model Files (*.joblib);;All Files (*.*)")

        if not file_path:
            return

        try:
            # Save model
            self.ml_analyzer.save_model(file_path)

            # Show success message
            self.statusBar().showMessage(f"Model saved to {file_path}")
            self.log_console.log(f"Saved ML model to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
            self.log_console.log(f"Error saving model: {str(e)}", LogConsole.ERROR)

    def load_ml_model(self):
        """Load a trained ML model from a file"""
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load ML Model", os.getcwd(), "Model Files (*.joblib);;All Files (*.*)")

        if not file_path:
            return

        try:
            # Load model
            self.ml_analyzer.load_model(file_path)

            # Show success message
            self.statusBar().showMessage(f"Model loaded from {file_path}")
            self.log_console.log(f"Loaded ML model from {file_path}")

            # Update feature importances
            self.update_feature_importances()

            # Apply to current frame
            self.apply_ml_model()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.log_console.log(f"Error loading model: {str(e)}", LogConsole.ERROR)


    def create_popup_curvature_plot(self, curvature_type="normalized"):
        """
        Create a pop-up window with a curvature profile plot

        Parameters:
        -----------
        curvature_type : str
            Type of curvature to display ("sign", "magnitude", or "normalized")
        """
        # Check if data is loaded
        if not self.loaded_data:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        # Check if results are available for current frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            self.current_frame not in self.loaded_data['results']):
            QMessageBox.warning(self, "Warning", "No analysis results for current frame.")
            return

        # Get frame results
        frame_results = self.loaded_data['results'][self.current_frame]

        # Check if curvature data is available
        if 'curvatures' not in frame_results:
            QMessageBox.warning(self, "Warning", "No curvature data for current frame.")
            return

        # Get curvature data based on type
        sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

        if curvature_type == "sign":
            curvatures = sign_curvatures
            title = f"Sign Curvature Profile (Frame {self.current_frame+1})"
            y_label = "Curvature Sign"
        elif curvature_type == "magnitude":
            curvatures = magnitude_curvatures
            title = f"Magnitude Curvature Profile (Frame {self.current_frame+1})"
            y_label = "Curvature Magnitude"
        else:  # normalized
            curvatures = normalized_curvatures
            title = f"Normalized Curvature Profile (Frame {self.current_frame+1})"
            y_label = "Normalized Curvature"

        # Create pop-up window
        self.create_popup_plot(curvatures, title, "Point Index", y_label)

    def create_popup_intensity_plot(self):
        """Create a pop-up window with an intensity profile plot"""
        # Check if data is loaded
        if not self.loaded_data:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        # Check if results are available for current frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            self.current_frame not in self.loaded_data['results']):
            QMessageBox.warning(self, "Warning", "No analysis results for current frame.")
            return

        # Get frame results
        frame_results = self.loaded_data['results'][self.current_frame]

        # Check if intensity data is available
        if 'intensities' not in frame_results or 'valid_points' not in frame_results:
            QMessageBox.warning(self, "Warning", "No intensity data for current frame.")
            return

        # Get data
        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Create masked array for invalid points
        masked_intensities = np.ma.array(intensities, mask=~valid_points)

        # Create pop-up window
        self.create_popup_plot(masked_intensities,
                             f"Intensity Profile (Frame {self.current_frame+1})",
                             "Point Index", "Intensity")

    def create_popup_plot(self, data, title, x_label, y_label):
        """
        Create a pop-up window with a plot

        Parameters:
        -----------
        data : ndarray
            Data to plot
        title : str
            Plot title
        x_label : str
            X-axis label
        y_label : str
            Y-axis label
        """
        # Create a custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)

        # Create layout
        layout = QVBoxLayout(dialog)

        # Create matplotlib figure and canvas
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dialog)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Add a close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        # Add plot to figure
        ax = fig.add_subplot(111)

        # Plot data
        x = np.arange(len(data))
        ax.plot(x, data, 'b-')

        # If data is a masked array, add points
        if isinstance(data, np.ma.MaskedArray):
            valid_points = ~data.mask
            ax.scatter(x[valid_points], data[valid_points], c='b', s=30, alpha=0.7)

        # Add labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Draw canvas
        canvas.draw()

        # Show dialog
        dialog.exec_()

    def create_popup_combined_plot(self):
        """Create a pop-up window with combined intensity and curvature profiles"""
        # Check if data is loaded
        if not self.loaded_data:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        # Check if results are available for current frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            self.current_frame not in self.loaded_data['results']):
            QMessageBox.warning(self, "Warning", "No analysis results for current frame.")
            return

        # Get frame results
        frame_results = self.loaded_data['results'][self.current_frame]

        # Check if necessary data is available
        if ('curvatures' not in frame_results or
            'intensities' not in frame_results or
            'valid_points' not in frame_results):
            QMessageBox.warning(self, "Warning", "Missing data for combined plot.")
            return

        # Get data
        sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']
        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Create masked arrays
        masked_curvatures = np.ma.array(normalized_curvatures, mask=~valid_points)
        masked_intensities = np.ma.array(intensities, mask=~valid_points)

        # Create a custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Combined Profile (Frame {self.current_frame+1})")
        dialog.resize(800, 600)

        # Create layout
        layout = QVBoxLayout(dialog)

        # Create matplotlib figure and canvas
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dialog)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Add a close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        # Add plot to figure
        ax1 = fig.add_subplot(111)

        # Plot curvature
        x = np.arange(len(normalized_curvatures))
        line1, = ax1.plot(x, masked_curvatures, 'b-', label='Normalized Curvature')
        ax1.scatter(x[valid_points], normalized_curvatures[valid_points], c='b', s=30, alpha=0.7)

        # Create second y-axis for intensity
        ax2 = ax1.twinx()
        line2, = ax2.plot(x, masked_intensities, 'r-', label='Intensity')
        ax2.scatter(x[valid_points], intensities[valid_points], c='r', s=30, alpha=0.7)

        # Add labels and title
        ax1.set_xlabel('Point Index')
        ax1.set_ylabel('Normalized Curvature', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2.set_ylabel('Intensity', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add legend
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Set title
        fig.suptitle(f"Combined Curvature and Intensity Profile (Frame {self.current_frame+1})")
        plt.tight_layout()

        # Draw canvas
        canvas.draw()

        # Show dialog
        dialog.exec_()


