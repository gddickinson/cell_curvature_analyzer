"""Builds the Visualization and Analysis tabs of MainWindow."""
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox, QComboBox, QGroupBox,
                             QSplitter, QGridLayout, QSlider)
from PyQt5.QtCore import Qt
from modules.widgets import ImageViewer, OverlayCanvas, ResultsTable


class VisualizationTabsMixin:
    """MainWindow mixin: builds the Visualization and Analysis tabs of MainWindow."""

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
