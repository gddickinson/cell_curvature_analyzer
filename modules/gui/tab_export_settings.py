"""Builds the Export and Settings tabs of MainWindow."""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QComboBox, QGroupBox, QGridLayout, QFormLayout, QLineEdit)
from modules.analysis_defaults import DEFAULTS


class ExportSettingsTabsMixin:
    """MainWindow mixin: builds the Export and Settings tabs of MainWindow."""

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
        self.n_points_spin.setValue(DEFAULTS["n_points"])
        self.n_points_spin.setSingleStep(5)
        analysis_layout.addWidget(QLabel("Number of Points:"), 0, 0)
        analysis_layout.addWidget(self.n_points_spin, 0, 1)

        # Sampling depth
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 200)
        self.depth_spin.setValue(DEFAULTS["depth"])
        analysis_layout.addWidget(QLabel("Sampling Depth:"), 1, 0)
        analysis_layout.addWidget(self.depth_spin, 1, 1)

        # Sampling width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100)
        self.width_spin.setValue(DEFAULTS["width"])
        analysis_layout.addWidget(QLabel("Sampling Width:"), 2, 0)
        analysis_layout.addWidget(self.width_spin, 2, 1)

        # Minimum cell coverage
        self.min_coverage_spin = QDoubleSpinBox()
        self.min_coverage_spin.setRange(0.1, 1.0)
        self.min_coverage_spin.setValue(DEFAULTS["min_cell_coverage"])
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
