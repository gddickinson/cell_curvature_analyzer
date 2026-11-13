"""Analysis parameter panel (initial values from analysis_defaults.DEFAULTS)."""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QFormLayout, QPushButton, QComboBox,
                             QCheckBox)
from PyQt5.QtCore import pyqtSignal
from ..analysis_defaults import DEFAULTS


class ParameterPanel(QWidget):
    """
    Widget for displaying and editing analysis parameters
    """

    # Signal emitted when parameters are changed
    parameters_changed = pyqtSignal(dict)

    SPACING_LABELS = {"equal": "Equal arc length", "legacy": "Legacy (pre-2026-11)"}

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current parameters - define this BEFORE calling init_ui
        self.parameters = {
            'n_points': DEFAULTS['n_points'],
            'depth': DEFAULTS['depth'],
            'width': DEFAULTS['width'],
            'min_cell_coverage': DEFAULTS['min_cell_coverage']
        }

        # Now call init_ui
        self.init_ui()


    def init_ui(self):
        # Create layout
        layout = QVBoxLayout(self)

        # Create form layout for parameters
        form_layout = QFormLayout()

        # Number of points
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(10, 500)
        self.n_points_spin.setValue(DEFAULTS["n_points"])  # Default value
        self.n_points_spin.setSingleStep(5)
        self.n_points_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Number of Points:", self.n_points_spin)

        # Sampling depth
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 200)
        self.depth_spin.setValue(DEFAULTS["depth"])  # Default value
        self.depth_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Sampling Depth:", self.depth_spin)

        # Sampling width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100)
        self.width_spin.setValue(DEFAULTS["width"])  # Default value
        self.width_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Sampling Width:", self.width_spin)

        # Minimum cell coverage
        self.coverage_spin = QDoubleSpinBox()
        self.coverage_spin.setRange(0.1, 1.0)
        self.coverage_spin.setValue(DEFAULTS["min_cell_coverage"])  # Default value
        self.coverage_spin.setSingleStep(0.05)
        self.coverage_spin.setDecimals(2)
        self.coverage_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Min. Cell Coverage:", self.coverage_spin)

        # Try to rotate sample region
        self.try_rotation_cb = QCheckBox("Try rotating rejected sampling regions")
        self.try_rotation_cb.setChecked(DEFAULTS["try_rotation"])  # Default to off
        self.try_rotation_cb.toggled.connect(self.on_parameter_changed)
        form_layout.addRow("", self.try_rotation_cb)

        # Exclude endpoints
        self.exclude_endpoints_cb = QCheckBox("Exclude first and last edge points")
        self.exclude_endpoints_cb.setChecked(DEFAULTS["exclude_endpoints"])  # Default to off
        self.exclude_endpoints_cb.toggled.connect(self.on_parameter_changed)
        form_layout.addRow("", self.exclude_endpoints_cb)

        # Angular distribution
        self.angular_distribution_cb = QCheckBox("Angular distribution of sample points")
        self.angular_distribution_cb.setChecked(False)  # Default to off
        self.angular_distribution_cb.toggled.connect(self.on_parameter_changed)
        self.angular_distribution_cb.setToolTip("Distribute points evenly by angle rather than by contour distance")
        form_layout.addRow("", self.angular_distribution_cb)

        # Axis-based distribution
        self.axis_based_distribution_cb = QCheckBox("Axis-based distribution (reduce bunching)")
        self.axis_based_distribution_cb.setChecked(False)  # Default to off
        self.axis_based_distribution_cb.toggled.connect(self.on_parameter_changed)
        self.axis_based_distribution_cb.setToolTip("Distribute points evenly along image axes to reduce bunching in extrusions")
        form_layout.addRow("", self.axis_based_distribution_cb)

        # X-axis based distribution
        self.x_axis_distribution_cb = QCheckBox("X-axis distribution (horizontal spacing)")
        self.x_axis_distribution_cb.setChecked(False)  # Default to off
        self.x_axis_distribution_cb.toggled.connect(self.on_parameter_changed)
        self.x_axis_distribution_cb.setToolTip("Distribute points evenly along x-axis to reduce bunching")
        form_layout.addRow("", self.x_axis_distribution_cb)

        # Point spacing along the edge
        self.point_spacing_combo = QComboBox()
        self.point_spacing_combo.addItems(list(self.SPACING_LABELS.values()))
        self.point_spacing_combo.setCurrentText(self.SPACING_LABELS[DEFAULTS["point_spacing"]])
        self.point_spacing_combo.setToolTip("Equal arc-length steps, or the pre-2026-11 sampler (reproduces old results)")
        self.point_spacing_combo.currentIndexChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Point Spacing:", self.point_spacing_combo)

        # Curvature smoothing (Gaussian sigma along the edge, px); 0 = none (legacy three-point curvature)
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 50.0)
        self.smoothing_spin.setDecimals(1)
        self.smoothing_spin.setSingleStep(0.5)
        self.smoothing_spin.setSpecialValueText("none (legacy)")
        self.smoothing_spin.setSuffix(" px")
        self.smoothing_spin.setValue(DEFAULTS["smoothing_px"])
        self.smoothing_spin.setToolTip("Gaussian smoothing of the edge before curvature (sigma along the edge). "
                                       "0 = none: legacy three-point curvature")
        self.smoothing_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Curvature Smoothing:", self.smoothing_spin)

        # Dead-band: |curvature| below this is straight (sign 0)
        self.straight_threshold_spin = QDoubleSpinBox()
        self.straight_threshold_spin.setRange(0.0, 1.0)
        self.straight_threshold_spin.setDecimals(4)
        self.straight_threshold_spin.setSingleStep(0.001)
        self.straight_threshold_spin.setSuffix(" 1/px")
        self.straight_threshold_spin.setValue(DEFAULTS["straight_threshold"])
        self.straight_threshold_spin.setToolTip("Points with |curvature| below this are straight (sign 0). "
                                                "0 with smoothing 'none' = legacy (straight counted as -1)")
        self.straight_threshold_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Straight Below:", self.straight_threshold_spin)

        # Add temporal analysis mode selector
        self.temporal_mode_combo = QComboBox()
        self.temporal_mode_combo.addItems(["Current Frame", "Current-Previous Frames", "Current-Random Frames"])
        self.temporal_mode_combo.currentIndexChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Temporal Mode:", self.temporal_mode_combo)

        # Reference frame selector (for random mode)
        self.reference_frame_spin = QSpinBox()
        self.reference_frame_spin.setRange(0, 0)  # Will be updated when data is loaded
        self.reference_frame_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Reference Frame:", self.reference_frame_spin)

        # Add random mode selector
        self.random_mode_combo = QComboBox()
        self.random_mode_combo.addItems(["Fixed Reference", "Random Each Time"])
        self.random_mode_combo.currentIndexChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Random Mode:", self.random_mode_combo)

        # Add form layout to main layout
        layout.addLayout(form_layout)

        # Add reset button
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_parameters)
        layout.addWidget(self.reset_btn)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Update parameters dictionary with current values
        self.parameters = {
            'n_points': self.n_points_spin.value(),
            'depth': self.depth_spin.value(),
            'width': self.width_spin.value(),
            'min_cell_coverage': self.coverage_spin.value(),
            'try_rotation': self.try_rotation_cb.isChecked(),
            'exclude_endpoints': self.exclude_endpoints_cb.isChecked(),
            'temporal_mode': self.temporal_mode_combo.currentText(),
            'reference_frame': self.reference_frame_spin.value(),
            'random_mode': self.random_mode_combo.currentText(),
            'angular_distribution': self.angular_distribution_cb.isChecked(),
            'axis_based_distribution': self.axis_based_distribution_cb.isChecked(),
            'x_axis_distribution': self.x_axis_distribution_cb.isChecked(),
            'point_spacing': self.point_spacing(),
            'smoothing_px': self.smoothing_spin.value(),
            'straight_threshold': self.straight_threshold_spin.value(),
        }


        # Initially hide the reference frame spinner (only visible in random mode)
        self.update_reference_frame_visibility()

    def update_reference_frame_visibility(self):
        """Show/hide reference frame selector and random mode based on temporal mode"""
        temporal_mode = self.temporal_mode_combo.currentText()
        is_random_mode = temporal_mode == "Current-Random Frames"

        # Show/hide reference frame spinner and random mode selector
        self.reference_frame_spin.setVisible(is_random_mode)
        self.random_mode_combo.setVisible(is_random_mode)

        # Update label next to reference_frame_spin
        if is_random_mode and self.random_mode_combo.currentText() == "Random Each Time":
            # Find the label widget for the reference frame and update its text
            for i in range(self.layout().count()):
                item = self.layout().itemAt(i)
                if isinstance(item, QFormLayout):
                    for row in range(item.rowCount()):
                        label_item = item.itemAt(row, QFormLayout.LabelRole)
                        if label_item and isinstance(label_item.widget(), QLabel):
                            if label_item.widget().text() == "Reference Frame:":
                                label_item.widget().setText("Initial Reference:")
                                break

    def on_parameter_changed(self):
        """Handle parameter changes"""
        # Update parameters
        self.parameters['n_points'] = self.n_points_spin.value()
        self.parameters['depth'] = self.depth_spin.value()
        self.parameters['width'] = self.width_spin.value()
        self.parameters['min_cell_coverage'] = self.coverage_spin.value()
        self.parameters['temporal_mode'] = self.temporal_mode_combo.currentText()
        self.parameters['reference_frame'] = self.reference_frame_spin.value()
        self.parameters['exclude_endpoints'] = self.exclude_endpoints_cb.isChecked()
        self.parameters['x_axis_distribution'] = self.x_axis_distribution_cb.isChecked()
        self.parameters['point_spacing'] = self.point_spacing()
        self.parameters['smoothing_px'] = self.smoothing_spin.value()
        self.parameters['straight_threshold'] = self.straight_threshold_spin.value()

        # Make distribution options mutually exclusive
        if self.sender() == self.x_axis_distribution_cb and self.x_axis_distribution_cb.isChecked():
            # Turn off other distribution options
            self.angular_distribution_cb.blockSignals(True)
            self.angular_distribution_cb.setChecked(False)
            self.angular_distribution_cb.blockSignals(False)

            self.axis_based_distribution_cb.blockSignals(True)
            self.axis_based_distribution_cb.setChecked(False)
            self.axis_based_distribution_cb.blockSignals(False)

            print("DEBUG: X-axis distribution selected, turning off other distribution methods")

        elif self.sender() == self.angular_distribution_cb and self.angular_distribution_cb.isChecked():
            # Turn off other distribution options
            self.x_axis_distribution_cb.blockSignals(True)
            self.x_axis_distribution_cb.setChecked(False)
            self.x_axis_distribution_cb.blockSignals(False)

            self.axis_based_distribution_cb.blockSignals(True)
            self.axis_based_distribution_cb.setChecked(False)
            self.axis_based_distribution_cb.blockSignals(False)

            print("DEBUG: Angular distribution selected, turning off other distribution methods")

        elif self.sender() == self.axis_based_distribution_cb and self.axis_based_distribution_cb.isChecked():
            # Turn off other distribution options
            self.x_axis_distribution_cb.blockSignals(True)
            self.x_axis_distribution_cb.setChecked(False)
            self.x_axis_distribution_cb.blockSignals(False)

            self.angular_distribution_cb.blockSignals(True)
            self.angular_distribution_cb.setChecked(False)
            self.angular_distribution_cb.blockSignals(False)

            print("DEBUG: Axis-based distribution selected, turning off other distribution methods")

        # Update the parameters
        self.parameters['x_axis_distribution'] = self.x_axis_distribution_cb.isChecked()
        self.parameters['angular_distribution'] = self.angular_distribution_cb.isChecked()
        self.parameters['axis_based_distribution'] = self.axis_based_distribution_cb.isChecked()


        # Update try_rotation parameter
        try_rotation_value = self.try_rotation_cb.isChecked()
        self.parameters['try_rotation'] = try_rotation_value
        print(f"DEBUG: Parameter try_rotation changed to {try_rotation_value}")

        # Update reference frame visibility
        self.update_reference_frame_visibility()

        # Emit signal
        self.parameters_changed.emit(self.parameters)

    def reset_parameters(self):
        """Reset parameters to defaults"""
        # Set default values
        self.n_points_spin.setValue(DEFAULTS["n_points"])
        self.depth_spin.setValue(DEFAULTS["depth"])
        self.width_spin.setValue(DEFAULTS["width"])
        self.coverage_spin.setValue(DEFAULTS["min_cell_coverage"])
        self.temporal_mode_combo.setCurrentIndex(0)  # Current Frame
        self.try_rotation_cb.setChecked(DEFAULTS["try_rotation"])  # Reset to default (off)
        self.exclude_endpoints_cb.setChecked(DEFAULTS["exclude_endpoints"])  # Reset to default (off)
        self.angular_distribution_cb.setChecked(False)  # Reset to default (off)
        self.axis_based_distribution_cb.setChecked(False)  # Reset to default (off)
        self.x_axis_distribution_cb.setChecked(False)  # Reset to default (off)
        self.point_spacing_combo.setCurrentText(self.SPACING_LABELS[DEFAULTS["point_spacing"]])
        self.smoothing_spin.setValue(DEFAULTS["smoothing_px"])
        self.straight_threshold_spin.setValue(DEFAULTS["straight_threshold"])
        # Don't reset reference frame as it depends on data

    def update_parameters(self, parameters):
        """
        Update parameters from dict

        Parameters:
        -----------
        parameters : dict
            Dictionary of parameter values
        """
        # Update UI without triggering signals
        self.n_points_spin.blockSignals(True)
        self.depth_spin.blockSignals(True)
        self.width_spin.blockSignals(True)
        self.coverage_spin.blockSignals(True)
        self.temporal_mode_combo.blockSignals(True)
        self.reference_frame_spin.blockSignals(True)

        # Set values
        if 'n_points' in parameters:
            self.n_points_spin.setValue(parameters['n_points'])
        if 'depth' in parameters:
            self.depth_spin.setValue(parameters['depth'])
        if 'width' in parameters:
            self.width_spin.setValue(parameters['width'])
        if 'min_cell_coverage' in parameters:
            self.coverage_spin.setValue(parameters['min_cell_coverage'])
        if 'temporal_mode' in parameters:
            index = self.temporal_mode_combo.findText(parameters['temporal_mode'])
            if index >= 0:
                self.temporal_mode_combo.setCurrentIndex(index)
        if 'reference_frame' in parameters:
            self.reference_frame_spin.setValue(parameters['reference_frame'])
        if 'try_rotation' in parameters:
            self.try_rotation_cb.setChecked(parameters['try_rotation'])
        if 'exclude_endpoints' in parameters:
            self.exclude_endpoints_cb.setChecked(parameters['exclude_endpoints'])
        if 'angular_distribution' in parameters:
            self.angular_distribution_cb.setChecked(parameters['angular_distribution'])
        if 'axis_based_distribution' in parameters:
            self.axis_based_distribution_cb.setChecked(parameters['axis_based_distribution'])
        if 'x_axis_distribution' in parameters:
            self.x_axis_distribution_cb.setChecked(parameters['x_axis_distribution'])
        if 'point_spacing' in parameters:
            self.point_spacing_combo.setCurrentText(self.SPACING_LABELS[parameters['point_spacing']])
        if 'smoothing_px' in parameters:
            self.smoothing_spin.setValue(parameters['smoothing_px'])
        if 'straight_threshold' in parameters:
            self.straight_threshold_spin.setValue(parameters['straight_threshold'])


        # Re-enable signals
        self.n_points_spin.blockSignals(False)
        self.depth_spin.blockSignals(False)
        self.width_spin.blockSignals(False)
        self.coverage_spin.blockSignals(False)
        self.temporal_mode_combo.blockSignals(False)
        self.reference_frame_spin.blockSignals(False)

        # Update parameters dict
        self.parameters.update(parameters)

        # Update reference frame visibility
        self.update_reference_frame_visibility()

    def point_spacing(self):
        """'equal' or 'legacy' from the point-spacing combo box."""
        labels = {v: k for k, v in self.SPACING_LABELS.items()}
        return labels[self.point_spacing_combo.currentText()]

    def update_frame_range(self, max_frame):
        """
        Update the maximum value for the reference frame spinner

        Parameters:
        -----------
        max_frame : int
            Maximum frame index
        """
        self.reference_frame_spin.setMaximum(max_frame)
