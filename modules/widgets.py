import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
                            QTableWidgetItem, QHeaderView, QTextEdit, QGroupBox,
                            QDoubleSpinBox, QSpinBox, QFormLayout, QPushButton, QMessageBox,
                            QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QImage
import datetime

class ImageViewer(QWidget):
    """
    Widget for displaying microscope images
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Create layout
        layout = QVBoxLayout(self)

        # Create figure
        self.figure = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Create axes
        self.ax = self.figure.add_subplot(111)

        # Image title
        self.title_label = QLabel("No image loaded")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Set layout margins
        layout.setContentsMargins(0, 0, 0, 0)

    def set_image(self, image, cmap='gray'):
        """
        Set the image to display

        Parameters:
        -----------
        image : ndarray
            Image to display
        cmap : str
            Colormap to use
        """
        # Clear previous image
        self.ax.clear()

        # Plot new image
        self.ax.imshow(image, cmap=cmap)
        self.ax.axis('off')

        # Refresh canvas
        self.canvas.draw()

    def set_title(self, title):
        """
        Set the image title

        Parameters:
        -----------
        title : str
            Image title
        """
        self.title_label.setText(title)


class OverlayCanvas(QWidget):
    """
    Widget for displaying analysis overlays
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # Initialize analysis layers
        self.background_image = None
        self.contour_data = None
        self.curvature_points = None
        self.intensity_points = None
        self.sampling_regions = None

        # Initialize figure attributes
        self.contour_color = 'y'
        self.contour_width = 1.0

    def init_ui(self):
        # Create layout
        layout = QVBoxLayout(self)

        # Create figure
        self.figure = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Create axes
        self.ax = self.figure.add_subplot(111)

        # Image title
        self.title_label = QLabel("No overlay data")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Set layout margins
        layout.setContentsMargins(0, 0, 0, 0)

    def clear(self):
        """Clear all overlay layers"""
        self.background_image = None
        self.contour_data = None
        self.curvature_points = None
        self.intensity_points = None
        self.sampling_regions = None

        # Clear figure
        self.ax.clear()
        self.canvas.draw()

    def add_background(self, image):
        """
        Add background image

        Parameters:
        -----------
        image : ndarray
            Background image
        """
        self.background_image = image

    def add_contour(self, contour, color='r', linewidth=2.0):
        """
        Add a cell contour

        Parameters:
        -----------
        contour : ndarray
            Cell contour points
        color : str
            Line color
        linewidth : float
            Line width
        """
        # Make sure we have a valid contour
        if contour is None or len(contour) == 0:
            return

        # Store contour data and styling
        self.contour_data = contour
        self.contour_color = color
        self.contour_width = linewidth

        # The actual drawing will happen in update_canvas()

    def add_curvature_points(self, points, curvatures, cmap='coolwarm', marker_size=10):
        """
        Add curvature points

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        curvatures : ndarray
            Array of curvature values
        cmap : str
            Colormap to use
        marker_size : int
            Size of markers
        """
        self.curvature_points = {
            'points': points,
            'curvatures': curvatures,
            'cmap': cmap,
            'marker_size': marker_size
        }

    def add_intensity_points(self, points, intensities, cmap='viridis', marker_size=10):
        """
        Add intensity points

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        intensities : ndarray
            Array of intensity values
        cmap : str
            Colormap to use
        marker_size : int
            Size of markers
        """
        self.intensity_points = {
            'points': points,
            'intensities': intensities,
            'cmap': cmap,
            'marker_size': marker_size
        }

    def add_sampling_regions(self, sampling_regions, valid_points=None):
        """
        Add sampling regions

        Parameters:
        -----------
        sampling_regions : list
            List of sampling region polygons
        valid_points : ndarray
            Boolean array indicating which points are valid
        """
        self.sampling_regions = {
            'regions': sampling_regions,
            'valid_points': valid_points
        }

    def update_canvas(self):
        """Update the canvas with all layers"""
        # Clear previous content
        self.ax.clear()

        # Add background image if available
        if self.background_image is not None:
            self.ax.imshow(self.background_image, cmap='gray')

            # If we have a mask, create an overlay
            if hasattr(self, 'mask') and self.mask is not None:
                # Create red mask overlay
                h, w = self.mask.shape
                red_mask = np.zeros((h, w, 4))  # RGBA
                red_mask[self.mask > 0, 0] = 1.0  # Red channel
                red_mask[self.mask > 0, 3] = self.mask_opacity  # Alpha channel

                self.ax.imshow(red_mask)

        # Add contour if available - MAKE SURE THIS WORKS
        if hasattr(self, 'contour_data') and self.contour_data is not None:
            # Debug print to ensure contour data exists
            print(f"Drawing contour with {len(self.contour_data)} points, color={self.contour_color}, width={self.contour_width}")

            # Make sure the contour is drawn on top with higher zorder
            self.ax.plot(self.contour_data[:, 1], self.contour_data[:, 0],
                        color=self.contour_color, linewidth=self.contour_width, zorder=10)



        # Add sampling regions if available
        if self.sampling_regions is not None:
            regions = self.sampling_regions['regions']
            valid_points = self.sampling_regions.get('valid_points')

            for i, region in enumerate(regions):
                # Convert to x, y coordinates for plotting
                polygon_y = region[:, 1]
                polygon_x = region[:, 0]

                if valid_points is not None:
                    if i < len(valid_points) and valid_points[i]:
                        self.ax.fill(polygon_y, polygon_x, alpha=0.3, color='cyan')
                    else:
                        self.ax.fill(polygon_y, polygon_x, alpha=0.15, color='gray')
                else:
                    self.ax.fill(polygon_y, polygon_x, alpha=0.3, color='cyan')

        # Add curvature points if available
        if self.curvature_points is not None:
            points = self.curvature_points['points']
            curvatures = self.curvature_points['curvatures']
            cmap = self.curvature_points['cmap']
            marker_size = self.curvature_points['marker_size']

            self.ax.scatter(points[:, 1], points[:, 0],
                          c=curvatures, cmap=cmap, vmin=-1, vmax=1,
                          s=marker_size, edgecolor='k')

        # Add intensity points if available
        if self.intensity_points is not None:
            points = self.intensity_points['points']
            intensities = self.intensity_points['intensities']
            cmap = self.intensity_points['cmap']
            marker_size = self.intensity_points['marker_size']

            # Calculate min and max for better color scaling
            # Using percentiles to avoid extreme outliers affecting the scale
            vmin = np.nanpercentile(intensities, 5)
            vmax = np.nanpercentile(intensities, 95)

            self.ax.scatter(points[:, 1], points[:, 0],
                          c=intensities, cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          s=marker_size, edgecolor='k')

        # Turn off axes
        self.ax.axis('off')

        # Refresh canvas
        self.canvas.draw()

    def set_title(self, title):
        """
        Set the overlay title

        Parameters:
        -----------
        title : str
            Overlay title
        """
        self.title_label.setText(title)

    def add_mask(self, mask, opacity=0.5):
        """
        Add a mask overlay with opacity control

        Parameters:
        -----------
        mask : ndarray
            Binary mask
        opacity : float
            Opacity of the mask (0-1)
        """
        # Store the mask and opacity
        self.mask = mask
        self.mask_opacity = opacity

        # The actual drawing will happen in update_canvas()

class ResultsTable(QTableWidget):
    """
    Widget for displaying analysis results in a table
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Set table properties
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def add_result(self, parameter, value):
        """
        Add a result to the table

        Parameters:
        -----------
        parameter : str
            Parameter name
        value : Any
            Parameter value
        """
        # Convert value to string representation
        if isinstance(value, float):
            value_str = f"{value:.6g}"
        else:
            value_str = str(value)

        # Add row
        row = self.rowCount()
        self.insertRow(row)

        # Add items
        parameter_item = QTableWidgetItem(parameter)
        value_item = QTableWidgetItem(value_str)

        # Add to table
        self.setItem(row, 0, parameter_item)
        self.setItem(row, 1, value_item)

        # Set background color based on parameter type
        if "error" in parameter.lower():
            parameter_item.setBackground(QColor(255, 200, 200))
            value_item.setBackground(QColor(255, 200, 200))
        elif "correlation" in parameter.lower() or "r²" in parameter.lower().replace("r2", "r²"):
            parameter_item.setBackground(QColor(200, 255, 200))
            value_item.setBackground(QColor(200, 255, 200))


class LogConsole(QTextEdit):
    """
    Widget for displaying log messages
    """

    # Log levels
    INFO = 0
    WARNING = 1
    ERROR = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Set properties
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)

        # Set font
        font = self.font()
        font.setFamily("Courier")
        self.setFont(font)

        # Add timestamp
        self.log("Log started", self.INFO)

    def log(self, message, level=INFO):
        """
        Add a log message

        Parameters:
        -----------
        message : str
            Log message
        level : int
            Log level (INFO, WARNING, ERROR)
        """
        # Get timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format message based on level
        if level == self.INFO:
            formatted_message = f"[{timestamp}] INFO: {message}"
            color = "black"
        elif level == self.WARNING:
            formatted_message = f"[{timestamp}] WARNING: {message}"
            color = "orange"
        elif level == self.ERROR:
            formatted_message = f"[{timestamp}] ERROR: {message}"
            color = "red"
        else:
            formatted_message = f"[{timestamp}] {message}"
            color = "black"

        # Add formatted message
        self.append(f'<span style="color: {color};">{formatted_message}</span>')

        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class ParameterPanel(QWidget):
    """
    Widget for displaying and editing analysis parameters
    """

    # Signal emitted when parameters are changed
    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current parameters - define this BEFORE calling init_ui
        self.parameters = {
            'n_points': 100,
            'depth': 20,
            'width': 5,
            'min_cell_coverage': 0.8
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
        self.n_points_spin.setValue(100)  # Default value
        self.n_points_spin.setSingleStep(5)
        self.n_points_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Number of Points:", self.n_points_spin)

        # Sampling depth
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 200)
        self.depth_spin.setValue(20)  # Default value
        self.depth_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Sampling Depth:", self.depth_spin)

        # Sampling width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100)
        self.width_spin.setValue(5)  # Default value
        self.width_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Sampling Width:", self.width_spin)

        # Minimum cell coverage
        self.coverage_spin = QDoubleSpinBox()
        self.coverage_spin.setRange(0.1, 1.0)
        self.coverage_spin.setValue(0.8)  # Default value
        self.coverage_spin.setSingleStep(0.05)
        self.coverage_spin.setDecimals(2)
        self.coverage_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Min. Cell Coverage:", self.coverage_spin)

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
            'temporal_mode': self.temporal_mode_combo.currentText(),
            'reference_frame': self.reference_frame_spin.value(),
            'random_mode': self.random_mode_combo.currentText()
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

        # Update reference frame visibility
        self.update_reference_frame_visibility()

        # Emit signal
        self.parameters_changed.emit(self.parameters)

    def reset_parameters(self):
        """Reset parameters to defaults"""
        # Set default values
        self.n_points_spin.setValue(100)
        self.depth_spin.setValue(20)
        self.width_spin.setValue(5)
        self.coverage_spin.setValue(0.8)
        self.temporal_mode_combo.setCurrentIndex(0)  # Current Frame

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

    def update_frame_range(self, max_frame):
        """
        Update the maximum value for the reference frame spinner

        Parameters:
        -----------
        max_frame : int
            Maximum frame index
        """
        self.reference_frame_spin.setMaximum(max_frame)
