"""Pop-up windows with curvature, intensity and combined profile plots."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QMessageBox, QDialog


class PopupPlotsMixin:
    """MainWindow mixin: pop-up windows with curvature, intensity and combined profile plots."""

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
