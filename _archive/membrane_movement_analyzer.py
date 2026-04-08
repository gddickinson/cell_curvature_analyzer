#!/usr/bin/env python3
"""
Membrane Movement Analyzer

A GUI application for analyzing cell membrane movement using the binary mask difference method.
Features visualization of microscope images with overlays, correlation analysis,
and adjustable analysis parameters.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from skimage import measure, draw, morphology
from scipy import stats
import tifffile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QSplitter, QLabel, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QComboBox, QGroupBox, QFormLayout, QTableWidget,
                           QTableWidgetItem, QHeaderView, QProgressBar, QStatusBar, QDockWidget,
                           QTabWidget, QMessageBox, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QColor


class ImageViewer(QWidget):
    """Widget for displaying microscope images with overlays"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create axis for image display
        self.ax = self.figure.add_subplot(111)
        
        # Initialize data
        self.image = None
        self.mask = None
        self.points = None
        self.valid_points = None
        self.movement_scores = None
        self.colormap = plt.cm.get_cmap('coolwarm')
        
        # Add info label
        self.info_label = QLabel("No image loaded")
        layout.addWidget(self.info_label)
        
        # Configure layout
        layout.setContentsMargins(0, 0, 0, 0)
    
    def set_image(self, image, mask=None):
        """Set the current image and mask"""
        self.image = image
        self.mask = mask
        self.points = None
        self.valid_points = None
        self.movement_scores = None
        self.update_display()
    
    def set_movement_data(self, points, valid_points, movement_scores):
        """Set movement analysis data"""
        self.points = points
        self.valid_points = valid_points
        self.movement_scores = movement_scores
        self.update_display()
    
    def update_display(self):
        """Update the image display with current data"""
        # Clear the axis
        self.ax.clear()
        
        if self.image is None:
            self.info_label.setText("No image loaded")
            self.canvas.draw()
            return
        
        # Create overlay
        overlay = self.create_overlay()
        
        # Display the overlay
        self.ax.imshow(overlay)
        
        # Add points if available
        if (self.points is not None and 
            self.valid_points is not None and 
            self.movement_scores is not None):
            
            # Plot points colored by movement score
            for i, point in enumerate(self.points):
                if self.valid_points[i]:
                    # Map score to color range (-1 to 1) -> (0 to 1)
                    color = self.colormap((self.movement_scores[i] + 1) / 2)
                    
                    # Scale point size based on movement magnitude
                    size = 20 + 60 * abs(self.movement_scores[i])
                    
                    self.ax.scatter(point[1], point[0], color=color, s=size, alpha=0.8, 
                                  edgecolors='black', linewidths=0.5)
        
        # Turn off axis
        self.ax.axis('off')
        
        # Update info label
        if self.mask is not None:
            self.info_label.setText(f"Image shape: {self.image.shape}, Mask sum: {np.sum(self.mask)}")
        else:
            self.info_label.setText(f"Image shape: {self.image.shape}")
        
        # Refresh canvas
        self.canvas.draw()
    
    def create_overlay(self):
        """Create an overlay with the image and mask"""
        # Normalize image to 0-1 range
        image_norm = self.image.astype(float)
        image_norm = (image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm) + 1e-10)
        
        # Create an RGB image with grayscale original
        overlay = np.stack([image_norm, image_norm, image_norm], axis=2)
        
        # Add red tint to masked areas if mask is available
        if self.mask is not None:
            overlay[self.mask > 0, 0] = np.maximum(overlay[self.mask > 0, 0], 0.7)  # Red channel
            overlay[self.mask > 0, 1] *= 0.7  # Reduce green
            overlay[self.mask > 0, 2] *= 0.7  # Reduce blue
        
        return overlay


class PlotViewer(QWidget):
    """Widget for displaying correlation plots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create axis for plot display
        self.ax = self.figure.add_subplot(111)
        
        # Add statistics label
        self.stats_label = QLabel("No data available")
        layout.addWidget(self.stats_label)
        
        # Configure layout
        layout.setContentsMargins(0, 0, 0, 0)
    
    def plot_correlation(self, intensities, movement_scores, valid_points):
        """Create a correlation plot of movement vs intensity"""
        # Clear the axis
        self.ax.clear()
        
        # Check if we have valid data
        if not np.any(valid_points):
            self.stats_label.setText("No valid data points")
            self.canvas.draw()
            return {}
        
        # Filter for valid points
        valid_intensities = intensities[valid_points]
        valid_movement = movement_scores[valid_points]
        
        # Create scatter plot
        scatter = self.ax.scatter(valid_movement, valid_intensities, alpha=0.7, 
                                c=valid_movement, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = self.figure.colorbar(scatter, ax=self.ax)
        cbar.set_label('Movement Score (Blue=Extending, Red=Retracting)')
        
        # Add regression line if enough points
        if len(valid_movement) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_movement, valid_intensities)
            
            # Create line data
            x_line = np.linspace(min(valid_movement), max(valid_movement), 100)
            y_line = slope * x_line + intercept
            
            # Plot line
            self.ax.plot(x_line, y_line, 'k--', linewidth=2,
                       label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')
            
            # Add legend
            self.ax.legend()
            
            # Set stats label
            self.stats_label.setText(
                f"R² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_movement)}, "
                f"slope = {slope:.3f}, intercept = {intercept:.3f}")
            
            # Return statistics
            stats_dict = {
                'r_squared': r_value**2,
                'p_value': p_value,
                'slope': slope,
                'intercept': intercept,
                'sample_size': len(valid_movement)
            }
        else:
            self.stats_label.setText(f"Insufficient data for regression (n = {len(valid_movement)})")
            stats_dict = {
                'r_squared': None,
                'p_value': None,
                'slope': None,
                'intercept': None,
                'sample_size': len(valid_movement) if len(valid_movement) > 0 else 0
            }
        
        # Set labels and title
        self.ax.set_xlabel('Movement Score')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title('Movement-Intensity Correlation')
        
        # Refresh canvas
        self.canvas.draw()
        
        return stats_dict


class SummaryViewer(QWidget):
    """Widget for displaying summary plots of all frames"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create tab widget for different summary views
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create correlation plot tab
        self.correlation_widget = QWidget()
        correlation_layout = QVBoxLayout(self.correlation_widget)
        
        # Create matplotlib figure and canvas for correlation
        self.corr_figure = Figure(figsize=(6, 5))
        self.corr_canvas = FigureCanvas(self.corr_figure)
        self.corr_toolbar = NavigationToolbar(self.corr_canvas, self)
        
        correlation_layout.addWidget(self.corr_toolbar)
        correlation_layout.addWidget(self.corr_canvas)
        
        self.tabs.addTab(self.correlation_widget, "All Frames Correlation")
        
        # Create statistics tab
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(["Frame", "R²", "p-value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.tabs.addTab(self.stats_table, "Statistics")
        
        # Create R² comparison plot tab
        self.r2_widget = QWidget()
        r2_layout = QVBoxLayout(self.r2_widget)
        
        # Create matplotlib figure and canvas for R² comparison
        self.r2_figure = Figure(figsize=(6, 5))
        self.r2_canvas = FigureCanvas(self.r2_figure)
        self.r2_toolbar = NavigationToolbar(self.r2_canvas, self)
        
        r2_layout.addWidget(self.r2_toolbar)
        r2_layout.addWidget(self.r2_canvas)
        
        self.tabs.addTab(self.r2_widget, "R² Comparison")
        
        # Configure layout
        layout.setContentsMargins(0, 0, 0, 0)
    
    def update_all_frames_correlation(self, all_intensities, all_movement_scores, all_valid_points):
        """Create a correlation plot combining data from all frames"""
        # Clear the figure
        self.corr_figure.clear()
        ax = self.corr_figure.add_subplot(111)
        
        # Combine data from all frames
        combined_intensities = []
        combined_movement = []
        
        for intensities, movement_scores, valid_points in zip(all_intensities, all_movement_scores, all_valid_points):
            # Filter for valid points
            if np.any(valid_points):
                combined_intensities.extend(intensities[valid_points])
                combined_movement.extend(movement_scores[valid_points])
        
        if not combined_intensities:
            ax.set_title("No valid data points across frames")
            self.corr_canvas.draw()
            return
        
        # Convert to numpy arrays
        combined_intensities = np.array(combined_intensities)
        combined_movement = np.array(combined_movement)
        
        # Create scatter plot
        scatter = ax.scatter(combined_movement, combined_intensities, alpha=0.5, 
                           c=combined_movement, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = self.corr_figure.colorbar(scatter, ax=ax)
        cbar.set_label('Movement Score (Blue=Extending, Red=Retracting)')
        
        # Add regression line
        if len(combined_movement) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_movement, combined_intensities)
            
            # Create line data
            x_line = np.linspace(min(combined_movement), max(combined_movement), 100)
            y_line = slope * x_line + intercept
            
            # Plot line
            ax.plot(x_line, y_line, 'k--', linewidth=2,
                   label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')
            
            # Add legend
            ax.legend()
            
            # Add stats to title
            ax.set_title(f"All Frames Movement-Intensity Correlation\n"
                        f"R² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_movement)}")
        else:
            ax.set_title("All Frames Movement-Intensity Correlation\nInsufficient data for regression")
        
        # Set labels
        ax.set_xlabel('Movement Score')
        ax.set_ylabel('Intensity')
        
        # Refresh canvas
        self.corr_canvas.draw()
    
    def update_statistics_table(self, frame_stats):
        """Update the statistics table with per-frame correlation statistics"""
        # Clear the table
        self.stats_table.setRowCount(0)
        
        # Add rows for each frame
        for i, stats in enumerate(frame_stats):
            if stats:
                row = self.stats_table.rowCount()
                self.stats_table.insertRow(row)
                
                # Add frame number
                frame_item = QTableWidgetItem(f"{i+1}")
                frame_item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(row, 0, frame_item)
                
                # Add R² value
                r2 = stats.get('r_squared')
                if r2 is not None:
                    r2_item = QTableWidgetItem(f"{r2:.4f}")
                    
                    # Color based on significance
                    p_value = stats.get('p_value', 1.0)
                    if p_value < 0.05:
                        r2_item.setBackground(QColor(200, 255, 200))  # Light green for significant
                else:
                    r2_item = QTableWidgetItem("N/A")
                
                r2_item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(row, 1, r2_item)
                
                # Add p-value
                p_value = stats.get('p_value')
                if p_value is not None:
                    p_item = QTableWidgetItem(f"{p_value:.4f}")
                    
                    # Color based on significance
                    if p_value < 0.001:
                        p_item.setBackground(QColor(150, 255, 150))  # Darker green for highly significant
                    elif p_value < 0.01:
                        p_item.setBackground(QColor(200, 255, 200))  # Medium green
                    elif p_value < 0.05:
                        p_item.setBackground(QColor(230, 255, 230))  # Light green
                else:
                    p_item = QTableWidgetItem("N/A")
                
                p_item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(row, 2, p_item)
    
    def update_r2_comparison(self, frame_stats):
        """Create a bar plot comparing R² values across frames"""
        # Clear the figure
        self.r2_figure.clear()
        ax = self.r2_figure.add_subplot(111)
        
        # Extract R² values
        frames = []
        r2_values = []
        p_values = []
        
        for i, stats in enumerate(frame_stats):
            if stats and stats.get('r_squared') is not None:
                frames.append(i+1)
                r2_values.append(stats['r_squared'])
                p_values.append(stats.get('p_value', 1.0))
        
        if not frames:
            ax.set_title("No valid R² values available")
            self.r2_canvas.draw()
            return
        
        # Create bar colors based on p-values
        colors = []
        for p in p_values:
            if p < 0.001:
                colors.append('darkgreen')  # Highly significant
            elif p < 0.01:
                colors.append('green')  # Very significant
            elif p < 0.05:
                colors.append('lightgreen')  # Significant
            else:
                colors.append('gray')  # Not significant
        
        # Create bars
        bars = ax.bar(frames, r2_values, color=colors, alpha=0.7)
        
        # Add value labels above bars
        for i, (bar, value) in enumerate(zip(bars, r2_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add significance indicators
        for i, p in enumerate(p_values):
            if p < 0.001:
                ax.text(frames[i], r2_values[i] + 0.04, '***', ha='center')
            elif p < 0.01:
                ax.text(frames[i], r2_values[i] + 0.04, '**', ha='center')
            elif p < 0.05:
                ax.text(frames[i], r2_values[i] + 0.04, '*', ha='center')
        
        # Add legend
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(color='darkgreen', alpha=0.7, label='p < 0.001 (***)'),
            mpatches.Patch(color='green', alpha=0.7, label='p < 0.01 (**)'),
            mpatches.Patch(color='lightgreen', alpha=0.7, label='p < 0.05 (*)'),
            mpatches.Patch(color='gray', alpha=0.7, label='Not significant')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Set labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel('R² (Coefficient of Determination)')
        ax.set_title('Movement-Intensity Correlation by Frame')
        
        # Set y-axis limit slightly above the maximum value
        max_val = max(r2_values, default=0.1)
        ax.set_ylim(0, max_val * 1.1 + 0.05)
        
        # Set integer ticks for x-axis
        ax.set_xticks(frames)
        
        # Refresh canvas
        self.r2_canvas.draw()


class ParameterPanel(QWidget):
    """Widget for adjusting analysis parameters"""
    
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout for parameters
        form_layout = QFormLayout()
        
        # Number of points
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(10, 500)
        self.n_points_spin.setValue(100)
        self.n_points_spin.setSingleStep(5)
        self.n_points_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Number of Points:", self.n_points_spin)
        
        # Sampling depth
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 100)
        self.depth_spin.setValue(20)
        self.depth_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Sampling Depth:", self.depth_spin)
        
        # Sampling width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 50)
        self.width_spin.setValue(5)
        self.width_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Sampling Width:", self.width_spin)
        
        # Minimum cell coverage
        self.coverage_spin = QDoubleSpinBox()
        self.coverage_spin.setRange(0.1, 1.0)
        self.coverage_spin.setValue(0.8)
        self.coverage_spin.setSingleStep(0.05)
        self.coverage_spin.setDecimals(2)
        self.coverage_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Min. Cell Coverage:", self.coverage_spin)
        
        # Local radius for movement calculation
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 20)
        self.radius_spin.setValue(5)
        self.radius_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Movement Analysis Radius:", self.radius_spin)
        
        # Movement threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.5)
        self.threshold_spin.setValue(0.1)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("Movement Threshold:", self.threshold_spin)
        
        # Try rotation
        self.try_rotation_cb = QCheckBox("Try rotating rejected sampling regions")
        self.try_rotation_cb.setChecked(True)
        self.try_rotation_cb.toggled.connect(self.on_parameter_changed)
        form_layout.addRow("", self.try_rotation_cb)
        
        # Exclude endpoints
        self.exclude_endpoints_cb = QCheckBox("Exclude first and last contour points")
        self.exclude_endpoints_cb.setChecked(False)
        self.exclude_endpoints_cb.toggled.connect(self.on_parameter_changed)
        form_layout.addRow("", self.exclude_endpoints_cb)
        
        # Add layout to main layout
        layout.addLayout(form_layout)
        
        # Add a reset button
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_parameters)
        layout.addWidget(self.reset_btn)
        
        # Add stretch to push controls to the top
        layout.addStretch()
        
        # Store current parameters
        self.parameters = {
            'n_points': self.n_points_spin.value(),
            'depth': self.depth_spin.value(),
            'width': self.width_spin.value(),
            'min_cell_coverage': self.coverage_spin.value(),
            'movement_radius': self.radius_spin.value(),
            'movement_threshold': self.threshold_spin.value(),
            'try_rotation': self.try_rotation_cb.isChecked(),
            'exclude_endpoints': self.exclude_endpoints_cb.isChecked()
        }
    
    def on_parameter_changed(self):
        """Handle parameter changes"""
        # Update parameters dictionary
        self.parameters['n_points'] = self.n_points_spin.value()
        self.parameters['depth'] = self.depth_spin.value()
        self.parameters['width'] = self.width_spin.value()
        self.parameters['min_cell_coverage'] = self.coverage_spin.value()
        self.parameters['movement_radius'] = self.radius_spin.value()
        self.parameters['movement_threshold'] = self.threshold_spin.value()
        self.parameters['try_rotation'] = self.try_rotation_cb.isChecked()
        self.parameters['exclude_endpoints'] = self.exclude_endpoints_cb.isChecked()
        
        # Emit signal
        self.parameters_changed.emit()
    
    def reset_parameters(self):
        """Reset parameters to defaults"""
        self.n_points_spin.setValue(100)
        self.depth_spin.setValue(20)
        self.width_spin.setValue(5)
        self.coverage_spin.setValue(0.8)
        self.radius_spin.setValue(5)
        self.threshold_spin.setValue(0.1)
        self.try_rotation_cb.setChecked(True)
        self.exclude_endpoints_cb.setChecked(False)
    
    def get_parameters(self):
        """Get current parameters"""
        return self.parameters


class AnalysisWorker(QThread):
    """Worker thread for running analysis without freezing the UI"""
    
    progress_signal = pyqtSignal(int, str)
    complete_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, analysis_function, params):
        super().__init__()
        self.analysis_function = analysis_function
        self.params = params
    
    def run(self):
        try:
            results = self.analysis_function(self.params, self.progress_callback)
            self.complete_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def progress_callback(self, percentage, message):
        self.progress_signal.emit(percentage, message)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Membrane Movement Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data
        self.images = None
        self.masks = None
        self.current_frame = 0
        self.analysis_results = None
        self.frame_stats = []
        
        # Set up UI
        self.init_ui()
        
        # Restore settings
        self.settings = QSettings("MembraneMovementAnalyzer", "App")
        self.restore_settings()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create central splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create parameter dock widget
        self.create_parameter_dock()
        
        # Create left panel with image viewer
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_viewer = ImageViewer()
        left_layout.addWidget(self.image_viewer)
        
        splitter.addWidget(left_panel)
        
        # Create right panel with tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        
        # Create correlation plot tab
        self.plot_viewer = PlotViewer()
        self.tabs.addTab(self.plot_viewer, "Correlation")
        
        # Create summary tab
        self.summary_viewer = SummaryViewer()
        self.tabs.addTab(self.summary_viewer, "All Frames")
        
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([600, 600])
        
        # Create status bar with progress
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
        # Set initial status
        self.statusBar.showMessage("Ready")
    
    def create_toolbar(self):
        """Create the main toolbar"""
        # Create toolbar
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)
        
        # Open images action
        open_images_action = toolbar.addAction("Open Images")
        open_images_action.triggered.connect(self.open_images)
        
        # Open masks action
        open_masks_action = toolbar.addAction("Open Masks")
        open_masks_action.triggered.connect(self.open_masks)
        
        toolbar.addSeparator()
        
        # Run analysis action
        analyze_action = toolbar.addAction("Analyze Current Frame")
        analyze_action.triggered.connect(self.analyze_current_frame)
        
        # Run all frames analysis action
        analyze_all_action = toolbar.addAction("Analyze All Frames")
        analyze_all_action.triggered.connect(self.analyze_all_frames)
        
        toolbar.addSeparator()
        
        # Add frame navigation
        self.frame_label = QLabel("Frame: 0/0")
        toolbar.addWidget(self.frame_label)
        
        self.prev_frame_btn = toolbar.addAction("Previous")
        self.prev_frame_btn.triggered.connect(self.previous_frame)
        
        self.next_frame_btn = toolbar.addAction("Next")
        self.next_frame_btn.triggered.connect(self.next_frame)
    
    def create_parameter_dock(self):
        """Create parameter dock widget"""
        # Create dock widget
        self.parameter_dock = QDockWidget("Analysis Parameters", self)
        self.parameter_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create parameter panel
        self.parameter_panel = ParameterPanel()
        self.parameter_panel.parameters_changed.connect(self.on_parameters_changed)
        
        # Set widget
        self.parameter_dock.setWidget(self.parameter_panel)
        
        # Add dock to main window
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameter_dock)
    
    def open_images(self):
        """Open a microscope image stack"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image Stack", "", "TIFF Files (*.tif *.tiff);;All Files (*.*)")
        
        if not file_path:
            return
        
        try:
            # Load image stack
            images = self.load_tiff_stack(file_path)
            
            # Store images
            self.images = images
            self.current_frame = 0
            
            # Update UI
            self.update_frame_display()
            
            # Update status
            self.statusBar.showMessage(f"Loaded {images.shape[0]} frames from {os.path.basename(file_path)}")
            
            # Store file path in settings
            self.settings.setValue("last_image_path", file_path)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image stack: {str(e)}")
    
    def open_masks(self):
        """Open a binary mask stack"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Stack", "", "TIFF Files (*.tif *.tiff);;All Files (*.*)")
        
        if not file_path:
            return
        
        try:
            # Load mask stack
            masks = self.load_tiff_stack(file_path)
            
            # Ensure binary
            masks = (masks > 0).astype(np.uint8)
            
            # Check if masks match image dimensions
            if self.images is not None:
                if masks.shape[0] != self.images.shape[0]:
                    QMessageBox.warning(self, "Warning", 
                                     f"Mask stack has {masks.shape[0]} frames but image stack has {self.images.shape[0]} frames.")
                if masks.shape[1:] != self.images.shape[1:]:
                    QMessageBox.warning(self, "Warning",
                                     f"Mask dimensions ({masks.shape[1:]}) do not match image dimensions ({self.images.shape[1:]}).")
            
            # Store masks
            self.masks = masks
            
            # Update UI
            self.update_frame_display()
            
            # Update status
            self.statusBar.showMessage(f"Loaded {masks.shape[0]} mask frames from {os.path.basename(file_path)}")
            
            # Store file path in settings
            self.settings.setValue("last_mask_path", file_path)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mask stack: {str(e)}")
    
    def load_tiff_stack(self, file_path):
        """Load a TIFF stack and handle single images"""
        images = tifffile.imread(file_path)
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        return images
    
    def update_frame_display(self):
        """Update the display for the current frame"""
        if self.images is None:
            return
        
        # Get current frame data
        frame = self.current_frame
        image = self.images[frame]
        
        # Get mask if available
        mask = None
        if self.masks is not None and frame < self.masks.shape[0]:
            mask = self.masks[frame]
        
        # Update image viewer
        self.image_viewer.set_image(image, mask)
        
        # Update frame label
        self.frame_label.setText(f"Frame: {frame+1}/{self.images.shape[0]}")
        
        # Update movement data if available
        if self.analysis_results is not None and frame in self.analysis_results:
            result = self.analysis_results[frame]
            
            self.image_viewer.set_movement_data(
                result['points'],
                result['valid_points'],
                result['movement_scores']
            )
            
            # Update correlation plot
            if frame > 0:  # Need previous frame for movement
                self.plot_viewer.plot_correlation(
                    result['intensities'],
                    result['movement_scores'],
                    result['valid_points']
                )
    
    def next_frame(self):
        """Go to next frame"""
        if self.images is not None and self.current_frame < self.images.shape[0] - 1:
            self.current_frame += 1
            self.update_frame_display()
    
    def previous_frame(self):
        """Go to previous frame"""
        if self.images is not None and self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame_display()
    
    def on_parameters_changed(self):
        """Handle parameter changes"""
        # Clear existing results
        self.analysis_results = None
        self.frame_stats = []
        
        # Update display
        self.update_frame_display()
        
        # Update status
        self.statusBar.showMessage("Parameters updated. Run analysis to see changes.")
    
    def analyze_current_frame(self):
        """Analyze the current frame"""
        if self.images is None or self.masks is None:
            QMessageBox.warning(self, "Warning", "Please load both images and masks first.")
            return
        
        if self.current_frame == 0:
            QMessageBox.warning(self, "Warning", "Cannot analyze first frame - no previous frame for movement calculation.")
            return
        
        # Get parameters
        params = self.parameter_panel.get_parameters()
        
        # Add frame information
        params['current_frame'] = self.current_frame
        params['previous_frame'] = self.current_frame - 1
        params['images'] = self.images
        params['masks'] = self.masks
        params['single_frame'] = True
        
        # Show progress
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar.showMessage("Analyzing frame...")
        
        # Create and start worker thread
        self.worker = AnalysisWorker(self.run_analysis, params)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.complete_signal.connect(self.analysis_completed)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.start()
    
    def analyze_all_frames(self):
        """Analyze all frames"""
        if self.images is None or self.masks is None:
            QMessageBox.warning(self, "Warning", "Please load both images and masks first.")
            return
        
        # Get parameters
        params = self.parameter_panel.get_parameters()
        
        # Add frame information
        params['images'] = self.images
        params['masks'] = self.masks
        params['single_frame'] = False
        
        # Show progress
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar.showMessage("Analyzing all frames...")
        
        # Create and start worker thread
        self.worker = AnalysisWorker(self.run_analysis, params)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.complete_signal.connect(self.analysis_completed)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.start()
    
    def run_analysis(self, params, progress_callback):
        """Run the analysis (called from worker thread)"""
        # Extract parameters
        images = params['images']
        masks = params['masks']
        n_points = params['n_points']
        depth = params['depth']
        width = params['width']
        min_cell_coverage = params['min_cell_coverage']
        movement_radius = params['movement_radius']
        movement_threshold = params['movement_threshold']
        try_rotation = params['try_rotation']
        exclude_endpoints = params['exclude_endpoints']
        single_frame = params['single_frame']
        
        # Initialize results
        results = {}
        frame_stats = []
        
        # Determine frames to analyze
        if single_frame:
            current_frame = params['current_frame']
            previous_frame = params['previous_frame']
            frames_to_analyze = [(current_frame, previous_frame)]
        else:
            # Skip first frame since it has no previous frame for movement calculation
            frames_to_analyze = [(i, i-1) for i in range(1, images.shape[0])]
        
        # Process each frame
        for i, (current_idx, previous_idx) in enumerate(frames_to_analyze):
            # Update progress
            progress = int(i / len(frames_to_analyze) * 100)
            progress_callback(progress, f"Analyzing frame {current_idx+1}...")
            
            # Get frame data
            current_image = images[current_idx]
            current_mask = masks[current_idx]
            previous_mask = masks[previous_idx]
            
            # Detect cell edge
            contour = self.detect_cell_edge(current_mask)
            if len(contour) == 0:
                continue
            
            # Sample points along contour
            points = self.sample_equidistant_points(contour, n_points)
            
            # Calculate inward normals
            normals = self.calculate_inward_normal(points, current_mask)
            
            # Measure intensity
            intensities, sampling_regions, valid_points, recovered = self.measure_intensity(
                current_image, current_mask, points, normals, 
                depth, width, min_cell_coverage, try_rotation, exclude_endpoints
            )
            
            # Calculate movement from masks
            movement_scores, movement_types = self.calculate_movement_from_masks(
                current_mask, previous_mask, points, valid_points, 
                movement_radius, movement_threshold
            )
            
            # Store results
            frame_result = {
                'points': points,
                'intensities': intensities,
                'valid_points': valid_points,
                'movement_scores': movement_scores,
                'movement_types': movement_types,
                'sampling_regions': sampling_regions
            }
            
            results[current_idx] = frame_result
            
            # Calculate correlation statistics
            if np.any(valid_points):
                # Plot correlation (within worker thread, just to get statistics)
                valid_intensities = intensities[valid_points]
                valid_movement = movement_scores[valid_points]
                
                if len(valid_intensities) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        valid_movement, valid_intensities)
                    
                    stats_dict = {
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'slope': slope,
                        'intercept': intercept,
                        'sample_size': len(valid_intensities)
                    }
                else:
                    stats_dict = None
                
                frame_stats.append(stats_dict)
            else:
                frame_stats.append(None)
        
        # Combine results and statistics
        combined_results = {
            'results': results,
            'frame_stats': frame_stats
        }
        
        # Final progress update
        progress_callback(100, "Analysis complete")
        
        return combined_results
    
    def update_progress(self, percentage, message):
        """Update progress bar during analysis"""
        self.progress_bar.setValue(percentage)
        self.statusBar.showMessage(message)
        QApplication.processEvents()  # Keep UI responsive
    
    def analysis_completed(self, results):
        """Handle analysis completion"""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Store results
        self.analysis_results = results['results']
        self.frame_stats = results['frame_stats']
        
        # Update display
        self.update_frame_display()
        
        # Update summary displays
        self.update_summary_displays()
        
        # Update status
        self.statusBar.showMessage("Analysis complete")
    
    def analysis_error(self, error_message):
        """Handle analysis error"""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Show error message
        QMessageBox.critical(self, "Analysis Error", error_message)
        
        # Update status
        self.statusBar.showMessage("Analysis failed")
    
    def update_summary_displays(self):
        """Update summary displays with all frame data"""
        if not self.analysis_results:
            return
        
        # Collect data from all frames
        all_intensities = []
        all_movement_scores = []
        all_valid_points = []
        
        for frame_idx, result in self.analysis_results.items():
            all_intensities.append(result['intensities'])
            all_movement_scores.append(result['movement_scores'])
            all_valid_points.append(result['valid_points'])
        
        # Update summary viewer
        self.summary_viewer.update_all_frames_correlation(
            all_intensities, all_movement_scores, all_valid_points)
        
        self.summary_viewer.update_statistics_table(self.frame_stats)
        
        self.summary_viewer.update_r2_comparison(self.frame_stats)
    
    def detect_cell_edge(self, mask):
        """Detect the cell edge from a binary mask"""
        contours = measure.find_contours(mask, 0.5)
        if len(contours) > 0:
            contour = max(contours, key=len)
            return contour
        else:
            return np.array([])
    
    def sample_equidistant_points(self, contour, n_points=100):
        """Sample equidistant points along contour"""
        # Calculate total contour length
        contour_length = 0
        for i in range(len(contour) - 1):
            contour_length += np.linalg.norm(contour[i+1] - contour[i])
        
        # Distance between equidistant points
        step_size = contour_length / n_points
        
        # Sample equidistant points
        sampled_points = []
        current_length = 0
        
        i = 0
        sampled_points.append(contour[0])
        
        while len(sampled_points) < n_points and i < len(contour) - 1:
            # Distance to next point on contour
            dist = np.linalg.norm(contour[i+1] - contour[i])
            
            if current_length + dist >= step_size:
                # Interpolate to find next equidistant point
                t = (step_size - current_length) / dist
                next_point = contour[i] + t * (contour[i+1] - contour[i])
                sampled_points.append(next_point)
                current_length = 0
                # Don't increment i, stay on same segment
            else:
                # Move to next segment
                current_length += dist
                i += 1
        
        # If we couldn't sample enough points, return what we have
        return np.array(sampled_points)
    
    def calculate_inward_normal(self, points, mask):
        """Calculate the inward normal vector at each point"""
        n_points = len(points)
        normals = np.zeros((n_points, 2))
        mask_shape = mask.shape
        
        for i in range(n_points):
            # Get adjacent points (with wrapping for closed contours)
            left_idx = (i - 1) % n_points
            right_idx = (i + 1) % n_points
            
            p = points[i]
            p_left = points[left_idx]
            p_right = points[right_idx]
            
            # Calculate tangent vector
            tangent = p_right - p_left
            tangent_norm = np.linalg.norm(tangent)
            
            # Check if tangent has non-zero length
            if tangent_norm > 1e-10:  # Only normalize if length is non-zero
                tangent = tangent / tangent_norm
            else:
                # If tangent is too small, try using more distant points
                left_idx2 = (i - 2) % n_points
                right_idx2 = (i + 2) % n_points
                p_left2 = points[left_idx2]
                p_right2 = points[right_idx2]
                tangent = p_right2 - p_left2
                tangent_norm = np.linalg.norm(tangent)
                
                if tangent_norm > 1e-10:
                    tangent = tangent / tangent_norm
                else:
                    # If still too small, use a default tangent along x-axis
                    tangent = np.array([1.0, 0.0])
            
            # Calculate normal vectors (both possible directions)
            normal1 = np.array([-tangent[1], tangent[0]])
            normal2 = -normal1
            
            # Check which normal points into the cell
            test_point1 = p + 3 * normal1
            test_point2 = p + 3 * normal2
            
            # Ensure test points are within image bounds
            test_point1 = np.clip(test_point1, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])
            test_point2 = np.clip(test_point2, [0, 0], [mask_shape[0]-1, mask_shape[1]-1])
            
            # Convert to integers
            tp1_y, tp1_x = int(round(test_point1[0])), int(round(test_point1[1]))
            tp2_y, tp2_x = int(round(test_point2[0])), int(round(test_point2[1]))
            
            try:
                # Check mask values
                val1 = mask[tp1_y, tp1_x]
                val2 = mask[tp2_y, tp2_x]
                
                # Choose the normal that points into the cell (value 1 in mask)
                if val1 == 1:
                    normals[i] = normal1
                elif val2 == 1:
                    normals[i] = normal2
                else:
                    # If neither point is inside the cell, use a default
                    normals[i] = normal2
            except IndexError:
                # Use a reasonable default if out of bounds
                normals[i] = normal1
        
        return normals
    
    def measure_intensity(self, image, mask, points, normals, depth=20, width=5, 
                        min_cell_coverage=0.8, try_rotation=False, exclude_endpoints=False):
        """Measure mean intensity within rectangular regions extending from each point"""
        n_points = len(points)
        intensities = np.full(n_points, np.nan)  # Initialize with NaN
        sampling_regions = []
        valid_points = np.zeros(n_points, dtype=bool)
        recovered_by_rotation = np.zeros(n_points, dtype=bool)
        
        image_shape = image.shape
        
        for i in range(n_points):
            # Check if this is an endpoint that should be excluded
            if exclude_endpoints and (i == 0 or i == n_points - 1):
                # Add empty region for visualization consistency
                sampling_regions.append(np.zeros((4, 2)))
                continue
            
            p = points[i]
            normal = normals[i]
            
            # Calculate end point of the vector
            end_point = p + depth * normal
            
            # Calculate perpendicular direction (for width)
            perp = np.array([-normal[1], normal[0]])
            
            # Calculate corner points of the rectangle
            corner1 = p + width/2 * perp
            corner2 = p - width/2 * perp
            corner3 = end_point - width/2 * perp
            corner4 = end_point + width/2 * perp
            
            # Create a polygon for the rectangle
            vertices = np.array([corner1, corner4, corner3, corner2])
            
            # Get all pixels within the polygon
            rr, cc = draw.polygon(vertices[:, 0], vertices[:, 1], image_shape)
            
            # Check if any pixels are within bounds
            valid_pixels = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
            if np.any(valid_pixels):
                rr = rr[valid_pixels]
                cc = cc[valid_pixels]
                
                # Check cell coverage
                total_pixels = len(rr)
                cell_pixels = np.sum(mask[rr, cc])
                cell_coverage = cell_pixels / total_pixels
                
                # Only include points with sufficient cell coverage
                if cell_coverage >= min_cell_coverage:
                    # Calculate mean intensity
                    intensities[i] = np.mean(image[rr, cc])
                    valid_points[i] = True
                
                # If coverage is insufficient and rotation is enabled, try rotating the region
                elif try_rotation:
                    # Rotate normal vector 180 degrees
                    rotated_normal = -normal
                    
                    # Recalculate end point with rotated normal
                    rotated_end_point = p + depth * rotated_normal
                    
                    # Calculate corner points of the rotated rectangle
                    rotated_corner1 = p + width/2 * perp
                    rotated_corner2 = p - width/2 * perp
                    rotated_corner3 = rotated_end_point - width/2 * perp
                    rotated_corner4 = rotated_end_point + width/2 * perp
                    
                    # Create a polygon for the rotated rectangle
                    rotated_vertices = np.array([rotated_corner1, rotated_corner4, rotated_corner3, rotated_corner2])
                    
                    # Get all pixels within the rotated polygon
                    rotated_rr, rotated_cc = draw.polygon(rotated_vertices[:, 0], rotated_vertices[:, 1], image_shape)
                    
                    # Check if any pixels are within bounds
                    rotated_valid_pixels = (rotated_rr >= 0) & (rotated_rr < image_shape[0]) & (rotated_cc >= 0) & (rotated_cc < image_shape[1])
                    if np.any(rotated_valid_pixels):
                        rotated_rr = rotated_rr[rotated_valid_pixels]
                        rotated_cc = rotated_cc[rotated_valid_pixels]
                        
                        # Check cell coverage of rotated region
                        rotated_total_pixels = len(rotated_rr)
                        rotated_cell_pixels = np.sum(mask[rotated_rr, rotated_cc])
                        rotated_cell_coverage = rotated_cell_pixels / rotated_total_pixels
                        
                        # If rotated region has sufficient coverage, use it
                        if rotated_cell_coverage >= min_cell_coverage:
                            intensities[i] = np.mean(image[rotated_rr, rotated_cc])
                            valid_points[i] = True
                            recovered_by_rotation[i] = True
                            
                            # Use the rotated vertices for visualization
                            vertices = rotated_vertices
            
            # Save sampling region for visualization (even if not valid)
            sampling_regions.append(vertices)
        
        return intensities, sampling_regions, valid_points, recovered_by_rotation
    
    def calculate_movement_from_masks(self, current_mask, previous_mask, points, valid_points, 
                                    radius=5, threshold=0.1):
        """Calculate movement from binary mask difference"""
        n_points = len(points)
        movement_scores = np.zeros(n_points)
        movement_types = np.full(n_points, 'stable', dtype=object)
        
        # Calculate difference mask
        diff_mask = current_mask.astype(int) - previous_mask.astype(int)
        
        # For each valid point, calculate local movement
        for i in range(n_points):
            if not valid_points[i]:
                continue
            
            # Get a local region around the point
            point = points[i].astype(int)
            y, x = point
            
            # Define a circular region around the point
            y_min = max(0, y - radius)
            y_max = min(current_mask.shape[0] - 1, y + radius)
            x_min = max(0, x - radius)
            x_max = min(current_mask.shape[1] - 1, x + radius)
            
            # Extract local region of the difference mask
            local_diff = diff_mask[y_min:y_max+1, x_min:x_max+1]
            
            # Count extension and retraction pixels in the local region
            extension_pixels = np.sum(local_diff == 1)
            retraction_pixels = np.sum(local_diff == -1)
            
            # Calculate local movement score
            total_changed_pixels = extension_pixels + retraction_pixels
            if total_changed_pixels > 0:
                movement_scores[i] = (extension_pixels - retraction_pixels) / total_changed_pixels
            else:
                movement_scores[i] = 0
            
            # Classify movement type
            if movement_scores[i] > threshold:
                movement_types[i] = 'extending'
            elif movement_scores[i] < -threshold:
                movement_types[i] = 'retracting'
            else:
                movement_types[i] = 'stable'
        
        return movement_scores, movement_types
    
    def save_settings(self):
        """Save application settings"""
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        
        # Save parameters
        params = self.parameter_panel.get_parameters()
        for key, value in params.items():
            self.settings.setValue(f"params/{key}", value)
    
    def restore_settings(self):
        """Restore application settings"""
        # Restore window geometry
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        if self.settings.contains("windowState"):
            self.restoreState(self.settings.value("windowState"))
        
        # Restore parameters if panel exists
        if hasattr(self, 'parameter_panel'):
            params = {}
            for key in self.parameter_panel.parameters.keys():
                if self.settings.contains(f"params/{key}"):
                    # Get correct type
                    if isinstance(self.parameter_panel.parameters[key], bool):
                        params[key] = self.settings.value(f"params/{key}", type=bool)
                    elif isinstance(self.parameter_panel.parameters[key], int):
                        params[key] = self.settings.value(f"params/{key}", type=int)
                    elif isinstance(self.parameter_panel.parameters[key], float):
                        params[key] = self.settings.value(f"params/{key}", type=float)
                    else:
                        params[key] = self.settings.value(f"params/{key}")
            
            # Update panel controls
            if params:
                self.parameter_panel.n_points_spin.setValue(params.get('n_points', 100))
                self.parameter_panel.depth_spin.setValue(params.get('depth', 20))
                self.parameter_panel.width_spin.setValue(params.get('width', 5))
                self.parameter_panel.coverage_spin.setValue(params.get('min_cell_coverage', 0.8))
                self.parameter_panel.radius_spin.setValue(params.get('movement_radius', 5))
                self.parameter_panel.threshold_spin.setValue(params.get('movement_threshold', 0.1))
                self.parameter_panel.try_rotation_cb.setChecked(params.get('try_rotation', True))
                self.parameter_panel.exclude_endpoints_cb.setChecked(params.get('exclude_endpoints', False))
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save settings before closing
        self.save_settings()
        event.accept()


if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Membrane Movement Analyzer")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())
