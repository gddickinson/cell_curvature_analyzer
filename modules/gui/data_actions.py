"""Loading image / mask stacks and running the analysis (whole stack and current frame)."""
import os
import numpy as np
import tifffile
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from modules.widgets import LogConsole
from modules.analysis_defaults import DEFAULTS
from modules.gui.workers import AnalysisWorker, FunctionWorker


class DataActionsMixin:
    """MainWindow mixin: loading image / mask stacks and running the analysis (whole stack and current frame)."""

    def analysis_parameters(self):
        """Analysis parameters from the settings spin boxes and the parameter panel (fallbacks: DEFAULTS)."""
        pp, d = self.parameter_panel.parameters, DEFAULTS
        return {
            'n_points': self.n_points_spin.value(),
            'depth': self.depth_spin.value(),
            'width': self.width_spin.value(),
            'min_cell_coverage': self.min_coverage_spin.value(),
            'try_rotation': pp.get('try_rotation', d['try_rotation']),
            'exclude_endpoints': pp.get('exclude_endpoints', d['exclude_endpoints']),
            'angular_distribution': pp.get('angular_distribution', False),
            'axis_based_distribution': pp.get('axis_based_distribution', False),
            'x_axis_distribution': pp.get('x_axis_distribution', False),
            'point_spacing': pp.get('point_spacing', d['point_spacing']),
            'smoothing_px': pp.get('smoothing_px', d['smoothing_px']),
            'straight_threshold': pp.get('straight_threshold', d['straight_threshold']),
        }

    def analyze_current_frame(self):
        """Analyze only the current frame (in a worker thread; the result is shown by current_frame_analyzed)."""
        # Check if data is loaded
        if (self.loaded_data.get('images') is None or
            self.loaded_data.get('masks') is None):
            QMessageBox.warning(self, "Warning", "Please load image and mask stacks first.")
            return
        if getattr(self, 'frame_worker', None) is not None and self.frame_worker.isRunning():
            self.statusBar().showMessage("A frame analysis is already running")
            return

        params = self.analysis_parameters()
        print(f"DEBUG: Analyzing current frame with {params}")

        # Get current frame data
        frame = self.current_frame
        image = self.loaded_data['images'][frame]
        mask = self.loaded_data['masks'][frame]

        # Log
        self.log_console.log(f"Analyzing frame {frame+1} with parameters: {params}")
        self.statusBar().showMessage(f"Analyzing frame {frame+1}...")

        self.frame_worker = FunctionWorker(self.curvature_analyzer.analyze_frame, image, mask, **params)
        self.frame_worker.frame = frame
        # bound methods of the window (a QObject on the main thread): the slots run on the main thread
        self.frame_worker.finished_signal.connect(self.current_frame_analyzed)
        self.frame_worker.error_signal.connect(self.current_frame_failed)
        self.frame_worker.start()

    def current_frame_analyzed(self, frame_results):
        """Store a single-frame result (main thread) and refresh the displays."""
        frame = self.frame_worker.frame
        if self.loaded_data.get('results') is None:
            self.loaded_data['results'] = {}
        self.loaded_data['results'][frame] = frame_results
        self.update_overlay_display()
        self.update_results_display()
        self.log_console.log(f"Frame {frame+1} analysis completed successfully")
        self.statusBar().showMessage(f"Frame {frame+1} analysis completed")

    def current_frame_failed(self, message):
        frame = self.frame_worker.frame
        QMessageBox.critical(self, "Error", f"Failed to analyze frame: {message}")
        self.log_console.log(f"Error analyzing frame {frame+1}: {message}", LogConsole.ERROR)
        self.statusBar().showMessage("Error analyzing frame")

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

        params = {
            'image_path': self.loaded_data.get('image_path', ''),
            'mask_path': self.loaded_data.get('mask_path', ''),
            'output_dir': os.path.join(os.getcwd(), "results"),
            **self.analysis_parameters(),
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
