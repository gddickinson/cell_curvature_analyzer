"""Export of the current frame and of all data, export preview, saving the log and the results."""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import datetime
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from modules.widgets import LogConsole


class ExportActionsMixin:
    """MainWindow mixin: export of the current frame and of all data, export preview, saving the log and the results."""

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
