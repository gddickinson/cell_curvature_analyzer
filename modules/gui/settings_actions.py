"""Saving, restoring and resetting the window's QSettings."""
import os
from modules.analysis_defaults import DEFAULTS


class SettingsActionsMixin:
    """MainWindow mixin: saving, restoring and resetting the window's QSettings."""

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
        self.n_points_spin.setValue(DEFAULTS["n_points"])
        self.depth_spin.setValue(DEFAULTS["depth"])
        self.width_spin.setValue(DEFAULTS["width"])
        self.min_coverage_spin.setValue(DEFAULTS["min_cell_coverage"])

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
