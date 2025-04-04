from PyQt5.QtCore import QSettings, QSize, QPoint

class SettingsManager:
    """
    Class for managing application settings
    """
    
    def __init__(self):
        """Initialize SettingsManager"""
        self.settings = QSettings("CellCurvatureAnalyzer", "Application")
    
    def save_window_state(self, window):
        """
        Save window state
        
        Parameters:
        -----------
        window : QMainWindow
            Main application window
        """
        self.settings.setValue("window/geometry", window.saveGeometry())
        self.settings.setValue("window/state", window.saveState())
        self.settings.setValue("window/size", window.size())
        self.settings.setValue("window/position", window.pos())
        
        # Save dock widget states
        self.settings.setValue("docks/parameter_visible", window.parameter_dock.isVisible())
        self.settings.setValue("docks/info_visible", window.info_dock.isVisible())
    
    def restore_window_state(self, window):
        """
        Restore window state
        
        Parameters:
        -----------
        window : QMainWindow
            Main application window
        """
        if self.settings.contains("window/geometry"):
            window.restoreGeometry(self.settings.value("window/geometry"))
        
        if self.settings.contains("window/state"):
            window.restoreState(self.settings.value("window/state"))
        
        if self.settings.contains("window/size"):
            size = self.settings.value("window/size", QSize(1280, 800))
            window.resize(size)
        
        if self.settings.contains("window/position"):
            pos = self.settings.value("window/position", QPoint(100, 100))
            window.move(pos)
        
        # Restore dock widget states
        if self.settings.contains("docks/parameter_visible"):
            visible = self.settings.value("docks/parameter_visible", True, type=bool)
            window.parameter_dock.setVisible(visible)
        
        if self.settings.contains("docks/info_visible"):
            visible = self.settings.value("docks/info_visible", True, type=bool)
            window.info_dock.setVisible(visible)
    
    def save_analysis_parameters(self, parameters):
        """
        Save analysis parameters
        
        Parameters:
        -----------
        parameters : dict
            Dictionary of analysis parameters
        """
        for key, value in parameters.items():
            self.settings.setValue(f"analysis/{key}", value)
    
    def restore_analysis_parameters(self):
        """
        Restore analysis parameters
        
        Returns:
        --------
        parameters : dict
            Dictionary of analysis parameters
        """
        parameters = {
            "n_points": self.settings.value("analysis/n_points", 100, type=int),
            "depth": self.settings.value("analysis/depth", 20, type=int),
            "width": self.settings.value("analysis/width", 5, type=int),
            "min_cell_coverage": self.settings.value("analysis/min_cell_coverage", 0.8, type=float)
        }
        
        return parameters
    
    def save_visualization_settings(self, settings):
        """
        Save visualization settings
        
        Parameters:
        -----------
        settings : dict
            Dictionary of visualization settings
        """
        for key, value in settings.items():
            self.settings.setValue(f"visualization/{key}", value)
    
    def restore_visualization_settings(self):
        """
        Restore visualization settings
        
        Returns:
        --------
        settings : dict
            Dictionary of visualization settings
        """
        settings = {
            "curvature_cmap": self.settings.value("visualization/curvature_cmap", "coolwarm"),
            "intensity_cmap": self.settings.value("visualization/intensity_cmap", "viridis"),
            "marker_size": self.settings.value("visualization/marker_size", 10, type=int),
            "contour_width": self.settings.value("visualization/contour_width", 1.0, type=float)
        }
        
        return settings
    
    def save_export_settings(self, settings):
        """
        Save export settings
        
        Parameters:
        -----------
        settings : dict
            Dictionary of export settings
        """
        for key, value in settings.items():
            self.settings.setValue(f"export/{key}", value)
    
    def restore_export_settings(self):
        """
        Restore export settings
        
        Returns:
        --------
        settings : dict
            Dictionary of export settings
        """
        settings = {
            "directory": self.settings.value("export/directory", ""),
            "prefix": self.settings.value("export/prefix", "cell_analysis_"),
            "export_images": self.settings.value("export/export_images", True, type=bool),
            "export_raw_data": self.settings.value("export/export_raw_data", True, type=bool),
            "export_stats": self.settings.value("export/export_stats", True, type=bool),
            "export_figures": self.settings.value("export/export_figures", True, type=bool),
            "export_report": self.settings.value("export/export_report", True, type=bool)
        }
        
        return settings
    
    def clear_settings(self):
        """Clear all settings"""
        self.settings.clear()
