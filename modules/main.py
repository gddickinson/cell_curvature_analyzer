"""
Main module for Cell Curvature Analyzer application
"""

def main():
    """
    Entry point function for the application
    """
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    # Import main application window
    from .main_window import MainWindow
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Cell Curvature Analyzer")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
