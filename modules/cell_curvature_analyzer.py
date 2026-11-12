"""MainWindow of the desktop app: construction, toolbar, docks and log tab.

The window's behaviour is split over mixins in modules/gui/ (one concern per file, see INTERFACE.md):
tabs are built in tab_*.py, frame plots in frame_plots.py, the correlation tab in correlation_*.py, navigation and
display in navigation.py / display.py, loading and analysis in data_actions.py (work runs in gui/workers.py threads),
export, settings, ML and pop-up plots in their own files.
"""
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox,
                             QTableWidgetItem, QAction, QToolBar, QProgressBar, QDockWidget, QFormLayout)
from PyQt5.QtCore import Qt, QSettings, QSize
from modules.curvature_analysis import CurvatureAnalyzer
from modules.file_io import FileManager
from modules.visualization import VisualizationManager
from modules.data_export import ExportManager
from modules.settings import SettingsManager
from modules.correlation_analysis import CorrelationAnalyzer
from modules.temporal_analysis import TemporalAnalyzer
from modules.widgets import LogConsole, ParameterPanel
from modules.gui.tab_visualization import VisualizationTabsMixin
from modules.gui.tab_analysis import AnalysisTabsMixin
from modules.gui.tab_export_settings import ExportSettingsTabsMixin
from modules.gui.frame_plots import FramePlotsMixin
from modules.gui.correlation_display import CorrelationDisplayMixin
from modules.gui.correlation_pooled import PooledCorrelationMixin
from modules.gui.correlation_temporal import TemporalCorrelationMixin
from modules.gui.correlation_movement_temporal import MovementTemporalCorrelationMixin
from modules.gui.navigation import NavigationMixin
from modules.gui.display import DisplayMixin
from modules.gui.data_actions import DataActionsMixin
from modules.gui.export_actions import ExportActionsMixin
from modules.gui.settings_actions import SettingsActionsMixin
from modules.gui.ml_tab import MLTabMixin
from modules.gui.popup_plots import PopupPlotsMixin


class MainWindow(VisualizationTabsMixin,
                 AnalysisTabsMixin,
                 ExportSettingsTabsMixin,
                 FramePlotsMixin,
                 CorrelationDisplayMixin,
                 PooledCorrelationMixin,
                 TemporalCorrelationMixin,
                 MovementTemporalCorrelationMixin,
                 NavigationMixin,
                 DisplayMixin,
                 DataActionsMixin,
                 ExportActionsMixin,
                 SettingsActionsMixin,
                 MLTabMixin,
                 PopupPlotsMixin,
                 QMainWindow):
    """Main application window for CellCurvatureAnalyzer"""

    def __init__(self):
        super().__init__()

        # Initialize application components
        self.file_manager = FileManager()
        self.curvature_analyzer = CurvatureAnalyzer()
        self.visualization_manager = VisualizationManager()
        self.export_manager = ExportManager()
        self.settings_manager = SettingsManager()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()

        # Application state
        self.current_frame = 0
        self.loaded_data = {
            'images': None,
            'masks': None,
            'results': None
        }

        # Create parameter panel first (before connecting signals)
        self.parameter_panel = None

        # Setup UI
        self.init_ui()

        # Connect parameter panel signals AFTER UI is initialized
        if self.parameter_panel:
            self.parameter_panel.parameters_changed.connect(self.on_parameters_changed)

        # Restore previous settings
        self.settings = QSettings("CellCurvatureAnalyzer", "Application")
        self.restore_settings()

        # Status message
        self.statusBar().showMessage("Ready")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Cell Curvature & PIEZO1 Analyzer")
        self.setGeometry(100, 100, 1280, 800)

        # Create main toolbar
        self.create_toolbar()

        # Create dock widgets first (so parameter panel is available)
        self.create_parameter_dock()
        self.create_info_dock()

        # Create main layout with central tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add main tabs
        self.create_visualization_tab()
        self.create_analysis_tab()
        self.create_correlation_tab()
        self.create_temporal_tab()
        self.create_movement_tab()
        self.create_export_tab()
        self.create_settings_tab()
        self.create_log_tab()

        # Create status bar with progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        # Connect signals and slots
        self.connect_signals()

    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Open action
        open_action = QAction("Open Images", self)
        open_action.setStatusTip("Open microscope image stack")
        open_action.triggered.connect(self.open_image_stack)
        toolbar.addAction(open_action)

        # Open masks action
        masks_action = QAction("Open Masks", self)
        masks_action.setStatusTip("Open binary mask stack")
        masks_action.triggered.connect(self.open_mask_stack)
        toolbar.addAction(masks_action)

        toolbar.addSeparator()

        # Run analysis action
        run_action = QAction("Run Analysis", self)
        run_action.setStatusTip("Run full analysis")
        run_action.triggered.connect(self.run_analysis)
        toolbar.addAction(run_action)

        # Save results action
        save_action = QAction("Save Results", self)
        save_action.setStatusTip("Save analysis results")
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Frame control
        self.frame_label = QLabel("Frame: 0/0")
        toolbar.addWidget(self.frame_label)

        prev_frame_action = QAction("Previous", self)
        prev_frame_action.setStatusTip("Previous frame")
        prev_frame_action.triggered.connect(self.previous_frame)
        toolbar.addAction(prev_frame_action)

        next_frame_action = QAction("Next", self)
        next_frame_action.setStatusTip("Next frame")
        next_frame_action.triggered.connect(self.next_frame)
        toolbar.addAction(next_frame_action)

    def closeEvent(self, event):
        """Handle application closing"""
        # Save settings
        self.save_settings()
        event.accept()


    def create_log_tab(self):
        """Create the log tab with console output"""
        log_tab = QWidget()
        self.tabs.addTab(log_tab, "Log")

        layout = QVBoxLayout(log_tab)

        # Create log console
        self.log_console = LogConsole()
        layout.addWidget(self.log_console)

        # Add buttons
        button_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_console.clear)

        save_btn = QPushButton("Save Log")
        save_btn.clicked.connect(self.save_log)

        button_layout.addStretch()
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)


    def create_parameter_dock(self):
        """Create the parameter dock widget"""
        # Create dock widget
        self.parameter_dock = QDockWidget("Analysis Parameters", self)
        self.parameter_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create parameter panel
        self.parameter_panel = ParameterPanel()

        # Set widget
        self.parameter_dock.setWidget(self.parameter_panel)

        # Add dock to main window
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameter_dock)

    def create_info_dock(self):
        """Create the info dock widget"""
        # Create dock widget
        self.info_dock = QDockWidget("Frame Information", self)
        self.info_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create widget
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # Frame info
        frame_group = QGroupBox("Frame")
        frame_layout = QVBoxLayout()
        frame_group.setLayout(frame_layout)

        self.frame_info_label = QLabel("No frame loaded")
        frame_layout.addWidget(self.frame_info_label)

        info_layout.addWidget(frame_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        stats_group.setLayout(stats_layout)

        self.curvature_stats_label = QLabel("N/A")
        self.intensity_stats_label = QLabel("N/A")
        self.valid_points_label = QLabel("N/A")
        self.movement_type_label = QLabel("N/A")

        stats_layout.addRow("Curvature (mean ± std):", self.curvature_stats_label)
        stats_layout.addRow("Intensity (mean ± std):", self.intensity_stats_label)
        stats_layout.addRow("Valid Points:", self.valid_points_label)
        stats_layout.addRow("Movement Type:", self.movement_type_label)

        info_layout.addWidget(stats_group)

        # Add plot buttons panel
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)

        # Curvature buttons
        curvature_layout = QHBoxLayout()
        self.sign_curvature_btn = QPushButton("Sign Curvature")
        self.sign_curvature_btn.clicked.connect(lambda: self.create_popup_curvature_plot("sign"))

        self.magnitude_curvature_btn = QPushButton("Magnitude Curvature")
        self.magnitude_curvature_btn.clicked.connect(lambda: self.create_popup_curvature_plot("magnitude"))

        self.normalized_curvature_btn = QPushButton("Normalized Curvature")
        self.normalized_curvature_btn.clicked.connect(lambda: self.create_popup_curvature_plot("normalized"))

        curvature_layout.addWidget(self.sign_curvature_btn)
        curvature_layout.addWidget(self.magnitude_curvature_btn)
        curvature_layout.addWidget(self.normalized_curvature_btn)

        plot_layout.addLayout(curvature_layout)

        # Intensity and combined buttons
        intensity_layout = QHBoxLayout()
        self.intensity_btn = QPushButton("Intensity Profile")
        self.intensity_btn.clicked.connect(self.create_popup_intensity_plot)

        self.combined_btn = QPushButton("Combined Plot")
        self.combined_btn.clicked.connect(self.create_popup_combined_plot)

        intensity_layout.addWidget(self.intensity_btn)
        intensity_layout.addWidget(self.combined_btn)

        plot_layout.addLayout(intensity_layout)

        info_layout.addWidget(plot_group)

        # Add stretch to push everything to the top
        info_layout.addStretch()

        # Set widget
        self.info_dock.setWidget(info_widget)

        # Add dock to main window
        self.addDockWidget(Qt.RightDockWidgetArea, self.info_dock)

    def connect_signals(self):
        """Connect signals and slots"""
        # Movement frame spin
        self.movement_frame_spin.valueChanged.connect(self.update_movement_display)

        # Temporal frame spins
        #self.current_frame_spin.valueChanged.connect(self.update_temporal_display)
        #self.reference_frame_spin.valueChanged.connect(self.update_temporal_display)

        # Settings controls
        self.curvature_cmap_combo.currentIndexChanged.connect(self.update_visualization_settings)
        self.intensity_cmap_combo.currentIndexChanged.connect(self.update_visualization_settings)
        self.marker_size_spin.valueChanged.connect(self.update_visualization_settings)
        self.contour_width_spin.valueChanged.connect(self.update_visualization_settings)

    def add_table_row(self, table, parameter, value):
        """
        Add a row to a table widget (works with both ResultsTable and QTableWidget)

        Parameters:
        -----------
        table : QTableWidget or ResultsTable
            Table widget to add row to
        parameter : str
            Parameter name/label
        value : Any
            Parameter value
        """
        # Check if the table has the add_result method (ResultsTable)
        if hasattr(table, 'add_result'):
            table.add_result(parameter, value)
        else:
            # Regular QTableWidget - add row manually
            row = table.rowCount()
            table.insertRow(row)

            # Convert value to string representation
            if isinstance(value, float):
                value_str = f"{value:.6g}"
            else:
                value_str = str(value)

            # Create items
            parameter_item = QTableWidgetItem(parameter)
            value_item = QTableWidgetItem(value_str)

            # Add to table
            table.setItem(row, 0, parameter_item)
            table.setItem(row, 1, value_item)

            # Make sure the table has the right number of columns
            if table.columnCount() < 2:
                table.setColumnCount(2)
                table.setHorizontalHeaderLabels(["Parameter", "Value"])




