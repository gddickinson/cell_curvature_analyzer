"""Builds the Correlation, Temporal and Movement tabs of MainWindow and the comparison chart."""
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox, QSplitter,
                             QTableWidget, QHeaderView)
from PyQt5.QtCore import Qt


class AnalysisTabsMixin:
    """MainWindow mixin: builds the Correlation, Temporal and Movement tabs of MainWindow and the comparison chart."""

    def create_correlation_tab(self):
        """Create the correlation analysis tab"""
        correlation_tab = QWidget()
        self.tabs.addTab(correlation_tab, "Correlation Analysis")

        layout = QVBoxLayout(correlation_tab)

        # Controls
        control_panel = QGroupBox("Correlation Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Add metric type selector (NEW)
        self.metric_type_combo = QComboBox()
        self.metric_type_combo.addItems(["Curvature", "Movement"])
        self.metric_type_combo.currentIndexChanged.connect(self.on_metric_type_changed)

        self.metric_type_label = QLabel("Metric Type:")
        control_layout.addWidget(self.metric_type_label)
        control_layout.addWidget(self.metric_type_combo)

        # Existing curvature type combo
        self.correlation_type_combo = QComboBox()
        self.correlation_type_combo.addItems(["Sign Curvature", "Magnitude Curvature", "Normalized Curvature"])
        self.correlation_type_combo.currentIndexChanged.connect(self.update_correlation_display)

        self.curvature_type_label = QLabel("Curvature Type:")
        control_layout.addWidget(self.curvature_type_label)
        control_layout.addWidget(self.correlation_type_combo)

        # NEW: Add movement type combo
        self.movement_type_combo = QComboBox()
        self.movement_type_combo.addItems(["Movement Score", "Extension/Retraction"])
        self.movement_type_combo.currentIndexChanged.connect(self.update_correlation_display)
        self.movement_type_combo.setVisible(False)  # Initially hidden

        self.movement_type_label = QLabel("Movement Type:")
        self.movement_type_label.setVisible(False)  # Initially hidden
        control_layout.addWidget(self.movement_type_label)
        control_layout.addWidget(self.movement_type_combo)

        self.correlation_mode_combo = QComboBox()
        self.correlation_mode_combo.addItems(["Current Frame", "All Frames", "By Movement Type"])
        self.correlation_mode_combo.currentIndexChanged.connect(self.update_correlation_display)

        control_layout.addWidget(QLabel("Analysis Mode:"))
        control_layout.addWidget(self.correlation_mode_combo)

        # Add comparison metric selector
        self.comparison_metric_combo = QComboBox()
        self.comparison_metric_combo.addItems(["R²", "p-value", "Slope", "Sample Size"])
        self.comparison_metric_combo.currentIndexChanged.connect(self.update_comparison_chart)

        control_layout.addWidget(QLabel("Comparison Metric:"))
        control_layout.addWidget(self.comparison_metric_combo)

        # Add a button to generate movement data for all frames
        self.generate_movement_btn = QPushButton("Generate Movement Data")
        self.generate_movement_btn.clicked.connect(self.generate_all_frames_movement_data)
        self.generate_movement_btn.setVisible(False)  # Initially hidden
        control_layout.addWidget(self.generate_movement_btn)

        layout.addWidget(control_panel)

        # Create correlation display with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Scatter plot
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        correlation_plot_group = QGroupBox("Correlation Plot")
        correlation_plot_layout = QVBoxLayout()
        correlation_plot_group.setLayout(correlation_plot_layout)

        self.correlation_plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.correlation_plot_canvas = FigureCanvas(self.correlation_plot_figure)
        self.correlation_plot_toolbar = NavigationToolbar(self.correlation_plot_canvas, self)
        correlation_plot_layout.addWidget(self.correlation_plot_toolbar)
        correlation_plot_layout.addWidget(self.correlation_plot_canvas)

        left_layout.addWidget(correlation_plot_group)
        splitter.addWidget(left_widget)

        # Right panel - Statistics and summary
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Statistics table
        stats_group = QGroupBox("Correlation Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.correlation_stats_table = QTableWidget()
        self.correlation_stats_table.setColumnCount(4)
        self.correlation_stats_table.setHorizontalHeaderLabels(["Parameter", "Value", "Significance", "Notes"])
        self.correlation_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.correlation_stats_table)

        right_layout.addWidget(stats_group)

        # Comparison chart (for movement types)
        comparison_group = QGroupBox("Comparison Chart")
        comparison_layout = QVBoxLayout()
        comparison_group.setLayout(comparison_layout)

        self.comparison_figure = Figure(figsize=(6, 4), dpi=100)
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        self.comparison_toolbar = NavigationToolbar(self.comparison_canvas, self)
        comparison_layout.addWidget(self.comparison_toolbar)
        comparison_layout.addWidget(self.comparison_canvas)

        right_layout.addWidget(comparison_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([600, 400])

    def create_temporal_tab(self):
        """Create the temporal analysis tab"""
        temporal_tab = QWidget()
        self.tabs.addTab(temporal_tab, "Temporal Analysis")

        layout = QVBoxLayout(temporal_tab)

        # Controls
        control_panel = QGroupBox("Temporal Analysis Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Keep only the temporal visualization type and measurement type
        self.temporal_type_combo = QComboBox()
        self.temporal_type_combo.addItems(["Heatmap Visualization"])
        self.temporal_type_combo.currentIndexChanged.connect(self.update_temporal_display)

        control_layout.addWidget(QLabel("Visualization Type:"))
        control_layout.addWidget(self.temporal_type_combo)

        self.temporal_measure_combo = QComboBox()
        self.temporal_measure_combo.addItems(["Sign Curvature", "Normalized Curvature", "Intensity"])
        self.temporal_measure_combo.currentIndexChanged.connect(self.update_temporal_display)

        control_layout.addWidget(QLabel("Measure:"))
        control_layout.addWidget(self.temporal_measure_combo)

        layout.addWidget(control_panel)

        # Create analysis display with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Visualization
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        temporal_plot_group = QGroupBox("Temporal Visualization")
        temporal_plot_layout = QVBoxLayout()
        temporal_plot_group.setLayout(temporal_plot_layout)

        self.temporal_plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.temporal_plot_canvas = FigureCanvas(self.temporal_plot_figure)
        self.temporal_plot_toolbar = NavigationToolbar(self.temporal_plot_canvas, self)
        temporal_plot_layout.addWidget(self.temporal_plot_toolbar)
        temporal_plot_layout.addWidget(self.temporal_plot_canvas)

        left_layout.addWidget(temporal_plot_group)
        splitter.addWidget(left_widget)

        # Right panel - Statistics and summary
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Statistics
        temporal_stats_group = QGroupBox("Temporal Statistics")
        temporal_stats_layout = QVBoxLayout()
        temporal_stats_group.setLayout(temporal_stats_layout)

        self.temporal_stats_table = QTableWidget()
        self.temporal_stats_table.setColumnCount(3)
        self.temporal_stats_table.setHorizontalHeaderLabels(["Parameter", "Value", "Notes"])
        self.temporal_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        temporal_stats_layout.addWidget(self.temporal_stats_table)

        right_layout.addWidget(temporal_stats_group)

        # Summary plot
        temporal_summary_group = QGroupBox("Temporal Summary")
        temporal_summary_layout = QVBoxLayout()
        temporal_summary_group.setLayout(temporal_summary_layout)

        self.temporal_summary_figure = Figure(figsize=(6, 4), dpi=100)
        self.temporal_summary_canvas = FigureCanvas(self.temporal_summary_figure)
        temporal_summary_layout.addWidget(self.temporal_summary_canvas)

        right_layout.addWidget(temporal_summary_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([600, 400])

    def update_comparison_chart(self):
        """Update the comparison chart based on the selected metric"""
        # Only update if in "By Movement Type" mode
        if self.correlation_mode_combo.currentText() != "By Movement Type":
            return

        # Get the current regression stats (this might be stored from the last analysis)
        # We'll need to add a class variable to store the regression stats
        if not hasattr(self, 'current_regression_stats') or not self.current_regression_stats:
            return

        # Get selected metric
        metric = self.comparison_metric_combo.currentText()

        # Map metric to data key
        metric_key = {
            "R²": 'r_squared',
            "p-value": 'p_value',
            "Slope": 'slope',
            "Sample Size": 'sample_size'
        }.get(metric)

        if not metric_key:
            return

        # Create bar chart
        self.comparison_figure.clear()
        ax = self.comparison_figure.add_subplot(111)

        # Get data for selected metric
        values = []
        labels = []
        colors_list = []

        # Colors for different movement types
        colors = {
            'extending': 'blue',
            'retracting': 'red',
            'stable': 'gray'
        }

        for movement in ['extending', 'retracting', 'stable']:
            if movement in self.current_regression_stats and metric_key in self.current_regression_stats[movement]:
                values.append(self.current_regression_stats[movement][metric_key])
                labels.append(movement.capitalize())
                colors_list.append(colors[movement])

        # Create bar chart
        bars = ax.bar(labels, values, color=colors_list, alpha=0.7)

        # Add value annotations
        for i, value in enumerate(values):
            # Format value based on metric
            if metric == "R²" or metric == "p-value":
                value_text = f"{value:.3f}"
            elif metric == "Slope":
                value_text = f"{value:.3f}"
            else:  # Sample Size
                value_text = f"{int(value)}"

            # Position annotation
            y_pos = value + (max(values) * 0.02)
            ax.annotate(value_text, (i, y_pos), ha='center', va='bottom', fontsize=9)

        # Add p-value annotations for certain metrics
        if metric == "R²":
            for i, movement in enumerate(['extending', 'retracting', 'stable']):
                if movement in self.current_regression_stats and i < len(values):
                    p_value = self.current_regression_stats[movement].get('p_value', 1.0)
                    annotation = f"p={p_value:.3f}"
                    if p_value < 0.001:
                        annotation = "p<0.001"
                    elif p_value < 0.01:
                        annotation = "p<0.01"
                    elif p_value < 0.05:
                        annotation = "p<0.05"

                    ax.annotate(annotation, (i, values[i] + max(values) * 0.05),
                                ha='center', va='bottom', fontsize=9, rotation=90)

        # Set labels and title based on metric
        ax.set_xlabel('Movement Type')
        ax.set_ylabel(metric)
        curvature_type = self.correlation_type_combo.currentText()
        ax.set_title(f'Comparison of {metric} by Movement Type for {curvature_type}')

        # Refresh canvas
        self.comparison_canvas.draw()

    def create_movement_tab(self):
        """Create the movement analysis tab"""
        movement_tab = QWidget()
        self.tabs.addTab(movement_tab, "Edge Movement")

        layout = QVBoxLayout(movement_tab)

        # Controls
        control_panel = QGroupBox("Movement Analysis Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        self.movement_display_combo = QComboBox()
        self.movement_display_combo.addItems(["Movement Map", "Edge Comparison", "Movement Over Time"])
        self.movement_display_combo.currentIndexChanged.connect(self.update_movement_display)

        control_layout.addWidget(QLabel("Display Type:"))
        control_layout.addWidget(self.movement_display_combo)

        self.movement_frame_spin = QSpinBox()
        self.movement_frame_spin.setMinimum(1)  # First frame has no movement data
        self.movement_frame_spin.setMaximum(1)
        self.movement_frame_spin.valueChanged.connect(self.on_movement_frame_changed)  # Use a separate handler

        control_layout.addWidget(QLabel("Frame:"))
        control_layout.addWidget(self.movement_frame_spin)

        layout.addWidget(control_panel)

        # Create analysis display with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Visualization
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        movement_plot_group = QGroupBox("Movement Visualization")
        movement_plot_layout = QVBoxLayout()
        movement_plot_group.setLayout(movement_plot_layout)

        self.movement_plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.movement_plot_canvas = FigureCanvas(self.movement_plot_figure)
        self.movement_plot_toolbar = NavigationToolbar(self.movement_plot_canvas, self)
        movement_plot_layout.addWidget(self.movement_plot_toolbar)
        movement_plot_layout.addWidget(self.movement_plot_canvas)

        left_layout.addWidget(movement_plot_group)
        splitter.addWidget(left_widget)

        # Right panel - Statistics and summary
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Movement statistics
        movement_stats_group = QGroupBox("Movement Statistics")
        movement_stats_layout = QVBoxLayout()
        movement_stats_group.setLayout(movement_stats_layout)

        self.movement_stats_table = QTableWidget()
        self.movement_stats_table.setColumnCount(3)
        self.movement_stats_table.setHorizontalHeaderLabels(["Parameter", "Value", "Notes"])
        self.movement_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        movement_stats_layout.addWidget(self.movement_stats_table)

        right_layout.addWidget(movement_stats_group)

        # Movement summary
        movement_summary_group = QGroupBox("Movement Summary")
        movement_summary_layout = QVBoxLayout()
        movement_summary_group.setLayout(movement_summary_layout)

        self.movement_summary_figure = Figure(figsize=(6, 4), dpi=100)
        self.movement_summary_canvas = FigureCanvas(self.movement_summary_figure)
        self.movement_summary_toolbar = NavigationToolbar(self.movement_summary_canvas, self)
        movement_summary_layout.addWidget(self.movement_summary_toolbar)
        movement_summary_layout.addWidget(self.movement_summary_canvas)

        right_layout.addWidget(movement_summary_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([600, 400])
