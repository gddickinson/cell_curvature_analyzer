"""ML Analysis tab: building it, training / applying / saving / loading the model, overlay and importances."""
import os
import numpy as np
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QGroupBox, QSplitter, QTableWidget,
                             QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt
from modules.widgets import LogConsole


class MLTabMixin:
    """MainWindow mixin: mL Analysis tab: building it, training / applying / saving / loading the model, overlay and importances."""

    def create_ml_analysis_tab(self):
        """Create the machine learning analysis tab"""
        ml_tab = QWidget()
        self.tabs.addTab(ml_tab, "ML Analysis")

        layout = QVBoxLayout(ml_tab)

        # Controls
        control_panel = QGroupBox("ML Controls")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Train button
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_ml_model)
        control_layout.addWidget(self.train_model_btn)

        # Apply to current frame button
        self.apply_ml_btn = QPushButton("Apply to Current Frame")
        self.apply_ml_btn.clicked.connect(self.apply_ml_model)
        control_layout.addWidget(self.apply_ml_btn)

        # Save/load model buttons
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_ml_model)
        control_layout.addWidget(self.save_model_btn)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_ml_model)
        control_layout.addWidget(self.load_model_btn)

        layout.addWidget(control_panel)

        # Create visualization with splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - ML overlay
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        ml_overlay_group = QGroupBox("ML Classification Overlay")
        ml_overlay_layout = QVBoxLayout()
        ml_overlay_group.setLayout(ml_overlay_layout)

        self.ml_overlay_figure = Figure(figsize=(5, 5))
        self.ml_overlay_canvas = FigureCanvas(self.ml_overlay_figure)
        self.ml_overlay_toolbar = NavigationToolbar(self.ml_overlay_canvas, self)
        ml_overlay_layout.addWidget(self.ml_overlay_toolbar)
        ml_overlay_layout.addWidget(self.ml_overlay_canvas)

        left_layout.addWidget(ml_overlay_group)
        splitter.addWidget(left_widget)

        # Right panel - Feature importances and stats
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Feature importances
        feature_group = QGroupBox("Feature Importances")
        feature_layout = QVBoxLayout()
        feature_group.setLayout(feature_layout)

        self.feature_figure = Figure(figsize=(5, 4))
        self.feature_canvas = FigureCanvas(self.feature_figure)
        feature_layout.addWidget(self.feature_canvas)

        right_layout.addWidget(feature_group)

        # ML statistics
        stats_group = QGroupBox("ML Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.ml_stats_table = QTableWidget()
        self.ml_stats_table.setColumnCount(2)
        self.ml_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.ml_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.ml_stats_table)

        right_layout.addWidget(stats_group)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([500, 300])

    def train_ml_model(self):
        """Train the ML model on all analyzed frames"""
        if self.loaded_data.get('results') is None:
            QMessageBox.warning(self, "Warning", "Please run analysis first.")
            return

        try:
            # Prepare training data
            features, labels = self.ml_analyzer.prepare_training_data(self.loaded_data['results'])

            if len(features) == 0:
                QMessageBox.warning(self, "Warning", "No valid data points for training.")
                return

            # Train model
            accuracy = self.ml_analyzer.train_model(features, labels)

            # Update UI
            self.statusBar().showMessage(f"Model trained with accuracy: {accuracy:.2f}")
            self.log_console.log(f"Trained ML model with accuracy: {accuracy:.2f}")

            # Update feature importances visualization
            self.update_feature_importances()

            # Update ML stats table
            self.ml_stats_table.setRowCount(0)
            self.ml_stats_table.add_result("Training Accuracy", f"{accuracy:.4f}")
            self.ml_stats_table.add_result("Training Samples", len(features))
            self.ml_stats_table.add_result("Positive Class Samples", np.sum(labels))
            self.ml_stats_table.add_result("Negative Class Samples", len(labels) - np.sum(labels))

            # Apply to current frame
            self.apply_ml_model()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error training model: {str(e)}")
            self.log_console.log(f"Error training model: {str(e)}", LogConsole.ERROR)

    def apply_ml_model(self):
        """Apply the ML model to the current frame"""
        if not self.ml_analyzer.trained:
            QMessageBox.warning(self, "Warning", "Please train the model first.")
            return

        # Get current frame
        frame = self.current_frame

        # Check if results exist for this frame
        if (self.loaded_data.get('results') is None or
            frame not in self.loaded_data['results']):
            QMessageBox.warning(self, "Warning", f"No analysis results for frame {frame+1}.")
            return

        try:
            # Get frame results
            frame_results = self.loaded_data['results'][frame]

            # Classify points
            classifications, probabilities = self.ml_analyzer.classify_points(frame_results)

            # Add to frame results
            frame_results['ml_classifications'] = classifications
            frame_results['ml_probabilities'] = probabilities

            # Update visualization
            self.update_ml_overlay()

            # Update stats
            valid_classifications = classifications[classifications >= 0]
            if len(valid_classifications) > 0:
                self.ml_stats_table.add_result("High Interest Points", np.sum(valid_classifications == 1))
                self.ml_stats_table.add_result("Low Interest Points", np.sum(valid_classifications == 0))
                self.ml_stats_table.add_result("High Interest Percentage",
                                             f"{np.sum(valid_classifications == 1) / len(valid_classifications) * 100:.1f}%")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying model: {str(e)}")
            self.log_console.log(f"Error applying model: {str(e)}", LogConsole.ERROR)

    def update_ml_overlay(self):
        """Update the ML classification overlay visualization"""
        # Get current frame
        frame = self.current_frame

        # Check if ML results exist for this frame
        if (self.loaded_data.get('results') is None or
            frame not in self.loaded_data['results'] or
            'ml_classifications' not in self.loaded_data['results'][frame]):
            return

        # Get frame results
        frame_results = self.loaded_data['results'][frame]

        # Get data
        points = frame_results['points']
        valid_points = frame_results['valid_points']
        classifications = frame_results['ml_classifications']
        probabilities = frame_results['ml_probabilities']

        # Get image if available
        image = None
        if self.loaded_data.get('images') is not None:
            if frame < self.loaded_data['images'].shape[0]:
                image = self.loaded_data['images'][frame]

        # Clear figure
        self.ml_overlay_figure.clear()
        ax = self.ml_overlay_figure.add_subplot(111)

        # Display image if available
        if image is not None:
            ax.imshow(image, cmap='gray')

        # Create scatter points for classifications
        high_interest_mask = (classifications == 1)
        low_interest_mask = (classifications == 0)

        # Plot points
        if np.any(high_interest_mask):
            ax.scatter(points[high_interest_mask, 1], points[high_interest_mask, 0],
                     c='r', s=50, alpha=0.7, label='High Interest')

        if np.any(low_interest_mask):
            ax.scatter(points[low_interest_mask, 1], points[low_interest_mask, 0],
                     c='b', s=20, alpha=0.5, label='Low Interest')

        # Add title and legend
        ax.set_title(f"ML Classification (Frame {frame+1})")
        ax.legend()

        # Hide axes
        ax.axis('off')

        # Refresh canvas
        self.ml_overlay_canvas.draw()

    def update_feature_importances(self):
        """Update the feature importances visualization"""
        if not self.ml_analyzer.trained:
            return

        # Get feature importances
        importances = self.ml_analyzer.get_feature_importances()

        # Clear figure
        self.feature_figure.clear()
        ax = self.feature_figure.add_subplot(111)

        # Create bar plot
        features = list(importances.keys())
        values = list(importances.values())

        # Sort by importance
        sorted_indices = np.argsort(values)
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # Create horizontal bar plot
        bars = ax.barh(features, values, color='skyblue')

        # Add values as text
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')

        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')

        # Refresh canvas
        self.feature_canvas.draw()

    def save_ml_model(self):
        """Save the trained ML model to a file"""
        if not self.ml_analyzer.trained:
            QMessageBox.warning(self, "Warning", "Please train the model first.")
            return

        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ML Model", os.getcwd(), "Model Files (*.joblib);;All Files (*.*)")

        if not file_path:
            return

        try:
            # Save model
            self.ml_analyzer.save_model(file_path)

            # Show success message
            self.statusBar().showMessage(f"Model saved to {file_path}")
            self.log_console.log(f"Saved ML model to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
            self.log_console.log(f"Error saving model: {str(e)}", LogConsole.ERROR)

    def load_ml_model(self):
        """Load a trained ML model from a file"""
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load ML Model", os.getcwd(), "Model Files (*.joblib);;All Files (*.*)")

        if not file_path:
            return

        try:
            # Load model
            self.ml_analyzer.load_model(file_path)

            # Show success message
            self.statusBar().showMessage(f"Model loaded from {file_path}")
            self.log_console.log(f"Loaded ML model from {file_path}")

            # Update feature importances
            self.update_feature_importances()

            # Apply to current frame
            self.apply_ml_model()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.log_console.log(f"Error loading model: {str(e)}", LogConsole.ERROR)
