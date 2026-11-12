"""Correlation tab: dispatch on metric / mode / temporal mode, per-frame movement proxy, single-frame correlations."""
import numpy as np
from scipy import stats


class CorrelationDisplayMixin:
    """MainWindow mixin: correlation tab: dispatch on metric / mode / temporal mode, per-frame movement proxy, single-frame correlations."""

    def on_movement_frame_changed(self, frame_index):
        """Handle changes to the movement frame spinner"""
        # Update the current frame for movement analysis only
        # Note: This does not change the current_frame variable which affects other tabs
        self.update_movement_display()

        # Log frame change
        self.log_console.log(f"Movement frame changed to {frame_index}")

    def generate_all_frames_movement_data(self):
        """Generate movement data for all frames"""
        results = self.loaded_data.get('results')
        if results is None:
            return

        # Process each frame
        for frame_idx in list(results.keys()):  # Create a copy of keys
            if not isinstance(frame_idx, int):
                continue

            self.generate_frame_movement_data(frame_idx)

        # Log completion
        self.log_console.log(f"Generated movement data for all frames")

    def on_metric_type_changed(self):
        """Handle changes to the metric type selector"""
        metric_type = self.metric_type_combo.currentText()

        # Show/hide appropriate controls
        if metric_type == "Curvature":
            self.correlation_type_combo.setVisible(True)
            self.curvature_type_label.setVisible(True)
            self.movement_type_combo.setVisible(False)
            self.movement_type_label.setVisible(False)
            self.generate_movement_btn.setVisible(False)
        else:  # Movement
            self.correlation_type_combo.setVisible(False)
            self.curvature_type_label.setVisible(False)
            self.movement_type_combo.setVisible(True)
            self.movement_type_label.setVisible(True)
            self.generate_movement_btn.setVisible(True)

        # Update the display
        self.update_correlation_display()

    def update_correlation_display(self):
        """Update the correlation analysis tab display"""
        results = self.loaded_data.get('results')

        if results is None:
            return

        # Get current selections
        metric_type = self.metric_type_combo.currentText()

        if metric_type == "Curvature":
            subtype = self.correlation_type_combo.currentText()
        else:  # Movement
            subtype = self.movement_type_combo.currentText()

        analysis_mode = self.correlation_mode_combo.currentText()

        # Get temporal mode from parameter panel
        temporal_mode = self.parameter_panel.parameters.get('temporal_mode', "Current Frame")
        reference_frame = self.parameter_panel.parameters.get('reference_frame', 0)

        # Update based on analysis mode
        if analysis_mode == "Current Frame":
            # Clear plot
            self.correlation_plot_figure.clear()
            ax = self.correlation_plot_figure.add_subplot(111)

            # Get current frame data
            frame = self.current_frame

            if frame in results:
                frame_results = results[frame]

                # For movement analysis, generate data if needed
                if metric_type == "Movement" and 'movement_data' not in frame_results:
                    self.generate_frame_movement_data(frame)

                if temporal_mode == "Current Frame":
                    # Standard single-frame correlation
                    self.analyze_current_frame_correlation(frame_results, metric_type, subtype, ax)

                elif temporal_mode == "Current-Previous Frames" and frame > 0:
                    previous_frame = frame - 1
                    if previous_frame in results:
                        previous_results = results[previous_frame]

                        # For movement analysis, generate data for previous frame if needed
                        if metric_type == "Movement" and 'movement_data' not in previous_results:
                            self.generate_frame_movement_data(previous_frame)

                        self.analyze_temporal_frame_correlation(frame_results, previous_results,
                                                              metric_type, subtype, "Previous Frame", ax)

                elif temporal_mode == "Current-Random Frames" and reference_frame != frame:
                    if reference_frame in results:
                        reference_results = results[reference_frame]

                        # For movement analysis, generate data for reference frame if needed
                        if metric_type == "Movement" and 'movement_data' not in reference_results:
                            self.generate_frame_movement_data(reference_frame)

                        self.analyze_temporal_frame_correlation(frame_results, reference_results,
                                                              metric_type, subtype, f"Frame {reference_frame+1}", ax)

                # Refresh canvas
                self.correlation_plot_canvas.draw()

                # Clear comparison chart
                self.comparison_figure.clear()
                self.comparison_canvas.draw()

        elif analysis_mode == "All Frames":
            # For movement analysis, generate data for all frames if needed
            if metric_type == "Movement":
                self.generate_all_frames_movement_data()

            # Apply temporal mode to "All Frames" analysis
            if temporal_mode == "Current Frame":
                # Standard all-frames analysis (within each frame)
                self.analyze_all_frames_standard_correlation(results, metric_type, subtype)

            elif temporal_mode == "Current-Previous Frames":
                # Analyze correlation between current frame curvature and previous frame intensity for all frames
                self.analyze_all_frames_temporal_correlation(results, metric_type, subtype, "previous")

            elif temporal_mode == "Current-Random Frames":
                # Analyze correlation between current frame curvature and random frame intensity
                self.analyze_all_frames_temporal_correlation(results, metric_type, subtype, "random", reference_frame)

        elif analysis_mode == "By Movement Type":
            # For movement analysis, generate data for all frames if needed
            if metric_type == "Movement":
                self.generate_all_frames_movement_data()

            # Apply temporal mode to "By Movement Type" analysis
            if temporal_mode == "Current Frame":
                # Standard movement-type analysis (within each frame)
                self.analyze_movement_standard_correlation(results, metric_type, subtype)

            elif temporal_mode == "Current-Previous Frames":
                # Analyze correlation between current frame curvature and previous frame intensity by movement type
                self.analyze_movement_temporal_correlation(results, metric_type, subtype, "previous")

            elif temporal_mode == "Current-Random Frames":
                # Analyze correlation between current frame curvature and random frame intensity by movement type
                self.analyze_movement_temporal_correlation(results, metric_type, subtype, "random", reference_frame)

    def generate_frame_movement_data(self, frame_idx):
        """
        Generate movement data for a single frame

        Parameters:
        -----------
        frame_idx : int
            Frame index
        """
        results = self.loaded_data.get('results')
        if results is None or frame_idx not in results:
            return

        frame_results = results[frame_idx]

        # Skip if we already have movement data
        if 'movement_data' in frame_results:
            return

        # We need points and contour data
        if 'points' not in frame_results or 'contour_data' not in frame_results:
            return

        points = frame_results['points']
        contour = frame_results['contour_data']

        # Initialize movement data with default values
        movement_scores = np.zeros(len(points))
        movement_types = np.array(['stable'] * len(points), dtype=object)

        # For each point, calculate local curvature change as a proxy for movement
        try:
            if 'curvatures' in frame_results:
                sign_curvature, magnitude_curvature, normalized_curvature = frame_results['curvatures']

                # Calculate the gradient of curvature along the contour
                for i in range(len(points)):
                    # Get neighboring points (with wrapping)
                    prev_idx = (i - 1) % len(points)
                    next_idx = (i + 1) % len(points)

                    # Calculate curvature gradient (central difference)
                    curvature_gradient = (normalized_curvature[next_idx] - normalized_curvature[prev_idx]) / 2
                    movement_scores[i] = curvature_gradient

                    # Classify as extending, retracting, or stable
                    if curvature_gradient > 0.1:
                        movement_types[i] = 'extending'
                    elif curvature_gradient < -0.1:
                        movement_types[i] = 'retracting'
                    else:
                        movement_types[i] = 'stable'
        except Exception as e:
            # Log error but continue
            print(f"Error generating movement data for frame {frame_idx}: {str(e)}")
            return

        # Store movement data in frame results
        frame_results['movement_data'] = {
            'scores': movement_scores,
            'types': movement_types
        }

    def analyze_current_frame_correlation(self, frame_results, metric_type, subtype, ax):
        """Analyze correlation for a single frame"""
        # Check if required data is available
        if ('intensities' not in frame_results or
            'valid_points' not in frame_results):
            return

        # Get data based on metric type and subtype
        if metric_type == "Curvature":
            if 'curvatures' not in frame_results:
                return

            # Map subtype to curvature index
            curvature_index = {
                "Sign Curvature": 0,
                "Magnitude Curvature": 1,
                "Normalized Curvature": 2
            }.get(subtype, 2)

            metric_data = frame_results['curvatures'][curvature_index]
            metric_name = subtype

        elif metric_type == "Movement":
            if 'movement_data' not in frame_results:
                # Try to generate movement data
                self.generate_frame_movement_data(self.current_frame)

                if 'movement_data' not in frame_results:
                    return

            if subtype == "Movement Score":
                metric_data = frame_results['movement_data']['scores']
                metric_name = "Movement Score"
            elif subtype == "Extension/Retraction":
                # Convert categorical to numerical
                movement_types = frame_results['movement_data']['types']
                metric_data = np.zeros(len(movement_types))
                for i, mtype in enumerate(movement_types):
                    if mtype == 'extending':
                        metric_data[i] = 1
                    elif mtype == 'retracting':
                        metric_data[i] = -1
                metric_name = "Extension/Retraction"
            else:
                return
        else:
            return

        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Filter data for valid points
        valid_metric = metric_data[valid_points]
        valid_intensities = intensities[valid_points]

        # Create scatter plot
        ax.scatter(valid_metric, valid_intensities, alpha=0.7)

        # Add regression line if enough points
        if len(valid_metric) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_metric, valid_intensities)

            # Create line data
            x_line = np.linspace(min(valid_metric), max(valid_metric), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"{metric_name}-Intensity Correlation (Current Frame)"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_metric)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            title = f"{metric_name}-Intensity Correlation (Current Frame)\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(valid_metric) > 1:
            ax.legend(loc='best')

    def analyze_temporal_frame_correlation(self, current_results, reference_results, metric_type, subtype,
                                         reference_label, ax):
        """Analyze correlation between current frame metric and reference frame intensity"""
        # Check if required data is available
        if ('intensities' not in reference_results or
            'valid_points' not in current_results or
            'valid_points' not in reference_results):
            return

        # Get data based on metric type and subtype
        if metric_type == "Curvature":
            if 'curvatures' not in current_results:
                return

            # Map subtype to curvature index
            curvature_index = {
                "Sign Curvature": 0,
                "Magnitude Curvature": 1,
                "Normalized Curvature": 2
            }.get(subtype, 2)

            metric_data = current_results['curvatures'][curvature_index]
            metric_name = subtype

        elif metric_type == "Movement":
            if 'movement_data' not in current_results:
                # Try to generate movement data
                self.generate_frame_movement_data(self.current_frame)

                if 'movement_data' not in current_results:
                    return

            if subtype == "Movement Score":
                metric_data = current_results['movement_data']['scores']
                metric_name = "Movement Score"
            elif subtype == "Extension/Retraction":
                # Convert categorical to numerical
                movement_types = current_results['movement_data']['types']
                metric_data = np.zeros(len(movement_types))
                for i, mtype in enumerate(movement_types):
                    if mtype == 'extending':
                        metric_data[i] = 1
                    elif mtype == 'retracting':
                        metric_data[i] = -1
                metric_name = "Extension/Retraction"
            else:
                return
        else:
            return

        current_valid = current_results['valid_points']
        reference_intensities = reference_results['intensities']
        reference_valid = reference_results['valid_points']

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, reference_valid)

        # Filter data for valid points
        valid_metric = metric_data[valid_points]
        valid_intensities = reference_intensities[valid_points]

        # Create scatter plot
        ax.scatter(valid_metric, valid_intensities, alpha=0.7)

        # Add regression line if enough points
        if len(valid_metric) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_metric, valid_intensities)

            # Create line data
            x_line = np.linspace(min(valid_metric), max(valid_metric), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"Current Frame {metric_name} vs {reference_label} Intensity"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_metric)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Current vs {reference_label}")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Current Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_label)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            title = f"Current Frame {metric_name} vs {reference_label} Intensity\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Current vs {reference_label}")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Current Frame", self.current_frame+1)
            self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_label)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(valid_metric))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(f"Current Frame {metric_name}")
        ax.set_ylabel(f'{reference_label} Intensity')
        ax.set_title(title)
        if len(valid_metric) > 1:
            ax.legend(loc='best')
