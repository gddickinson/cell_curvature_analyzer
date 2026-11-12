"""Correlation tab: all-frames correlations between a frame's curvature and another frame's intensity."""
import numpy as np
from scipy import stats


class TemporalCorrelationMixin:
    """MainWindow mixin: correlation tab: all-frames correlations between a frame's curvature and another frame's intensity."""

    def analyze_all_frames_temporal_correlation(self, results, metric_type, subtype, mode, reference_frame=None):
        """
        Temporal correlation analysis combining data from all frames

        Parameters:
        -----------
        results : dict
            Analysis results
        metric_type : str
            Type of metric to analyze ("Curvature" or "Movement")
        subtype : str
            Subtype of the metric (e.g., "Normalized Curvature" or "Movement Score")
        mode : str
            "previous" or "random"
        reference_frame : int, optional
            Reference frame for random mode
        """
        # Combine data from all frames
        combined_metric_data = []
        combined_intensities = []

        # Find valid frames (with results)
        valid_frames = []
        for frame_idx in results.keys():
            if isinstance(frame_idx, int):
                valid_frames.append(frame_idx)

        # Sort the valid frames
        valid_frames.sort()

        # Process each frame with its temporal comparison
        for frame_idx in valid_frames:
            if frame_idx == 0 and mode == "previous":
                # Skip first frame for previous mode
                continue

            # Get current frame data
            frame_results = results[frame_idx]

            # Check for required data
            if ('valid_points' not in frame_results):
                continue

            # Get data based on metric type and subtype
            if metric_type == "Curvature":
                if 'curvatures' not in frame_results:
                    continue

                # Map subtype to curvature index
                if subtype == "Sign Curvature":
                    metric_data = frame_results['curvatures'][0]  # Sign curvature
                elif subtype == "Magnitude Curvature":
                    metric_data = frame_results['curvatures'][1]  # Magnitude curvature
                elif subtype == "Normalized Curvature":
                    metric_data = frame_results['curvatures'][2]  # Normalized curvature
                else:
                    continue

                metric_name = subtype

            elif metric_type == "Movement":
                if 'movement_data' not in frame_results:
                    # Try to generate movement data
                    self.generate_frame_movement_data(frame_idx)

                    if 'movement_data' not in frame_results:
                        continue

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
                    continue
            else:
                continue

            current_valid = frame_results['valid_points']

            # Determine reference frame based on mode
            if mode == "previous":
                ref_frame = frame_idx - 1
            elif mode == "random":
                if reference_frame is not None and reference_frame in valid_frames and reference_frame != frame_idx:
                    ref_frame = reference_frame
                else:
                    # Pick a random frame that's not the current one
                    available_frames = [f for f in valid_frames if f != frame_idx]
                    if not available_frames:
                        continue
                    ref_frame = np.random.choice(available_frames)
            else:
                continue

            # Get intensity data from reference frame
            if ref_frame not in results:
                continue

            ref_results = results[ref_frame]
            if ('intensities' not in ref_results or
                'valid_points' not in ref_results):
                continue

            ref_intensities = ref_results['intensities']
            ref_valid = ref_results['valid_points']

            # Use logical AND of valid points
            valid_points = np.logical_and(current_valid, ref_valid)

            # Skip if no valid points
            if not np.any(valid_points):
                continue

            # Filter data for valid points
            valid_metric_data = metric_data[valid_points]
            valid_intensities = ref_intensities[valid_points]

            # Add to combined data
            combined_metric_data.extend(valid_metric_data)
            combined_intensities.extend(valid_intensities)

        # Convert to numpy arrays
        combined_metric_data = np.array(combined_metric_data)
        combined_intensities = np.array(combined_intensities)

        # Clear plot
        self.correlation_plot_figure.clear()
        ax = self.correlation_plot_figure.add_subplot(111)

        # Create scatter plot
        ax.scatter(combined_metric_data, combined_intensities, alpha=0.5)

        # Add regression line if enough points
        if len(combined_metric_data) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_metric_data, combined_intensities)

            # Create line data
            x_line = np.linspace(
                min(combined_metric_data),
                max(combined_metric_data),
                100
            )
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-',
                    label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Create title based on mode
            if mode == "previous":
                title = f"Current Frame {metric_name} vs Previous Frame Intensity (All Frames)"
            else:
                if reference_frame is not None:
                    title = f"Current Frame {metric_name} vs Frame {reference_frame+1} Intensity (All Frames)"
                else:
                    title = f"Current Frame {metric_name} vs Random Frame Intensity (All Frames)"

            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_metric_data)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Temporal ({mode})")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            if mode == "random" and reference_frame is not None:
                self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            # Create title based on mode
            if mode == "previous":
                title = f"Current Frame {metric_name} vs Previous Frame Intensity (All Frames)"
            else:
                if reference_frame is not None:
                    title = f"Current Frame {metric_name} vs Frame {reference_frame+1} Intensity (All Frames)"
                else:
                    title = f"Current Frame {metric_name} vs Random Frame Intensity (All Frames)"

            title += "\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Temporal ({mode})")
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            if mode == "random" and reference_frame is not None:
                self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_frame+1)
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(f"Current Frame {metric_name}")
        if mode == "previous":
            ax.set_ylabel('Previous Frame Intensity')
        else:
            if reference_frame is not None:
                ax.set_ylabel(f'Frame {reference_frame+1} Intensity')
            else:
                ax.set_ylabel('Random Frame Intensity')
        ax.set_title(title)
        if len(combined_metric_data) > 1:
            ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Clear comparison chart
        self.comparison_figure.clear()
        self.comparison_canvas.draw()

    def update_temporal_comparison_plots(self, current_results, reference_results, reference_label):
        """
        Update plots with temporal comparison between current and reference frames
        """
        # Check if we're using "Random Each Time" mode and need a different reference frame for each plot
        random_each_time = (self.parameter_panel.parameters.get('temporal_mode') == "Current-Random Frames" and
                          self.parameter_panel.parameters.get('random_mode') == "Random Each Time")

        # Update curvature plot (showing both frames)
        if 'curvatures' in current_results and 'curvatures' in reference_results:
            if not random_each_time:
                self.update_comparative_curvature_plot(current_results, reference_results, reference_label)
            else:
                # Get a random reference frame for this plot
                new_ref_frame = self.get_random_reference_frame()
                if new_ref_frame is not None and new_ref_frame in self.loaded_data['results']:
                    new_ref_results = self.loaded_data['results'][new_ref_frame]
                    self.update_comparative_curvature_plot(current_results, new_ref_results, f"Random Frame {new_ref_frame+1}")

        # Update intensity plot (showing both frames)
        if ('intensities' in current_results and 'valid_points' in current_results and
            'intensities' in reference_results and 'valid_points' in reference_results):
            if not random_each_time:
                self.update_comparative_intensity_plot(current_results, reference_results, reference_label)
            else:
                # Get a random reference frame for this plot
                new_ref_frame = self.get_random_reference_frame()
                if new_ref_frame is not None and new_ref_frame in self.loaded_data['results']:
                    new_ref_results = self.loaded_data['results'][new_ref_frame]
                    self.update_comparative_intensity_plot(current_results, new_ref_results, f"Random Frame {new_ref_frame+1}")

        # Update correlation plot between curvature and reference intensity
        if ('curvatures' in current_results and
            'intensities' in reference_results and
            'valid_points' in current_results and
            'valid_points' in reference_results):
            if not random_each_time:
                self.update_temporal_correlation_plot(current_results, reference_results, reference_label)
            else:
                # Get a random reference frame for this plot
                new_ref_frame = self.get_random_reference_frame()
                if new_ref_frame is not None and new_ref_frame in self.loaded_data['results']:
                    new_ref_results = self.loaded_data['results'][new_ref_frame]
                    self.update_temporal_correlation_plot(current_results, new_ref_results, f"Random Frame {new_ref_frame+1}")

    # Add a helper method to get a random reference frame

    def get_random_reference_frame(self):
        """Get a random reference frame that's different from the current frame"""
        # Find all valid frame indices
        valid_frames = []
        for frame_idx in self.loaded_data['results'].keys():
            if isinstance(frame_idx, int) and frame_idx != self.current_frame:
                valid_frames.append(frame_idx)

        if not valid_frames:
            return None

        # Return a random frame
        return np.random.choice(valid_frames)
