"""Per-frame curvature, intensity and correlation plots, temporal comparison plots and the frame tables."""
import os
import numpy as np
import scipy.stats


class FramePlotsMixin:
    """MainWindow mixin: per-frame curvature, intensity and correlation plots, temporal comparison plots and the frame tables."""

    def update_curvature_plot(self, frame_results):
        """Update the curvature profile plot with frame data"""
        if 'points' not in frame_results or 'curvatures' not in frame_results:
            return

        # Get data
        points = frame_results['points']
        sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

        # Clear figure
        self.curvature_figure.clear()
        ax = self.curvature_figure.add_subplot(111)

        # Plot sign curvature
        ax.plot(sign_curvatures, 'b-', label='Sign')

        # Plot magnitude curvature on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(magnitude_curvatures, 'r-', label='Magnitude')

        # Plot normalized curvature
        ax.plot(normalized_curvatures, 'g-', label='Normalized')

        # Add labels and legend
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Curvature Sign / Normalized')
        ax2.set_ylabel('Curvature Magnitude')
        ax.set_title('Curvature Profile')

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Refresh canvas
        self.curvature_canvas.draw()

    def update_intensity_plot(self, frame_results):
        """Update the intensity profile plot with frame data"""
        if ('points' not in frame_results or
            'intensities' not in frame_results or
            'valid_points' not in frame_results):
            return

        # Get data
        points = frame_results['points']
        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Clear figure
        self.intensity_figure.clear()
        ax = self.intensity_figure.add_subplot(111)

        # Create x-axis (point indices)
        x = np.arange(len(points))

        # Create masked array for invalid points
        masked_intensities = np.ma.array(intensities, mask=~valid_points)

        # Plot intensity
        ax.plot(x, masked_intensities, 'b-', label='Intensity')
        ax.scatter(x[valid_points], intensities[valid_points], c='b', s=30, alpha=0.7)

        # Add labels and legend
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Intensity')
        ax.set_title('Intensity Profile')
        ax.legend(loc='best')

        # Refresh canvas
        self.intensity_canvas.draw()

    def update_correlation_plot(self, frame_results):
        """Update the correlation plot with frame data"""
        if ('curvatures' not in frame_results or
            'intensities' not in frame_results or
            'valid_points' not in frame_results):
            return

        # Get data
        sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']
        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']

        # Filter data for valid points
        valid_norm_curvatures = normalized_curvatures[valid_points]
        valid_intensities = intensities[valid_points]

        # Clear figure
        self.correlation_figure.clear()
        ax = self.correlation_figure.add_subplot(111)

        # Plot scatter plot
        ax.scatter(valid_norm_curvatures, valid_intensities, alpha=0.7)

        # Add regression line if there are enough points
        if len(valid_norm_curvatures) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                valid_norm_curvatures, valid_intensities)

            # Create line data
            x_line = np.linspace(
                min(valid_norm_curvatures),
                max(valid_norm_curvatures),
                100
            )
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-',
                    label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = "Curvature-Intensity Correlation"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_norm_curvatures)}"
        else:
            title = "Curvature-Intensity Correlation\nInsufficient data for regression"

        # Add labels and title
        ax.set_xlabel('Normalized Curvature')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(valid_norm_curvatures) > 1:
            ax.legend(loc='best')

        # Refresh canvas
        self.correlation_canvas.draw()

    def update_comparative_curvature_plot(self, current_results, reference_results, reference_label):
        """Update curvature plot with data from both current and reference frames"""
        # Clear figure
        self.curvature_figure.clear()
        ax = self.curvature_figure.add_subplot(111)

        # Get current frame curvature data
        _, _, current_norm_curvatures = current_results['curvatures']

        # Get reference frame curvature data
        _, _, reference_norm_curvatures = reference_results['curvatures']

        # Plot both curvature profiles
        ax.plot(current_norm_curvatures, 'b-', label='Current Frame')
        ax.plot(reference_norm_curvatures, 'r-', label=reference_label)

        # Add labels and title
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Normalized Curvature')
        ax.set_title('Comparative Curvature Profile')
        ax.legend()

        # Refresh canvas
        self.curvature_canvas.draw()

    def update_comparative_intensity_plot(self, current_results, reference_results, reference_label):
        """Update intensity plot with data from both current and reference frames"""
        # Clear figure
        self.intensity_figure.clear()
        ax = self.intensity_figure.add_subplot(111)

        # Get current frame intensity data
        current_intensities = current_results['intensities']
        current_valid = current_results['valid_points']

        # Get reference frame intensity data
        reference_intensities = reference_results['intensities']
        reference_valid = reference_results['valid_points']

        # Create x-axis (point indices)
        x = np.arange(len(current_intensities))

        # Create masked arrays for invalid points
        current_masked = np.ma.array(current_intensities, mask=~current_valid)
        reference_masked = np.ma.array(reference_intensities, mask=~reference_valid)

        # Plot intensity profiles
        ax.plot(x, current_masked, 'b-', label='Current Frame')
        ax.scatter(x[current_valid], current_intensities[current_valid], c='b', s=30, alpha=0.7)

        ax.plot(x, reference_masked, 'r-', label=reference_label)
        ax.scatter(x[reference_valid], reference_intensities[reference_valid], c='r', s=30, alpha=0.7)

        # Add labels and title
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Intensity')
        ax.set_title('Comparative Intensity Profile')
        ax.legend()

        # Refresh canvas
        self.intensity_canvas.draw()

    def add_temporal_correlation_stats(self, current_results, reference_results, reference_label, table):
        """
        Add temporal correlation statistics to the given table

        Parameters:
        -----------
        current_results : dict
            Results for current frame
        reference_results : dict
            Results for reference frame
        reference_label : str
            Label for the reference frame
        table : ResultsTable
            Table to add statistics to
        """
        # Get current frame curvature data
        _, _, current_norm_curvatures = current_results['curvatures']

        # Get reference frame intensity data
        reference_intensities = reference_results['intensities']

        # Get valid points (both frames)
        current_valid = current_results['valid_points']
        reference_valid = reference_results['valid_points']

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, reference_valid)

        # Skip if no valid points
        if not np.any(valid_points):
            table.add_result("Status", "Insufficient valid points")
            return

        # Filter data for valid points
        valid_curvatures = current_norm_curvatures[valid_points]
        valid_intensities = reference_intensities[valid_points]

        # Calculate correlation if enough points
        if len(valid_curvatures) > 1:
            # Calculate regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_curvatures, valid_intensities)

            table.add_result("R-squared", r_value**2)
            table.add_result("p-value", p_value)
            table.add_result("Slope", slope)
            table.add_result("Sample Size", len(valid_curvatures))
        else:
            table.add_result("Status", "Insufficient data for correlation")

    def update_temporal_correlation_plot(self, current_results, reference_results, reference_label):
        """Update correlation plot with current curvature vs reference intensity"""
        # Clear figure
        self.correlation_figure.clear()
        ax = self.correlation_figure.add_subplot(111)

        # Get current frame curvature data
        _, _, current_norm_curvatures = current_results['curvatures']

        # Get reference frame intensity data
        reference_intensities = reference_results['intensities']

        # Get valid points (both frames)
        current_valid = current_results['valid_points']
        reference_valid = reference_results['valid_points']

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, reference_valid)

        # Skip if no valid points
        if not np.any(valid_points):
            ax.set_title("Insufficient data for temporal correlation")
            self.correlation_canvas.draw()
            return

        # Filter data for valid points
        valid_curvatures = current_norm_curvatures[valid_points]
        valid_intensities = reference_intensities[valid_points]

        # Create scatter plot
        ax.scatter(valid_curvatures, valid_intensities, alpha=0.7)

        # Add regression line if enough points
        if len(valid_curvatures) > 1:
            # Calculate regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_curvatures, valid_intensities)

            # Create line data
            x_line = np.linspace(min(valid_curvatures), max(valid_curvatures), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"Current Frame Curvature vs {reference_label} Intensity"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_curvatures)}"
        else:
            title = f"Current Frame Curvature vs {reference_label} Intensity\nInsufficient data for regression"

        # Set labels and title
        ax.set_xlabel('Current Frame Normalized Curvature')
        ax.set_ylabel(f'{reference_label} Intensity')
        ax.set_title(title)
        if len(valid_curvatures) > 1:
            ax.legend()

        # Refresh canvas
        self.correlation_canvas.draw()

    def update_frame_results_table(self, frame_results):
        """Update the frame results table with current frame data"""
        self.frame_results_table.setRowCount(0)

        # Add basic information
        if 'points' in frame_results:
            self.frame_results_table.add_result("Number of Points", len(frame_results['points']))

        if 'valid_points' in frame_results:
            valid_count = np.sum(frame_results['valid_points'])
            total_count = len(frame_results['valid_points'])
            self.frame_results_table.add_result("Valid Points", f"{valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")

        # Add curvature statistics
        if 'curvatures' in frame_results:
            sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

            self.frame_results_table.add_result("Mean Sign Curvature", np.mean(sign_curvatures))
            self.frame_results_table.add_result("Mean Magnitude Curvature", np.mean(magnitude_curvatures))
            self.frame_results_table.add_result("Mean Normalized Curvature", np.mean(normalized_curvatures))

        # Add intensity statistics
        if 'intensities' in frame_results and 'valid_points' in frame_results:
            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']

            if np.any(valid_points):
                valid_intensities = intensities[valid_points]
                self.frame_results_table.add_result("Mean Intensity", np.mean(valid_intensities))
                self.frame_results_table.add_result("Min Intensity", np.min(valid_intensities))
                self.frame_results_table.add_result("Max Intensity", np.max(valid_intensities))

        # Get temporal mode
        temporal_mode = self.parameter_panel.parameters['temporal_mode']

        # Add temporal correlation statistics if applicable
        if temporal_mode == "Current-Previous Frames" and self.current_frame > 0:
            previous_frame = self.current_frame - 1

            if previous_frame in self.loaded_data['results']:
                # Get previous frame results
                previous_results = self.loaded_data['results'][previous_frame]

                # Calculate correlation if both frames have valid data
                if ('curvatures' in frame_results and 'intensities' in previous_results and
                    'valid_points' in frame_results and 'valid_points' in previous_results):

                    # Add section header
                    self.frame_results_table.add_result("Temporal Correlation", "Current vs Previous Frame")

                    # Calculate correlation
                    self.add_temporal_correlation_stats(frame_results, previous_results,
                                                     "Previous Frame", self.frame_results_table)

        elif temporal_mode == "Current-Random Frames":
            reference_frame = self.parameter_panel.parameters['reference_frame']

            if reference_frame in self.loaded_data['results'] and reference_frame != self.current_frame:
                # Get reference frame results
                reference_results = self.loaded_data['results'][reference_frame]

                # Calculate correlation if both frames have valid data
                if ('curvatures' in frame_results and 'intensities' in reference_results and
                    'valid_points' in frame_results and 'valid_points' in reference_results):

                    # Add section header
                    self.frame_results_table.add_result("Temporal Correlation", f"Current vs Frame {reference_frame+1}")

                    # Calculate correlation
                    self.add_temporal_correlation_stats(frame_results, reference_results,
                                                     f"Frame {reference_frame+1}", self.frame_results_table)

    def update_frame_info(self, frame_results):
        """Update the frame information panel with current frame data"""
        # Update frame info
        image_path = self.loaded_data.get('image_path', 'N/A')
        mask_path = self.loaded_data.get('mask_path', 'N/A')
        frame = self.current_frame

        self.frame_info_label.setText(
            f"Frame: {frame + 1}\n"
            f"Image: {os.path.basename(image_path)}\n"
            f"Mask: {os.path.basename(mask_path)}"
        )

        # Update statistics
        if 'curvatures' in frame_results:
            _, _, normalized_curvatures = frame_results['curvatures']
            mean_curvature = np.mean(normalized_curvatures)
            std_curvature = np.std(normalized_curvatures)
            self.curvature_stats_label.setText(f"{mean_curvature:.3f} ± {std_curvature:.3f}")
        else:
            self.curvature_stats_label.setText("N/A")

        if 'intensities' in frame_results and 'valid_points' in frame_results:
            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']
            if np.any(valid_points):
                valid_intensities = intensities[valid_points]
                mean_intensity = np.mean(valid_intensities)
                std_intensity = np.std(valid_intensities)
                self.intensity_stats_label.setText(f"{mean_intensity:.3f} ± {std_intensity:.3f}")
            else:
                self.intensity_stats_label.setText("N/A")
        else:
            self.intensity_stats_label.setText("N/A")

        if 'valid_points' in frame_results:
            valid_count = np.sum(frame_results['valid_points'])
            total_count = len(frame_results['valid_points'])
            self.valid_points_label.setText(f"{valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        else:
            self.valid_points_label.setText("N/A")

        # Update movement type if available
        results = self.loaded_data.get('results')
        movement_type = "N/A"

        if results and 'movement_types' in results and frame > 0:
            movement_types = results.get('movement_types', [])
            if frame - 1 < len(movement_types):  # Subtract 1 because first frame has no movement
                movement_type = movement_types[frame - 1].capitalize()

        self.movement_type_label.setText(movement_type)
