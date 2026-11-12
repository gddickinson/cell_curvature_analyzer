"""Correlation tab: all-frames and by-movement-type correlations within each frame."""
import numpy as np
from scipy import stats


class PooledCorrelationMixin:
    """MainWindow mixin: correlation tab: all-frames and by-movement-type correlations within each frame."""

    def analyze_all_frames_standard_correlation(self, results, metric_type, subtype):
        """Standard correlation analysis combining data from all frames (within each frame)"""
        # Combine data from all frames
        combined_metric_data = []
        combined_intensities = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            # Check if required data is available
            if ('intensities' not in frame_results or
                'valid_points' not in frame_results):
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

            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']

            # Filter valid points
            valid_metric = metric_data[valid_points]
            valid_intensities = intensities[valid_points]

            # Add to combined data
            combined_metric_data.extend(valid_metric)
            combined_intensities.extend(valid_intensities)

        # Convert to numpy arrays
        combined_metric_data = np.array(combined_metric_data)
        combined_intensities = np.array(combined_intensities)

        # Clear figure
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

            # Add stats to title
            title = f"{metric_name}-Intensity Correlation (All Frames)"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_metric_data)}"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frames", "All")
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "R²", r_value**2)
            self.add_table_row(self.correlation_stats_table, "p-value", p_value)
            self.add_table_row(self.correlation_stats_table, "Slope", slope)
            self.add_table_row(self.correlation_stats_table, "Intercept", intercept)
            self.add_table_row(self.correlation_stats_table, "Standard Error", std_err)
        else:
            title = f"{metric_name}-Intensity Correlation (All Frames)\nInsufficient data for regression"

            # Update stats table
            self.correlation_stats_table.setRowCount(0)
            self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
            self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
            self.add_table_row(self.correlation_stats_table, "Frames", "All")
            self.add_table_row(self.correlation_stats_table, "Sample Size", len(combined_metric_data))
            self.add_table_row(self.correlation_stats_table, "Status", "Insufficient data")

        # Set labels and title
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(combined_metric_data) > 1:
            ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Clear comparison chart
        self.comparison_figure.clear()
        self.comparison_canvas.draw()

    def analyze_movement_standard_correlation(self, results, metric_type, subtype):
        """Standard correlation analysis by movement type (within each frame)"""
        if 'movement_types' not in results:
            return

        movement_types = results.get('movement_types', [])

        # Group frames by movement type
        extending_frames = []
        retracting_frames = []
        stable_frames = []

        for i, movement_type in enumerate(movement_types):
            # Movement type is for transitions, so the frame index is i+1
            frame = i + 1
            if movement_type == 'extending':
                extending_frames.append(frame)
            elif movement_type == 'retracting':
                retracting_frames.append(frame)
            elif movement_type == 'stable':
                stable_frames.append(frame)

        # Collect data by movement type
        movement_data = {
            'extending': {'metric_data': [], 'intensities': []},
            'retracting': {'metric_data': [], 'intensities': []},
            'stable': {'metric_data': [], 'intensities': []}
        }

        # Process each movement type
        for movement_type_name, frames in [
            ('extending', extending_frames),
            ('retracting', retracting_frames),
            ('stable', stable_frames)
        ]:
            for frame in frames:
                if frame not in results:
                    continue

                frame_results = results[frame]

                # Check for required data
                if ('intensities' not in frame_results or
                    'valid_points' not in frame_results):
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
                        self.generate_frame_movement_data(frame)

                        if 'movement_data' not in frame_results:
                            continue

                    if subtype == "Movement Score":
                        metric_data = frame_results['movement_data']['scores']
                        metric_name = "Movement Score"
                    elif subtype == "Extension/Retraction":
                        # Convert categorical to numerical
                        frame_movement_types = frame_results['movement_data']['types']
                        metric_data = np.zeros(len(frame_movement_types))
                        for i, mtype in enumerate(frame_movement_types):
                            if mtype == 'extending':
                                metric_data[i] = 1
                            elif mtype == 'retracting':
                                metric_data[i] = -1
                        metric_name = "Extension/Retraction"
                    else:
                        continue
                else:
                    continue

                intensities = frame_results['intensities']
                valid_points = frame_results['valid_points']

                # Filter for valid points
                valid_metric = metric_data[valid_points]
                valid_intensities = intensities[valid_points]

                # Add to movement data
                movement_data[movement_type_name]['metric_data'].extend(valid_metric)
                movement_data[movement_type_name]['intensities'].extend(valid_intensities)

        # Create scatter plot with all movement types
        self.correlation_plot_figure.clear()
        ax = self.correlation_plot_figure.add_subplot(111)

        # Colors for different movement types
        colors = {
            'extending': 'blue',
            'retracting': 'red',
            'stable': 'gray'
        }

        # Plot each movement type
        for movement, color in colors.items():
            metric_values = np.array(movement_data[movement]['metric_data'])
            intensity_values = np.array(movement_data[movement]['intensities'])

            if len(metric_values) > 0:
                ax.scatter(metric_values, intensity_values,
                          color=color, alpha=0.5, label=movement.capitalize())

        # Add regression lines for each type
        regression_stats = {}

        for movement, color in colors.items():
            metric_values = np.array(movement_data[movement]['metric_data'])
            intensity_values = np.array(movement_data[movement]['intensities'])

            if len(metric_values) > 1:
                # Calculate regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    metric_values, intensity_values)

                # Create line data
                try:
                    x_line = np.linspace(
                        min(metric_values),
                        max(metric_values),
                        100
                    )
                    y_line = slope * x_line + intercept

                    # Plot line
                    ax.plot(x_line, y_line, color=color, linestyle='-',
                            label=f'{movement.capitalize()}: R²={r_value**2:.3f}')
                except:
                    # Handle case where metric values have no range
                    pass

                # Store stats
                regression_stats[movement] = {
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'slope': slope,
                    'intercept': intercept,
                    'sample_size': len(metric_values)
                }

        # Set labels and title
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Intensity')
        ax.set_title(f"{metric_name}-Intensity Correlation by Movement Type")
        ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Update stats table
        self.correlation_stats_table.setRowCount(0)
        self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
        self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)

        for movement in ['extending', 'retracting', 'stable']:
            if movement in regression_stats:
                stats_dict = regression_stats[movement]
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} R²", stats_dict['r_squared'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} p-value", stats_dict['p_value'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} Slope", stats_dict['slope'])
                self.add_table_row(self.correlation_stats_table, f"{movement.capitalize()} Sample Size", stats_dict['sample_size'])

        # Store regression stats for comparison chart updates
        self.current_regression_stats = regression_stats

        # Create comparison bar chart
        self.update_comparison_chart()
