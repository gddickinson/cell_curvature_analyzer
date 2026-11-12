"""Correlation tab: by-movement-type correlations between a frame and a previous / reference frame."""
import numpy as np
from scipy import stats


class MovementTemporalCorrelationMixin:
    """MainWindow mixin: correlation tab: by-movement-type correlations between a frame and a previous / reference frame."""

    def analyze_movement_temporal_correlation(self, results, metric_type, subtype, mode, reference_frame=None):
        """
        Temporal correlation analysis by movement type

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
        # Check if using "Random Each Time" mode
        random_each_time = (mode == "random" and
                          self.parameter_panel.parameters.get('random_mode') == "Random Each Time")

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

        # Find valid frames (with results)
        valid_frames = []
        for frame_idx in results.keys():
            if isinstance(frame_idx, int):
                valid_frames.append(frame_idx)

        # Sort the valid frames
        valid_frames.sort()

        # Collect data by movement type using temporal relationship
        movement_data = {
            'extending': {'metric_data': [], 'intensities': []},
            'retracting': {'metric_data': [], 'intensities': []},
            'stable': {'metric_data': [], 'intensities': []}
        }

        # Process each frame by movement type, using temporal comparison
        for movement_type_name, frames in [
            ('extending', extending_frames),
            ('retracting', retracting_frames),
            ('stable', stable_frames)
        ]:
            for frame in frames:
                if frame == 0 and mode == "previous":
                    # Skip first frame for previous mode
                    continue

                # Get current frame data
                if frame not in results:
                    continue

                frame_results = results[frame]

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

                current_valid = frame_results['valid_points']

                # Determine reference frame based on mode
                if mode == "previous":
                    ref_frame = frame - 1
                elif mode == "random":
                    if random_each_time:
                        # Pick a random frame for each comparison
                        available_frames = [f for f in valid_frames if f != frame]
                        if not available_frames:
                            continue
                        ref_frame = np.random.choice(available_frames)
                    elif reference_frame is not None and reference_frame in valid_frames and reference_frame != frame:
                        ref_frame = reference_frame
                    else:
                        # Pick a random frame that's not the current one
                        available_frames = [f for f in valid_frames if f != frame]
                        if not available_frames:
                            continue
                        ref_frame = np.random.choice(available_frames)
                else:
                    continue

                # Get reference frame data
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

                # Add to movement data
                movement_data[movement_type_name]['metric_data'].extend(valid_metric_data)
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

        # Set labels and title based on mode
        ax.set_xlabel(f"Current Frame {metric_name}")
        if mode == "previous":
            ax.set_ylabel('Previous Frame Intensity')
            title_mode = "Previous Frame"
        else:
            if random_each_time:
                ax.set_ylabel('Random Frames Intensity')
                title_mode = "Random Frames"
            elif reference_frame is not None:
                ax.set_ylabel(f'Frame {reference_frame+1} Intensity')
                title_mode = f"Frame {reference_frame+1}"
            else:
                ax.set_ylabel('Random Frame Intensity')
                title_mode = "Random Frame"

        ax.set_title(f"Current Frame {metric_name} vs {title_mode} Intensity by Movement Type")
        ax.legend(loc='best')

        # Refresh canvas
        self.correlation_plot_canvas.draw()

        # Update stats table
        self.correlation_stats_table.setRowCount(0)
        self.add_table_row(self.correlation_stats_table, "Analysis Type", f"Temporal ({mode}) by Movement")
        self.add_table_row(self.correlation_stats_table, "Metric Type", metric_type)
        self.add_table_row(self.correlation_stats_table, "Metric Subtype", subtype)
        if mode == "random":
            if random_each_time:
                self.add_table_row(self.correlation_stats_table, "Reference Frames", "Different for each point")
            elif reference_frame is not None:
                self.add_table_row(self.correlation_stats_table, "Reference Frame", reference_frame+1)

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
