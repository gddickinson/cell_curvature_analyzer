import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class TemporalAnalyzer:
    """
    Class for analyzing temporal relationships in cell curvature and intensity
    """

    def __init__(self):
        """Initialize TemporalAnalyzer"""
        pass

    def plot_current_previous_correlation(self, results, current_frame, measure_type,
                                         figure, canvas, stats_table=None,
                                         summary_figure=None, summary_canvas=None):
        """
        Plot correlation between current frame and previous frame measurements

        Parameters:
        -----------
        results : dict
            Analysis results
        current_frame : int
            Current frame index
        measure_type : str
            Type of measurement to analyze (Sign Curvature, Normalized Curvature, Intensity)
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        stats_table : QTableWidget, optional
            Table to display statistics
        summary_figure : matplotlib.figure.Figure, optional
            Figure for summary visualization
        summary_canvas : FigureCanvas, optional
            Canvas for summary visualization
        """
        # Verify frames are valid
        if current_frame <= 0 or current_frame not in results:
            return

        previous_frame = current_frame - 1
        if previous_frame not in results:
            return

        # Get current and previous frame data
        current_results = results[current_frame]
        previous_results = results[previous_frame]

        # Check if required data is available
        if 'curvatures' not in current_results or 'intensities' not in previous_results:
            return

        # Get data based on measure type
        if measure_type == "Sign Curvature":
            current_data = current_results['curvatures'][0]  # Sign curvature
            measurement_label = "Current Frame Sign Curvature"
        elif measure_type == "Normalized Curvature":
            current_data = current_results['curvatures'][2]  # Normalized curvature
            measurement_label = "Current Frame Normalized Curvature"
        elif measure_type == "Intensity":
            current_data = current_results['intensities']
            measurement_label = "Current Frame Intensity"
        else:
            return

        previous_data = previous_results['intensities']

        # Get valid points (both frames)
        current_valid = current_results.get('valid_points', np.ones(len(current_data), dtype=bool))
        previous_valid = previous_results.get('valid_points', np.ones(len(previous_data), dtype=bool))

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, previous_valid)

        # Skip if no valid points
        if not np.any(valid_points):
            return

        # Filter data for valid points
        valid_current = current_data[valid_points]
        valid_previous = previous_data[valid_points]

        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Create scatter plot
        ax.scatter(valid_current, valid_previous, alpha=0.7)

        # Add regression line if enough points
        if len(valid_current) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_current, valid_previous)

            # Create line data
            x_line = np.linspace(min(valid_current), max(valid_current), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"{measurement_label} vs. Previous Frame Intensity"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_current)}"
        else:
            title = f"{measurement_label} vs. Previous Frame Intensity\nInsufficient data for regression"

        # Set labels and title
        ax.set_xlabel(measurement_label)
        ax.set_ylabel('Previous Frame Intensity')
        ax.set_title(title)
        if len(valid_current) > 1:
            ax.legend()

        # Refresh canvas
        canvas.draw()

        # Update stats table if provided
        if stats_table is not None:
            stats_table.setRowCount(0)

            stats_table.add_result("Analysis Type", "Temporal Correlation")
            stats_table.add_result("Current Frame", current_frame)
            stats_table.add_result("Previous Frame", previous_frame)
            stats_table.add_result("Measurement Type", measure_type)

            if len(valid_current) > 1:
                stats_table.add_result("R-squared", r_value**2)
                stats_table.add_result("p-value", p_value)
                stats_table.add_result("Slope", slope)
                stats_table.add_result("Intercept", intercept)
                stats_table.add_result("Standard Error", std_err)

            stats_table.add_result("Sample Size", len(valid_current))
            stats_table.add_result("Valid Points", f"{np.sum(valid_points)}/{len(valid_points)}")

        # Update summary figure if provided
        if summary_figure is not None and summary_canvas is not None:
            self.plot_temporal_summary(results, measure_type,
                                      summary_figure, summary_canvas)

    def plot_current_random_correlation(self, results, current_frame, random_frame, measure_type,
                                      figure, canvas, stats_table=None,
                                      summary_figure=None, summary_canvas=None):
        """
        Plot correlation between current frame and a random frame measurements

        Parameters:
        -----------
        results : dict
            Analysis results
        current_frame : int
            Current frame index
        random_frame : int
            Random frame index
        measure_type : str
            Type of measurement to analyze (Sign Curvature, Normalized Curvature, Intensity)
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        stats_table : QTableWidget, optional
            Table to display statistics
        summary_figure : matplotlib.figure.Figure, optional
            Figure for summary visualization
        summary_canvas : FigureCanvas, optional
            Canvas for summary visualization
        """
        # Verify frames are valid
        if current_frame not in results or random_frame not in results:
            return

        # Skip if frames are the same
        if current_frame == random_frame:
            return

        # Get frame data
        current_results = results[current_frame]
        random_results = results[random_frame]

        # Check if required data is available
        if 'curvatures' not in current_results or 'intensities' not in random_results:
            return

        # Get data based on measure type
        if measure_type == "Sign Curvature":
            current_data = current_results['curvatures'][0]  # Sign curvature
            measurement_label = "Current Frame Sign Curvature"
        elif measure_type == "Normalized Curvature":
            current_data = current_results['curvatures'][2]  # Normalized curvature
            measurement_label = "Current Frame Normalized Curvature"
        elif measure_type == "Intensity":
            current_data = current_results['intensities']
            measurement_label = "Current Frame Intensity"
        else:
            return

        random_data = random_results['intensities']

        # Get valid points (both frames)
        current_valid = current_results.get('valid_points', np.ones(len(current_data), dtype=bool))
        random_valid = random_results.get('valid_points', np.ones(len(random_data), dtype=bool))

        # Use logical AND of valid points
        valid_points = np.logical_and(current_valid, random_valid)

        # Skip if no valid points
        if not np.any(valid_points):
            return

        # Filter data for valid points
        valid_current = current_data[valid_points]
        valid_random = random_data[valid_points]

        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Create scatter plot
        ax.scatter(valid_current, valid_random, alpha=0.7)

        # Add regression line if enough points
        if len(valid_current) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_current, valid_random)

            # Create line data
            x_line = np.linspace(min(valid_current), max(valid_current), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = f"{measurement_label} vs. Random Frame {random_frame+1} Intensity"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_current)}"
        else:
            title = f"{measurement_label} vs. Random Frame {random_frame+1} Intensity\nInsufficient data for regression"

        # Set labels and title
        ax.set_xlabel(measurement_label)
        ax.set_ylabel(f'Random Frame {random_frame+1} Intensity')
        ax.set_title(title)
        if len(valid_current) > 1:
            ax.legend()

        # Refresh canvas
        canvas.draw()

        # Update stats table if provided
        if stats_table is not None:
            stats_table.setRowCount(0)

            stats_table.add_result("Analysis Type", "Random Control Correlation")
            stats_table.add_result("Current Frame", current_frame+1)
            stats_table.add_result("Random Frame", random_frame+1)
            stats_table.add_result("Measurement Type", measure_type)

            if len(valid_current) > 1:
                stats_table.add_result("R-squared", r_value**2)
                stats_table.add_result("p-value", p_value)
                stats_table.add_result("Slope", slope)
                stats_table.add_result("Intercept", intercept)
                stats_table.add_result("Standard Error", std_err)

            stats_table.add_result("Sample Size", len(valid_current))
            stats_table.add_result("Valid Points", f"{np.sum(valid_points)}/{len(valid_points)}")

        # Update summary figure if provided
        if summary_figure is not None and summary_canvas is not None:
            self.plot_random_vs_temporal_summary(results, measure_type,
                                               summary_figure, summary_canvas)

    def plot_temporal_heatmap(self, results, measure_type,
                            figure, canvas,
                            summary_figure=None, summary_canvas=None):
        """
        Plot a temporal heatmap of measurements across all frames

        Parameters:
        -----------
        results : dict
            Analysis results
        measure_type : str
            Type of measurement to analyze (Sign Curvature, Normalized Curvature, Intensity)
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        summary_figure : matplotlib.figure.Figure, optional
            Figure for summary visualization
        summary_canvas : FigureCanvas, optional
            Canvas for summary visualization
        """
        # Collect data from all frames
        frame_data = []
        valid_masks = []
        frame_indices = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            # Get data based on measure type
            if measure_type == "Sign Curvature" and 'curvatures' in frame_results:
                data = frame_results['curvatures'][0]  # Sign curvature
                measurement_label = "Sign Curvature"
            elif measure_type == "Normalized Curvature" and 'curvatures' in frame_results:
                data = frame_results['curvatures'][2]  # Normalized curvature
                measurement_label = "Normalized Curvature"
            elif measure_type == "Intensity" and 'intensities' in frame_results:
                data = frame_results['intensities']
                measurement_label = "Intensity"
            else:
                continue

            # Get valid points
            valid_points = frame_results.get('valid_points', np.ones(len(data), dtype=bool))

            # Add to arrays
            frame_data.append(data)
            valid_masks.append(valid_points)
            frame_indices.append(frame_idx)

        # Skip if no data
        if not frame_data:
            return

        # Create masked arrays
        masked_data = []
        for i in range(len(frame_data)):
            masked_data.append(np.ma.array(frame_data[i], mask=~valid_masks[i]))

        # Convert to a 2D masked array
        data_array = np.ma.stack(masked_data)

        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Create colormap appropriate for the measure type
        if measure_type in ["Sign Curvature", "Normalized Curvature"]:
            cmap = 'coolwarm'
            vmin, vmax = -1, 1
        else:
            cmap = 'viridis'
            vmin = np.nanpercentile(data_array.compressed(), 5)
            vmax = np.nanpercentile(data_array.compressed(), 95)

        # Create heatmap
        im = ax.imshow(data_array, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        # Add colorbar
        figure.colorbar(im, ax=ax, label=measurement_label)

        # Add labels and title
        ax.set_title(f'{measurement_label} Over Time')
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Frame')

        # Set y-ticks to frame indices
        ax.set_yticks(range(len(frame_indices)))
        ax.set_yticklabels([str(i+1) for i in frame_indices])

        # Refresh canvas
        canvas.draw()

        # Update summary figure if provided
        if summary_figure is not None and summary_canvas is not None:
            self.plot_temporal_trend_summary(results, measure_type,
                                          summary_figure, summary_canvas)

    def plot_temporal_summary(self, results, measure_type, figure, canvas):
        """
        Plot a summary of temporal correlations across all frames

        Parameters:
        -----------
        results : dict
            Analysis results
        measure_type : str
            Type of measurement to analyze (Sign Curvature, Normalized Curvature, Intensity)
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        """
        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Calculate correlations for all frame pairs
        correlation_data = []

        for current_frame in range(1, max(results.keys()) + 1):
            if current_frame not in results:
                continue

            previous_frame = current_frame - 1
            if previous_frame not in results:
                continue

            # Get current and previous frame data
            current_results = results[current_frame]
            previous_results = results[previous_frame]

            # Check if required data is available
            if 'curvatures' not in current_results or 'intensities' not in previous_results:
                continue

            # Get data based on measure type
            if measure_type == "Sign Curvature":
                current_data = current_results['curvatures'][0]  # Sign curvature
            elif measure_type == "Normalized Curvature":
                current_data = current_results['curvatures'][2]  # Normalized curvature
            elif measure_type == "Intensity":
                current_data = current_results['intensities']
            else:
                continue

            previous_data = previous_results['intensities']

            # Get valid points (both frames)
            current_valid = current_results.get('valid_points', np.ones(len(current_data), dtype=bool))
            previous_valid = previous_results.get('valid_points', np.ones(len(previous_data), dtype=bool))

            # Use logical AND of valid points
            valid_points = np.logical_and(current_valid, previous_valid)

            # Skip if no valid points
            if not np.any(valid_points):
                continue

            # Filter data for valid points
            valid_current = current_data[valid_points]
            valid_previous = previous_data[valid_points]

            # Calculate correlation if enough points
            if len(valid_current) > 1:
                # Calculate regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_current, valid_previous)

                # Add to correlation data
                correlation_data.append({
                    'current_frame': current_frame,
                    'previous_frame': previous_frame,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'sample_size': len(valid_current)
                })

        # Skip if no correlation data
        if not correlation_data:
            return

        # Create bar plot of R-squared values
        frames = [d['current_frame'] for d in correlation_data]
        r_squared = [d['r_squared'] for d in correlation_data]
        p_values = [d['p_value'] for d in correlation_data]

        # Color bars based on significance
        colors = []
        for p in p_values:
            if p < 0.001:
                colors.append('darkgreen')  # Highly significant
            elif p < 0.01:
                colors.append('green')  # Very significant
            elif p < 0.05:
                colors.append('lightgreen')  # Significant
            else:
                colors.append('gray')  # Not significant

        # Create bar plot
        bars = ax.bar(frames, r_squared, alpha=0.7, color=colors)

        # Add significance stars
        for i, p in enumerate(p_values):
            if p < 0.001:
                ax.text(frames[i], r_squared[i] + 0.02, '***', ha='center')
            elif p < 0.01:
                ax.text(frames[i], r_squared[i] + 0.02, '**', ha='center')
            elif p < 0.05:
                ax.text(frames[i], r_squared[i] + 0.02, '*', ha='center')

        # Add labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel('R² (Coefficient of Determination)')
        ax.set_title(f'Temporal Correlation: {measure_type} vs. Previous Frame Intensity')

        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add legend for significance
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(color='darkgreen', alpha=0.7, label='p < 0.001 (***)'),
            mpatches.Patch(color='green', alpha=0.7, label='p < 0.01 (**)'),
            mpatches.Patch(color='lightgreen', alpha=0.7, label='p < 0.05 (*)'),
            mpatches.Patch(color='gray', alpha=0.7, label='Not significant')
        ]
        ax.legend(handles=legend_elements, loc='best')

        # Set y-range to [0, max_value*1.1]
        max_value = max(r_squared, default=0.1)
        ax.set_ylim([0, max_value * 1.1])

        # Refresh canvas
        canvas.draw()

    def plot_random_vs_temporal_summary(self, results, measure_type, figure, canvas):
        """
        Plot a comparison of random vs temporal correlations

        Parameters:
        -----------
        results : dict
            Analysis results
        measure_type : str
            Type of measurement to analyze (Sign Curvature, Normalized Curvature, Intensity)
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        """
        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Calculate temporal correlations for all sequential frame pairs
        temporal_correlations = []

        for current_frame in range(1, max(results.keys()) + 1):
            if current_frame not in results:
                continue

            previous_frame = current_frame - 1
            if previous_frame not in results:
                continue

            # Get correlation data for this frame pair
            temporal_data = self.calculate_frame_pair_correlation(
                results, current_frame, previous_frame, measure_type)

            if temporal_data:
                temporal_correlations.append(temporal_data)

        # Calculate random correlations for random frame pairs
        random_correlations = []
        available_frames = [f for f in results.keys() if isinstance(f, int)]

        if len(available_frames) >= 2:
            import random

            # Use same number of random pairs as temporal pairs
            for _ in range(len(temporal_correlations)):
                # Select two random frames
                random_pair = random.sample(available_frames, 2)
                current_frame = max(random_pair)
                random_frame = min(random_pair)

                # Skip if same as sequential frames
                if current_frame - random_frame == 1:
                    continue

                # Get correlation data for this frame pair
                random_data = self.calculate_frame_pair_correlation(
                    results, current_frame, random_frame, measure_type)

                if random_data:
                    random_correlations.append(random_data)

        # Skip if insufficient data
        if not temporal_correlations or not random_correlations:
            return

        # Calculate average statistics
        temporal_r2_values = [d['r_squared'] for d in temporal_correlations]
        temporal_avg_r2 = np.mean(temporal_r2_values)
        temporal_std_r2 = np.std(temporal_r2_values)

        random_r2_values = [d['r_squared'] for d in random_correlations]
        random_avg_r2 = np.mean(random_r2_values)
        random_std_r2 = np.std(random_r2_values)

        # Calculate p-value for difference
        t_stat, p_value = stats.ttest_ind(temporal_r2_values, random_r2_values)

        # Create bar plot
        bar_width = 0.35
        index = np.array([0, 1])

        # Create bars
        bar1 = ax.bar(index[0], temporal_avg_r2, bar_width,
                     yerr=temporal_std_r2, label='Temporal',
                     alpha=0.7, color='blue', capsize=7)

        bar2 = ax.bar(index[1], random_avg_r2, bar_width,
                     yerr=random_std_r2, label='Random',
                     alpha=0.7, color='gray', capsize=7)

        # Add labels and title
        ax.set_ylabel('Average R² Value')
        ax.set_title(f'Temporal vs. Random Correlation ({measure_type})')
        ax.set_xticks(index)
        ax.set_xticklabels(['Temporal', 'Random'])
        ax.legend()

        # Add p-value annotation
        stars = ""
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"

        if stars:
            # Add bracket with stars
            ax.plot([0, 1], [temporal_avg_r2 + temporal_std_r2 + 0.03, random_avg_r2 + random_std_r2 + 0.03],
                   'k-', linewidth=1.5)
            ax.text(0.5, max(temporal_avg_r2 + temporal_std_r2, random_avg_r2 + random_std_r2) + 0.05,
                   stars, ha='center', va='bottom', fontsize=14)

        # Add text with p-value and sample sizes
        ax.text(0.5, -0.1,
               f"p-value = {p_value:.4f}\nTemporal n={len(temporal_correlations)}, Random n={len(random_correlations)}",
               ha='center', transform=ax.transAxes)

        # Refresh canvas
        canvas.draw()

    def plot_temporal_trend_summary(self, results, measure_type, figure, canvas):
        """
        Plot a summary of temporal trends in measurements

        Parameters:
        -----------
        results : dict
            Analysis results
        measure_type : str
            Type of measurement to analyze (Sign Curvature, Normalized Curvature, Intensity)
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        """
        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Collect mean and std values per frame
        frame_indices = []
        mean_values = []
        std_values = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            # Get data based on measure type
            if measure_type == "Sign Curvature" and 'curvatures' in frame_results:
                data = frame_results['curvatures'][0]  # Sign curvature
                measurement_label = "Sign Curvature"
            elif measure_type == "Normalized Curvature" and 'curvatures' in frame_results:
                data = frame_results['curvatures'][2]  # Normalized curvature
                measurement_label = "Normalized Curvature"
            elif measure_type == "Intensity" and 'intensities' in frame_results:
                data = frame_results['intensities']
                measurement_label = "Intensity"
            else:
                continue

            # Get valid points
            valid_points = frame_results.get('valid_points', np.ones(len(data), dtype=bool))

            # Calculate statistics for valid points
            if np.any(valid_points):
                valid_data = data[valid_points]
                mean_values.append(np.mean(valid_data))
                std_values.append(np.std(valid_data))
                frame_indices.append(frame_idx)

        # Skip if no data
        if not frame_indices:
            return

        # Plot mean values with error bars
        ax.errorbar(frame_indices, mean_values, yerr=std_values,
                   fmt='o-', capsize=5, linewidth=2, markersize=8)

        # Add regression line if enough points
        if len(frame_indices) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                frame_indices, mean_values)

            # Create line data
            x_line = np.linspace(min(frame_indices), max(frame_indices), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r--', label=f'Trend: R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add annotation if significant
            if p_value < 0.05:
                trend_text = "Significant Increasing Trend" if slope > 0 else "Significant Decreasing Trend"
                ax.text(0.5, 0.95, trend_text,
                       ha='center', transform=ax.transAxes,
                       bbox=dict(facecolor='yellow', alpha=0.3))

        # Add labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'Mean {measurement_label}')
        ax.set_title(f'Temporal Trend in {measurement_label}')

        # Set x-ticks to frame indices
        ax.set_xticks(frame_indices)
        ax.set_xticklabels([str(i+1) for i in frame_indices])

        # Add legend if regression line was added
        if len(frame_indices) > 1:
            ax.legend()

        # Refresh canvas
        canvas.draw()

    def calculate_frame_pair_correlation(self, results, frame1, frame2, measure_type):
        """
        Calculate correlation between measurements in two frames

        Parameters:
        -----------
        results : dict
            Analysis results
        frame1 : int
            First frame index
        frame2 : int
            Second frame index
        measure_type : str
            Type of measurement to analyze (Sign Curvature, Normalized Curvature, Intensity)

        Returns:
        --------
        correlation_data : dict or None
            Dictionary with correlation statistics, or None if calculation failed
        """
        # Check if frames exist
        if frame1 not in results or frame2 not in results:
            return None

        # Get frame data
        frame1_results = results[frame1]
        frame2_results = results[frame2]

        # Check if required data is available
        if 'curvatures' not in frame1_results or 'intensities' not in frame2_results:
            return None

        # Get data based on measure type
        if measure_type == "Sign Curvature":
            frame1_data = frame1_results['curvatures'][0]  # Sign curvature
        elif measure_type == "Normalized Curvature":
            frame1_data = frame1_results['curvatures'][2]  # Normalized curvature
        elif measure_type == "Intensity":
            frame1_data = frame1_results['intensities']
        else:
            return None

        frame2_data = frame2_results['intensities']

        # Get valid points (both frames)
        frame1_valid = frame1_results.get('valid_points', np.ones(len(frame1_data), dtype=bool))
        frame2_valid = frame2_results.get('valid_points', np.ones(len(frame2_data), dtype=bool))

        # Use logical AND of valid points
        valid_points = np.logical_and(frame1_valid, frame2_valid)

        # Skip if no valid points
        if not np.any(valid_points):
            return None

        # Filter data for valid points
        valid_frame1 = frame1_data[valid_points]
        valid_frame2 = frame2_data[valid_points]

        # Calculate correlation if enough points
        if len(valid_frame1) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_frame1, valid_frame2)

            # Return correlation data
            return {
                'frame1': frame1,
                'frame2': frame2,
                'r_squared': r_value**2,
                'p_value': p_value,
                'slope': slope,
                'intercept': intercept,
                'sample_size': len(valid_frame1)
            }

        return None
