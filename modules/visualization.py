import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from skimage import measure, morphology
from scipy import stats
import os
from PyQt5.QtWidgets import QTableWidgetItem

class VisualizationManager:
    """
    Class for visualizing cell curvature, intensity, and edge movement results.
    Provides methods for creating overlays, plots, and summary visualizations.
    """

    def __init__(self):
        """Initialize VisualizationManager"""
        # Create custom colormaps
        self.curvature_cmap = LinearSegmentedColormap.from_list(
            'curvature', ['blue', 'white', 'red'])

    def create_overlay(self, image, mask, min_brightness=0.0, max_brightness=1.0, mask_opacity=0.5):
        """
        Create an RGB overlay with the image in grayscale and mask as a semi-transparent overlay

        Parameters:
        -----------
        image : ndarray
            Original microscope image
        mask : ndarray
            Binary mask
        min_brightness : float
            Minimum brightness value (0-1)
        max_brightness : float
            Maximum brightness value (0-1)
        mask_opacity : float
            Opacity of the mask overlay (0-1)

        Returns:
        --------
        overlay : ndarray
            RGB overlay image
        """
        # Normalize image to 0-1 range
        image_norm = image.astype(float)
        if image_norm.max() > 0:
            image_norm = image_norm / image_norm.max()

        # Apply min/max brightness adjustment
        image_norm = np.clip((image_norm - min_brightness) / (max_brightness - min_brightness + 1e-8), 0, 1)

        # Create an RGB image with grayscale original
        overlay = np.stack([image_norm, image_norm, image_norm], axis=2)

        # Create a red mask
        red_mask = np.zeros_like(overlay)
        red_mask[mask > 0, 0] = 1.0  # Red channel

        # Blend using mask opacity
        overlay = overlay * (1 - mask_opacity * mask[:, :, np.newaxis]) + red_mask * mask_opacity * mask[:, :, np.newaxis]

        return overlay

    def plot_curvature_profile(self, points, curvatures, figure, canvas):
        """
        Plot the curvature profile

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        curvatures : tuple
            Tuple of (sign_curvatures, magnitude_curvatures, normalized_curvatures)
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        """
        sign_curvatures, magnitude_curvatures, normalized_curvatures = curvatures

        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Plot the three curvature types
        ax.plot(sign_curvatures, 'b-', label='Sign')
        ax.plot(normalized_curvatures, 'g-', label='Normalized')

        # Create secondary axis for magnitude (which has a different scale)
        ax2 = ax.twinx()
        ax2.plot(magnitude_curvatures, 'r-', label='Magnitude')

        # Add labels and titles
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Curvature Sign/Normalized')
        ax2.set_ylabel('Curvature Magnitude')
        ax.set_title('Curvature Profile')

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Refresh canvas
        canvas.draw()

    def plot_intensity_profile(self, points, intensities, valid_points, figure, canvas):
        """
        Plot the intensity profile

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        intensities : ndarray
            Array of intensity values
        valid_points : ndarray
            Boolean array indicating which points are valid
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        """
        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Create x-axis (point indices)
        x = np.arange(len(points))

        # Create masked array for invalid points
        masked_intensities = np.ma.array(intensities, mask=~valid_points)

        # Plot intensity profile
        ax.plot(x, masked_intensities, 'b-', label='Intensity')
        ax.scatter(x[valid_points], intensities[valid_points], c='b', s=30, alpha=0.7)

        # Add labels and title
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Intensity')
        ax.set_title('Intensity Profile')
        ax.legend()

        # Refresh canvas
        canvas.draw()

    def plot_correlation(self, curvatures, intensities, valid_points, figure, canvas, title=None):
        """
        Plot correlation between curvature and intensity

        Parameters:
        -----------
        curvatures : ndarray
            Array of curvature values
        intensities : ndarray
            Array of intensity values
        valid_points : ndarray
            Boolean array indicating which points are valid
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        title : str, optional
            Plot title
        """
        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Filter data for valid points
        valid_curvatures = curvatures[valid_points]
        valid_intensities = intensities[valid_points]

        # Plot scatter points
        ax.scatter(valid_curvatures, valid_intensities, alpha=0.7)

        # Add regression line if there are enough points
        if len(valid_curvatures) > 1:
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_curvatures, valid_intensities)

            # Create line data
            x_line = np.linspace(min(valid_curvatures), max(valid_curvatures), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Set title with correlation stats
            if title:
                title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_curvatures)}"
            else:
                title = f"Curvature-Intensity Correlation\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(valid_curvatures)}"
        else:
            if title:
                title += "\nInsufficient data for regression"
            else:
                title = "Curvature-Intensity Correlation\nInsufficient data for regression"

        # Add labels and title
        ax.set_xlabel('Curvature')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(valid_curvatures) > 1:
            ax.legend()

        # Refresh canvas
        canvas.draw()

    def plot_movement_map(self, results, frame, images, masks, figure, canvas, stats_table=None):
        """
        Plot the edge movement map for a specific frame

        Parameters:
        -----------
        results : dict
            Analysis results
        frame : int
            Frame index
        images : ndarray
            Array of microscope images
        masks : ndarray
            Array of binary masks
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        stats_table : QTableWidget, optional
            Table to display statistics
        """
        # Skip if no movement data or invalid frame
        if ('movement_types' not in results or
            frame <= 0 or frame >= len(images)):
            return

        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Get current and previous frame data
        image = images[frame]
        mask = masks[frame]
        prev_mask = masks[frame - 1]

        # Calculate movement map
        diff_mask = mask.astype(int) - prev_mask.astype(int)
        extension_regions = (diff_mask == 1)
        retraction_regions = (diff_mask == -1)

        movement_map = np.zeros_like(mask, dtype=float)
        movement_map[extension_regions] = 1.0  # Extension
        movement_map[retraction_regions] = -1.0  # Retraction

        # Create custom colormap for movement
        movement_cmap = LinearSegmentedColormap.from_list('movement', ['red', 'white', 'blue'])

        # Plot the image
        ax.imshow(image, cmap='gray')

        # Only show movement at cell edges
        edge_movement = np.zeros_like(movement_map)
        dilated_current = morphology.binary_dilation(mask)
        dilated_previous = morphology.binary_dilation(prev_mask)
        edge_regions = np.logical_or(
            np.logical_and(dilated_current, ~mask),
            np.logical_and(dilated_previous, ~prev_mask)
        )
        edge_movement[edge_regions] = movement_map[edge_regions]

        # Plot movement map overlay
        movement_img = ax.imshow(edge_movement, cmap=movement_cmap, alpha=0.7, vmin=-1, vmax=1)

        # Add colorbar
        cbar = figure.colorbar(movement_img, ax=ax)
        cbar.set_label('Edge Movement (Blue=Extending, Red=Retracting)')

        # Get movement type
        if frame - 1 < len(results['movement_types']):
            movement_type = results['movement_types'][frame - 1]
            movement_score = results['movement_scores'][frame - 1]

            # Set color based on movement type
            if movement_type == 'extending':
                color = 'blue'
            elif movement_type == 'retracting':
                color = 'red'
            else:
                color = 'gray'

            # Add title with movement type
            ax.set_title(f'Edge Movement (Frame {frame})\nOverall: {movement_type.upper()}',
                        color=color, fontweight='bold')
        else:
            ax.set_title(f'Edge Movement (Frame {frame})')

        # Hide axes
        ax.axis('off')

        # Add legend
        extension_patch = mpatches.Patch(color='blue', label='Extension')
        retraction_patch = mpatches.Patch(color='red', label='Retraction')
        ax.legend(handles=[extension_patch, retraction_patch], loc='lower right')

        # Update stats table if provided
        if stats_table is not None:
            # Check if the table has the add_result method
            if hasattr(stats_table, 'add_result'):
                stats_table.setRowCount(0)

                # Add movement stats
                if frame - 1 < len(results['movement_types']):
                    movement_type = results['movement_types'][frame - 1]
                    movement_score = results['movement_scores'][frame - 1]

                    stats_table.add_result("Frame", frame)
                    stats_table.add_result("Movement Type", movement_type.capitalize())
                    stats_table.add_result("Movement Score", movement_score)

                    # Add pixel counts
                    extension_pixels = np.sum(extension_regions)
                    retraction_pixels = np.sum(retraction_regions)
                    stats_table.add_result("Extension Pixels", extension_pixels)
                    stats_table.add_result("Retraction Pixels", retraction_pixels)
                    stats_table.add_result("Total Changed Pixels", extension_pixels + retraction_pixels)
            else:
                # Clear table
                stats_table.setRowCount(0)
                stats_table.setColumnCount(2)
                stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])

                # Add movement stats
                if frame - 1 < len(results['movement_types']):
                    movement_type = results['movement_types'][frame - 1]
                    movement_score = results['movement_scores'][frame - 1]

                    self._add_table_row(stats_table, "Frame", str(frame))
                    self._add_table_row(stats_table, "Movement Type", movement_type.capitalize())
                    self._add_table_row(stats_table, "Movement Score", f"{movement_score:.4f}")

                    # Add pixel counts
                    extension_pixels = np.sum(extension_regions)
                    retraction_pixels = np.sum(retraction_regions)
                    self._add_table_row(stats_table, "Extension Pixels", str(extension_pixels))
                    self._add_table_row(stats_table, "Retraction Pixels", str(retraction_pixels))
                    self._add_table_row(stats_table, "Total Changed Pixels", str(extension_pixels + retraction_pixels))

        # Refresh canvas
        canvas.draw()

    def plot_edge_comparison(self, results, frame, images, masks, figure, canvas, stats_table=None):
        """
        Plot a comparison of cell edges between consecutive frames

        Parameters:
        -----------
        results : dict
            Analysis results
        frame : int
            Frame index
        images : ndarray
            Array of microscope images
        masks : ndarray
            Array of binary masks
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        stats_table : QTableWidget, optional
            Table to display statistics
        """
        # Skip if invalid frame
        if frame <= 0 or frame >= len(images):
            return

        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Get current and previous frame data
        image = images[frame]
        mask = masks[frame]
        prev_mask = masks[frame - 1]

        # Plot the original image
        ax.imshow(image, cmap='gray')

        # Create an RGB mask comparison
        mask_vis = np.zeros((image.shape[0], image.shape[1], 3), dtype=float)
        mask_vis[..., 0] = prev_mask * 0.7  # Red channel - previous mask
        mask_vis[..., 1] = mask * 0.7  # Green channel - current mask

        # Apply mask comparison as overlay
        ax.imshow(mask_vis, alpha=0.5)

        # Get contours
        current_contour = measure.find_contours(mask, 0.5)
        prev_contour = measure.find_contours(prev_mask, 0.5)

        # Plot contours
        if current_contour:
            for c in current_contour:
                ax.plot(c[:, 1], c[:, 0], 'g-', linewidth=1.0, alpha=0.8)

        if prev_contour:
            for c in prev_contour:
                ax.plot(c[:, 1], c[:, 0], 'r-', linewidth=1.0, alpha=0.8)

        # Set title
        ax.set_title(f'Edge Comparison (Frames {frame-1} and {frame})\nPrevious (Red), Current (Green)')

        # Hide axes
        ax.axis('off')

        # Add legend
        prev_patch = mpatches.Patch(color='red', label='Previous Mask')
        current_patch = mpatches.Patch(color='green', label='Current Mask')
        overlap_patch = mpatches.Patch(color='yellow', label='Overlap')
        ax.legend(handles=[prev_patch, current_patch, overlap_patch], loc='lower right')

        # Update stats table if provided
        if stats_table is not None:
            # Check if the table has the add_result method
            if hasattr(stats_table, 'add_result'):
                stats_table.setRowCount(0)

                # Add basic information
                stats_table.add_result("Current Frame", frame)
                stats_table.add_result("Previous Frame", frame-1)

                # Add area stats
                current_area = np.sum(mask)
                prev_area = np.sum(prev_mask)
                area_change = int(current_area) - int(prev_area)
                area_change_percent = (area_change / prev_area * 100) if prev_area > 0 else 0

                stats_table.add_result("Current Area (pixels)", current_area)
                stats_table.add_result("Previous Area (pixels)", prev_area)
                stats_table.add_result("Area Change (pixels)", area_change)
                stats_table.add_result("Area Change (%)", area_change_percent)

                # Add overlap stats
                overlap = np.sum(np.logical_and(mask, prev_mask))
                overlap_percent = (overlap / prev_area * 100) if prev_area > 0 else 0

                stats_table.add_result("Overlap Area (pixels)", overlap)
                stats_table.add_result("Overlap (%)", overlap_percent)
            else:
                # Clear table
                stats_table.setRowCount(0)
                stats_table.setColumnCount(2)
                stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])

                # Add basic information
                self._add_table_row(stats_table, "Current Frame", str(frame))
                self._add_table_row(stats_table, "Previous Frame", str(frame-1))

                # Add area stats
                current_area = np.sum(mask)
                prev_area = np.sum(prev_mask)
                area_change = int(current_area) - int(prev_area)
                area_change_percent = (area_change / prev_area * 100) if prev_area > 0 else 0

                self._add_table_row(stats_table, "Current Area (pixels)", str(current_area))
                self._add_table_row(stats_table, "Previous Area (pixels)", str(prev_area))
                self._add_table_row(stats_table, "Area Change (pixels)", str(area_change))
                self._add_table_row(stats_table, "Area Change (%)", f"{area_change_percent:.2f}")

                # Add overlap stats
                overlap = np.sum(np.logical_and(mask, prev_mask))
                overlap_percent = (overlap / prev_area * 100) if prev_area > 0 else 0

                self._add_table_row(stats_table, "Overlap Area (pixels)", str(overlap))
                self._add_table_row(stats_table, "Overlap (%)", f"{overlap_percent:.2f}")

        # Refresh canvas
        canvas.draw()

    def _add_table_row(self, table, parameter, value):
        """Helper method to add a row to a QTableWidget"""
        row = table.rowCount()
        table.insertRow(row)

        parameter_item = QTableWidgetItem(parameter)
        value_item = QTableWidgetItem(value)

        table.setItem(row, 0, parameter_item)
        table.setItem(row, 1, value_item)

    def plot_movement_over_time(self, results, figure, canvas, summary_figure=None, summary_canvas=None, stats_table=None):
        """
        Plot movement scores over time

        Parameters:
        -----------
        results : dict
            Analysis results
        figure : matplotlib.figure.Figure
            Matplotlib figure to plot on
        canvas : FigureCanvas
            Canvas to render the figure
        summary_figure : matplotlib.figure.Figure, optional
            Figure for summary visualization
        summary_canvas : FigureCanvas, optional
            Canvas for summary visualization
        stats_table : QTableWidget, optional
            Table to display statistics
        """
        # Skip if no movement data
        if 'movement_scores' not in results or 'movement_types' not in results:
            return

        movement_scores = results['movement_scores']
        movement_types = results['movement_types']

        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)

        # Convert movement types to colors
        movement_colors = []
        for movement_type in movement_types:
            if movement_type == 'extending':
                movement_colors.append('blue')
            elif movement_type == 'retracting':
                movement_colors.append('red')
            else:
                movement_colors.append('gray')

        # Plot movement scores
        frame_indices = range(len(movement_scores))
        ax.bar(frame_indices, movement_scores, color=movement_colors, alpha=0.7)

        # Add horizontal line at zero
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Add thresholds for classification
        ax.axhline(y=0.1, color='blue', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.3)

        # Annotate bars with movement type
        for i, score in enumerate(movement_scores):
            offset = 0.05 if score >= 0 else -0.15
            ax.text(i, score + offset, movement_types[i],
                  ha='center', rotation=90, fontsize=8, fontweight='bold',
                  color=movement_colors[i])

        # Add legend
        extending_patch = mpatches.Patch(color='blue', label='Extending')
        retracting_patch = mpatches.Patch(color='red', label='Retracting')
        stable_patch = mpatches.Patch(color='gray', label='Stable')
        ax.legend(handles=[extending_patch, retracting_patch, stable_patch])

        # Set labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel('Movement Score')
        ax.set_title('Cell Edge Movement Over Time')

        # Set x-ticks to frame indices
        ax.set_xticks(frame_indices)
        ax.set_xticklabels([str(i+2) for i in frame_indices])  # +2 because first frame is 0, and movement is for frame transitions

        # Refresh canvas
        canvas.draw()

        # Create summary pie chart if summary figure is provided
        if summary_figure is not None and summary_canvas is not None:
            # Count movement types
            extending_count = movement_types.count('extending')
            retracting_count = movement_types.count('retracting')
            stable_count = movement_types.count('stable')

            # Clear figure
            summary_figure.clear()
            ax = summary_figure.add_subplot(111)

            # Create pie chart
            labels = ['Extending', 'Retracting', 'Stable']
            sizes = [extending_count, retracting_count, stable_count]
            colors = ['blue', 'red', 'gray']
            explode = (0.1, 0.1, 0.1)

            # Only include non-zero values
            filtered_labels = []
            filtered_sizes = []
            filtered_colors = []
            filtered_explode = []

            for i in range(len(sizes)):
                if sizes[i] > 0:
                    filtered_labels.append(labels[i])
                    filtered_sizes.append(sizes[i])
                    filtered_colors.append(colors[i])
                    filtered_explode.append(explode[i])

            if filtered_sizes:
                wedges, texts, autotexts = ax.pie(
                    filtered_sizes, explode=filtered_explode, labels=filtered_labels,
                    colors=filtered_colors, autopct='%1.1f%%', shadow=True, startangle=90)

                # Set font properties for text elements
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    autotext.set_fontweight('bold')

            # Set title
            total_frames = extending_count + retracting_count + stable_count
            predominant = max(['Extending', 'Retracting', 'Stable'],
                             key=lambda x: [extending_count, retracting_count, stable_count][['Extending', 'Retracting', 'Stable'].index(x)])

            if total_frames > 0:
                predominant_percentage = sizes[labels.index(predominant)]/total_frames*100
                ax.set_title(f'Movement Type Distribution\nPredominant: {predominant} ({predominant_percentage:.1f}% of frames)')
            else:
                ax.set_title('Movement Type Distribution')

            # Equal aspect ratio
            ax.axis('equal')

            # Refresh canvas
            summary_canvas.draw()

        # Update stats table if provided
        if stats_table is not None:
            # Add movement type counts
            extending_count = movement_types.count('extending')
            retracting_count = movement_types.count('retracting')
            stable_count = movement_types.count('stable')
            total_frames = extending_count + retracting_count + stable_count

            # Check if the table has the add_result method
            if hasattr(stats_table, 'add_result'):
                stats_table.setRowCount(0)

                stats_table.add_result("Total Frames with Movement", total_frames)
                stats_table.add_result("Extending Frames", extending_count)
                stats_table.add_result("Extending (%)", extending_count / total_frames * 100 if total_frames > 0 else 0)
                stats_table.add_result("Retracting Frames", retracting_count)
                stats_table.add_result("Retracting (%)", retracting_count / total_frames * 100 if total_frames > 0 else 0)
                stats_table.add_result("Stable Frames", stable_count)
                stats_table.add_result("Stable (%)", stable_count / total_frames * 100 if total_frames > 0 else 0)

                # Add movement score stats
                if movement_scores:
                    stats_table.add_result("Average Movement Score", np.mean(movement_scores))
                    stats_table.add_result("Movement Score Std Dev", np.std(movement_scores))
                    stats_table.add_result("Max Extension Score", np.max(movement_scores))
                    stats_table.add_result("Max Retraction Score", np.min(movement_scores))

                    # Calculate movement trends
                    direction_changes = 0
                    for i in range(1, len(movement_scores)):
                        if np.sign(movement_scores[i]) != np.sign(movement_scores[i-1]):
                            direction_changes += 1

                    stats_table.add_result("Direction Changes", direction_changes)
                    stats_table.add_result("Direction Changes (%)", direction_changes / (len(movement_scores) - 1) * 100 if len(movement_scores) > 1 else 0)
            else:
                # Clear table
                stats_table.setRowCount(0)
                stats_table.setColumnCount(2)
                stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])

                self._add_table_row(stats_table, "Total Frames with Movement", str(total_frames))
                self._add_table_row(stats_table, "Extending Frames", str(extending_count))
                self._add_table_row(stats_table, "Extending (%)", f"{extending_count / total_frames * 100:.2f}" if total_frames > 0 else "0.00")
                self._add_table_row(stats_table, "Retracting Frames", str(retracting_count))
                self._add_table_row(stats_table, "Retracting (%)", f"{retracting_count / total_frames * 100:.2f}" if total_frames > 0 else "0.00")
                self._add_table_row(stats_table, "Stable Frames", str(stable_count))
                self._add_table_row(stats_table, "Stable (%)", f"{stable_count / total_frames * 100:.2f}" if total_frames > 0 else "0.00")

                # Add movement score stats
                if movement_scores:
                    self._add_table_row(stats_table, "Average Movement Score", f"{np.mean(movement_scores):.4f}")
                    self._add_table_row(stats_table, "Movement Score Std Dev", f"{np.std(movement_scores):.4f}")
                    self._add_table_row(stats_table, "Max Extension Score", f"{np.max(movement_scores):.4f}")
                    self._add_table_row(stats_table, "Max Retraction Score", f"{np.min(movement_scores):.4f}")

                    # Calculate movement trends
                    direction_changes = 0
                    for i in range(1, len(movement_scores)):
                        if np.sign(movement_scores[i]) != np.sign(movement_scores[i-1]):
                            direction_changes += 1

                    self._add_table_row(stats_table, "Direction Changes", str(direction_changes))
                change_percent = direction_changes / (len(movement_scores) - 1) * 100 if len(movement_scores) > 1 else 0
                self._add_table_row(stats_table, "Direction Changes (%)", f"{change_percent:.2f}")

    def update_overlay_display(self):
        """Update the overlay display based on selected options"""
        # Check if data is loaded
        if not self.loaded_data:
            return

        # Get current frame
        frame = self.current_frame

        # Check if we have results for this frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            frame not in self.loaded_data['results']):
            return

        # Get frame results
        frame_results = self.loaded_data['results'][frame]

        # Clear overlay canvas
        self.overlay_canvas.clear()

        # Add background image if available
        if 'images' in self.loaded_data and self.loaded_data['images'] is not None:
            images = self.loaded_data['images']
            if frame < images.shape[0]:
                # Get brightness settings
                min_brightness = self.min_brightness_slider.value() / 100.0
                max_brightness = self.max_brightness_slider.value() / 100.0

                # Apply brightness adjustment to background image
                image = images[frame].copy().astype(float)
                if np.max(image) > 0:
                    image = image / np.max(image)
                image = np.clip((image - min_brightness) / (max_brightness - min_brightness + 1e-8), 0, 1)

                self.overlay_canvas.add_background(image)

        # Add contour if available and checkbox is checked
        if (self.show_contour_cb.isChecked() and
            'contour_data' in frame_results):
            self.overlay_canvas.add_contour(
                frame_results['contour_data'],
                color='y',
                linewidth=self.contour_width_spin.value()
            )

        # Add curvature points if available and checkbox is checked
        if (self.show_curvature_cb.isChecked() and
            'points' in frame_results and
            'curvatures' in frame_results):
            self.overlay_canvas.add_curvature_points(
                frame_results['points'],
                frame_results['curvatures'][2],  # Use normalized curvature
                cmap=self.curvature_cmap_combo.currentText(),
                marker_size=self.marker_size_spin.value()
            )

        # Add intensity points if available and checkbox is checked
        if (self.show_intensity_cb.isChecked() and
            'points' in frame_results and
            'intensities' in frame_results):
            self.overlay_canvas.add_intensity_points(
                frame_results['points'],
                frame_results['intensities'],
                cmap=self.intensity_cmap_combo.currentText(),
                marker_size=self.marker_size_spin.value()
            )

        # Add sampling regions if available and checkbox is checked
        if (self.show_sampling_cb.isChecked() and
            'sampling_regions' in frame_results and
            'valid_points' in frame_results):
            self.overlay_canvas.add_sampling_regions(
                frame_results['sampling_regions'],
                frame_results['valid_points']
            )

        # Update canvas
        self.overlay_canvas.update_canvas()

        # Set title
        self.overlay_canvas.set_title(f"Analysis Overlay (Frame {frame+1})")
