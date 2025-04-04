import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

class ExportManager:
    """
    Class for exporting analysis results
    """

    def __init__(self):
        """Initialize ExportManager"""
        pass

    def export_data(self, data, output_dir, prefix="cell_analysis_",
                  export_images=True, export_raw_data=True, export_stats=True,
                  export_figures=True, export_report=True, progress_callback=None):
        """
        Export data and results

        Parameters:
        -----------
        data : dict
            Dictionary of loaded data and results
        output_dir : str
            Directory to save exported data
        prefix : str
            Prefix for output files
        export_images : bool
            Whether to export images
        export_raw_data : bool
            Whether to export raw data
        export_stats : bool
            Whether to export statistics
        export_figures : bool
            Whether to export figures
        export_report : bool
            Whether to export summary report
        progress_callback : function
            Callback function for progress updates (percentage, message)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize progress
        progress = 0
        if progress_callback:
            progress_callback(progress, "Starting export")

        # Export images
        if export_images and 'images' in data and data['images'] is not None:
            if progress_callback:
                progress_callback(progress, "Exporting images")

            self.export_images(data, output_dir, prefix)
            progress += 20
            if progress_callback:
                progress_callback(progress, "Images exported")

        # Export raw data
        if export_raw_data and 'results' in data and data['results'] is not None:
            if progress_callback:
                progress_callback(progress, "Exporting raw data")

            self.export_raw_data(data['results'], output_dir, prefix)
            progress += 20
            if progress_callback:
                progress_callback(progress, "Raw data exported")

        # Export statistics
        if export_stats and 'results' in data and data['results'] is not None:
            if progress_callback:
                progress_callback(progress, "Exporting statistics")

            self.export_statistics(data['results'], output_dir, prefix)
            progress += 20
            if progress_callback:
                progress_callback(progress, "Statistics exported")

        # Export figures
        if export_figures and 'results' in data and data['results'] is not None:
            if progress_callback:
                progress_callback(progress, "Exporting figures")

            self.export_figures(data['results'], output_dir, prefix)
            progress += 20
            if progress_callback:
                progress_callback(progress, "Figures exported")

        # Export summary report
        if export_report and 'results' in data and data['results'] is not None:
            if progress_callback:
                progress_callback(progress, "Generating summary report")

            self.export_summary_report(data, output_dir, prefix)
            progress += 20
            if progress_callback:
                progress_callback(progress, "Summary report generated")

        # Complete
        if progress_callback:
            progress_callback(100, "Export completed")

    def export_images(self, data, output_dir, prefix):
        """
        Export original images and overlays

        Parameters:
        -----------
        data : dict
            Dictionary of loaded data and results
        output_dir : str
            Directory to save exported data
        prefix : str
            Prefix for output files
        """
        # Create subdirectories
        original_dir = os.path.join(output_dir, f"{prefix}original_images")
        mask_dir = os.path.join(output_dir, f"{prefix}masks")
        overlay_dir = os.path.join(output_dir, f"{prefix}overlays")

        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        # Get images and masks
        images = data.get('images')
        masks = data.get('masks')

        if images is None:
            return

        # Export images
        for i in range(images.shape[0]):
            # Save original image
            plt.imsave(os.path.join(original_dir, f"frame_{i+1:04d}.png"), images[i], cmap='gray')

            # Save mask if available
            if masks is not None and i < masks.shape[0]:
                plt.imsave(os.path.join(mask_dir, f"frame_{i+1:04d}.png"), masks[i], cmap='gray')

                # Create and save overlay
                overlay = self.create_overlay(images[i], masks[i])
                plt.imsave(os.path.join(overlay_dir, f"frame_{i+1:04d}.png"), overlay)

    def export_raw_data(self, results, output_dir, prefix):
        """
        Export raw data as CSV files

        Parameters:
        -----------
        results : dict
            Analysis results
        output_dir : str
            Directory to save exported data
        prefix : str
            Prefix for output files
        """
        # Extract frame data and combine into a DataFrame
        frames_data = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            # Extract key data from frame results
            if ('points' in frame_results and
                'curvatures' in frame_results and
                'intensities' in frame_results and
                'valid_points' in frame_results):

                # Get curvatures
                sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

                # Get data
                points = frame_results['points']
                intensities = frame_results['intensities']
                valid_points = frame_results['valid_points']

                # Add to frames data
                for i in range(len(points)):
                    frames_data.append({
                        'frame': frame_idx,
                        'point_index': i,
                        'x': points[i, 1],  # x coordinate
                        'y': points[i, 0],  # y coordinate
                        'valid': valid_points[i],
                        'curvature_sign': sign_curvatures[i],
                        'curvature_magnitude': magnitude_curvatures[i],
                        'curvature_normalized': normalized_curvatures[i],
                        'intensity': intensities[i]
                    })

                # Save individual frame data
                frame_df = pd.DataFrame([
                    {
                        'point_index': i,
                        'x': points[i, 1],
                        'y': points[i, 0],
                        'valid': valid_points[i],
                        'curvature_sign': sign_curvatures[i],
                        'curvature_magnitude': magnitude_curvatures[i],
                        'curvature_normalized': normalized_curvatures[i],
                        'intensity': intensities[i]
                    }
                    for i in range(len(points))
                ])

                frame_df.to_csv(os.path.join(output_dir, f"{prefix}frame_{frame_idx+1:04d}_data.csv"), index=False)

        # Save combined data
        if frames_data:
            combined_df = pd.DataFrame(frames_data)
            combined_df.to_csv(os.path.join(output_dir, f"{prefix}combined_data.csv"), index=False)

        # Save movement data if available
        if 'movement_scores' in results and 'movement_types' in results:
            movement_data = []
            for i, (score, movement_type) in enumerate(zip(results['movement_scores'], results['movement_types'])):
                movement_data.append({
                    'frame': i+2,  # +2 because first frame is 0, and movement is for frame transitions
                    'movement_score': score,
                    'movement_type': movement_type
                })

            movement_df = pd.DataFrame(movement_data)
            movement_df.to_csv(os.path.join(output_dir, f"{prefix}movement_data.csv"), index=False)

    def export_statistics(self, results, output_dir, prefix):
        """
        Export statistics as JSON and text files

        Parameters:
        -----------
        results : dict
            Analysis results
        output_dir : str
            Directory to save exported data
        prefix : str
            Prefix for output files
        """
        # Convert results to serializable format
        serializable_results = self.convert_results_for_json(results)

        # Save JSON
        with open(os.path.join(output_dir, f"{prefix}metadata.json"), 'w') as f:
            json.dump(serializable_results, f, indent=4)

        # Create human-readable text file
        with open(os.path.join(output_dir, f"{prefix}metadata.txt"), 'w') as f:
            f.write("===== CELL CURVATURE AND INTENSITY ANALYSIS METADATA =====\n\n")

            # Add timestamp
            if 'timestamp' in results:
                f.write(f"Analysis performed: {results['timestamp']}\n\n")

            # Add parameters
            if 'parameters' in results:
                f.write("PARAMETERS:\n")
                for key, value in results['parameters'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

            # Add statistics
            if 'statistics' in results:
                f.write("ANALYSIS STATISTICS:\n")
                self.write_stats_recursively(f, results['statistics'])
                f.write("\n")

        # Save correlation statistics separately
        if 'statistics' in results:
            stats = results['statistics']
            correlation_stats = {}

            # Extract correlation statistics
            if 'sign_curvature_correlation' in stats:
                correlation_stats['sign_curvature'] = stats['sign_curvature_correlation']

            if 'normalized_curvature_correlation' in stats:
                correlation_stats['normalized_curvature'] = stats['normalized_curvature_correlation']

            # Add movement correlation stats if available
            if 'movement_grouped_correlation' in stats:
                correlation_stats['movement_grouped'] = stats['movement_grouped_correlation']

            # Save as JSON
            with open(os.path.join(output_dir, f"{prefix}correlation_stats.json"), 'w') as f:
                json.dump(correlation_stats, f, indent=4)

    def write_stats_recursively(self, file, stats, indent=""):
        """
        Recursively write statistics to a text file

        Parameters:
        -----------
        file : file
            Open file to write to
        stats : dict
            Statistics dictionary
        indent : str
            Indentation string
        """
        for key, value in stats.items():
            if isinstance(value, dict):
                file.write(f"{indent}- {key}:\n")
                self.write_stats_recursively(file, value, indent + "  ")
            else:
                file.write(f"{indent}- {key}: {value}\n")

    def export_figures(self, results, output_dir, prefix):
        """
        Export summary figures

        Parameters:
        -----------
        results : dict
            Analysis results
        output_dir : str
            Directory to save exported data
        prefix : str
            Prefix for output files
        """
        # Create summary correlation figures
        self.create_summary_correlation_figure(results, os.path.join(output_dir, f"{prefix}summary_correlation.png"))

        # Create summary curvature figure
        self.create_summary_curvature_figure(results, os.path.join(output_dir, f"{prefix}summary_curvature.png"))

        # Create summary intensity figure
        self.create_summary_intensity_figure(results, os.path.join(output_dir, f"{prefix}summary_intensity.png"))

        # Create movement summary figure if applicable
        if 'movement_scores' in results and 'movement_types' in results:
            self.create_movement_summary_figure(results, os.path.join(output_dir, f"{prefix}movement_summary.png"))

    def create_summary_correlation_figure(self, results, output_path):
        """
        Create a summary correlation figure

        Parameters:
        -----------
        results : dict
            Analysis results
        output_path : str
            Path to save figure
        """
        # Collect correlation data
        curvatures = []
        intensities = []
        valid_points = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            if ('curvatures' in frame_results and
                'intensities' in frame_results and
                'valid_points' in frame_results):

                # Get curvatures (use normalized)
                _, _, normalized_curvatures = frame_results['curvatures']

                # Store data
                curvatures.append(normalized_curvatures)
                intensities.append(frame_results['intensities'])
                valid_points.append(frame_results['valid_points'])

        # Skip if no data
        if not curvatures:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Combine data from all frames
        combined_curvatures = []
        combined_intensities = []

        for frame_idx in range(len(curvatures)):
            curr_curvatures = curvatures[frame_idx]
            curr_intensities = intensities[frame_idx]
            curr_valid_points = valid_points[frame_idx]

            # Filter for valid points
            valid_curvatures = curr_curvatures[curr_valid_points]
            valid_intensities = curr_intensities[curr_valid_points]

            combined_curvatures.extend(valid_curvatures)
            combined_intensities.extend(valid_intensities)

        # Convert to numpy arrays
        combined_curvatures = np.array(combined_curvatures)
        combined_intensities = np.array(combined_intensities)

        # Create scatter plot
        ax.scatter(combined_curvatures, combined_intensities, alpha=0.5)

        # Add regression line if enough points
        if len(combined_curvatures) > 1:
            from scipy import stats

            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                combined_curvatures, combined_intensities)

            # Create line data
            x_line = np.linspace(min(combined_curvatures), max(combined_curvatures), 100)
            y_line = slope * x_line + intercept

            # Plot line
            ax.plot(x_line, y_line, 'r-', linewidth=2,
                   label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

            # Add stats to title
            title = "Curvature-Intensity Correlation (All Frames)"
            title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_curvatures)}"
        else:
            title = "Curvature-Intensity Correlation (All Frames)\nInsufficient data for regression"

        # Set labels and title
        ax.set_xlabel('Normalized Curvature')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if len(combined_curvatures) > 1:
            ax.legend()

        # Save figure
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def create_summary_curvature_figure(self, results, output_path):
        """
        Create a summary curvature figure

        Parameters:
        -----------
        results : dict
            Analysis results
        output_path : str
            Path to save figure
        """
        # Collect curvature data
        all_sign_curvatures = []
        all_normalized_curvatures = []
        frame_indices = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            if 'curvatures' in frame_results:
                sign_curvatures, _, normalized_curvatures = frame_results['curvatures']

                all_sign_curvatures.append(sign_curvatures)
                all_normalized_curvatures.append(normalized_curvatures)
                frame_indices.append(frame_idx)

        # Skip if no data
        if not all_sign_curvatures:
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot sign curvature heatmap
        sign_array = np.array(all_sign_curvatures)
        sign_img = ax1.imshow(sign_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(sign_img, ax=ax1, label='Curvature Sign')

        # Plot normalized curvature heatmap
        norm_array = np.array(all_normalized_curvatures)
        norm_img = ax2.imshow(norm_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(norm_img, ax=ax2, label='Normalized Curvature')

        # Add labels and titles
        ax1.set_title('Sign Curvature Across Frames')
        ax2.set_title('Normalized Curvature Across Frames')

        ax1.set_ylabel('Frame')
        ax2.set_ylabel('Frame')
        ax2.set_xlabel('Point Index')

        # Set y-ticks to frame indices
        ax1.set_yticks(range(len(frame_indices)))
        ax1.set_yticklabels([str(i+1) for i in frame_indices])
        ax2.set_yticks(range(len(frame_indices)))
        ax2.set_yticklabels([str(i+1) for i in frame_indices])

        # Add overall title
        plt.suptitle('Curvature Analysis Summary', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save figure
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def create_summary_intensity_figure(self, results, output_path):
        """
        Create a summary intensity figure

        Parameters:
        -----------
        results : dict
            Analysis results
        output_path : str
            Path to save figure
        """
        # Collect intensity data
        all_intensities = []
        all_valid_points = []
        frame_indices = []

        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            if 'intensities' in frame_results and 'valid_points' in frame_results:
                intensities = frame_results['intensities']
                valid_points = frame_results['valid_points']

                # Create masked array
                masked_intensities = np.ma.array(intensities, mask=~valid_points)

                all_intensities.append(masked_intensities)
                all_valid_points.append(valid_points)
                frame_indices.append(frame_idx)

        # Skip if no data
        if not all_intensities:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot intensity heatmap
        intensity_array = np.ma.stack(all_intensities)

        # Calculate min and max for better color scaling
        # Using percentiles to avoid extreme outliers affecting the scale
        intensity_min = np.nanpercentile(intensity_array.compressed(), 5)
        intensity_max = np.nanpercentile(intensity_array.compressed(), 95)

        intensity_img = ax.imshow(intensity_array, cmap='viridis', aspect='auto',
                                 vmin=intensity_min, vmax=intensity_max)
        plt.colorbar(intensity_img, ax=ax, label='Intensity')

        # Add labels and title
        ax.set_title('Intensity Measurements Across Frames')
        ax.set_ylabel('Frame')
        ax.set_xlabel('Point Index')

        # Set y-ticks to frame indices
        ax.set_yticks(range(len(frame_indices)))
        ax.set_yticklabels([str(i+1) for i in frame_indices])

        # Save figure
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def create_movement_summary_figure(self, results, output_path):
        """
        Create a movement summary figure

        Parameters:
        -----------
        results : dict
            Analysis results
        output_path : str
            Path to save figure
        """
        # Check if movement data is available
        if 'movement_scores' not in results or 'movement_types' not in results:
            return

        movement_scores = results['movement_scores']
        movement_types = results['movement_types']

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

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
        import matplotlib.patches as mpatches
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
        ax.set_xticklabels([str(i+2) for i in frame_indices])

        # Save figure
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def export_summary_report(self, data, output_dir, prefix):
        """
        Export a comprehensive summary report as PDF

        Parameters:
        -----------
        data : dict
            Dictionary of loaded data and results
        output_dir : str
            Directory to save exported data
        prefix : str
            Prefix for output files
        """
        # Create PDF file
        pdf_path = os.path.join(output_dir, f"{prefix}summary_report.pdf")

        with PdfPages(pdf_path) as pdf:
            # Add title page
            self.add_title_page(pdf, data)

            # Add parameters page
            self.add_parameters_page(pdf, data)

            # Add correlation analysis page
            self.add_correlation_page(pdf, data)

            # Add curvature analysis page
            self.add_curvature_page(pdf, data)

            # Add intensity analysis page
            self.add_intensity_page(pdf, data)

            # Add movement analysis page if applicable
            if 'results' in data and 'movement_scores' in data['results']:
                self.add_movement_page(pdf, data)

            # Add sample frames page
            self.add_sample_frames_page(pdf, data)

    def add_title_page(self, pdf, data):
        """
        Add title page to PDF report

        Parameters:
        -----------
        pdf : PdfPages
            PDF document
        data : dict
            Dictionary of loaded data and results
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))

        # Hide axes
        ax.axis('off')

        # Add title
        ax.text(0.5, 0.8, "Cell Curvature & PIEZO1 Analysis",
               ha='center', fontsize=24, fontweight='bold')

        # Add subtitle
        ax.text(0.5, 0.7, "Microscopy Image Analysis Report",
               ha='center', fontsize=18)

        # Add timestamp
        timestamp = plt.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.6, f"Generated on {timestamp}",
               ha='center', fontsize=12)

        # Add file information
        if 'image_path' in data:
            image_path = data['image_path']
            ax.text(0.5, 0.5, f"Image Stack: {os.path.basename(image_path)}",
                   ha='center', fontsize=10)

        if 'mask_path' in data:
            mask_path = data['mask_path']
            ax.text(0.5, 0.45, f"Mask Stack: {os.path.basename(mask_path)}",
                   ha='center', fontsize=10)

        # Add frame count
        if 'images' in data and data['images'] is not None:
            frame_count = data['images'].shape[0]
            ax.text(0.5, 0.4, f"Total Frames: {frame_count}",
                   ha='center', fontsize=10)

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    def add_parameters_page(self, pdf, data):
        """
        Add parameters page to PDF report

        Parameters:
        -----------
        pdf : PdfPages
            PDF document
        data : dict
            Dictionary of loaded data and results
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))

        # Hide axes
        ax.axis('off')

        # Add title
        ax.text(0.5, 0.95, "Analysis Parameters",
               ha='center', fontsize=18, fontweight='bold')

        # Add parameters
        if 'results' in data and 'parameters' in data['results']:
            parameters = data['results']['parameters']

            y_pos = 0.85
            for key, value in parameters.items():
                ax.text(0.3, y_pos, f"{key}:", fontsize=12, fontweight='bold')
                ax.text(0.7, y_pos, str(value), fontsize=12)
                y_pos -= 0.05

        # Add statistics
        if 'results' in data and 'statistics' in data['results']:
            stats = data['results']['statistics']

            # Add section title
            ax.text(0.5, 0.6, "Overall Statistics",
                   ha='center', fontsize=14, fontweight='bold')

            y_pos = 0.55

            # Add frame statistics
            ax.text(0.3, y_pos, "Total Frames:", fontsize=12, fontweight='bold')
            ax.text(0.7, y_pos, str(stats.get('total_frames', 'N/A')), fontsize=12)
            y_pos -= 0.04

            ax.text(0.3, y_pos, "Processed Frames:", fontsize=12, fontweight='bold')
            ax.text(0.7, y_pos, str(stats.get('processed_frames', 'N/A')), fontsize=12)
            y_pos -= 0.04

            ax.text(0.3, y_pos, "Frames with No Contour:", fontsize=12, fontweight='bold')
            ax.text(0.7, y_pos, str(stats.get('frames_with_no_contour', 'N/A')), fontsize=12)
            y_pos -= 0.04

            # Add point statistics
            ax.text(0.3, y_pos, "Total Points:", fontsize=12, fontweight='bold')
            ax.text(0.7, y_pos, str(stats.get('total_points', 'N/A')), fontsize=12)
            y_pos -= 0.04

            ax.text(0.3, y_pos, "Valid Points:", fontsize=12, fontweight='bold')
            ax.text(0.7, y_pos, str(stats.get('valid_points', 'N/A')), fontsize=12)
            y_pos -= 0.04

            ax.text(0.3, y_pos, "Invalid Points:", fontsize=12, fontweight='bold')
            ax.text(0.7, y_pos, str(stats.get('invalid_points', 'N/A')), fontsize=12)
            y_pos -= 0.04

            ax.text(0.3, y_pos, "Avg. Valid Points/Frame:", fontsize=12, fontweight='bold')
            ax.text(0.7, y_pos, f"{stats.get('average_valid_points_per_frame', 0):.2f}", fontsize=12)
            y_pos -= 0.04

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    def add_correlation_page(self, pdf, data):
        """
        Add correlation analysis page to PDF report

        Parameters:
        -----------
        pdf : PdfPages
            PDF document
        data : dict
            Dictionary of loaded data and results
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))

        # Add title
        fig.suptitle("Correlation Analysis", fontsize=18, fontweight='bold', y=0.98)

        # Add correlation plot
        if 'results' in data:
            results = data['results']

            # Collect correlation data
            curvatures = []
            intensities = []
            valid_points = []

            for frame_idx, frame_results in results.items():
                if not isinstance(frame_idx, int):
                    continue

                if ('curvatures' in frame_results and
                    'intensities' in frame_results and
                    'valid_points' in frame_results):

                    # Get curvatures (use normalized)
                    _, _, normalized_curvatures = frame_results['curvatures']

                    # Store data
                    curvatures.append(normalized_curvatures)
                    intensities.append(frame_results['intensities'])
                    valid_points.append(frame_results['valid_points'])

            # Create scatter plot if data available
            if curvatures:
                # Combine data from all frames
                combined_curvatures = []
                combined_intensities = []

                for frame_idx in range(len(curvatures)):
                    curr_curvatures = curvatures[frame_idx]
                    curr_intensities = intensities[frame_idx]
                    curr_valid_points = valid_points[frame_idx]

                    # Filter for valid points
                    valid_curvatures = curr_curvatures[curr_valid_points]
                    valid_intensities = curr_intensities[curr_valid_points]

                    combined_curvatures.extend(valid_curvatures)
                    combined_intensities.extend(valid_intensities)

                # Convert to numpy arrays
                combined_curvatures = np.array(combined_curvatures)
                combined_intensities = np.array(combined_intensities)

                # Create scatter plot
                ax1.scatter(combined_curvatures, combined_intensities, alpha=0.5)

                # Add regression line if enough points
                if len(combined_curvatures) > 1:
                    from scipy import stats

                    # Calculate regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        combined_curvatures, combined_intensities)

                    # Create line data
                    x_line = np.linspace(min(combined_curvatures), max(combined_curvatures), 100)
                    y_line = slope * x_line + intercept

                    # Plot line
                    ax1.plot(x_line, y_line, 'r-', linewidth=2,
                           label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')

                    # Add stats to title
                    title = "Curvature-Intensity Correlation (All Frames)"
                    title += f"\nR² = {r_value**2:.3f}, p-value = {p_value:.3f}, n = {len(combined_curvatures)}"
                else:
                    title = "Curvature-Intensity Correlation (All Frames)\nInsufficient data for regression"

                # Set labels and title
                ax1.set_xlabel('Normalized Curvature')
                ax1.set_ylabel('Intensity')
                ax1.set_title(title)
                if len(combined_curvatures) > 1:
                    ax1.legend()

        # Add correlation statistics table
        if 'results' in data and 'statistics' in data['results']:
            stats = data['results']['statistics']

            # Hide axes for the table
            ax2.axis('off')

            # Create table data
            table_data = []

            # Add sign curvature correlation
            if 'sign_curvature_correlation' in stats:
                sign_stats = stats['sign_curvature_correlation']
                r_squared = sign_stats.get('r_squared', 'N/A')
                p_value = sign_stats.get('p_value', 'N/A')
                sample_size = sign_stats.get('sample_size', 'N/A')

                table_data.append(["Sign Curvature:", f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else f"R² = {r_squared}",
                                 f"p = {p_value:.4f}" if isinstance(p_value, float) else f"p = {p_value}",
                                 f"n = {sample_size}"])

            # Add normalized curvature correlation
            if 'normalized_curvature_correlation' in stats:
                norm_stats = stats['normalized_curvature_correlation']
                r_squared = norm_stats.get('r_squared', 'N/A')
                p_value = norm_stats.get('p_value', 'N/A')
                sample_size = norm_stats.get('sample_size', 'N/A')

                table_data.append(["Normalized Curvature:", f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else f"R² = {r_squared}",
                                 f"p = {p_value:.4f}" if isinstance(p_value, float) else f"p = {p_value}",
                                 f"n = {sample_size}"])

            # Add movement grouped correlation if available
            if 'movement_grouped_correlation' in stats:
                movement_stats = stats['movement_grouped_correlation']

                # Add section title
                ax2.text(0.5, 0.5, "Movement-Grouped Correlation Statistics",
                       ha='center', fontsize=12, fontweight='bold')

                # Add extending correlation
                if 'extending' in movement_stats:
                    extending_stats = movement_stats['extending']

                    if 'normalized_curvature' in extending_stats:
                        norm_stats = extending_stats['normalized_curvature']
                        r_squared = norm_stats.get('r_squared', 'N/A')
                        p_value = norm_stats.get('p_value', 'N/A')
                        sample_size = norm_stats.get('sample_size', 'N/A')

                        table_data.append(["Extending Frames:", f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else f"R² = {r_squared}",
                                         f"p = {p_value:.4f}" if isinstance(p_value, float) else f"p = {p_value}",
                                         f"n = {sample_size}"])

                # Add retracting correlation
                if 'retracting' in movement_stats:
                    retracting_stats = movement_stats['retracting']

                    if 'normalized_curvature' in retracting_stats:
                        norm_stats = retracting_stats['normalized_curvature']
                        r_squared = norm_stats.get('r_squared', 'N/A')
                        p_value = norm_stats.get('p_value', 'N/A')
                        sample_size = norm_stats.get('sample_size', 'N/A')

                        table_data.append(["Retracting Frames:", f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else f"R² = {r_squared}",
                                         f"p = {p_value:.4f}" if isinstance(p_value, float) else f"p = {p_value}",
                                         f"n = {sample_size}"])

                # Add stable correlation
                if 'stable' in movement_stats:
                    stable_stats = movement_stats['stable']

                    if 'normalized_curvature' in stable_stats:
                        norm_stats = stable_stats['normalized_curvature']
                        r_squared = norm_stats.get('r_squared', 'N/A')
                        p_value = norm_stats.get('p_value', 'N/A')
                        sample_size = norm_stats.get('sample_size', 'N/A')

                        table_data.append(["Stable Frames:", f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else f"R² = {r_squared}",
                                         f"p = {p_value:.4f}" if isinstance(p_value, float) else f"p = {p_value}",
                                         f"n = {sample_size}"])

            # Create table
            if table_data:
                ax2.set_title("Correlation Statistics Summary", fontsize=14)

                table = ax2.table(
                    cellText=table_data,
                    colLabels=["Analysis Type", "R-squared", "p-value", "Sample Size"],
                    cellLoc='center',
                    loc='center'
                )

                # Adjust table appearance
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    def add_curvature_page(self, pdf, data):
        """
        Add curvature analysis page to PDF report

        Parameters:
        -----------
        pdf : PdfPages
            PDF document
        data : dict
            Dictionary of loaded data and results
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))

        # Add title
        fig.suptitle("Curvature Analysis", fontsize=18, fontweight='bold', y=0.98)

        # Add curvature heatmap
        if 'results' in data:
            results = data['results']

            # Collect curvature data
            all_sign_curvatures = []
            all_normalized_curvatures = []
            frame_indices = []

            for frame_idx, frame_results in results.items():
                if not isinstance(frame_idx, int):
                    continue

                if 'curvatures' in frame_results:
                    sign_curvatures, _, normalized_curvatures = frame_results['curvatures']

                    all_sign_curvatures.append(sign_curvatures)
                    all_normalized_curvatures.append(normalized_curvatures)
                    frame_indices.append(frame_idx)

            # Plot curvature heatmaps if data available
            if all_sign_curvatures:
                # Plot sign curvature heatmap
                sign_array = np.array(all_sign_curvatures)
                sign_img = ax1.imshow(sign_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                plt.colorbar(sign_img, ax=ax1, label='Curvature Sign')

                # Plot normalized curvature heatmap
                norm_array = np.array(all_normalized_curvatures)
                norm_img = ax2.imshow(norm_array, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                plt.colorbar(norm_img, ax=ax2, label='Normalized Curvature')

                # Add labels and titles
                ax1.set_title('Sign Curvature Across Frames')
                ax2.set_title('Normalized Curvature Across Frames')

                ax1.set_ylabel('Frame')
                ax2.set_ylabel('Frame')
                ax2.set_xlabel('Point Index')

                # Set y-ticks to frame indices
                ax1.set_yticks(range(len(frame_indices)))
                ax1.set_yticklabels([str(i+1) for i in frame_indices])
                ax2.set_yticks(range(len(frame_indices)))
                ax2.set_yticklabels([str(i+1) for i in frame_indices])

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    def add_intensity_page(self, pdf, data):
        """
        Add intensity analysis page to PDF report

        Parameters:
        -----------
        pdf : PdfPages
            PDF document
        data : dict
            Dictionary of loaded data and results
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))

        # Add title
        fig.suptitle("Intensity Analysis", fontsize=18, fontweight='bold', y=0.98)

        # Add intensity heatmap
        if 'results' in data:
            results = data['results']

            # Collect intensity data
            all_intensities = []
            all_valid_points = []
            frame_indices = []

            for frame_idx, frame_results in results.items():
                if not isinstance(frame_idx, int):
                    continue

                if 'intensities' in frame_results and 'valid_points' in frame_results:
                    intensities = frame_results['intensities']
                    valid_points = frame_results['valid_points']

                    # Create masked array
                    masked_intensities = np.ma.array(intensities, mask=~valid_points)

                    all_intensities.append(masked_intensities)
                    all_valid_points.append(valid_points)
                    frame_indices.append(frame_idx)

            # Plot intensity heatmap if data available
            if all_intensities:
                # Create masked array stack
                intensity_array = np.ma.stack(all_intensities)

                # Calculate min and max for better color scaling
                # Using percentiles to avoid extreme outliers affecting the scale
                intensity_min = np.nanpercentile(intensity_array.compressed(), 5)
                intensity_max = np.nanpercentile(intensity_array.compressed(), 95)

                intensity_img = ax.imshow(intensity_array, cmap='viridis', aspect='auto',
                                         vmin=intensity_min, vmax=intensity_max)
                plt.colorbar(intensity_img, ax=ax, label='Intensity')

                # Add labels and title
                ax.set_title('Intensity Measurements Across Frames')
                ax.set_ylabel('Frame')
                ax.set_xlabel('Point Index')

                # Set y-ticks to frame indices
                ax.set_yticks(range(len(frame_indices)))
                ax.set_yticklabels([str(i+1) for i in frame_indices])

                # Add statistics table
                if 'statistics' in results:
                    stats = results['statistics']
                    valid_points = stats.get('valid_points', 0)
                    total_points = stats.get('total_points', 0)
                    valid_percentage = (valid_points / total_points * 100) if total_points > 0 else 0

                    # Create table
                    ax.text(0.5, -0.1,
                           f"Valid Intensity Measurements: {valid_points}/{total_points} ({valid_percentage:.1f}%)",
                           ha='center', transform=ax.transAxes, fontsize=10)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    def add_movement_page(self, pdf, data):
        """
        Add movement analysis page to PDF report

        Parameters:
        -----------
        pdf : PdfPages
            PDF document
        data : dict
            Dictionary of loaded data and results
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))

        # Add title
        fig.suptitle("Edge Movement Analysis", fontsize=18, fontweight='bold', y=0.98)

        # Add movement bar chart
        if 'results' in data and 'movement_scores' in data['results'] and 'movement_types' in data['results']:
            results = data['results']
            movement_scores = results['movement_scores']
            movement_types = results['movement_types']

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
            ax1.bar(frame_indices, movement_scores, color=movement_colors, alpha=0.7)

            # Add horizontal line at zero
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)

            # Add thresholds for classification
            ax1.axhline(y=0.1, color='blue', linestyle='--', alpha=0.3)
            ax1.axhline(y=-0.1, color='red', linestyle='--', alpha=0.3)

            # Annotate bars with movement type
            for i, score in enumerate(movement_scores):
                offset = 0.05 if score >= 0 else -0.15
                ax1.text(i, score + offset, movement_types[i],
                      ha='center', rotation=90, fontsize=8, fontweight='bold',
                      color=movement_colors[i])

            # Add legend
            import matplotlib.patches as mpatches
            extending_patch = mpatches.Patch(color='blue', label='Extending')
            retracting_patch = mpatches.Patch(color='red', label='Retracting')
            stable_patch = mpatches.Patch(color='gray', label='Stable')
            ax1.legend(handles=[extending_patch, retracting_patch, stable_patch])

            # Set labels and title
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Movement Score')
            ax1.set_title('Cell Edge Movement Over Time')

            # Set x-ticks to frame indices
            ax1.set_xticks(frame_indices)
            ax1.set_xticklabels([str(i+2) for i in frame_indices])

            # Add movement type pie chart
            # Count movement types
            extending_count = movement_types.count('extending')
            retracting_count = movement_types.count('retracting')
            stable_count = movement_types.count('stable')

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
                wedges, texts, autotexts = ax2.pie(
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
                ax2.set_title(f'Movement Type Distribution\nPredominant: {predominant} ({predominant_percentage:.1f}% of frames)')
            else:
                ax2.set_title('Movement Type Distribution')

            # Equal aspect ratio
            ax2.axis('equal')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    def add_sample_frames_page(self, pdf, data):
        """
        Add sample frames page to PDF report

        Parameters:
        -----------
        pdf : PdfPages
            PDF document
        data : dict
            Dictionary of loaded data and results
        """
        # Create figure
        fig, axs = plt.subplots(3, 2, figsize=(8.5, 11))

        # Add title
        fig.suptitle("Sample Frames", fontsize=18, fontweight='bold', y=0.98)

        # Flatten axes for easier indexing
        axs = axs.flatten()

        # Get images and masks
        images = data.get('images')
        masks = data.get('masks')

        if images is None:
            return

        # Select sample frames
        n_frames = images.shape[0]
        if n_frames <= 6:
            sample_indices = range(n_frames)
        else:
            # Select first, last, and 4 evenly spaced frames in between
            sample_indices = [0]
            if n_frames > 2:
                step = (n_frames - 1) / 5
                for i in range(1, 5):
                    sample_indices.append(int(i * step))
            sample_indices.append(n_frames - 1)

        # Display samples
        for i, ax in enumerate(axs):
            if i < len(sample_indices):
                frame_idx = sample_indices[i]

                # Get image
                image = images[frame_idx]

                # Create overlay if mask available
                if masks is not None and frame_idx < masks.shape[0]:
                    mask = masks[frame_idx]
                    overlay = self.create_overlay(image, mask)
                    ax.imshow(overlay)
                else:
                    ax.imshow(image, cmap='gray')

                # Add title
                ax.set_title(f"Frame {frame_idx+1}")

                # Hide axes
                ax.axis('off')
            else:
                # Hide unused subplots
                ax.axis('off')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save figure to PDF
        pdf.savefig(fig)
        plt.close(fig)

    def create_overlay(self, image, mask):
        """
        Create an RGB overlay with the image in grayscale and mask as a semi-transparent overlay

        Parameters:
        -----------
        image : ndarray
            Original microscope image
        mask : ndarray
            Binary mask

        Returns:
        --------
        overlay : ndarray
            RGB overlay image
        """
        # Normalize image to 0-1 range
        image_norm = image.astype(float)
        if image_norm.max() > 0:
            image_norm = image_norm / image_norm.max()

        # Create an RGB image with grayscale original
        overlay = np.stack([image_norm, image_norm, image_norm], axis=2)

        # Add red tint to masked areas
        overlay[mask > 0, 0] = np.maximum(overlay[mask > 0, 0], 0.5)  # Red channel
        overlay[mask > 0, 1] *= 0.7  # Reduce green
        overlay[mask > 0, 2] *= 0.7  # Reduce blue

        return overlay

    def convert_results_for_json(self, results):
        """
        Convert analysis results to JSON-serializable format

        Parameters:
        -----------
        results : dict
            Analysis results

        Returns:
        --------
        serializable_results : dict
            JSON-serializable version of the results
        """
        # Convert numpy types to Python types
        def numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_python(item) for item in obj]
            elif isinstance(obj, tuple):
                return [numpy_to_python(item) for item in obj]
            else:
                return obj

        # Convert results
        serializable_results = numpy_to_python(results)

        # Remove large data arrays for better performance
        for key, value in serializable_results.items():
            if isinstance(key, int) and isinstance(value, dict):
                # Remove large data arrays from per-frame results
                if 'points' in value:
                    # Keep only count of points
                    points = value.pop('points')
                    value['points_count'] = len(points)

                if 'sampling_regions' in value:
                    # Remove sampling regions completely
                    value.pop('sampling_regions')

        return serializable_results
