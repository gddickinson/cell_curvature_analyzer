"""Debug plots of the sampling methods for CurvatureAnalyzer (matplotlib, not used by the GUI)."""
import numpy as np
import datetime


class SamplingDebugMixin:
    """CurvatureAnalyzer mixin: debug plots of the sampling methods."""

    def debug_visualize_sampling_methods(self, contour, n_points=100):
        """
        Create a debug visualization comparing standard and angular sampling methods

        Parameters:
        -----------
        contour : ndarray
            Cell contour
        n_points : int
            Number of points to sample

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Debug visualization figure
        """
        import matplotlib.pyplot as plt

        # Get both types of sampled points
        std_points = self.sample_equidistant_points(contour, n_points)
        ang_points = self.sample_angular_points(contour, n_points)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot standard sampling
        ax1.plot(contour[:, 1], contour[:, 0], 'b-', alpha=0.5, label='Contour')
        ax1.scatter(std_points[:, 1], std_points[:, 0], c='r', s=30, label='Sampled Points')
        ax1.set_title('Standard Equidistant Sampling')
        ax1.legend()
        ax1.axis('equal')

        # Plot angular sampling
        ax2.plot(contour[:, 1], contour[:, 0], 'b-', alpha=0.5, label='Contour')
        ax2.scatter(ang_points[:, 1], ang_points[:, 0], c='g', s=30, label='Sampled Points')
        ax2.set_title('Angular Sampling')
        ax2.legend()
        ax2.axis('equal')

        plt.tight_layout()

        # Save comparison to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"sampling_comparison_{timestamp}.png", dpi=150)

        print(f"DEBUG: Saved sampling method comparison to sampling_comparison_{timestamp}.png")

        return fig

    def visualize_sampling(self, image, mask, points, method_name=""):
        """
        Create a visualization of the sampled points on the cell contour
        """
        import matplotlib.pyplot as plt

        # Extract the contour
        contour = self.detect_cell_edge(mask)

        # Create figure
        plt.figure(figsize=(10, 8))

        # Show original image
        plt.imshow(image, cmap='gray')

        # Plot the contour
        if len(contour) > 0:
            plt.plot(contour[:, 1], contour[:, 0], 'b-', alpha=0.5, linewidth=1)

        # Plot the sampled points
        plt.scatter(points[:, 1], points[:, 0], c='r', s=30)

        # Add point indices (every 10th point)
        for i in range(0, len(points), 10):
            plt.text(points[i, 1] + 3, points[i, 0] + 3, str(i), color='white', fontsize=9,
                     bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

        # If using X-axis distribution, add vertical lines to show x-positions
        if "X-Axis" in method_name:
            # Get the min and max y values to define line lengths
            min_y = np.min(contour[:, 0])
            max_y = np.max(contour[:, 0])
            y_range = max_y - min_y

            # Draw vertical lines at each x-position
            for point in points:
                x = point[1]  # x-coordinate
                # Draw a vertical line from above to below the cell
                plt.plot([x, x], [min_y - 0.1*y_range, max_y + 0.1*y_range],
                        'g--', alpha=0.3, linewidth=0.5)

        # Add title
        plt.title(f"{method_name} Point Sampling")

        # Add center point if angular method
        if "Angular" in method_name:
            center = np.mean(contour, axis=0)
            plt.scatter([center[1]], [center[0]], c='cyan', s=100, marker='x')

            # Add a few radial lines to show the angular divisions
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                r = np.sqrt(image.shape[0]**2 + image.shape[1]**2) / 2  # Radius to edge of image
                x_end = center[1] + r * np.cos(angle)
                y_end = center[0] + r * np.sin(angle)
                plt.plot([center[1], x_end], [center[0], y_end], 'cyan', alpha=0.3, linestyle='--')

        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])

        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{method_name.lower().replace(' ', '_')}_sampling_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"DEBUG: Saved sampling visualization to {filename}")
