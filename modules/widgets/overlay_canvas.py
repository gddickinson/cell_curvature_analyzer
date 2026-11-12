"""Overlay canvas: image, mask, contour, curvature / intensity points, sampling regions, movement."""
import numpy as np
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class OverlayCanvas(QWidget):
    """
    Widget for displaying analysis overlays
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # Initialize analysis layers
        self.background_image = None
        self.contour_data = None
        self.curvature_points = None
        self.intensity_points = None
        self.sampling_regions = None

        # Initialize figure attributes
        self.contour_color = 'y'
        self.contour_width = 1.0

    def init_ui(self):
        # Create layout
        layout = QVBoxLayout(self)

        # Create figure
        self.figure = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Create axes
        self.ax = self.figure.add_subplot(111)

        # Image title
        self.title_label = QLabel("No overlay data")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Add position and intensity label
        self.position_label = QLabel("Position: -- | Intensity: --")
        self.position_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.position_label)

        # Set layout margins
        layout.setContentsMargins(0, 0, 0, 0)

        # Connect mouse movement event
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def clear(self):
        """Clear all overlay layers"""
        self.background_image = None
        self.contour_data = None
        self.curvature_points = None
        self.intensity_points = None
        self.sampling_regions = None

        # Clear figure
        self.ax.clear()
        self.canvas.draw()

    def add_background(self, image):
        """
        Add background image

        Parameters:
        -----------
        image : ndarray
            Background image
        """
        # Store the raw image data without normalization
        self.background_image = image

    def add_contour(self, contour, color='r', linewidth=2.0):
        """
        Add a cell contour

        Parameters:
        -----------
        contour : ndarray
            Cell contour points
        color : str
            Line color
        linewidth : float
            Line width
        """
        # Make sure we have a valid contour
        if contour is None or len(contour) == 0:
            return

        # Store contour data and styling
        self.contour_data = contour
        self.contour_color = color
        self.contour_width = linewidth

        # The actual drawing will happen in update_canvas()

    def add_curvature_points(self, points, curvatures, cmap='coolwarm', marker_size=10):
        """
        Add curvature points

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        curvatures : ndarray
            Array of curvature values
        cmap : str
            Colormap to use
        marker_size : int
            Size of markers
        """
        self.curvature_points = {
            'points': points,
            'curvatures': curvatures,
            'cmap': cmap,
            'marker_size': marker_size
        }

    def add_intensity_points(self, points, intensities, cmap='viridis', marker_size=10):
        """
        Add intensity points

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        intensities : ndarray
            Array of intensity values
        cmap : str
            Colormap to use
        marker_size : int
            Size of markers
        """
        self.intensity_points = {
            'points': points,
            'intensities': intensities,
            'cmap': cmap,
            'marker_size': marker_size
        }

    def add_sampling_regions(self, sampling_regions, valid_points=None, recovered_by_rotation=None, exclude_endpoints=False):
        """
        Add sampling regions

        Parameters:
        -----------
        sampling_regions : list
            List of sampling region polygons
        valid_points : ndarray
            Boolean array indicating which points are valid
        recovered_by_rotation : ndarray, optional
            Boolean array indicating which valid points were recovered by rotation
        exclude_endpoints : bool
            Whether endpoints are excluded from analysis
        """
        self.sampling_regions = {
            'regions': sampling_regions,
            'valid_points': valid_points,
            'recovered_by_rotation': recovered_by_rotation,
            'exclude_endpoints': exclude_endpoints
        }


    def add_movement_visualization(self, points, movement_scores, colors, marker_size=10):
        """
        Add movement visualization to the overlay

        Parameters:
        -----------
        points : ndarray
            Array of contour points
        movement_scores : ndarray
            Array of movement scores
        colors : list
            List of colors for each point
        marker_size : int
            Size of markers
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return

        try:
            # Add points with colors representing movement
            self.ax.scatter(points[:, 1], points[:, 0], c=colors,
                          s=marker_size, marker='o', zorder=15)

            # Add arrows indicating movement direction and magnitude
            for i, (point, score) in enumerate(zip(points, movement_scores)):
                if abs(score) > 0.05:  # Only show arrows for significant movement
                    # Skip if this is the last or first point
                    if i == 0 or i == len(points) - 1:
                        continue

                    # Get normal direction (perpendicular to contour)
                    prev_idx = (i - 1) % len(points)
                    next_idx = (i + 1) % len(points)

                    # Vector along contour
                    tangent = points[next_idx] - points[prev_idx]
                    tangent_norm = np.linalg.norm(tangent)

                    if tangent_norm > 1e-10:
                        tangent = tangent / tangent_norm

                        # Normal vector (perpendicular to tangent)
                        normal = np.array([-tangent[1], tangent[0]])

                        # Scale by movement score
                        arrow_size = min(abs(score) * 10, 20)  # Limit maximum size
                        if score < 0:
                            normal = -normal

                        # Draw arrow
                        self.ax.arrow(point[1], point[0],
                                    normal[1] * arrow_size, normal[0] * arrow_size,
                                    head_width=2, head_length=3, fc=colors[i], ec=colors[i],
                                    zorder=20)
        except Exception as e:
            # Log error but continue
            print(f"Error adding movement visualization: {str(e)}")


    def update_canvas(self):
        """Update the canvas with all layers"""
        # Clear previous content
        self.ax.clear()

        # Add background image if available
        if self.background_image is not None:
            # Get brightness settings if available
            min_brightness = getattr(self, 'min_brightness', 0.0)
            max_brightness = getattr(self, 'max_brightness', 1.0)

            # Calculate display min/max for windowing
            actual_min = np.min(self.background_image)
            actual_max = np.max(self.background_image)
            display_min = actual_min + min_brightness * (actual_max - actual_min)
            display_max = actual_min + max_brightness * (actual_max - actual_min)

            # Create display-only copy for visualization with window applied
            display_image = np.clip(self.background_image.copy(), display_min, display_max)

            # Normalize for display only
            if display_max > display_min:
                display_image = (display_image - display_min) / (display_max - display_min)
            else:
                display_image = np.zeros_like(self.background_image)

            self.ax.imshow(display_image, cmap='gray')

            # If we have a mask, create an overlay
            if hasattr(self, 'mask') and self.mask is not None:
                # Create red mask overlay
                h, w = self.mask.shape
                red_mask = np.zeros((h, w, 4))  # RGBA
                red_mask[self.mask > 0, 0] = 1.0  # Red channel
                red_mask[self.mask > 0, 3] = self.mask_opacity  # Alpha channel

                self.ax.imshow(red_mask)

        # Add contour if available
        if hasattr(self, 'contour_data') and self.contour_data is not None:
            # Debug print to ensure contour data exists
            print(f"Drawing contour with {len(self.contour_data)} points, color={self.contour_color}, width={self.contour_width}")

            # Make sure the contour is drawn on top with higher zorder
            self.ax.plot(self.contour_data[:, 1], self.contour_data[:, 0],
                        color=self.contour_color, linewidth=self.contour_width, zorder=10)

        # Add sampling regions if available
        if self.sampling_regions is not None:
            regions = self.sampling_regions['regions']
            valid_points = self.sampling_regions.get('valid_points')
            recovered_by_rotation = self.sampling_regions.get('recovered_by_rotation')
            exclude_endpoints = self.sampling_regions.get('exclude_endpoints', False)

            for i, region in enumerate(regions):
                # Check if this is an excluded endpoint
                is_endpoint = exclude_endpoints and (i == 0 or i == len(regions) - 1)

                # Skip if this is an empty region (endpoint)
                if is_endpoint or np.all(region == 0):
                    continue

                # Convert to x, y coordinates for plotting
                polygon_y = region[:, 1]
                polygon_x = region[:, 0]

                if valid_points is not None:
                    if i < len(valid_points) and valid_points[i]:
                        # Check if this region was recovered by rotation
                        if recovered_by_rotation is not None and i < len(recovered_by_rotation) and recovered_by_rotation[i]:
                            # Use a different color for regions recovered by rotation
                            self.ax.fill(polygon_y, polygon_x, alpha=0.4, color='magenta')
                        else:
                            # Normal valid region
                            self.ax.fill(polygon_y, polygon_x, alpha=0.3, color='cyan')
                    else:
                        # Invalid region
                        self.ax.fill(polygon_y, polygon_x, alpha=0.15, color='gray')
                else:
                    self.ax.fill(polygon_y, polygon_x, alpha=0.3, color='cyan')

        # Add curvature points if available
        if self.curvature_points is not None:
            points = self.curvature_points['points']
            curvatures = self.curvature_points['curvatures']
            cmap = self.curvature_points['cmap']
            marker_size = self.curvature_points['marker_size']

            self.ax.scatter(points[:, 1], points[:, 0],
                          c=curvatures, cmap=cmap, vmin=-1, vmax=1,
                          s=marker_size, edgecolor='k')

        # Add intensity points if available
        if self.intensity_points is not None:
            points = self.intensity_points['points']
            intensities = self.intensity_points['intensities']
            cmap = self.intensity_points['cmap']
            marker_size = self.intensity_points['marker_size']

            # Calculate min and max for better color scaling
            # Using percentiles to avoid extreme outliers affecting the scale
            vmin = np.nanpercentile(intensities, 5)
            vmax = np.nanpercentile(intensities, 95)

            self.ax.scatter(points[:, 1], points[:, 0],
                          c=intensities, cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          s=marker_size, edgecolor='k')

        # Turn off axes
        self.ax.axis('off')

        # Refresh canvas
        self.canvas.draw()

    def set_title(self, title):
        """
        Set the overlay title

        Parameters:
        -----------
        title : str
            Overlay title
        """
        self.title_label.setText(title)

    def add_mask(self, mask, opacity=0.5):
        """
        Add a mask overlay with opacity control

        Parameters:
        -----------
        mask : ndarray
            Binary mask
        opacity : float
            Opacity of the mask (0-1)
        """
        # Store the mask and opacity
        self.mask = mask
        self.mask_opacity = opacity

        # The actual drawing will happen in update_canvas()

    def on_mouse_move(self, event):
        """
        Handle mouse movement events to display position and intensity

        Parameters:
        -----------
        event : MouseEvent
            Matplotlib mouse event
        """
        if event.inaxes and self.background_image is not None:
            # Get mouse position
            x, y = int(event.xdata), int(event.ydata)

            # Check if position is within image bounds
            if (0 <= y < self.background_image.shape[0] and
                0 <= x < self.background_image.shape[1]):

                # Get raw intensity value
                intensity = self.background_image[y, x]

                # Get additional information if available
                additional_info = ""

                # Check for curvature data at this position
                if self.curvature_points is not None:
                    points = self.curvature_points['points']
                    curvatures = self.curvature_points['curvatures']

                    # Find closest point (simplified approach)
                    closest_idx = None
                    min_dist = float('inf')
                    for i, point in enumerate(points):
                        dist = (point[0] - y)**2 + (point[1] - x)**2
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i

                    # If close enough to a contour point, show curvature
                    if closest_idx is not None and min_dist < 100:  # Threshold for "close enough"
                        additional_info = f" | Curvature: {curvatures[closest_idx]:.3f}"

                # Update position label with raw intensity value
                self.position_label.setText(f"Position: x={x}, y={y} | Intensity: {intensity:.2f}{additional_info}")
            else:
                self.position_label.setText(f"Position: x={x}, y={y} | Out of bounds")
        else:
            self.position_label.setText("Position: -- | Intensity: --")
