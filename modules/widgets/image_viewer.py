"""Matplotlib image viewer widget with a pixel read-out."""
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class ImageViewer(QWidget):
    """
    Widget for displaying microscope images
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.image_data = None  # Store the raw image data

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
        self.title_label = QLabel("No image loaded")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Add position and intensity label
        self.position_label = QLabel("Position: -- Intensity: --")
        self.position_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.position_label)

        # Set layout margins
        layout.setContentsMargins(0, 0, 0, 0)

        # Connect mouse movement event
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def set_image(self, image, cmap='gray'):
        """
        Set the image to display

        Parameters:
        -----------
        image : ndarray
            Image to display
        cmap : str
            Colormap to use
        """
        # Store the raw image data
        self.image_data = image

        # Clear previous image
        self.ax.clear()

        # Plot new image
        self.ax.imshow(image, cmap=cmap)
        self.ax.axis('off')

        # Refresh canvas
        self.canvas.draw()

    def set_title(self, title):
        """
        Set the image title

        Parameters:
        -----------
        title : str
            Image title
        """
        self.title_label.setText(title)

    def on_mouse_move(self, event):
        """
        Handle mouse movement events to display position and intensity

        Parameters:
        -----------
        event : MouseEvent
            Matplotlib mouse event
        """
        if event.inaxes and self.image_data is not None:
            # Get mouse position
            x, y = int(event.xdata), int(event.ydata)

            # Check if position is within image bounds
            if (0 <= y < self.image_data.shape[0] and
                0 <= x < self.image_data.shape[1]):

                # Get raw intensity value (not normalized)
                intensity = self.image_data[y, x]

                # Update position label with raw value
                self.position_label.setText(f"Position: x={x}, y={y} | Intensity: {intensity:.2f}")
            else:
                self.position_label.setText(f"Position: x={x}, y={y} | Out of bounds")
        else:
            self.position_label.setText("Position: -- | Intensity: --")
