"""Results table and colour-coded log console widgets."""
import datetime
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit
from PyQt5.QtGui import QColor


class ResultsTable(QTableWidget):
    """
    Widget for displaying analysis results in a table
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Set table properties
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def add_result(self, parameter, value):
        """
        Add a result to the table

        Parameters:
        -----------
        parameter : str
            Parameter name
        value : Any
            Parameter value
        """
        # Convert value to string representation
        if isinstance(value, float):
            value_str = f"{value:.6g}"
        else:
            value_str = str(value)

        # Add row
        row = self.rowCount()
        self.insertRow(row)

        # Add items
        parameter_item = QTableWidgetItem(parameter)
        value_item = QTableWidgetItem(value_str)

        # Add to table
        self.setItem(row, 0, parameter_item)
        self.setItem(row, 1, value_item)

        # Set background color based on parameter type
        if "error" in parameter.lower():
            parameter_item.setBackground(QColor(255, 200, 200))
            value_item.setBackground(QColor(255, 200, 200))
        elif "correlation" in parameter.lower() or "r²" in parameter.lower().replace("r2", "r²"):
            parameter_item.setBackground(QColor(200, 255, 200))
            value_item.setBackground(QColor(200, 255, 200))


class LogConsole(QTextEdit):
    """
    Widget for displaying log messages
    """

    # Log levels
    INFO = 0
    WARNING = 1
    ERROR = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Set properties
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)

        # Set font
        font = self.font()
        font.setFamily("Courier")
        self.setFont(font)

        # Add timestamp
        self.log("Log started", self.INFO)

    def log(self, message, level=INFO):
        """
        Add a log message

        Parameters:
        -----------
        message : str
            Log message
        level : int
            Log level (INFO, WARNING, ERROR)
        """
        # Get timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format message based on level
        if level == self.INFO:
            formatted_message = f"[{timestamp}] INFO: {message}"
            color = "black"
        elif level == self.WARNING:
            formatted_message = f"[{timestamp}] WARNING: {message}"
            color = "orange"
        elif level == self.ERROR:
            formatted_message = f"[{timestamp}] ERROR: {message}"
            color = "red"
        else:
            formatted_message = f"[{timestamp}] {message}"
            color = "black"

        # Add formatted message
        self.append(f'<span style="color: {color};">{formatted_message}</span>')

        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
