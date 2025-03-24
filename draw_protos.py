#!/usr/bin/env python3

import sys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TimeSeriesCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)

        # Time series parameters
        self.x_min = 0
        self.x_max = 100
        self.y_min = -1
        self.y_max = 1
        self.margin_ratio = 0.1  # 10% margin on each side

        # Drawing state
        self.points = {}  # Dictionary to store {x: y} mapping
        self.is_drawing = False
        self.last_x = None
        self.setCursor(Qt.CrossCursor)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Connect matplotlib events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        self.setup_plot()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_plot(self):
        self.ax.clear()
        x_total = self.x_max - self.x_min
        y_total = self.y_max - self.y_min
        x_margin = x_total * self.margin_ratio
        y_margin = y_total * self.margin_ratio

        # Set the full range including margins
        self.ax.set_xlim(self.x_min - x_margin, self.x_max + x_margin)
        self.ax.set_ylim(self.y_min - y_margin, self.y_max + y_margin)

        # Add gray background for margins
        self.ax.axvspan(self.x_min - x_margin, self.x_min, color='gray', alpha=0.1)
        self.ax.axvspan(self.x_max, self.x_max + x_margin, color='gray', alpha=0.1)
        self.ax.axhspan(self.y_min - y_margin, self.y_min, color='gray', alpha=0.1)
        self.ax.axhspan(self.y_max, self.y_max + y_margin, color='gray', alpha=0.1)

        self.ax.grid(True)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Time Series Prototype')
        (self.line,) = self.ax.plot([], [], 'b-')
        self.canvas.draw()

    # Replace the Qt mouse events with matplotlib event handlers
    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left button
            self.is_drawing = True
            self.update_series(event)

    def on_mouse_move(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return
        self.update_series(event)

    def on_mouse_release(self, event):
        if event.button == 1:  # Left button
            self.is_drawing = False
            self.last_x = None

    def update_series(self, event):
        # Use matplotlib event coordinates directly
        x_pos = event.xdata
        y_pos = event.ydata

        if x_pos is None or y_pos is None:
            return

        # Constrain to valid range and round to nearest integer
        x_pos = round(x_pos)
        y_pos = max(self.y_min, min(self.y_max, y_pos))  # Constrain y to actual drawing area

        # Only store points within the actual drawing area
        if not (self.x_min <= x_pos <= self.x_max):
            return

        # We need to handle interpolation if there's a gap between current and last x
        if self.last_x is not None and abs(x_pos - self.last_x) > 1:
            # Interpolate points between last_x and x_pos
            x_values = np.arange(
                self.last_x + 1, x_pos + 1 if x_pos > self.last_x else x_pos - 1, 1 if x_pos > self.last_x else -1
            )

            # Get the y values for the last point
            last_y = self.points[self.last_x]

            # Linear interpolation
            for i, x in enumerate(x_values):
                ratio = (i + 1) / (len(x_values) + 1)
                interp_y = last_y + ratio * (y_pos - last_y)
                self.points[x] = interp_y

        # Update the point dictionary
        self.points[x_pos] = y_pos
        self.last_x = x_pos

        # Update plot
        x_sorted = sorted(self.points.keys())
        y_values = [self.points[x] for x in x_sorted]

        self.line.set_data(x_sorted, y_values)
        self.canvas.draw()

    def clear_drawing(self):
        self.points = {}
        self.last_x = None
        self.line.set_data([], [])
        self.canvas.draw()

    def get_time_series(self):
        # Returns the time series as sorted (x, y) pairs
        x_sorted = sorted(self.points.keys())
        y_values = [self.points[x] for x in x_sorted]
        return x_sorted, y_values

    def set_ranges(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Trim points outside the new range
        to_remove = []
        for x in self.points:
            if x < x_min or x > x_max:
                to_remove.append(x)

        for x in to_remove:
            del self.points[x]

        # Setup plot with new ranges
        self.setup_plot()

        # Update existing line if points exist
        if self.points:
            x_sorted = sorted(self.points.keys())
            y_values = [self.points[x] for x in x_sorted]
            self.line.set_data(x_sorted, y_values)
            self.canvas.draw()


class TimeSeriesDrawer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Series Prototype Drawer")
        self.setMinimumSize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Prototype length control
        length_layout = QVBoxLayout()
        length_layout.addWidget(QLabel("Prototype Length:"))
        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 10000)
        self.length_spin.setValue(100)
        length_layout.addWidget(self.length_spin)
        controls_layout.addLayout(length_layout)

        # Y range controls
        y_range_layout = QVBoxLayout()
        y_range_layout.addWidget(QLabel("Y Range:"))

        # Y max control
        y_max_layout = QHBoxLayout()
        y_max_layout.addWidget(QLabel("Max:"))
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-1000, 1000)
        self.y_max_spin.setValue(1)
        y_max_layout.addWidget(self.y_max_spin)
        y_range_layout.addLayout(y_max_layout)

        # Y min control
        y_min_layout = QHBoxLayout()
        y_min_layout.addWidget(QLabel("Min:"))
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1000, 1000)
        self.y_min_spin.setValue(-1)
        y_min_layout.addWidget(self.y_min_spin)
        y_range_layout.addLayout(y_min_layout)

        controls_layout.addLayout(y_range_layout)

        # Buttons
        buttons_layout = QVBoxLayout()

        self.apply_button = QPushButton("Apply Settings")
        self.apply_button.clicked.connect(self.apply_settings)
        buttons_layout.addWidget(self.apply_button)

        self.clear_button = QPushButton("Clear Drawing")
        self.clear_button.clicked.connect(self.clear_drawing)
        buttons_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save Time Series")
        self.save_button.clicked.connect(self.save_time_series)
        buttons_layout.addWidget(self.save_button)

        controls_layout.addLayout(buttons_layout)

        main_layout.addLayout(controls_layout)

        # Canvas
        self.canvas = TimeSeriesCanvas()
        main_layout.addWidget(self.canvas)

        # Status bar for instructions
        self.statusBar().showMessage("Click and drag to draw the time series.")

    def apply_settings(self):
        prototype_length = self.length_spin.value()
        y_min = self.y_min_spin.value()
        y_max = self.y_max_spin.value()

        if y_min >= y_max:
            self.statusBar().showMessage("Error: Y Min must be less than Y Max")
            return

        # Just set new ranges without clearing
        self.canvas.set_ranges(0, prototype_length, y_min, y_max)
        self.statusBar().showMessage("Settings applied")

    def clear_drawing(self):
        self.canvas.clear_drawing()
        self.statusBar().showMessage("Drawing cleared")

    def save_time_series(self):
        x_values, y_values = self.canvas.get_time_series()

        if not x_values:
            self.statusBar().showMessage("Error: No data to save")
            return

        try:
            fname = "saved_time_series.txt"
            np.savetxt(fname, X=np.array(y_values))

            self.statusBar().showMessage(f"Time series saved to {fname}")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving time series: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimeSeriesDrawer()
    window.show()
    sys.exit(app.exec_())
