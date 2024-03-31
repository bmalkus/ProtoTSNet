#!/usr/bin/env python3

import os
import sys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
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

        # Create lines for both current and ghost data
        (self.line,) = self.ax.plot([], [], 'b-')
        (self.ghost_line,) = self.ax.plot([], [], 'gray', alpha=0.5, linestyle='--')

        # Connect matplotlib events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        self.has_changes = False
        self.original_data = None

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
        (self.ghost_line,) = self.ax.plot([], [], 'gray', alpha=0.5, linestyle='--')
        self.canvas.draw()

    def set_ghost_data(self, x_values, y_values):
        """Set and display ghost data"""
        if len(x_values) > 0 and len(y_values) > 0:
            self.original_data = y_values  # Store the original data
            self.ghost_line.set_data(x_values, y_values)
        else:
            self.original_data = None
            self.ghost_line.set_data([], [])
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
            if x_pos > self.last_x:
                x_values = np.arange(self.last_x + 1, x_pos + 1)
            else:
                x_values = np.arange(self.last_x - 1, x_pos - 1, -1)

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

        self.has_changes = True
        # Always show ghost of original data when it exists
        if self.original_data is not None:
            x_values = np.arange(len(self.original_data))
            self.ghost_line.set_data(x_values, self.original_data)

        self.line.set_data(x_sorted, y_values)
        self.canvas.draw()

    def clear_drawing(self):
        """Clear current drawing and restore original data if it exists"""
        self.points = {}
        self.last_x = None
        self.line.set_data([], [])

        if self.original_data is not None:
            # Restore the original data
            self.load_from_array(self.original_data)
            self.has_changes = False
        else:
            # Just clear everything if no original data
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

    def load_from_array(self, data):
        """Convert numpy array to points dictionary and store original data"""
        self.original_data = np.array(data)  # Store the original data
        self.points = {i: float(v) for i, v in enumerate(data)}
        x_sorted = sorted(self.points.keys())
        y_values = [self.points[x] for x in x_sorted]
        self.line.set_data(x_sorted, y_values)
        # Show original data as ghost immediately
        x_values = np.arange(len(self.original_data))
        self.ghost_line.set_data(x_values, self.original_data)
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

        # File management controls
        file_controls = QGridLayout()

        # Directory selection
        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_directory)
        self.dir_label = QLabel("No directory selected")
        file_controls.addWidget(self.dir_button, 0, 0)
        file_controls.addWidget(self.dir_label, 0, 1)

        # Class selection
        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)
        # self.class_combo.lineEdit().editingFinished.connect(self.on_class_changed)
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        file_controls.addWidget(QLabel("Class:"), 1, 0)
        file_controls.addWidget(self.class_combo, 1, 1)

        # Prototype selection
        self.proto_combo = QComboBox()
        self.proto_combo.setEditable(True)
        # self.proto_combo.lineEdit().editingFinished.connect(self.on_prototype_changed)
        self.proto_combo.currentIndexChanged.connect(self.on_prototype_changed)
        file_controls.addWidget(QLabel("Prototype:"), 2, 0)
        file_controls.addWidget(self.proto_combo, 2, 1)

        main_layout.insertLayout(0, file_controls)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Prototype length control
        length_layout = QVBoxLayout()
        length_layout.addWidget(QLabel("Prototype Length:"))
        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 10000)
        self.length_spin.setValue(100)
        self.length_spin.lineEdit().editingFinished.connect(self.apply_settings)
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
        self.y_max_spin.lineEdit().editingFinished.connect(self.apply_settings)
        y_max_layout.addWidget(self.y_max_spin)
        y_range_layout.addLayout(y_max_layout)

        # Y min control
        y_min_layout = QHBoxLayout()
        y_min_layout.addWidget(QLabel("Min:"))
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1000, 1000)
        self.y_min_spin.setValue(-1)
        self.y_min_spin.lineEdit().editingFinished.connect(self.apply_settings)
        y_min_layout.addWidget(self.y_min_spin)
        y_range_layout.addLayout(y_min_layout)

        controls_layout.addLayout(y_range_layout)

        # Buttons
        buttons_layout = QVBoxLayout()

        self.apply_button = QPushButton("Apply Settings")
        self.apply_button.clicked.connect(self.apply_settings)
        buttons_layout.addWidget(self.apply_button)

        self.clear_button = QPushButton("Reset Drawing")
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

        # State variables
        self.working_dir = None
        self.current_file = None
        self.original_data = None

        # Ensure no widget gets initial focus
        self.setFocus()

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.working_dir = dir_path
            self.dir_label.setText(dir_path)
            self.proto_combo.clear()
            self.canvas.clear_drawing()
            self.canvas.set_ghost_data([], [])
            self.update_class_list()
            self.on_class_changed()

    def update_class_list(self):
        if not self.working_dir:
            return

        existing_classes = set()
        for fname in os.listdir(self.working_dir):
            if fname.startswith('class_') and fname.endswith('.txt'):
                try:
                    class_num = int(fname.split('_')[1])
                    existing_classes.add(str(class_num))
                except:
                    continue

        current_items = set(self.class_combo.itemText(i) for i in range(self.class_combo.count()))
        if existing_classes != current_items:
            self.class_combo.clear()
            self.class_combo.addItems(sorted(existing_classes))

    def on_class_changed(self):
        text = self.class_combo.currentText()
        if not text:  # Ignore empty selections
            return
        self.update_prototype_list()

        # Check if class exists
        if self.proto_combo.count() == 0:
            self.canvas.set_ghost_data([], [])
            self.original_data = None
            self.canvas.clear_drawing()
            self.proto_combo.addItem("0")
            self.proto_combo.setCurrentIndex(0)
            self.statusBar().showMessage("No prototypes found for this class, defaulting to prototype 0")
            return

        # Auto-select first prototype
        self.proto_combo.setCurrentIndex(0)
        self.on_prototype_changed()

    def update_prototype_list(self):
        current_proto = self.proto_combo.currentText()
        self.proto_combo.clear()
        if not self.working_dir or not self.class_combo.currentText():
            return

        class_num = self.class_combo.currentText()
        existing_protos = set()
        for fname in os.listdir(self.working_dir):
            if fname.startswith(f'class_{class_num}_proto_') and fname.endswith('.txt'):
                try:
                    proto_num = int(fname.split('_')[3].split('.')[0])
                    existing_protos.add(str(proto_num))
                except:
                    continue

        self.proto_combo.addItems(sorted(existing_protos))

        # Restore previous selection if it exists
        if current_proto in existing_protos:
            self.proto_combo.setCurrentText(current_proto)

        self.on_prototype_changed()

    def on_prototype_changed(self):
        text = self.proto_combo.currentText()
        if not text:  # Ignore empty selections
            return
        fname = self.get_current_filename()
        if fname and os.path.exists(fname):
            self.load_prototype(fname)
        else:
            self.canvas.set_ghost_data([], [])
            self.original_data = None
            self.canvas.clear_drawing()
            self.statusBar().showMessage("No prototype file exists")

    def get_current_filename(self):
        if not all([self.working_dir, self.class_combo.currentText(), self.proto_combo.currentText()]):
            return None
        return os.path.join(
            self.working_dir, f"class_{self.class_combo.currentText()}_proto_{self.proto_combo.currentText()}.txt"
        )

    def load_prototype(self, fname):
        try:
            data = np.loadtxt(fname)
            x_values = np.arange(len(data))

            # Set prototype length to match loaded data
            self.length_spin.setValue(len(data))
            data_y_range = np.max(data) - np.min(data)
            self.y_max_spin.setValue(np.max(data) + data_y_range * 0.1)
            self.y_min_spin.setValue(np.min(data) - data_y_range * 0.1)
            self.canvas.set_ranges(0, len(data) - 1, self.y_min_spin.value(), self.y_max_spin.value())

            # Store original data
            self.original_data = data
            self.current_file = fname

            # Load data into points and display
            self.canvas.load_from_array(data)
            self.canvas.has_changes = False
            self.statusBar().showMessage(f"Loaded {os.path.basename(fname)}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {str(e)}")

    def apply_settings(self):
        prototype_length = self.length_spin.value()
        y_min = self.y_min_spin.value()
        y_max = self.y_max_spin.value()

        if y_min >= y_max:
            self.statusBar().showMessage("Error: Y Min must be less than Y Max")
            return

        # Just set new ranges without clearing
        self.canvas.set_ranges(0, prototype_length - 1, y_min, y_max)
        self.statusBar().showMessage("Settings applied")

    def clear_drawing(self):
        self.canvas.clear_drawing()
        self.statusBar().showMessage(
            "Drawing cleared" if self.original_data is None else "Drawing reverted to original"
        )

    def save_time_series(self):
        if not self.working_dir:
            self.statusBar().showMessage("Error: No working directory selected")
            return

        if not all([self.class_combo.currentText(), self.proto_combo.currentText()]):
            self.statusBar().showMessage("Error: Please select class and prototype numbers")
            return

        fname = self.get_current_filename()
        # if os.path.exists(fname):
        #     reply = QMessageBox.question(
        #         self, 'Confirm Overwrite',
        #         'The file already exists. Do you want to overwrite it?',
        #         QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        #     )
        #     if reply == QMessageBox.No:
        #         return

        x_values, y_values = self.canvas.get_time_series()
        if not x_values:
            self.statusBar().showMessage("Error: No data to save")
            return
        if len(x_values) < self.length_spin.value():
            self.statusBar().showMessage("Error: Incomplete time series")
            return

        try:
            np.savetxt(fname, np.array(y_values))
            self.original_data = np.array(y_values)
            self.current_file = fname
            self.canvas.has_changes = False
            self.canvas.set_ghost_data(x_values, y_values)
            self.update_class_list()
            self.statusBar().showMessage(f"Time series saved to {fname}")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving time series: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimeSeriesDrawer()
    window.show()
    sys.exit(app.exec_())
