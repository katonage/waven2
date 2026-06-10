from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors

import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import QAction, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from GripSplitter import GripSplitter  # type: ignore

from data_model import CellDataModel, DISPLAY_FIELD_MAP, FilterSettings
from plotting_mpl import MatplotlibRenderer


class MainWindow(QMainWindow):
    def __init__(self, cells_path: str | None = None, background_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Visual Cortex Cell Browser")

        self.settings = QSettings("RCNS", "VisualCortexCellBrowser")
        self.model = CellDataModel()
        self.filter_settings = FilterSettings()
        self.selected_iloc: Optional[int] = None
        self._restoring = False

        self.is_dark_theme = QApplication.instance().palette().color(QPalette.Window).value() < 128
        if self.is_dark_theme:
            pg.setConfigOptions(background="k", foreground="w")
        else:
            pg.setConfigOptions(background="w", foreground="k")

        self._build_menu()
        self._build_ui()
        self._connect_signals()
        self._restore_settings()
        self._update_controls_enabled()

        if cells_path:
            self.open_cells(cells_path)
        if background_path:
            self.open_background(background_path)

    # ------------------------------------------------------------------ UI
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        self.action_open_cells = QAction("Open Cells...", self)
        self.action_open_cells.setShortcut("Ctrl+O")
        self.action_open_bg = QAction("Open Background...", self)
        self.action_open_bg.setShortcut("Ctrl+B")
        self.action_quit = QAction("Quit", self)
        self.action_quit.setShortcut("Ctrl+Q")
        file_menu.addAction(self.action_open_cells)
        file_menu.addAction(self.action_open_bg)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

    def _build_ui(self) -> None:
        main_splitter = GripSplitter(Qt.Horizontal, theme="dark" if self.is_dark_theme else "light")
        self.setCentralWidget(main_splitter)

        self.left_panel = self._build_left_panel()
        main_splitter.addWidget(self.left_panel)

        right_splitter = GripSplitter(Qt.Vertical, theme="dark" if self.is_dark_theme else "light")
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([260, 1200])

        self.spatial_panel = SpatialPanel(self)
        right_splitter.addWidget(self.spatial_panel)

        self.figure = Figure(figsize=(10, 6), constrained_layout=False)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_renderer = MatplotlibRenderer(self.figure)
        right_splitter.addWidget(self.canvas)
        right_splitter.setSizes([500, 500])

        self.main_splitter = main_splitter
        self.right_splitter = right_splitter
        self.mpl_renderer.draw_empty()
        self.canvas.draw_idle()

    def _build_left_panel(self) -> QWidget:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(245)
        scroll.setMaximumWidth(330)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        scroll.setWidget(content)

        # Load
        layout.addWidget(_section_label("Load"))
        self.open_cells_button = QPushButton("Open Cells...")
        self.open_background_button = QPushButton("Open Background...")
        self.cells_path_label = QLabel("Cells: --")
        self.cells_path_label.setWordWrap(True)
        self.background_path_label = QLabel("Background: --")
        self.background_path_label.setWordWrap(True)
        self.reset_bg_colorbar_button = QPushButton("Reset Background Colorbar")
        layout.addWidget(self.open_cells_button)
        layout.addWidget(self.open_background_button)
        layout.addWidget(self.cells_path_label)
        layout.addWidget(self.background_path_label)
        layout.addWidget(self.reset_bg_colorbar_button)
        layout.addWidget(_separator())

        # Filter
        layout.addWidget(_section_label("Filter"))
        self.filter_count_label = QLabel("Filtered cells: 0 / 0")
        self.filter_count_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.filter_count_label)

        self.snr_spin = _double_spin(-1e9, 1e9, -1e9, 0.1)
        self.cnn_spin = _double_spin(-1e9, 1e9, -1e9, 0.05)
        self.accepted_check = QCheckBox("Accepted only")
        self.repeat_spin = _double_spin(-1e9, 1e9, -1e9, 0.05)
        self.r_train_spin = _double_spin(-1e9, 1e9, -1e9, 0.05)
        self.r_test_spin = _double_spin(-1e9, 1e9, -1e9, 0.05)

        layout.addLayout(_labeled_widget("SNR >", self.snr_spin))
        layout.addLayout(_labeled_widget("CNN >", self.cnn_spin))
        layout.addWidget(self.accepted_check)
        layout.addLayout(_labeled_widget("Repeatability >", self.repeat_spin))
        layout.addLayout(_labeled_widget("r_train >", self.r_train_spin))
        layout.addLayout(_labeled_widget("r_test >", self.r_test_spin))
        layout.addWidget(_separator())

        # Display
        layout.addWidget(_section_label("Display"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["None"])
        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 100)
        self.marker_size_spin.setValue(7)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Selected", "Stat"])
        self.hover_text = QTextEdit()
        self.hover_text.setReadOnly(True)
        self.hover_text.setMinimumHeight(110)
        self.hover_text.setMaximumHeight(160)

        layout.addLayout(_labeled_widget("Color cells by:", self.color_by_combo))
        layout.addLayout(_labeled_widget("Marker size:", self.marker_size_spin))
        layout.addLayout(_labeled_widget("Mode:", self.mode_combo))
        layout.addWidget(QLabel("Hover info:"))
        layout.addWidget(self.hover_text)
        layout.addStretch(1)
        return outer

    def _connect_signals(self) -> None:
        self.action_open_cells.triggered.connect(lambda: self.open_cells())
        self.action_open_bg.triggered.connect(lambda: self.open_background())
        self.action_quit.triggered.connect(self.close)
        self.open_cells_button.clicked.connect(lambda: self.open_cells())
        self.open_background_button.clicked.connect(lambda: self.open_background())
        self.reset_bg_colorbar_button.clicked.connect(self.spatial_panel.reset_background_colorbar)

        for widget in [self.snr_spin, self.cnn_spin, self.repeat_spin, self.r_train_spin, self.r_test_spin]:
            widget.valueChanged.connect(self.on_filter_changed)
        self.accepted_check.stateChanged.connect(self.on_filter_changed)
        self.color_by_combo.currentTextChanged.connect(self.on_display_changed)
        self.marker_size_spin.valueChanged.connect(self.on_display_changed)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

    # ---------------------------------------------------------------- settings
    def _restore_settings(self) -> None:
        self._restoring = True
        try:
            geom = self.settings.value("window/geometry")
            if geom is not None:
                self.restoreGeometry(geom)
            state = self.settings.value("window/state")
            if state is not None:
                self.restoreState(state)
            self.snr_spin.setValue(float(self.settings.value("filter/snr", -1e9)))
            self.cnn_spin.setValue(float(self.settings.value("filter/cnn", -1e9)))
            self.repeat_spin.setValue(float(self.settings.value("filter/repeatability", -1e9)))
            self.r_train_spin.setValue(float(self.settings.value("filter/r_train", -1e9)))
            self.r_test_spin.setValue(float(self.settings.value("filter/r_test", -1e9)))
            self.accepted_check.setChecked(_settings_bool(self.settings.value("filter/accepted_only", False)))
            self.marker_size_spin.setValue(int(self.settings.value("display/marker_size", 7)))
            mode = str(self.settings.value("display/mode", "Selected"))
            if mode in [self.mode_combo.itemText(i) for i in range(self.mode_combo.count())]:
                self.mode_combo.setCurrentText(mode)
        finally:
            self._restoring = False

    def _save_settings(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("splitter/main", self.main_splitter.saveState())
        self.settings.setValue("splitter/right", self.right_splitter.saveState())
        if self.model.cells_path is not None:
            self.settings.setValue("paths/cells", str(self.model.cells_path))
        if self.model.background_path is not None:
            self.settings.setValue("paths/background", str(self.model.background_path))
        self.settings.setValue("filter/snr", self.snr_spin.value())
        self.settings.setValue("filter/cnn", self.cnn_spin.value())
        self.settings.setValue("filter/repeatability", self.repeat_spin.value())
        self.settings.setValue("filter/r_train", self.r_train_spin.value())
        self.settings.setValue("filter/r_test", self.r_test_spin.value())
        self.settings.setValue("filter/accepted_only", self.accepted_check.isChecked())
        self.settings.setValue("display/color_by", self.color_by_combo.currentText())
        self.settings.setValue("display/marker_size", self.marker_size_spin.value())
        self.settings.setValue("display/mode", self.mode_combo.currentText())
        if self.selected_iloc is not None:
            self.settings.setValue("display/selected_iloc", int(self.selected_iloc))

    def showEvent(self, event):
        super().showEvent(event)
        # Splitter restore must happen after the widgets have geometry.
        main_state = self.settings.value("splitter/main")
        if main_state is not None:
            self.main_splitter.restoreState(main_state)
        right_state = self.settings.value("splitter/right")
        if right_state is not None:
            self.right_splitter.restoreState(right_state)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    # ---------------------------------------------------------------- opening
    def open_cells(self, path: str | None = None) -> None:
        if path is None or path is False:
            start = str(self.settings.value("paths/cells", "."))
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open cell database",
                start,
                "Cell DB pickle (*.cellDB_pickle *.pickle *.pkl);;All files (*)",
            )
            if not path:
                return
        try:
            self.model.load_cells(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load cells", str(exc))
            return

        self.cells_path_label.setText(f"Cells: {self._short_path(path)}")
        self.setWindowTitle(f"Visual Cortex Cell Browser - {Path(path).name}")
        self._update_filter_spinbox_ranges()
        self._update_display_options()
        self._update_selected_after_filter()
        self._show_metadata_warnings()
        self._update_all()

    def open_background(self, path: str | None = None) -> None:
        if path is None or path is False:
            start = str(self.settings.value("paths/background", "."))
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open background image",
                start,
                "NumPy arrays (*.npy);;All files (*)",
            )
            if not path:
                return
        try:
            self.model.load_background(path, transpose=True)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load background", str(exc))
            return
        self.background_path_label.setText(f"Background: {self._short_path(path)}")
        self._update_all()

    def _show_metadata_warnings(self) -> None:
        if self.model.warnings:
            QMessageBox.warning(self, "Metadata warning", "\n".join(self.model.warnings))

    # ---------------------------------------------------------------- updates
    def on_filter_changed(self, *_args) -> None:
        if self._restoring:
            return
        self.filter_settings = FilterSettings(
            snr_min=self.snr_spin.value(),
            cnn_min=self.cnn_spin.value(),
            accepted_only=self.accepted_check.isChecked(),
            repeatability_min=self.repeat_spin.value(),
            r_train_min=self.r_train_spin.value(),
            r_test_min=self.r_test_spin.value(),
        )
        self.model.apply_filters(self.filter_settings)
        self._update_selected_after_filter()
        self._update_all()

    def on_display_changed(self, *_args) -> None:
        if self._restoring:
            return
        self.spatial_panel.update_spatial_view()
        self._save_settings()

    def on_mode_changed(self, *_args) -> None:
        if self._restoring:
            return
        self._update_matplotlib()
        self._save_settings()

    def _update_all(self) -> None:
        self._update_controls_enabled()
        self._update_filter_label()
        self.spatial_panel.update_spatial_view()
        self._update_matplotlib()
        self._save_settings()

    def _update_matplotlib(self) -> None:
        mode = self.mode_combo.currentText()
        if mode == "Stat":
            self.mpl_renderer.draw_stat(self.model)
        else:
            self.mpl_renderer.draw_selected(self.model, self.selected_iloc)
        self.canvas.draw_idle()

    def _update_filter_label(self) -> None:
        self.filter_count_label.setText(f"Filtered cells: {self.model.filtered_count} / {self.model.total_cells}")

    def _update_controls_enabled(self) -> None:
        has_cells = self.model.has_cells
        has_bg = self.model.background_image is not None
        for w in [self.snr_spin, self.cnn_spin, self.accepted_check, self.repeat_spin, self.r_train_spin, self.r_test_spin,
                  self.color_by_combo, self.marker_size_spin, self.mode_combo]:
            w.setEnabled(has_cells)
        self.reset_bg_colorbar_button.setEnabled(has_bg)

    def _update_filter_spinbox_ranges(self) -> None:
        # Keep the default very broad range but set more meaningful starts if possible.
        if self.model.df is None:
            return
        defaults = {
            self.snr_spin: "SNR",
            self.cnn_spin: "CNN",
            self.repeat_spin: "Repeatability",
            self.r_train_spin: "r_train",
            self.r_test_spin: "r_test",
        }
        for spin, field in defaults.items():
            if field in self.model.df.columns:
                vals = pd.to_numeric(self.model.df[field], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    lo = min(-1.0, float(np.floor(vals.min() * 10) / 10) - 1.0)
                    hi = max(1.0, float(np.ceil(vals.max() * 10) / 10) + 1.0)
                    spin.setRange(lo, hi)
            spin.setEnabled(field in self.model.df.columns)
        self.accepted_check.setEnabled("Accepted" in self.model.df.columns)

    def _update_display_options(self) -> None:
        previous = str(self.settings.value("display/color_by", self.color_by_combo.currentText()))
        opts = self.model.display_options()
        self.color_by_combo.blockSignals(True)
        self.color_by_combo.clear()
        self.color_by_combo.addItems(opts)
        self.color_by_combo.setCurrentText(previous if previous in opts else "None")
        self.color_by_combo.blockSignals(False)

    def _update_selected_after_filter(self) -> None:
        if self.model.filtered_indices.size == 0:
            self.selected_iloc = None
            return
        if self.selected_iloc is None or self.selected_iloc not in set(map(int, self.model.filtered_indices)):
            stored = self.settings.value("display/selected_iloc", None)
            try:
                stored_i = int(stored) if stored is not None else None
            except Exception:
                stored_i = None
            if stored_i is not None and stored_i in set(map(int, self.model.filtered_indices)):
                self.selected_iloc = stored_i
            else:
                self.selected_iloc = int(self.model.filtered_indices[0])

    def select_cell(self, iloc: int) -> None:
        self.selected_iloc = int(iloc)
        self.spatial_panel.update_spatial_view()
        if self.mode_combo.currentText() == "Selected":
            self._update_matplotlib()
        self._save_settings()

    def update_hover_text(self, text: str) -> None:
        self.hover_text.setPlainText(text)

    def _short_path(self, path: str | Path, max_len: int = 45) -> str:
        s = str(path)
        if len(s) <= max_len:
            return s
        return "..." + s[-max_len:]


class SpatialPanel(QWidget):
    def __init__(self, main_window: MainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_item: Optional[pg.ImageItem] = None
        self.scatter_item: Optional[pg.ScatterPlotItem] = None
        self.selected_item: Optional[pg.ScatterPlotItem] = None
        self.background_colorbar = None
        self._last_image_levels: Optional[tuple[float, float]] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setDefaultPadding(0.0)
        self.plot_widget.getPlotItem().setAspectLocked(True)
        self.plot_widget.getPlotItem().showAxes(True, showValues=(True, False, False, True))
        self.plot_widget.getPlotItem().invertY(True)
        layout.addWidget(self.plot_widget)

        self.plot_widget.sceneObj.sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.sceneObj.sigMouseClicked.connect(self._on_mouse_clicked)

    def update_spatial_view(self) -> None:
        model = self.main_window.model
        self.plot_widget.clear()
        self.image_item = None
        self.scatter_item = None
        self.selected_item = None
        if self.background_colorbar is not None:
            try:
                self.plot_widget.getPlotItem().layout.removeItem(self.background_colorbar)
                self.background_colorbar.deleteLater()
            except Exception:
                pass
            self.background_colorbar = None

        if model.background_image is None and not model.has_cells:
            text = pg.TextItem(text="No data loaded", anchor=(0.5, 0.5), color=_pg_text_color(self.main_window.is_dark_theme))
            self.plot_widget.addItem(text)
            self.plot_widget.getPlotItem().setTitle("Spatial view")
            return

        if model.background_image is not None:
            image = model.background_image
            self.image_item = pg.ImageItem()
            self.image_item.setImage(image, autoLevels=False)
            self.image_item.setRect(QRectF(0, 0, image.shape[0] * model.resolution_x_um, image.shape[1] * model.resolution_y_um))
            self.plot_widget.addItem(self.image_item)
            finite = image[np.isfinite(image)]
            if finite.size:
                levels = (float(np.nanmin(finite)), float(np.nanmax(finite)))
                self._last_image_levels = levels
                self.image_item.setLevels(levels)
            try:
                self.background_colorbar = self.plot_widget.getPlotItem().addColorBar(self.image_item, colorMap="viridis", rounding=1e-10)
            except Exception:
                self.background_colorbar = None

        if model.has_cells:
            self._draw_cell_scatter()

        title = "Spatial view"
        metric = self.main_window.color_by_combo.currentText()
        if metric != "None":
            title += f" | cells colored by {metric}"
        self.plot_widget.getPlotItem().setTitle(title)
        self.plot_widget.getPlotItem().setLabel("bottom", "x (µm)")
        self.plot_widget.getPlotItem().setLabel("left", "y (µm)")

    def _draw_cell_scatter(self) -> None:
        model = self.main_window.model
        x, y, ilocs = model.filtered_xy_um()
        if x.size == 0:
            return
        size = self.main_window.marker_size_spin.value()
        metric_label = self.main_window.color_by_combo.currentText()
        field = DISPLAY_FIELD_MAP.get(metric_label)

        if field is None:
            brushes = [pg.mkBrush(220, 30, 30, 210) for _ in ilocs]
        else:
            values = model.numeric_values_for_filtered(field)
            brushes = _brushes_from_values(values, circular=metric_label in ["angle", "phase"])

        spots = []
        for xi, yi, idx, brush in zip(x, y, ilocs, brushes):
            spots.append({"pos": (float(xi), float(yi)), "data": int(idx), "brush": brush, "pen": pg.mkPen(None), "size": size})
        self.scatter_item = pg.ScatterPlotItem(spots=spots, pxMode=True)
        self.plot_widget.addItem(self.scatter_item)

        if self.main_window.selected_iloc is not None and self.main_window.selected_iloc in set(map(int, ilocs)):
            row = model.cell_row_by_iloc(self.main_window.selected_iloc)
            sx = float(row["Soma_Xpix"]) * model.resolution_x_um
            sy = float(row["Soma_Ypix"]) * model.resolution_y_um
            self.selected_item = pg.ScatterPlotItem(
                x=[sx],
                y=[sy],
                size=max(size + 8, 14),
                pxMode=True,
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen(255, 255, 0, 255, width=2),
            )
            self.plot_widget.addItem(self.selected_item)

    def reset_background_colorbar(self) -> None:
        if self.image_item is not None and self.main_window.model.background_image is not None:
            image = self.main_window.model.background_image
            finite = image[np.isfinite(image)]
            if finite.size:
                levels = (float(np.nanmin(finite)), float(np.nanmax(finite)))
                self.image_item.setLevels(levels)
                if self.background_colorbar is not None:
                    self.background_colorbar.setLevels(levels)

    def _on_mouse_moved(self, scene_pos) -> None:
        model = self.main_window.model
        if not self.plot_widget.sceneBoundingRect().contains(scene_pos):
            return
        pos = self.plot_widget.getViewBox().mapSceneToView(scene_pos)
        x_um = float(pos.x())
        y_um = float(pos.y())
        text = f"X: {x_um:.2f} µm\nY: {y_um:.2f} µm"

        if model.has_cells and model.filtered_indices.size > 0:
            nearest, dist = model.nearest_filtered_cell(x_um, y_um)
            tolerance = max(8.0, self.main_window.marker_size_spin.value() * max(model.resolution_x_um, model.resolution_y_um))
            if nearest is not None and dist <= tolerance:
                row = model.cell_row_by_iloc(nearest)
                cell_id = row.get("cell_id", nearest)
                text += f"\nCell: {cell_id}"
                metric_label = self.main_window.color_by_combo.currentText()
                field = DISPLAY_FIELD_MAP.get(metric_label)
                if field is not None and field in row.index:
                    text += f"\n{metric_label}: {row.get(field)}"
                for f in ["Repeatability", "r_train", "r_test"]:
                    if f in row.index:
                        try:
                            text += f"\n{f}: {float(row.get(f)):.3f}"
                        except Exception:
                            text += f"\n{f}: {row.get(f)}"
        self.main_window.update_hover_text(text)

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        model = self.main_window.model
        if not model.has_cells or model.filtered_indices.size == 0:
            return
        pos = self.plot_widget.getViewBox().mapSceneToView(event.scenePos())
        nearest, dist = model.nearest_filtered_cell(float(pos.x()), float(pos.y()))
        tolerance = max(10.0, self.main_window.marker_size_spin.value() * max(model.resolution_x_um, model.resolution_y_um) * 1.5)
        if nearest is not None and dist <= tolerance:
            self.main_window.select_cell(nearest)
            event.accept()


def _brushes_from_values(values: np.ndarray, circular: bool = False) -> list:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [pg.mkBrush(128, 128, 128, 180) for _ in values]
    if circular:
        # Normalize circular variables in degrees. This works for 0-180 and 0-360 values.
        period = 360.0 if np.nanmax(finite) > 180 else 180.0
        normed = (values % period) / period
        cmap = mpl_cm.get_cmap("hsv")
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        normed = norm(values)
        cmap = mpl_cm.get_cmap("viridis")
    brushes = []
    for val, nval in zip(values, normed):
        if not np.isfinite(val):
            brushes.append(pg.mkBrush(128, 128, 128, 80))
            continue
        rgba = cmap(float(nval))
        brushes.append(pg.mkBrush(*(int(255 * c) for c in rgba[:3]), 220))
    return brushes


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: bold; margin-top: 8px;")
    return label


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _labeled_widget(label_text: str, widget: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    label = QLabel(label_text)
    label.setMinimumWidth(92)
    layout.addWidget(label)
    layout.addWidget(widget, stretch=1)
    return layout


def _double_spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setSingleStep(step)
    spin.setDecimals(4)
    return spin


def _settings_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1", "yes"]


def _pg_text_color(is_dark: bool):
    return QColor("white") if is_dark else QColor("black")
