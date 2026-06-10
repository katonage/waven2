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
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from GripSplitter import GripSplitter  # type: ignore
except Exception:  # package import fallback
    from .GripSplitter import GripSplitter  # type: ignore

try:
    from data_model import CellDataModel, DISPLAY_FIELD_MAP, FilterSettings
    from plotting_mpl import MatplotlibRenderer
except Exception:  # package import fallback
    from .data_model import CellDataModel, DISPLAY_FIELD_MAP, FilterSettings
    from .plotting_mpl import MatplotlibRenderer


class MainWindow(QMainWindow):
    def __init__(self, cells_path: str | None = None, background_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Visual Cortex Cell Browser")

        self.settings = QSettings("RCNS", "VisualCortexCellBrowser")
        self.model = CellDataModel()
        self.filter_settings = FilterSettings()
        self.selected_iloc: Optional[int] = None
        self._restoring = False
        self._splitters_restored = False

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
        main_splitter.setSizes([190, 1200])

        self.spatial_panel = SpatialPanel(self)
        right_splitter.addWidget(self.spatial_panel)

        self.bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(2)

        self.figure = Figure(figsize=(10, 4.1), constrained_layout=False)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_renderer = MatplotlibRenderer(self.figure)
        bottom_layout.addWidget(self.canvas, stretch=3)

        self.temporal_panel = TemporalTracePanel(self)
        bottom_layout.addWidget(self.temporal_panel, stretch=2)
        self.temporal_panel.hide()

        right_splitter.addWidget(self.bottom_panel)
        right_splitter.setSizes([520, 500])

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
        scroll.setMinimumWidth(130)
        #scroll.setMaximumWidth(225)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        scroll.setWidget(content)

        # Load
        layout.addWidget(_section_label("Load"))
        self.open_cells_button = QPushButton("Open Cells...")
        self.open_background_button = QPushButton("Open Background...")
        self.cells_path_label = QLabel("Cells: --\nSeriesID: --")
        self.cells_path_label.setWordWrap(True)
        self.background_path_label = QLabel("Background: --")
        self.background_path_label.setWordWrap(True)
        self.reset_bg_colorbar_button = QPushButton("Reset BG Colorbar")
        layout.addWidget(self.open_cells_button)
        layout.addWidget(self.open_background_button)
        layout.addWidget(self.cells_path_label)
        layout.addWidget(self.background_path_label)
        layout.addWidget(self.reset_bg_colorbar_button)
        layout.addWidget(_separator())

        # Filter
        layout.addWidget(_section_label("Filter"))
        self.filter_count_label = QLabel("Filtered cells: 0 / 0")
        #self.filter_count_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.filter_count_label)

        self.snr_spin = _double_spin(-999.0, 999.0, -999.0, 0.1)
        self.cnn_spin = _double_spin(-999.0, 999.0, -999.0, 0.05)
        self.accepted_check = QCheckBox("Accepted only")
        self.repeat_spin = _double_spin(-999.0, 999.0, -999.0, 0.05)
        self.r_train_spin = _double_spin(-999.0, 999.0, -999.0, 0.05)
        self.r_test_spin = _double_spin(-999.0, 999.0, -999.0, 0.05)

        layout.addLayout(_labeled_widget("SNR >", self.snr_spin))
        layout.addLayout(_labeled_widget("CNN >", self.cnn_spin))
        layout.addWidget(self.accepted_check)
        layout.addLayout(_labeled_widget("Repeat. >", self.repeat_spin))
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
        self.marker_size_spin.setMinimumWidth(90)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Selected", "Stat"])

        layout.addLayout(_labeled_widget("Color:", self.color_by_combo))
        layout.addLayout(_labeled_widget("Size:", self.marker_size_spin))
        layout.addLayout(_labeled_widget("Mode:", self.mode_combo))
        layout.addWidget(_separator())

        # Hover info as separate section
        layout.addWidget(_section_label("Hover info"))
        self.hover_text = QTextEdit()
        self.hover_text.setReadOnly(True)
        self.hover_text.setMinimumHeight(105)
        self.hover_text.setMaximumHeight(245)
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
            self.snr_spin.setValue(float(self.settings.value("filter/snr", -999.0)))
            self.cnn_spin.setValue(float(self.settings.value("filter/cnn", -999.0)))
            self.repeat_spin.setValue(float(self.settings.value("filter/repeatability", -999.0)))
            self.r_train_spin.setValue(float(self.settings.value("filter/r_train", -999.0)))
            self.r_test_spin.setValue(float(self.settings.value("filter/r_test", -999.0)))
            self.accepted_check.setChecked(_settings_bool(self.settings.value("filter/accepted_only", False)))
            self.marker_size_spin.setValue(int(self.settings.value("display/marker_size", 7)))
            mode = str(self.settings.value("display/mode", "Selected"))
            if mode in [self.mode_combo.itemText(i) for i in range(self.mode_combo.count())]:
                self.mode_combo.setCurrentText(mode)
            self.filter_settings = self._read_filter_settings_from_widgets()
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
        if self._splitters_restored:
            return
        self._splitters_restored = True
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

        progress = QProgressDialog("Opening cell database...", None, 0, 0, self)
        progress.setWindowTitle("Loading Cells")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setFixedWidth(360)
        progress.show()
        QApplication.processEvents()

        try:
            self.model.load_cells(path)
            progress.setRange(0, 100)
            progress.setValue(55)
            progress.setLabelText("Updating controls...")
            QApplication.processEvents()

            #self.cells_path_label.setText(f"Cells: {self._short_path(path)}\nSeriesID: {self.model.series_id}")
            self.cells_path_label.setText(f"SeriesID: {self.model.series_id}")
            self.setWindowTitle(f"Visual Cortex Cell Browser - {Path(path).name}")
            self._update_filter_spinbox_ranges()
            self._update_display_options()

            progress.setValue(75)
            progress.setLabelText("Applying filters...")
            QApplication.processEvents()
            self.filter_settings = self._read_filter_settings_from_widgets()
            self.model.apply_filters(self.filter_settings)
            self._update_selected_after_filter()

            progress.setValue(88)
            progress.setLabelText("Rendering...")
            QApplication.processEvents()
            self._show_metadata_warnings()
            self._update_all()
            progress.setValue(100)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load cells", str(exc))
        finally:
            progress.close()

    def open_background(self, path: str | None = None) -> None:
        if path is None or path is False:
            start_path = self._background_dialog_start_path()
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open background image",
                start_path,
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

    def _background_dialog_start_path(self) -> str:
        # Prefer the image folder indicated by the CaImAn HDF5 path stored in metadata.
        # This is intentionally the folder, not the HDF5 filename.
        hdf5_folder = self.model.hdf5_folder
        if hdf5_folder is not None:
            return str(hdf5_folder)
        bg_path = self.settings.value("paths/background", None)
        if bg_path:
            return str(bg_path)
        if self.model.cells_path is not None:
            return str(self.model.cells_path.parent)
        return "."

    def _show_metadata_warnings(self) -> None:
        if self.model.warnings:
            QMessageBox.warning(self, "Metadata warning", "\n".join(self.model.warnings))

    # ---------------------------------------------------------------- updates
    def _read_filter_settings_from_widgets(self) -> FilterSettings:
        return FilterSettings(
            snr_min=self.snr_spin.value(),
            cnn_min=self.cnn_spin.value(),
            accepted_only=self.accepted_check.isChecked(),
            repeatability_min=self.repeat_spin.value(),
            r_train_min=self.r_train_spin.value(),
            r_test_min=self.r_test_spin.value(),
        )

    def on_filter_changed(self, *_args) -> None:
        if self._restoring:
            return
        self.filter_settings = self._read_filter_settings_from_widgets()
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
        self._update_matplotlib_and_temporal()
        self._save_settings()

    def _update_all(self) -> None:
        self._update_controls_enabled()
        self._update_filter_label()
        self.spatial_panel.update_spatial_view()
        self._update_matplotlib_and_temporal()
        self._save_settings()

    def _update_matplotlib_and_temporal(self) -> None:
        mode = self.mode_combo.currentText()
        if mode == "Stat":
            self.temporal_panel.hide()
            self.mpl_renderer.draw_stat(self.model)
        else:
            self.temporal_panel.show()
            self.mpl_renderer.draw_selected(self.model, self.selected_iloc)
            self.temporal_panel.update_temporal_view()
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
            spin.blockSignals(True)
            old_value = spin.value()
            if field in self.model.df.columns:
                vals = pd.to_numeric(self.model.df[field], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    lo = min(-1.0, float(np.floor(vals.min() * 100) / 100) - 0.1)
                    hi = max(1.0, float(np.ceil(vals.max() * 100) / 100) + 0.1)
                    spin.setRange(lo, hi)
                    spin.setValue(max(lo, min(old_value, hi)))
                spin.setEnabled(True)
            else:
                spin.setEnabled(False)
            spin.blockSignals(False)
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
        filtered_set = set(map(int, self.model.filtered_indices))
        if self.selected_iloc is None or self.selected_iloc not in filtered_set:
            stored = self.settings.value("display/selected_iloc", None)
            try:
                stored_i = int(stored) if stored is not None else None
            except Exception:
                stored_i = None
            if stored_i is not None and stored_i in filtered_set:
                self.selected_iloc = stored_i
            else:
                self.selected_iloc = int(self.model.filtered_indices[0])

    def select_cell(self, iloc: int) -> None:
        self.selected_iloc = int(iloc)
        self.spatial_panel.update_spatial_view()
        if self.mode_combo.currentText() == "Selected":
            self._update_matplotlib_and_temporal()
        self._save_settings()

    def update_hover_text(self, text: str) -> None:
        self.hover_text.setPlainText(text)

    def _short_path(self, path: str | Path, max_len: int = 28) -> str:
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
            gray_cmap = pg.colormap.get("gray", source="matplotlib") 
            try:
                self.image_item.setLookupTable(gray_cmap.getLookupTable(nPts=256))
            except Exception:
                pass
            self.image_item.setImage(image, autoLevels=False)
            self.image_item.setRect(QRectF(0, 0, image.shape[0] * model.resolution_x_um, image.shape[1] * model.resolution_y_um))
            self.plot_widget.addItem(self.image_item)
            finite = image[np.isfinite(image)]
            if finite.size:
                levels = (float(np.nanmin(finite)), float(np.nanmax(finite)))
                self.image_item.setLevels(levels)
            try:
                self.background_colorbar = self.plot_widget.getPlotItem().addColorBar(self.image_item, colorMap=gray_cmap, rounding=1e-10)
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
            brushes = _brushes_from_values(values, metric_label=metric_label)

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
                    try:
                        self.background_colorbar.setLevels(levels)
                    except Exception:
                        pass

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


class PlotWidgetWithRightAxis(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.showAxis("right")
        self.RightViewBox = pg.ViewBox()
        self.plotItem.scene().addItem(self.RightViewBox)
        self.getAxis("right").linkToView(self.RightViewBox)
        self.RightViewBox.setXLink(self)
        self.plotItem.vb.sigResized.connect(self._update_views)
        self._update_views()

    def _update_views(self):
        self.RightViewBox.setGeometry(self.plotItem.vb.sceneBoundingRect())
        self.RightViewBox.linkedViewChanged(self.plotItem.vb, self.RightViewBox.XAxis)

    def clear_all(self):
        self.clear()
        self.RightViewBox.clear()


class TemporalTracePanel(QWidget):
    def __init__(self, main_window: MainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 4)
        self.temporal_view = PlotWidgetWithRightAxis()
        self.temporal_view.setDefaultPadding(0.0)
        self.temporal_view.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.temporal_view)

    def update_temporal_view(self) -> None:
        model = self.main_window.model
        selected = self.main_window.selected_iloc
        self.temporal_view.clear_all()
        self.temporal_view.getPlotItem().showAxes(True, showValues=(True, False, False, True))
        self.temporal_view.getAxis("right").setStyle(showValues=False)
        self.temporal_view.setLabel("bottom", "Time", units="s")
        self.temporal_view.setLabel("left", "Cell mean activity")
        self.temporal_view.setLabel("right", "")

        if not model.has_cells or selected is None:
            text = pg.TextItem(text="No selected cell", anchor=(0.5, 0.5), color=_pg_text_color(self.main_window.is_dark_theme))
            self.temporal_view.addItem(text)
            return

        row = model.cell_row_by_iloc(selected)
        cell_id = row.get("cell_id", selected)
        rep = _fmt(row.get("Repeatability", np.nan))
        r_train = _fmt(row.get("r_train", np.nan))
        r_test = _fmt(row.get("r_test", np.nan))
        self.temporal_view.getPlotItem().setTitle(f"Cell {cell_id} | Repeatability={rep} | r_train={r_train} | r_test={r_test}")

        activity = _as_numeric_array(row.get("Cell_activity", None))
        plotted = False
        if activity is not None:
            if activity.ndim == 1:
                activity = activity[None, :]
            n_time = activity.shape[-1]
            t = np.arange(n_time, dtype=float) / float(model.target_fps)
            trial_pen = pg.mkPen((80, 80, 80, 80), width=0.8)
            mean_pen = pg.mkPen("b", width=2)
            for trial in activity:
                self.temporal_view.plot(t, trial, pen=trial_pen)
            self.temporal_view.plot(t, np.nanmean(activity, axis=0), pen=mean_pen)
            plotted = True
        else:
            text = pg.TextItem(text="No Cell_activity", anchor=(0.02, 0.90), color=_pg_text_color(self.main_window.is_dark_theme))
            self.temporal_view.addItem(text)

        transient = _as_numeric_array(row.get("WL_transient_mod", None))
        if transient is not None:
            t2 = np.arange(len(transient), dtype=float) / float(model.target_fps)
            item = pg.PlotCurveItem(t2, transient, pen=pg.mkPen("r", width=1.3))
            self.temporal_view.RightViewBox.addItem(item)
            self.temporal_view.getAxis("right").setStyle(showValues=True)
            self.temporal_view.setLabel("right", "WL transient")
            plotted = True

        if plotted:
            self.temporal_view.getViewBox().autoRange()
            self.temporal_view.RightViewBox.autoRange()


def _brushes_from_values(values: np.ndarray, metric_label: str = "") -> list:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [pg.mkBrush(128, 128, 128, 180) for _ in values]

    metric = metric_label.lower()
    if metric in ["angle", "phase", "Angle_fit_ori"]:
        vmax = float(np.nanmax(np.abs(finite)))
        if metric == "angle":
            period = np.pi if vmax <= 2 * np.pi + 1e-6 else 180.0
        else:
            period = 2 * np.pi if vmax <= 2 * np.pi + 1e-6 else 360.0
        normed = (values % period) / period
        cmap = mpl_cm.get_cmap("hsv")
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        normed = norm(values)
        cmap = mpl_cm.get_cmap("rainbow")

    brushes = []
    for val, nval in zip(values, normed):
        if not np.isfinite(val):
            brushes.append(pg.mkBrush(128, 128, 128, 80))
            continue
        rgba = cmap(float(nval))
        brushes.append(pg.mkBrush(*(int(255 * c) for c in rgba[:3]), 220))
    return brushes


def _as_numeric_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    try:
        arr = arr.astype(float)
    except Exception:
        return None
    return arr


def _fmt(value) -> str:
    try:
        v = float(value)
        if not np.isfinite(v):
            return "nan"
        return f"{v:.3f}"
    except Exception:
        return "nan"


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: bold; margin-top: 6px;")
    return label


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _labeled_widget(label_text: str, widget: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(label_text)
    label.setMinimumWidth(66)
    layout.addWidget(label)
    layout.addWidget(widget, stretch=1)
    return layout


def _double_spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setSingleStep(step)
    spin.setDecimals(2)
    spin.setMinimumWidth(90)
    return spin


def _settings_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1", "yes"]


def _pg_text_color(is_dark: bool):
    return QColor("white") if is_dark else QColor("black")
