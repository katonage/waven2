from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from .plugin import WavenResultsWidget


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
    def __init__(self, results_widget: WavenResultsWidget):
        super().__init__(results_widget)
        self.results_widget = results_widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 4)
        self.temporal_view = PlotWidgetWithRightAxis()
        self.temporal_view.setDefaultPadding(0.0)
        self.temporal_view.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.temporal_view)

    def update_temporal_view(self) -> None:
        model = self.results_widget.model
        selected = self.results_widget.selected_iloc
        self.temporal_view.clear_all()
        self.temporal_view.getPlotItem().showAxes(True, showValues=(True, False, False, True))
        self.temporal_view.getAxis("right").setStyle(showValues=False)
        self.temporal_view.setLabel("bottom", "Time", units="s")
        self.temporal_view.setLabel("left", "Cell activity")
        self.temporal_view.setLabel("right", "")

        if not model.has_cells or selected is None:
            text = pg.TextItem(text="No selected cell", anchor=(0.5, 0.5), color=_pg_text_color(self.results_widget.is_dark_theme))
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
            text = pg.TextItem(text="No Cell_activity", anchor=(0.02, 0.90), color=_pg_text_color(self.results_widget.is_dark_theme))
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


def _pg_text_color(is_dark: bool):
    return QColor("white") if is_dark else QColor("black")