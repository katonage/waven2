from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from twop_analysis.cell_browser.plugins import BrowserContext, DisplayField, NumericFilter

from .data_view import WavenDataView


class WavenBrowserPlugin:
    id = "waven2"
    label = "Waven2"
    priority = 100
    modes = ("Selected", "Stat")
    filters = (
        NumericFilter("Repeatability", "Repeat. >"),
        NumericFilter("r_train", "r_train >"),
        NumericFilter("r_test", "r_test >"),
    )
    display_fields = {
        "x": DisplayField("Azimuth"),
        "y": DisplayField("Elevation"),
        "angle": DisplayField("Angle", "hsv", np.pi, 180.0),
        "size": DisplayField("Size"),
        "freq": DisplayField("Frequency"),
        "phase": DisplayField("Phase", "hsv", 2 * np.pi, 360.0),
        "Angle_fit_ori": DisplayField("Angle_fit_ori", "hsv", 2 * np.pi, 360.0),
        "Angle_fit_OSI": DisplayField("Angle_fit_OSI"),
        "Repeatability": DisplayField("Repeatability"),
        "r_train": DisplayField("r_train"),
        "r_test": DisplayField("r_test"),
    }
    hover_fields = ("Repeatability", "r_train", "r_test")

    def matches(self, model) -> bool:
        if model.df is None:
            return False
        return (
            "wavelet_params" in model.metadata
            or bool(set(model.df.columns) & {
                "tun_xs", "tun_ys", "tun_angles", "tun_sizes", "tun_freqs",
                "tun_phases", "tun_xy", "WL_transient_mod", "WL_transient_phase",
            })
        )

    def create_widget(self, context: BrowserContext, parent: QWidget) -> QWidget:
        return WavenResultsWidget(context, parent)


class WavenResultsWidget(QWidget):
    def __init__(self, context: BrowserContext, parent: QWidget):
        super().__init__(parent)
        # Plotting and analysis dependencies are only loaded for Waven databases.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from .plotting_mpl import MatplotlibRenderer
        from .temporal import TemporalTracePanel

        self.model = WavenDataView(context.model)
        self.selected_iloc = context.selected_iloc
        self.is_dark_theme = context.is_dark_theme
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.figure = Figure(figsize=(10, 4.1), constrained_layout=False)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_renderer = MatplotlibRenderer(self.figure)
        layout.addWidget(self.canvas, stretch=3)
        self.temporal_panel = TemporalTracePanel(self)
        layout.addWidget(self.temporal_panel, stretch=2)
        self.temporal_panel.hide()
        self.mpl_renderer.draw_empty()
        self.canvas.draw_idle()

    def update_view(self, context: BrowserContext) -> None:
        self.model = WavenDataView(context.model)
        self.selected_iloc = context.selected_iloc
        self.is_dark_theme = context.is_dark_theme
        if context.mode == "Stat":
            self.temporal_panel.hide()
            self.mpl_renderer.draw_stat(self.model)
        else:
            self.temporal_panel.show()
            self.mpl_renderer.draw_selected(self.model, self.selected_iloc)
            self.temporal_panel.update_temporal_view()
        self.canvas.draw_idle()
