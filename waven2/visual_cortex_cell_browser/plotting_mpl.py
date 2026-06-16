from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

try:
    from data_model import CellDataModel, TUNING_SPECS
except Exception:  # package import fallback
    from .data_model import CellDataModel, TUNING_SPECS
try:
    from ..analysis_utils import sine1x, fit_quadratic, restore_fit_quadratic
except Exception:  # package import fallback
    from pathlib import Path
    import sys

    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))
    from analysis_utils import sine1x, fit_quadratic, restore_fit_quadratic

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
        arr = np.atleast_1d(arr.astype(float))
    except Exception:
        return None
    return arr


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


class MatplotlibRenderer:
    def __init__(self, figure: Figure):
        self.figure = figure
        self._apply_small_fonts()

    def _apply_small_fonts(self) -> None:
        # Apply to this Figure only. Avoid global matplotlib style changes.
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.titlesize": 8,
            "axes.labelpad": 2,
        })

    def draw_empty(self, message: str = "No data loaded") -> None:
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        self.figure.tight_layout()

    def draw_selected(self, model: CellDataModel, selected_iloc: Optional[int]) -> None:
        """Draw only the 2 x 4 tuning-summary panel.

        The selected cell temporal trace is drawn by the PyQtGraph panel in main_window.py,
        because it needs interactive pan/zoom.
        """
        self.figure.clf()
        if not model.has_cells or selected_iloc is None:
            self.draw_empty("Load cells and select a cell")
            return

        row = model.cell_row_by_iloc(selected_iloc)
        gs = self.figure.add_gridspec(2, 4, hspace=0.50, wspace=0.38)
        axes = [self.figure.add_subplot(gs[i // 4, i % 4]) for i in range(8)]

        for ax, (label, best_field, tun_field, axis_key) in zip(axes[:7], TUNING_SPECS):
            self._plot_one_tuning_axis(ax, model, row, label, best_field, tun_field, axis_key)

        self._plot_tuning_map(axes[7], model, row)
        self.figure.subplots_adjust(left=0.045, right=0.985, top=0.93, bottom=0.10, hspace=0.55, wspace=0.36)

    def _plot_one_tuning_axis(self, ax, model: CellDataModel, row: pd.Series, label: str, best_field: str, tun_field: str, axis_key: str) -> None:
        if tun_field not in row.index:
            ax.text(0.5, 0.5, f"No {tun_field}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.axis("off")
            return
        y = _as_numeric_array(row.get(tun_field))
        if y is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.axis("off")
            return
        x = model.wavelet_axis(axis_key)
        if x is None or len(x) != len(y):
            x = np.arange(len(y), dtype=float)
            xlabel = "index"
        else:
            xlabel = axis_key
        ax.plot(x, y, marker="o", markersize=2.5, linewidth=0.8)
        ax.axhline(0, color="0.5", linewidth=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("r")
        best = row.get(best_field, np.nan)
        try:
            ax.axvline(float(best), color="red", linestyle="--", linewidth=0.8)
            ax.set_title(f"{label} | {float(best):.1f}")
        except Exception:
            ax.set_title(label)
        
        #fit plots
        if label == "angle":
            # The notebook stores angles and orientation in radians:
            amp = row.get("Angle_fit_amplitude", np.nan)
            const = row.get("Angle_fit_constant", np.nan)
            ori = row.get("Angle_fit_ori", np.nan)
            try:
                amp = float(amp)
                const = float(const)
                ori = float(ori)
                if not np.all(np.isfinite([amp, const, ori])) or np.asarray(x).size < 2:
                    return
                #x = np.asarray(x, dtype=float)
                xf = np.linspace(0, np.pi, 100)
                yf = sine1x(xf, const, amp, ori)
                #yf = const + amp * (np.cos(2.0 * (xf - ori)) + 1.0) / 2.0
                ax.plot(xf, yf, color="0.45", linewidth=0.9)
                ax.axvline(ori, color="0.45", linewidth=0.6)
                #ax.axhline(const, color="0.45", linestyle="--", linewidth=0.5)
                ax.set_title(f"angle | fitted ori={np.rad2deg(ori):.1f}°")
            except Exception:
                pass
        
        if label in ["x", "y", "size", "freq", "drift"]:
            best, fit_params = fit_quadratic(x, y)
            ax.axvline(best, color="0.45", linewidth=0.6)

            if fit_params is not None:
                x_fit, y_fit = restore_fit_quadratic(fit_params)
                ax.plot(x_fit, y_fit, color="0.45", linewidth=0.9)
                ax.set_title(f"{ax.get_title()} | fitted={best:.1f}")
        

    def _plot_tuning_map(self, ax, model: CellDataModel, row: pd.Series) -> None:
        ax.set_title("2D RF")
        arr = _as_numeric_array(row.get("tun_xy", None))
        if arr is None or arr.ndim != 2:
            ax.text(0.5, 0.5, "No tun_xy", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.axis("off")
            return
        extent = None
        if model.visual_coverage is not None and len(model.visual_coverage) >= 4:
            az_left, az_right, el_bottom, el_top = model.visual_coverage[:4]
            extent = [az_left, az_right, el_bottom, el_top]
        im = ax.imshow(arr.T, origin="lower", aspect="equal", extent=extent)
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation")
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    def draw_stat(self, model: CellDataModel) -> None:
        self.figure.clf()
        if not model.has_cells:
            self.draw_empty("No cells loaded")
            return
        if model.filtered_indices.size == 0:
            self.draw_empty("No cells pass current filters")
            return

        gs = self.figure.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.12], hspace=0.55, wspace=0.38)
        axes = [self.figure.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
        ax_train = self.figure.add_subplot(gs[2, 0:2])
        ax_test = self.figure.add_subplot(gs[2, 2:4])

        for ax, (label, field, _, axis_key) in zip(axes[:7], TUNING_SPECS):
            self._plot_hist(ax, model, label, field, axis_key)
        self._plot_rf_scatter(axes[7], model)
        self._plot_fit(ax_train, model, "Repeatability", "r_train", through_origin=False)
        self._plot_fit(ax_test, model, "Repeatability", "r_test", through_origin=True)
        self.figure.subplots_adjust(left=0.045, right=0.985, top=0.95, bottom=0.08, hspace=0.55, wspace=0.36)
        ax_train.set_aspect("equal", adjustable="box")
        ax_test.set_aspect("equal", adjustable="box")

    def _plot_hist(self, ax, model: CellDataModel, label: str, field: str, axis_key: str) -> None:
        ax.set_title(label)
        if model.df is None or field not in model.df.columns:
            ax.text(0.5, 0.5, f"No {field}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.axis("off")
            return
        vals = model.numeric_values_for_filtered(field)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.axis("off")
            return
        axis_values = model.wavelet_axis(axis_key)
        if axis_values is not None and len(axis_values) > 1:
            bins = _bins_from_centers(axis_values)
        else:
            bins = min(20, max(5, int(np.sqrt(vals.size))))
        ax.hist(vals, bins=bins)
        ax.set_xlabel(field)
        ax.set_ylabel("n")

    def _plot_rf_scatter(self, ax, model: CellDataModel) -> None:
        ax.set_title("RF positions")
        if model.df is None or "Azimuth" not in model.df.columns or "Elevation" not in model.df.columns:
            ax.text(0.5, 0.5, "No RF coordinates", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.axis("off")
            return
        x = model.numeric_values_for_filtered("Azimuth")
        y = model.numeric_values_for_filtered("Elevation")
        x, y = _clean_xy(x, y)
        ax.scatter(x, y, s=8, alpha=0.6)
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation")
        ax.set_aspect("equal", adjustable="box")
        if model.visual_coverage is not None and len(model.visual_coverage) >= 4:
            az_left, az_right, el_bottom, el_top = model.visual_coverage[:4]
            ax.set_xlim(az_left, az_right)
            ax.set_ylim(el_bottom, el_top)

    def _plot_fit(self, ax, model: CellDataModel, x_field: str, y_field: str, through_origin: bool) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(x_field)
        ax.set_ylabel(y_field)
        if model.df is None or x_field not in model.df.columns or y_field not in model.df.columns:
            ax.set_title(f"{y_field}: missing")
            return

        x_all = model.numeric_values_for_all(x_field)
        y_all = model.numeric_values_for_all(y_field)
        x_all, y_all = _clean_xy(x_all, y_all)
        ax.scatter(x_all, y_all, s=8, alpha=0.30, color="0.55", label="all")

        x_f = model.numeric_values_for_filtered(x_field)
        y_f = model.numeric_values_for_filtered(y_field)
        x_f, y_f = _clean_xy(x_f, y_f)
        ax.scatter(x_f, y_f, s=9, alpha=0.70, color="tab:blue", label="filtered")

        if x_all.size < 2:
            ax.set_title(f"{y_field}: few data")
            return

        xx = np.linspace(0, 1, 200)
        if through_origin:
            denom = float(np.sum(x_all ** 2))
            a = float(np.sum(x_all * y_all) / denom) if denom > 0 else np.nan
            ax.plot(xx, a * xx, linewidth=1.2, color="black")
            ax.set_title(f"{y_field}: y={a:.3f}x")
        else:
            a, b = np.polyfit(x_all, y_all, 1)
            ax.plot(xx, a * xx + b, linewidth=1.2, color="black")
            ax.set_title(f"{y_field}: y={a:.3f}x+{b:.3f}")


def _fmt(value) -> str:
    try:
        v = float(value)
        if not np.isfinite(v):
            return "nan"
        return f"{v:.3f}"
    except Exception:
        return "nan"


def _bins_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    centers = np.sort(centers[np.isfinite(centers)])
    if centers.size < 2:
        return 10
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges
