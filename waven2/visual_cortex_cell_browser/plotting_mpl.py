from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from data_model import CellDataModel, TUNING_SPECS


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


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


class MatplotlibRenderer:
    def __init__(self, figure: Figure):
        self.figure = figure

    def draw_empty(self, message: str = "No data loaded") -> None:
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        self.figure.tight_layout()

    def draw_selected(self, model: CellDataModel, selected_iloc: Optional[int]) -> None:
        self.figure.clf()
        if not model.has_cells or selected_iloc is None:
            self.draw_empty("Load cells and select a cell")
            return

        row = model.cell_row_by_iloc(selected_iloc)
        gs = self.figure.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.25], hspace=0.55, wspace=0.35)
        axes = [self.figure.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
        ax_time = self.figure.add_subplot(gs[2, :])

        for ax, (label, best_field, tun_field, axis_key) in zip(axes[:7], TUNING_SPECS):
            self._plot_one_tuning_axis(ax, model, row, label, best_field, tun_field, axis_key)

        self._plot_tuning_map(axes[7], model, row)
        self._plot_selected_temporal(ax_time, model, row)
        self.figure.tight_layout()

    def _plot_one_tuning_axis(self, ax, model: CellDataModel, row: pd.Series, label: str, best_field: str, tun_field: str, axis_key: str) -> None:
        ax.set_title(label)
        if tun_field not in row.index:
            ax.text(0.5, 0.5, f"No {tun_field}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            return
        y = _as_numeric_array(row.get(tun_field))
        if y is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            return
        x = model.wavelet_axis(axis_key)
        if x is None or len(x) != len(y):
            x = np.arange(len(y), dtype=float)
            ax.set_xlabel("index")
        else:
            ax.set_xlabel(axis_key)
        ax.plot(x, y, marker="o", linewidth=1)
        ax.set_ylabel("corr.")
        best = row.get(best_field, np.nan)
        try:
            if np.isfinite(float(best)):
                ax.axvline(float(best), linestyle="--", linewidth=1)
        except Exception:
            pass

        if label == "angle":
            amp = row.get("Angle_fit_amplitude", np.nan)
            const = row.get("Angle_fit_constant", np.nan)
            ori = row.get("Angle_fit_ori", np.nan)
            try:
                amp = float(amp); const = float(const); ori = float(ori)
                if np.all(np.isfinite([amp, const, ori])) and x.size > 1:
                    xf = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    # Orientation tuning: period 180 degrees, peak at ori.
                    yf = const + amp * (np.cos(np.deg2rad(2 * (xf - ori))) + 1) / 2
                    ax.plot(xf, yf, linewidth=1)
            except Exception:
                pass

    def _plot_tuning_map(self, ax, model: CellDataModel, row: pd.Series) -> None:
        ax.set_title("2D RF tuning")
        arr = _as_numeric_array(row.get("tun_xy", None))
        if arr is None or arr.ndim != 2:
            ax.text(0.5, 0.5, "No tun_xy", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            return
        extent = None
        if model.visual_coverage is not None and len(model.visual_coverage) >= 4:
            az_left, az_right, el_bottom, el_top = model.visual_coverage[:4]
            extent = [az_left, az_right, el_bottom, el_top]
        im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=extent)
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation")
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_selected_temporal(self, ax, model: CellDataModel, row: pd.Series) -> None:
        cell_id = row.get("cell_id", "?")
        rep = row.get("Repeatability", np.nan)
        r_train = row.get("r_train", np.nan)
        r_test = row.get("r_test", np.nan)
        ax.set_title(f"Cell {cell_id} | Repeatability={_fmt(rep)} | r_train={_fmt(r_train)} | r_test={_fmt(r_test)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cell activity")

        activity = _as_numeric_array(row.get("Cell_activity", None))
        if activity is not None:
            if activity.ndim == 1:
                activity = activity[None, :]
            n_time = activity.shape[-1]
            t = np.arange(n_time) / float(model.target_fps)
            for trial in activity:
                ax.plot(t, trial, alpha=0.25, linewidth=0.8)
            ax.plot(t, np.nanmean(activity, axis=0), linewidth=2)
        else:
            ax.text(0.02, 0.90, "No Cell_activity", transform=ax.transAxes, fontsize=8)

        transient = _as_numeric_array(row.get("WL_transient_mod", None))
        if transient is not None:
            ax2 = ax.twinx()
            t2 = np.arange(len(transient)) / float(model.target_fps)
            ax2.plot(t2, transient, linewidth=1)
            ax2.set_ylabel("WL transient")

    def draw_stat(self, model: CellDataModel) -> None:
        self.figure.clf()
        if not model.has_cells:
            self.draw_empty("No cells loaded")
            return
        if model.filtered_indices.size == 0:
            self.draw_empty("No cells pass current filters")
            return

        gs = self.figure.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.2], hspace=0.55, wspace=0.35)
        axes = [self.figure.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
        ax_train = self.figure.add_subplot(gs[2, 0:2])
        ax_test = self.figure.add_subplot(gs[2, 2:4])

        for ax, (label, field, _, axis_key) in zip(axes[:7], TUNING_SPECS):
            self._plot_hist(ax, model, label, field, axis_key)
        self._plot_rf_scatter(axes[7], model)
        self._plot_fit(ax_train, model, "Repeatability", "r_train", through_origin=False)
        self._plot_fit(ax_test, model, "Repeatability", "r_test", through_origin=True)
        #self.figure.tight_layout()

    def _plot_hist(self, ax, model: CellDataModel, label: str, field: str, axis_key: str) -> None:
        ax.set_title(label)
        if model.df is None or field not in model.df.columns:
            ax.text(0.5, 0.5, f"No {field}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            return
        vals = model.numeric_values_for_filtered(field)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
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
            ax.text(0.5, 0.5, "No RF coordinates", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            return
        x = model.numeric_values_for_filtered("Azimuth")
        y = model.numeric_values_for_filtered("Elevation")
        x, y = _clean_xy(x, y)
        ax.scatter(x, y, s=10, alpha=0.6)
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation")
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
            ax.set_title(f"{y_field}: missing data")
            return
        x = model.numeric_values_for_filtered(x_field)
        y = model.numeric_values_for_filtered(y_field)
        x, y = _clean_xy(x, y)
        ax.scatter(x, y, s=12, alpha=0.5)
        if x.size < 2:
            ax.set_title(f"{y_field}: not enough data")
            return
        xx = np.linspace(0, 1, 200)
        if through_origin:
            denom = float(np.sum(x ** 2))
            a = float(np.sum(x * y) / denom) if denom > 0 else np.nan
            ax.plot(xx, a * xx, linewidth=1.5)
            ax.set_title(f"{y_field} vs {x_field} | y={a:.3f}x")
        else:
            a, b = np.polyfit(x, y, 1)
            ax.plot(xx, a * xx + b, linewidth=1.5)
            ax.set_title(f"{y_field} vs {x_field} | y={a:.3f}x+{b:.3f}")


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
