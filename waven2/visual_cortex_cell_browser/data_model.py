from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd


DISPLAY_FIELD_MAP: dict[str, Optional[str]] = {
    "None": None,
    "x": "Azimuth",
    "y": "Elevation",
    "angle": "Angle",
    "size": "Size",
    "freq": "Frequency",
    "drift": "Drift",
    "phase": "Phase",
    "Angle_fit_ori": "Angle_fit_ori",
    "Angle_fit_OSI": "Angle_fit_OSI",
    "SNR": "SNR",
    "CNN": "CNN",
    "Repeatability": "Repeatability",
    "r_train": "r_train",
    "r_test": "r_test",
}

TUNING_SPECS = [
    ("x", "Azimuth", "tun_xs", "xs"),
    ("y", "Elevation", "tun_ys", "ys"),
    ("angle", "Angle", "tun_angles", "angles"),
    ("size", "Size", "tun_sizes", "sizes"),
    ("freq", "Frequency", "tun_freqs", "freqs"),
    ("drift", "Drift", "tun_drifts", "drifts"),
    ("phase", "Phase", "tun_phases", "phases"),
]


def _first_existing(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "good", "accepted"])


def _safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


@dataclass
class FilterSettings:
    snr_min: float = -np.inf
    cnn_min: float = -np.inf
    accepted_only: bool = False
    repeatability_min: float = -np.inf
    r_train_min: float = -np.inf
    r_test_min: float = -np.inf


@dataclass
class CellDataModel:
    df: Optional[pd.DataFrame] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cells_path: Optional[Path] = None
    background_path: Optional[Path] = None
    background_image: Optional[np.ndarray] = None
    resolution_x_um: float = 1.0
    resolution_y_um: float = 1.0
    target_fps: float = 30.0
    visual_coverage: Optional[list[float]] = None
    filtered_mask: Optional[np.ndarray] = None
    filtered_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    warnings: list[str] = field(default_factory=list)

    @property
    def has_cells(self) -> bool:
        return self.df is not None and len(self.df) > 0

    @property
    def total_cells(self) -> int:
        return 0 if self.df is None else len(self.df)

    @property
    def filtered_count(self) -> int:
        return int(len(self.filtered_indices))

    def load_cells(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("rb") as f:
            first = pickle.load(f)
            try:
                second = pickle.load(f)
            except EOFError:
                second = {}

        if not isinstance(first, pd.DataFrame):
            raise ValueError("The first object in the pickle file is not a pandas DataFrame.")
        if second is None:
            second = {}
        if not isinstance(second, dict):
            raise ValueError("The second object in the pickle file is not a metadata dictionary.")

        df = first.copy()
        if "cell_id" not in df.columns:
            df["cell_id"] = np.arange(len(df), dtype=int)
        # Keep row order stable, but allow fast .loc by cell_id when possible.
        df = df.reset_index(drop=True)

        missing_required = [c for c in ["Soma_Xpix", "Soma_Ypix"] if c not in df.columns]
        if missing_required:
            raise ValueError("Missing required cell coordinate field(s): " + ", ".join(missing_required))

        self.df = df
        self.metadata = second
        self.cells_path = path
        self._normalize_metadata()
        self.apply_filters(FilterSettings())

    def load_background(self, path: str | Path, transpose: bool = True) -> None:
        path = Path(path)
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Background image must be 2D, got shape {arr.shape}.")
        arr = np.asarray(arr, dtype=float)
        if transpose:
            arr = arr.T
        self.background_image = arr
        self.background_path = path

    def _normalize_metadata(self) -> None:
        self.warnings = []
        md = self.metadata or {}

        res = _first_existing(md, ["resolution", "Resolution_um", "pixel_size", "dxy"], None)
        if res is None:
            self.warnings.append("Missing resolution metadata; using 1.0 µm/pixel.")
            self.resolution_x_um = 1.0
            self.resolution_y_um = 1.0
        else:
            arr = np.asarray(res, dtype=float).ravel()
            if len(arr) == 0 or not np.isfinite(arr[0]):
                self.warnings.append("Invalid resolution metadata; using 1.0 µm/pixel.")
                self.resolution_x_um = 1.0
                self.resolution_y_um = 1.0
            elif len(arr) == 1:
                self.resolution_x_um = float(arr[0])
                self.resolution_y_um = float(arr[0])
            else:
                self.resolution_x_um = float(arr[0])
                self.resolution_y_um = float(arr[1])

        fps = _first_existing(md, ["target_fps", "fps", "framerate"], None)
        if fps is None:
            self.warnings.append("Missing target_fps metadata; using 30 Hz.")
            self.target_fps = 30.0
        else:
            self.target_fps = float(fps)

        wavelet_params = md.get("wavelet_params", {}) or {}
        vc = _first_existing(md, ["visual_coverage"], None)
        if vc is None and isinstance(wavelet_params, dict):
            vc = wavelet_params.get("visual_coverage", None)
        if vc is not None:
            vc_arr = list(np.asarray(vc, dtype=float).ravel())
            if len(vc_arr) >= 4:
                self.visual_coverage = vc_arr[:4]

    def wavelet_axis(self, key: str) -> Optional[np.ndarray]:
        md = self.metadata or {}
        wp = md.get("wavelet_params", {}) or {}
        value = None
        if isinstance(wp, dict):
            value = wp.get(key, None)
        if value is None:
            value = md.get(key, None)
        if value is None:
            return None
        arr = np.asarray(value, dtype=float).ravel()
        return arr if arr.size else None

    def has_phase_dimension(self) -> bool:
        if self.df is None:
            return False
        return "Phase" in self.df.columns or "tun_phases" in self.df.columns

    def display_options(self) -> list[str]:
        if self.df is None:
            return ["None"]
        opts: list[str] = []
        for label, field in DISPLAY_FIELD_MAP.items():
            if field is None or field in self.df.columns:
                opts.append(label)
        return opts

    def apply_filters(self, settings: FilterSettings) -> np.ndarray:
        if self.df is None:
            self.filtered_mask = np.zeros(0, dtype=bool)
            self.filtered_indices = np.array([], dtype=int)
            return self.filtered_mask

        df = self.df
        mask = np.ones(len(df), dtype=bool)

        if "SNR" in df.columns:
            mask &= (_safe_numeric(df["SNR"]).to_numpy() > settings.snr_min)
        if "CNN" in df.columns:
            mask &= (_safe_numeric(df["CNN"]).to_numpy() > settings.cnn_min)
        if "Repeatability" in df.columns:
            mask &= (_safe_numeric(df["Repeatability"]).to_numpy() > settings.repeatability_min)
        if "r_train" in df.columns:
            mask &= (_safe_numeric(df["r_train"]).to_numpy() > settings.r_train_min)
        if "r_test" in df.columns:
            mask &= (_safe_numeric(df["r_test"]).to_numpy() > settings.r_test_min)
        if settings.accepted_only and "Accepted" in df.columns:
            mask &= _to_bool_series(df["Accepted"]).to_numpy()

        self.filtered_mask = mask
        self.filtered_indices = np.flatnonzero(mask)
        return mask

    def cell_row_by_iloc(self, row_index: int) -> pd.Series:
        if self.df is None:
            raise IndexError("No cells loaded.")
        return self.df.iloc[int(row_index)]

    def nearest_filtered_cell(self, x_um: float, y_um: float) -> tuple[Optional[int], float]:
        if self.df is None or self.filtered_indices.size == 0:
            return None, np.inf
        dfv = self.df.iloc[self.filtered_indices]
        x = pd.to_numeric(dfv["Soma_Xpix"], errors="coerce").to_numpy() * self.resolution_x_um
        y = pd.to_numeric(dfv["Soma_Ypix"], errors="coerce").to_numpy() * self.resolution_y_um
        d2 = (x - x_um) ** 2 + (y - y_um) ** 2
        if not np.any(np.isfinite(d2)):
            return None, np.inf
        pos = int(np.nanargmin(d2))
        return int(self.filtered_indices[pos]), float(np.sqrt(d2[pos]))

    def filtered_xy_um(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.df is None or self.filtered_indices.size == 0:
            return np.array([]), np.array([]), np.array([], dtype=int)
        dfv = self.df.iloc[self.filtered_indices]
        x = pd.to_numeric(dfv["Soma_Xpix"], errors="coerce").to_numpy() * self.resolution_x_um
        y = pd.to_numeric(dfv["Soma_Ypix"], errors="coerce").to_numpy() * self.resolution_y_um
        return x, y, self.filtered_indices.copy()

    def numeric_values_for_filtered(self, field: str) -> np.ndarray:
        if self.df is None or field not in self.df.columns or self.filtered_indices.size == 0:
            return np.array([])
        return pd.to_numeric(self.df.iloc[self.filtered_indices][field], errors="coerce").to_numpy(dtype=float)
