from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from twop_analysis.cell_browser.data_model import CellDataModel


TUNING_SPECS = [
    ("x", "Azimuth", "tun_xs", "xs"),
    ("y", "Elevation", "tun_ys", "ys"),
    ("angle", "Angle", "tun_angles", "angles"),
    ("size", "Size", "tun_sizes", "sizes"),
    ("freq", "Frequency", "tun_freqs", "freqs"),
    ("phase", "Phase", "tun_phases", "phases"),
]


class WavenDataView:
    """Add Waven-specific metadata access without changing the shared model."""

    def __init__(self, model: CellDataModel):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def visual_coverage(self) -> list[float] | None:
        metadata = self.model.metadata
        value = metadata.get("visual_coverage")
        if value is None:
            params = metadata.get("wavelet_params", {})
            if isinstance(params, dict):
                value = params.get("visual_coverage")
        if value is None:
            return None
        coverage = list(np.asarray(value, dtype=float).ravel())
        return coverage[:4] if len(coverage) >= 4 else None

    def wavelet_axis(self, key: str) -> np.ndarray | None:
        metadata = self.model.metadata
        params = metadata.get("wavelet_params", {})
        value = params.get(key) if isinstance(params, dict) else None
        if value is None:
            value = metadata.get(key)
        if value is None:
            return None
        array = np.asarray(value, dtype=float).ravel()
        return array if array.size else None
