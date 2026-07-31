"""Run the S4 visual-feature estimation workflow outside Jupyter.

This module is the reusable counterpart of ``S4_feature_estimation_vS.ipynb``.
Call :func:`run_feature_estimation` from another application, or run the
module as a command-line program with ``python -m waven2.s4_feature_estimation``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


@dataclass(slots=True)
class FeatureEstimationConfig:
    """Inputs and analysis parameters for the S4 workflow."""

    spks_path: Path
    resampled_video_path: Path
    cell_db_path: Path | None = None
    output_folder: Path | None = None
    full_screen_coverage: tuple[float, float, float, float] = (
        -15.0, # left edge
        105.0, # right edge
        -17.5, # bottom edge
        50.0, # top edge
    )
    visual_coverage: tuple[float, float, float, float] = (
        -15.0, # left edge
        105.0, # right edge
        -17.5, # bottom edge
        50.0, # top edge
    )
    screen_x: int = 120 # calculation resolution in pixels
    nx: int = 20 # Gabor library: number of center positions
    n_thetas: int = 16 # Gabor library: number of angles
    theta_max: float = float(np.pi) # Gabor library: maximum angle
    size_min: float = 5.0 # Gabor library: minimum size in visual degrees
    size_max: float = 25.0 # Gabor library: maximum size in visual degrees
    n_sizes: int = 7 # Gabor library: number of sizes
    freq_min: float = 0.025 # Gabor library: minimum frequency
    freq_max: float = 0.15 # Gabor library: maximum frequency
    n_freqs: int = 7 # Gabor library: number of frequencies
    n_phases: int = 2 # Gabor library: number of phases
    phase_max: float = float(np.pi) # Gabor library: maximum phase
    target_fps: float | None = None # analysis frame rate in Hz video and activity data are sampled at this rate
    average_fwhm_sec: float = 0.1 # temporal smoothing of activity data in seconds
    shift_samples: int = 1 # temporal shift of activity data in samples
    dwt_mode: str = "complex" 
    train_split: float = 0.85 # train/test split ratio in time
    device: str = "cuda"
    force_downsample: bool = False # force calculating downsampled video even if it already exists
    force_dwt: bool = False # force calculating DWT even if it already exists

    def __post_init__(self) -> None:
        self.spks_path = Path(self.spks_path)
        self.resampled_video_path = Path(self.resampled_video_path)
        if self.cell_db_path is not None:
            self.cell_db_path = Path(self.cell_db_path)
        if self.output_folder is not None:
            self.output_folder = Path(self.output_folder)


@dataclass(frozen=True, slots=True)
class FeatureEstimationResult:
    """Files produced by :func:`run_feature_estimation`."""

    output_folder: Path
    cell_database_path: Path
    cell_table_path: Path
    statistics_path: Path
    metadata_path: Path


def _validate_config(config: FeatureEstimationConfig) -> None:
    if not config.spks_path.is_file():
        raise FileNotFoundError(f"Spike-response array not found: {config.spks_path}")
    if not config.resampled_video_path.is_file():
        raise FileNotFoundError(
            f"Resampled stimulus video not found: {config.resampled_video_path}"
        )
    if config.screen_x < 1 or config.nx < 1:
        raise ValueError("screen_x and nx must be positive integers")
    for name in ("n_thetas", "n_sizes", "n_freqs", "n_phases"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.size_min <= 0 or config.size_max < config.size_min:
        raise ValueError("sizes must satisfy 0 < size_min <= size_max")
    if config.freq_min <= 0 or config.freq_max < config.freq_min:
        raise ValueError("frequencies must satisfy 0 < freq_min <= freq_max")
    if not 0.0 < config.train_split < 1.0:
        raise ValueError("train_split must be strictly between 0 and 1")
    if config.average_fwhm_sec < 0:
        raise ValueError("average_fwhm_sec cannot be negative")
    if config.dwt_mode not in {"original", "positive", "absolute", "complex"}:
        raise ValueError(
            "dwt_mode must be one of: original, positive, absolute, complex"
        )
    for name, coverage in (
        ("full_screen_coverage", config.full_screen_coverage),
        ("visual_coverage", config.visual_coverage),
    ):
        if len(coverage) != 4:
            raise ValueError(f"{name} must contain four values")
        left, right, bottom, top = coverage
        if right <= left or top <= bottom:
            raise ValueError(f"{name} bounds must have positive width and height")
    full_left, full_right, full_bottom, full_top = config.full_screen_coverage
    vis_left, vis_right, vis_bottom, vis_top = config.visual_coverage
    if not (
        full_left <= vis_left < vis_right <= full_right
        and full_bottom <= vis_bottom < vis_top <= full_top
    ):
        raise ValueError("visual_coverage must lie within full_screen_coverage")


def _make_wavelet_params(config: FeatureEstimationConfig) -> dict[str, Any]:
    from waven2.wavelet_utils_vSpeed import makeFilterParamDict_vS

    az_left, az_right, el_bottom, el_top = config.visual_coverage
    aspect = (el_top - el_bottom) / (az_right - az_left)
    screen_y = max(1, int(config.screen_x * aspect))
    ny = max(1, int(config.nx * aspect))

    xs = np.linspace(az_left, az_right, config.nx, endpoint=False)
    xs += (az_right - az_left) / (2 * config.nx)
    ys = np.linspace(el_bottom, el_top, ny, endpoint=False)
    ys += (el_top - el_bottom) / (2 * ny)
    angles = np.linspace(0, config.theta_max, config.n_thetas, endpoint=False)
    sizes = np.logspace(
        np.log10(config.size_min), np.log10(config.size_max), config.n_sizes
    )
    freqs = np.logspace(
        np.log10(config.freq_min), np.log10(config.freq_max), config.n_freqs
    )
    phases = np.linspace(0, config.phase_max, config.n_phases, endpoint=False)

    total = (
        len(xs)
        * len(ys)
        * len(angles)
        * len(sizes)
        * len(freqs)
        * len(phases)
    )
    print(f"Screen size: {config.screen_x}x{screen_y} pixels")
    print(f"Gabor feature count: {total:,}")

    return makeFilterParamDict_vS(
        config.screen_x,
        screen_y,
        config.visual_coverage,
        config.full_screen_coverage,
        xs,
        ys,
        angles,
        sizes,
        freqs,
        phases,
    )


def _load_cell_database(
    path: Path, n_neurons: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if path.exists():
        with path.open("rb") as handle:
            try:
                df_cells = pickle.load(handle)
                metadata = pickle.load(handle)
            except EOFError as exc:
                raise ValueError(
                    f"Cell database must contain a DataFrame followed by metadata: {path}"
                ) from exc
        if not isinstance(df_cells, pd.DataFrame) or not isinstance(metadata, dict):
            raise ValueError(
                f"Cell database must contain a DataFrame followed by a dict: {path}"
            )
        print(f"Loaded cell database: {path}")
    else:
        df_cells = pd.DataFrame(
            {"cell_id": np.arange(n_neurons), "SeriesID": "unknown"}
        )
        metadata = {"target_fps": 30.0}
        print(f"No cell database found at {path}; created {n_neurons} rows")

    if len(df_cells) != n_neurons:
        raise ValueError(
            f"Cell database has {len(df_cells)} rows, but responses contain "
            f"{n_neurons} neurons"
        )
    if "cell_id" not in df_cells.columns:
        df_cells = df_cells.copy()
        df_cells.insert(0, "cell_id", np.arange(n_neurons))
    if df_cells["cell_id"].duplicated().any():
        raise ValueError("cell_id values must be unique")
    return df_cells.set_index("cell_id", drop=False), metadata


def _accepted_mask(df_cells: pd.DataFrame) -> np.ndarray:
    if "Accepted" not in df_cells.columns:
        return np.ones(len(df_cells), dtype=bool)
    values = df_cells["Accepted"]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    return (
        values.astype(str)
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes"})
        .to_numpy(dtype=bool)
    )


def _modify_dwt(
    dwt: np.ndarray,
    phases: np.ndarray,
    mode: str,
    sigma: float,
    device: str,
) -> tuple[np.ndarray, str]:
    from waven2.analysis_utils import (
        dwt_amp_phase_torch_batched,
        gaussian_filter1d_torch_axis0_chunked,
    )

    if mode == "original":
        comment = "orig"
    elif mode == "positive":
        dwt = dwt.copy()
        dwt[dwt < 0] = 0
        comment = "Gt0"
    elif mode == "absolute":
        dwt = np.abs(dwt)
        comment = "abs"
    else:
        if dwt.shape[-1] != len(phases):
            raise ValueError(
                "The last DWT dimension must match the configured phases for "
                "complex mode"
            )
        dwt = dwt_amp_phase_torch_batched(dwt, phases, device=device)
        comment = "complex"

    if sigma > 0:
        dwt = gaussian_filter1d_torch_axis0_chunked(
            dwt, sigma, device=device
        )
    return dwt, comment


def _best_feature_performance(
    rfs: np.ndarray,
    mean_spks: np.ndarray,
    dwt: np.ndarray,
    train_split_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_neurons = mean_spks.shape[1]
    max_idxs = np.empty((n_neurons, rfs.ndim - 1), dtype=int)
    r_train = np.empty(n_neurons, dtype=float)
    r_test = np.empty(n_neurons, dtype=float)

    for neuron_index in range(n_neurons):
        neuron_rfs = rfs[neuron_index]
        max_idx = tuple(int(i) for i in np.unravel_index(
            np.argmax(neuron_rfs), neuron_rfs.shape
        ))
        max_idxs[neuron_index] = max_idx
        r_train[neuron_index] = neuron_rfs[max_idx]
        r_test[neuron_index] = np.corrcoef(
            mean_spks[train_split_index:, neuron_index],
            dwt[train_split_index:, *max_idx],
        )[0, 1]
    return max_idxs, r_train, r_test


def _fit_tuning_curves(
    rfs: np.ndarray,
    max_idxs: np.ndarray,
    wavelet_params: Mapping[str, Any],
) -> tuple[list[dict[str, float]], dict[str, list[tuple[float, Any]]]]:
    from waven2.analysis_utils import fit_quadratic, fit_sine1x

    xs = np.asarray(wavelet_params["xs"])
    ys = np.asarray(wavelet_params["ys"])
    angles = np.asarray(wavelet_params["angles"])
    sizes = np.asarray(wavelet_params["sizes"])
    freqs = np.asarray(wavelet_params["freqs"])
    feature_dims = rfs.ndim - 1
    angle_fits: list[dict[str, float]] = []
    fits: dict[str, list[tuple[float, Any]]] = {
        "xs": [],
        "ys": [],
        "sizes": [],
        "freqs": [],
    }

    for neuron_index, max_idx_array in enumerate(max_idxs):
        max_idx = tuple(int(i) for i in max_idx_array)
        neuron_rfs = rfs[neuron_index]
        angle_curve = neuron_rfs[
            max_idx[0], max_idx[1], :, *max_idx[3:feature_dims]
        ]
        angle_fits.append(fit_sine1x(angles, angle_curve))

        curves = {
            "xs": neuron_rfs[:, *max_idx[1:feature_dims]],
            "ys": neuron_rfs[max_idx[0], :, *max_idx[2:feature_dims]],
            "sizes": neuron_rfs[
                max_idx[0], max_idx[1], max_idx[2], :, *max_idx[4:feature_dims]
            ],
            "freqs": neuron_rfs[
                max_idx[0],
                max_idx[1],
                max_idx[2],
                max_idx[3],
                :,
                *max_idx[5:feature_dims],
            ],
        }
        for name, values, curve in (
            ("xs", xs, curves["xs"]),
            ("ys", ys, curves["ys"]),
            ("sizes", sizes, curves["sizes"]),
            ("freqs", freqs, curves["freqs"]),
        ):
            fits[name].append(fit_quadratic(values, curve))
    return angle_fits, fits


def _safe_linear_fits(
    respcorr: np.ndarray, r_train: np.ndarray, r_test: np.ndarray
) -> tuple[float, float, float]:
    train_mask = np.isfinite(respcorr) & np.isfinite(r_train)
    if np.count_nonzero(train_mask) >= 2:
        train_a, train_b = np.polyfit(respcorr[train_mask], r_train[train_mask], 1)
    else:
        train_a = train_b = np.nan

    test_mask = np.isfinite(respcorr) & np.isfinite(r_test)
    denominator = np.sum(respcorr[test_mask] ** 2)
    test_a = (
        np.sum(respcorr[test_mask] * r_test[test_mask]) / denominator
        if denominator > 0
        else np.nan
    )
    return float(train_a), float(train_b), float(test_a)


def _save_performance_plot(
    output_folder: Path,
    respcorr: np.ndarray,
    r_train: np.ndarray,
    r_test: np.ndarray,
    top_indices: np.ndarray,
    train_a: float,
    train_b: float,
    test_a: float,
) -> None:
    fig = Figure(figsize=(14, 6))
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    xx = np.linspace(0, 1, 200)
    for ax, values, title in zip(axes, (r_train, r_test), ("Train", "Test")):
        ax.scatter(respcorr, values, alpha=0.5)
        for index in top_indices:
            if np.isfinite(respcorr[index]) and np.isfinite(values[index]):
                ax.annotate(str(index), (respcorr[index], values[index]))
        ax.set_title(title)
        ax.set_xlabel("Response correlation")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Prediction correlation")
    if np.isfinite(train_a):
        axes[0].plot(xx, train_a * xx + train_b, color="black")
    if np.isfinite(test_a):
        axes[1].plot(xx, test_a * xx, color="black")
    fig.tight_layout()
    fig.savefig(
        output_folder / "respcorr_vs_prediction_correlation_5.png", dpi=300
    )
    fig.clear()


def _save_reliability_plots(
    output_folder: Path,
    ccmax: np.ndarray,
    r_test: np.ndarray,
    accepted: np.ndarray,
    top_indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    ccnorm = np.full_like(r_test, np.nan, dtype=float)
    valid = np.isfinite(r_test) & np.isfinite(ccmax) & (ccmax > 0)
    ccnorm[valid] = r_test[valid] / ccmax[valid]
    denominator = np.sum(ccmax[valid] ** 2)
    population_ccnorm = (
        float(np.sum(ccmax[valid] * r_test[valid]) / denominator)
        if denominator > 0
        else np.nan
    )

    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    ax.scatter(ccmax[valid], r_test[valid], alpha=0.5)
    ax.scatter(ccmax[valid & accepted], r_test[valid & accepted], alpha=0.5)
    for index in top_indices:
        if valid[index]:
            ax.annotate(str(index), (ccmax[index], r_test[index]))
    if np.any(valid):
        xx = np.linspace(0, np.nanmax(ccmax[valid]), 200)
        ax.plot(xx, population_ccnorm * xx, color="black")
    ax.set_xlabel("CCmax from split-half reliability")
    ax.set_ylabel("Test correlation")
    ax.set_title("Test correlation vs noise ceiling")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_folder / "ccmax_vs_prediction_correlation.png", dpi=300)
    fig.clear()
    return ccnorm, population_ccnorm


def _save_fitted_rf_positions(
    output_folder: Path,
    df_cells: pd.DataFrame,
    visual_coverage: Sequence[float],
) -> None:
    """Plot fitted receptive-field centers in visual-degree coordinates."""

    azimuth = pd.to_numeric(df_cells["Azimuth_fit"], errors="coerce").to_numpy()
    elevation = pd.to_numeric(df_cells["Elevation_fit"], errors="coerce").to_numpy()
    repeatability = pd.to_numeric(
        df_cells["Repeatability"], errors="coerce"
    ).to_numpy()
    finite = np.isfinite(azimuth) & np.isfinite(elevation)
    good_finite = finite & np.isfinite(repeatability) & (repeatability > 0.2)

    az_left, az_right, el_bottom, el_top = visual_coverage
    figure = Figure(figsize=(8, 6))
    axis = figure.subplots()
    axis.scatter(
        azimuth[finite],
        elevation[finite],
        s=18,
        color="0.65",
        alpha=0.55,
        label=f"All cells (n={np.count_nonzero(finite)})",
    )
    axis.scatter(
        azimuth[good_finite],
        elevation[good_finite],
        s=24,
        color="tab:blue",
        alpha=0.8,
        label=f"Good cells: repeatability > 0.2 (n={np.count_nonzero(good_finite)})",
    )
    axis.set_xlim(az_left, az_right)
    axis.set_ylim(el_bottom, el_top)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Fitted azimuth (visual degrees)")
    axis.set_ylabel("Fitted elevation (visual degrees)")
    axis.set_title("Fitted receptive-field positions")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_folder / "fitted_rf_positions.png", dpi=300)
    figure.clear()


def _add_results_to_cells(
    df_cells: pd.DataFrame,
    spks: np.ndarray,
    dwt: np.ndarray,
    rfs: np.ndarray,
    max_idxs: np.ndarray,
    respcorr: np.ndarray,
    r_train: np.ndarray,
    r_test: np.ndarray,
    angle_fits: list[dict[str, float]],
    tuning_fits: Mapping[str, list[tuple[float, Any]]],
    params: Mapping[str, Any],
) -> pd.DataFrame:
    result = df_cells.copy()
    object_columns = [
        "RF_indexes",
        "WL_transient_mod",
        "Cell_activity",
        "tun_xs",
        "tun_ys",
        "tun_xy",
        "tun_angles",
        "tun_sizes",
        "tun_freqs",
        "tun_xs_params",
        "tun_ys_params",
        "tun_sizes_params",
        "tun_freqs_params",
    ]
    has_phase = rfs.ndim - 1 > 5
    if has_phase:
        object_columns.append("tun_phases")
    for column in object_columns:
        result[column] = pd.Series(
            [None] * len(result), index=result.index, dtype="object"
        )

    xs = np.asarray(params["xs"])
    ys = np.asarray(params["ys"])
    angles = np.asarray(params["angles"])
    sizes = np.asarray(params["sizes"])
    freqs = np.asarray(params["freqs"])
    phases = np.asarray(params["phases"])
    feature_dims = rfs.ndim - 1

    for neuron_index, cell_id in enumerate(tqdm(result.index, desc="Saving cells")):
        max_idx = tuple(int(i) for i in max_idxs[neuron_index])
        neuron_rfs = rfs[neuron_index]
        result.at[cell_id, "Repeatability"] = respcorr[neuron_index]
        result.at[cell_id, "RF_indexes"] = list(max_idx)
        result.at[cell_id, "r_train"] = r_train[neuron_index]
        result.at[cell_id, "r_test"] = r_test[neuron_index]
        result.at[cell_id, "Azimuth"] = xs[max_idx[0]]
        result.at[cell_id, "Elevation"] = ys[max_idx[1]]
        result.at[cell_id, "Angle"] = angles[max_idx[2]]
        result.at[cell_id, "Size"] = sizes[max_idx[3]]
        result.at[cell_id, "Frequency"] = freqs[max_idx[4]]
        if has_phase:
            result.at[cell_id, "Phase"] = phases[max_idx[5]]

        result.at[cell_id, "WL_transient_mod"] = dwt[:, *max_idx]
        result.at[cell_id, "Cell_activity"] = spks[:, :, neuron_index]
        result.at[cell_id, "tun_xs"] = neuron_rfs[:, *max_idx[1:feature_dims]]
        result.at[cell_id, "tun_ys"] = neuron_rfs[
            max_idx[0], :, *max_idx[2:feature_dims]
        ]
        result.at[cell_id, "tun_xy"] = neuron_rfs[
            :, :, *max_idx[2:feature_dims]
        ]
        result.at[cell_id, "tun_angles"] = neuron_rfs[
            max_idx[0], max_idx[1], :, *max_idx[3:feature_dims]
        ]
        result.at[cell_id, "tun_sizes"] = neuron_rfs[
            max_idx[0], max_idx[1], max_idx[2], :, *max_idx[4:feature_dims]
        ]
        result.at[cell_id, "tun_freqs"] = neuron_rfs[
            max_idx[0], max_idx[1], max_idx[2], max_idx[3], :,
            *max_idx[5:feature_dims]
        ]
        if has_phase:
            result.at[cell_id, "tun_phases"] = neuron_rfs[
                max_idx[0], max_idx[1], max_idx[2], max_idx[3], max_idx[4], :
            ]

        angle_fit = angle_fits[neuron_index]
        result.at[cell_id, "Angle_fit_ori"] = angle_fit["orientation"]
        result.at[cell_id, "Angle_fit_amplitude"] = angle_fit["amplitude"]
        result.at[cell_id, "Angle_fit_constant"] = angle_fit["constant"]
        denominator = angle_fit["amplitude"] + angle_fit["constant"]
        result.at[cell_id, "Angle_fit_OSI"] = (
            angle_fit["amplitude"] / denominator if denominator != 0 else np.nan
        )
        for name, output_name in (
            ("xs", "Azimuth"),
            ("ys", "Elevation"),
            ("sizes", "Size"),
            ("freqs", "Frequency"),
        ):
            best, fit_params = tuning_fits[name][neuron_index]
            result.at[cell_id, f"{output_name}_fit"] = best
            result.at[cell_id, f"tun_{name}_params"] = fit_params
    return result


_SKIP = object()


def make_yaml_readable(
    obj: Any, *, max_array_items: int = 200, skip_unknown: bool = True
) -> Any:
    """Convert nested metadata to a compact, human-readable YAML structure."""

    if isinstance(obj, np.generic):
        return make_yaml_readable(
            obj.item(), max_array_items=max_array_items, skip_unknown=skip_unknown
        )
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return "nan"
        if np.isposinf(obj):
            return "inf"
        if np.isneginf(obj):
            return "-inf"
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        if obj.size <= max_array_items:
            return make_yaml_readable(
                obj.tolist(),
                max_array_items=max_array_items,
                skip_unknown=skip_unknown,
            )
        summary: dict[str, Any] = {
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
        if np.issubdtype(obj.dtype, np.number):
            finite = obj[np.isfinite(obj)]
            if finite.size:
                summary.update(
                    min=float(np.nanmin(finite)),
                    max=float(np.nanmax(finite)),
                    mean=float(np.nanmean(finite)),
                )
        return summary
    if isinstance(obj, Mapping):
        output = {}
        for key, value in obj.items():
            safe_value = make_yaml_readable(
                value,
                max_array_items=max_array_items,
                skip_unknown=skip_unknown,
            )
            if safe_value is not _SKIP:
                output[str(key)] = safe_value
        return output
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        output = []
        for value in obj:
            safe_value = make_yaml_readable(
                value,
                max_array_items=max_array_items,
                skip_unknown=skip_unknown,
            )
            if safe_value is not _SKIP:
                output.append(safe_value)
        return output
    return _SKIP if skip_unknown else str(obj)


def _save_yaml_readable(params: Mapping[str, Any], filename: Path) -> None:
    with filename.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            make_yaml_readable(params),
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=120,
        )


def run_feature_estimation(
    config: FeatureEstimationConfig,
) -> FeatureEstimationResult:
    """Run S4 feature estimation and write the notebook-compatible outputs."""

    from waven2.analysis_utils import (
        FeatureSearch_correlation_batched,
        ccmax_from_cchalf_simple,
        compute_respcorr_split_half,
        downscale_binary_video,
    )
    from waven2.wavelet_utils_vSpeed import (
        compute_and_save_dwt_vS,
        filename_fromFilterParam,
        saveFilterParamDict_vS,
    )

    _validate_config(config)
    params = _make_wavelet_params(config)
    working_dir = config.spks_path.parent

    spks = np.load(config.spks_path)
    if spks.ndim != 3:
        raise ValueError(
            f"spks must have shape (trials, timepoints, neurons), got {spks.shape}"
        )
    n_trials, n_timepoints, n_neurons = spks.shape
    if n_trials < 2:
        raise ValueError("At least two response trials are required")
    if n_timepoints < 4:
        raise ValueError("At least four response timepoints are required")
    if n_neurons < 1:
        raise ValueError("At least one neuron is required")

    cell_db_path = config.cell_db_path or working_dir / "cells_caiman.cellDB_pickle"
    df_cells, metadata = _load_cell_database(cell_db_path, n_neurons)
    accepted = _accepted_mask(df_cells)
    target_fps = float(
        config.target_fps
        if config.target_fps is not None
        else metadata.get("target_fps", 30.0)
    )
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    downsampled_video_path = downscale_binary_video(
        config.resampled_video_path,
        config.full_screen_coverage,
        config.visual_coverage,
        config.screen_x,
        params["screen_y"],
        fps=target_fps,
        force=config.force_downsample,
    )
    _, paramname = filename_fromFilterParam(params)
    paramspath = Path(downsampled_video_path).parent / paramname
    saveFilterParamDict_vS(params, paramspath)
    dwt_path = compute_and_save_dwt_vS(
        downsampled_video_path,
        params,
        device=config.device,
        force=config.force_dwt,
    )

    average_samples = int(round(target_fps * config.average_fwhm_sec))
    sigma = average_samples / 2.355 if average_samples > 0 else 0.0
    if sigma > 0:
        spks = gaussian_filter1d(spks, sigma=sigma, axis=1)
    if config.shift_samples:
        spks = np.roll(spks, -config.shift_samples, axis=1)
    mean_spks = np.mean(spks, axis=0)

    dwt = np.load(dwt_path)
    dwt, mode_comment = _modify_dwt(
        dwt,
        np.asarray(params["phases"]),
        config.dwt_mode,
        sigma,
        config.device,
    )
    if dwt.shape[0] != n_timepoints:
        raise ValueError(
            f"DWT has {dwt.shape[0]} frames, responses have {n_timepoints} timepoints"
        )
    train_split_index = int(n_timepoints * config.train_split)
    if train_split_index < 2 or n_timepoints - train_split_index < 2:
        raise ValueError("train_split must leave at least two train and test samples")

    rfs = FeatureSearch_correlation_batched(
        dwt[:train_split_index],
        mean_spks[:train_split_index],
        device=config.device,
    )
    output_folder = config.output_folder or working_dir / (
        paramname.removesuffix(".json") + f"_{mode_comment}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    saveFilterParamDict_vS(params, output_folder / paramname)

    respcorr = compute_respcorr_split_half(spks)
    max_idxs, r_train, r_test = _best_feature_performance(
        rfs, mean_spks, dwt, train_split_index
    )
    top_indices = np.argsort(np.nan_to_num(respcorr, nan=-np.inf))[
        -min(20, n_neurons) :
    ]
    angle_fits, tuning_fits = _fit_tuning_curves(rfs, max_idxs, params)
    train_a, train_b, test_a = _safe_linear_fits(respcorr, r_train, r_test)
    _save_performance_plot(
        output_folder,
        respcorr,
        r_train,
        r_test,
        top_indices,
        train_a,
        train_b,
        test_a,
    )

    ccmax = ccmax_from_cchalf_simple(respcorr)
    ccnorm, population_ccnorm = _save_reliability_plots(
        output_folder,
        ccmax,
        r_test,
        accepted,
        top_indices,
    )

    stat = {
        "N_xs": len(params["xs"]),
        "N_ys": len(params["ys"]),
        "N_angles": len(params["angles"]),
        "N_sizes": len(params["sizes"]),
        "N_freqs": len(params["freqs"]),
        "N_phases": len(params["phases"]),
        "average_FWHM_samples": average_samples,
        "shift_samples": config.shift_samples,
        "mode": mode_comment,
        "mean_respcorr_top20": float(np.nanmean(respcorr[top_indices])),
        "mean_r_train_top20": float(np.nanmean(r_train[top_indices])),
        "mean_r_test_top20": float(np.nanmean(r_test[top_indices])),
        "train_fit_a": train_a,
        "train_fit_b": train_b,
        "test_fit_a": test_a,
        "population_CCnorm": population_ccnorm,
        "median_cellwise_CCnorm_accepted": float(np.nanmedian(ccnorm[accepted])),
    }
    metadata.update(
        spks_path=str(config.spks_path),
        resampled_video_file=str(config.resampled_video_path),
        downsampled_video_path=str(downsampled_video_path),
        wavelet_params=params,
        target_fps=target_fps,
        shift_samples=config.shift_samples,
        dwt_mode=mode_comment,
        feature_dim_number=dwt.ndim - 1,
        train_split=config.train_split,
        train_split_index=train_split_index,
        stat=stat,
    )

    df_cells = _add_results_to_cells(
        df_cells,
        spks,
        dwt,
        rfs,
        max_idxs,
        respcorr,
        r_train,
        r_test,
        angle_fits,
        tuning_fits,
        params,
    )
    _save_fitted_rf_positions(
        output_folder,
        df_cells,
        config.visual_coverage,
    )
    cell_database_path = output_folder / "cells_waven1.cellDB_pickle"
    with cell_database_path.open("wb") as handle:
        pickle.dump(df_cells, handle)
        pickle.dump(metadata, handle)

    object_columns = [
        "WL_transient_mod",
        "WL_transient_phase",
        "Cell_activity",
        "contour",
        "tun_xs",
        "tun_ys",
        "tun_xy",
        "tun_angles",
        "tun_sizes",
        "tun_freqs",
        "tun_drifts",
        "tun_phases",
        "tun_xs_params",
        "tun_ys_params",
        "tun_sizes_params",
        "tun_freqs_params",
        "tun_drifts_params",
    ]
    cell_table_path = output_folder / "cells_waven1.xlsx"
    df_cells.drop(columns=object_columns, errors="ignore").to_excel(
        cell_table_path, index=False
    )
    pd.DataFrame(
        {"respcorr": respcorr, "r_train": r_train, "r_test": r_test}
    ).to_csv(output_folder / "respcorr_max_values_vs4.csv", index=False)

    statistics_path = output_folder / "stat.xlsx"
    pd.DataFrame([stat]).to_excel(statistics_path, index=False, sheet_name="stat")
    metadata_path = output_folder / "metadata_humanreadable.yaml"
    _save_yaml_readable(metadata, metadata_path)
    print(f"Feature estimation complete: {output_folder}")
    return FeatureEstimationResult(
        output_folder=output_folder,
        cell_database_path=cell_database_path,
        cell_table_path=cell_table_path,
        statistics_path=statistics_path,
        metadata_path=metadata_path,
    )


def _coverage(values: list[float]) -> tuple[float, float, float, float]:
    return values[0], values[1], values[2], values[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate visual-cortex Gabor features from trial responses"
    )
    parser.add_argument(
        "spks_path", type=Path, help="Response array: trials x timepoints x neurons"
    )
    parser.add_argument("resampled_video_path", type=Path, help="Stimulus MP4 file")
    parser.add_argument("--cell-db", dest="cell_db_path", type=Path)
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument(
        "--full-screen-coverage",
        type=float,
        nargs=4,
        metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
        default=[-15.0, 105.0, -17.5, 50.0],
    )
    parser.add_argument(
        "--visual-coverage",
        type=float,
        nargs=4,
        metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
        default=[-15.0, 105.0, -17.5, 50.0],
    )
    parser.add_argument("--screen-x", type=int, default=120)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--n-thetas", type=int, default=16)
    parser.add_argument("--size-min", type=float, default=5.0)
    parser.add_argument("--size-max", type=float, default=25.0)
    parser.add_argument("--n-sizes", type=int, default=7)
    parser.add_argument("--freq-min", type=float, default=0.025)
    parser.add_argument("--freq-max", type=float, default=0.15)
    parser.add_argument("--n-freqs", type=int, default=7)
    parser.add_argument("--n-phases", type=int, default=2)
    parser.add_argument("--target-fps", type=float)
    parser.add_argument("--average-fwhm-sec", type=float, default=0.1)
    parser.add_argument("--shift-samples", type=int, default=1)
    parser.add_argument(
        "--dwt-mode",
        choices=("original", "positive", "absolute", "complex"),
        default="complex",
    )
    parser.add_argument("--train-split", type=float, default=0.85)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--force-downsample", action="store_true")
    parser.add_argument("--force-dwt", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> FeatureEstimationResult:
    args = build_parser().parse_args(argv)
    config = FeatureEstimationConfig(
        spks_path=args.spks_path,
        resampled_video_path=args.resampled_video_path,
        cell_db_path=args.cell_db_path,
        output_folder=args.output_folder,
        full_screen_coverage=_coverage(args.full_screen_coverage),
        visual_coverage=_coverage(args.visual_coverage),
        screen_x=args.screen_x,
        nx=args.nx,
        n_thetas=args.n_thetas,
        size_min=args.size_min,
        size_max=args.size_max,
        n_sizes=args.n_sizes,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        n_freqs=args.n_freqs,
        n_phases=args.n_phases,
        target_fps=args.target_fps,
        average_fwhm_sec=args.average_fwhm_sec,
        shift_samples=args.shift_samples,
        dwt_mode=args.dwt_mode,
        train_split=args.train_split,
        device=args.device,
        force_downsample=args.force_downsample,
        force_dwt=args.force_dwt,
    )
    return run_feature_estimation(config)


if __name__ == "__main__":
    main()
