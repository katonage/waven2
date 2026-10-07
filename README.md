# waven2
This project provides a Python package designed to analyze neuronal responses in the visual cortex to visual stimuli using a Gabor transform of the stimulus. The package enables users to extract tuning curves for key visual features.

Some of the calculations are based on the [skriabineSop/waven](https://github.com/skriabineSop/waven) Python package, see related publication: 

Skriabine S, Shinn M, Picard S, Harris KD, Carandini M. Mapping the visual cortex with Zebra noise and wavelets. J Vis. 2026 Jan 5;26(1):1. doi: [10.1167/jov.26.1.1](https://doi.org/10.1167/jov.26.1.1). 

## Installation
The package can be installed:
* download repo, navigate in it.
* `pip install -e .`.

## S4 feature estimation

The reusable Python version of `S4_feature_estimation_vS.ipynb` can be run
from the repository root:

```powershell
python -m waven2.s4_feature_estimation --help
python -m waven2.s4_feature_estimation <resps_all.npy> <resampled_video.mp4>
```

Other applications can call `run_feature_estimation()` with a
`FeatureEstimationConfig`, without running the command-line interface.

## Cell browser visualization plugin

The cell database browser framework now lives in
[`twop_analysis`](https://github.com/katonage/twop_analysis). Install that package
and use `cb --cells <cells_waven1.cellDB_pickle>` or
`python -m twop_analysis.cell_browser`. The former
`python -m waven2.visual_cortex_cell_browser` invocation forwards to the new browser.

Waven2 provides `waven2.cell_browser_plugin`, registered in the
`twop_analysis.cell_browser.plugins` entry point group. It recognizes Waven
results by their columns/metadata and supplies the tuning, receptive-field,
statistics, and temporal panels, plus Waven filters and spatial color choices.
CaImAn databases open with the framework's plain panel. The framework loads
plugins only when browsing; running Waven analyses does not import the browser.

When developing from sibling checkouts, reinstall both editable packages after
this migration so plugin discovery and console commands are refreshed:

```powershell
python -m pip install -e . --no-deps
python -m pip install -e ..\twop_analysis --no-deps
```

