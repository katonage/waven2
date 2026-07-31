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

