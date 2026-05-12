import numpy as np
import torch
import gc
from tqdm import tqdm
from torch_utils import handle_torch_device, print_cuda_tensors_mem
from itertools import combinations
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator

def correctNeuronPos(neuron_pos, resolution=2.14):
    """
    converts neuron position in micronshttps://chatgpt.com/

    Parameters:
        neuron_pos (array-like): shape (n_neurons x 2)
        resolution (float): resolution.

    Returns:
        neuron_pos_corrected[int]: new positions
    """

    #switch x and y positons
    neuron_pos[:, [0, 1]] = neuron_pos[:, [1, 0]]
    # flip x axis 
    neuron_pos[:, 1] = abs(neuron_pos[:, 1] - np.max(neuron_pos[:, 1]))
    # scale to microns
    neuron_pos=resolution*neuron_pos
    
    return neuron_pos


def compute_respcorr_split_half(resps_all):
    """
    Compute split-half reliability (trial-to-trial response correlation) per neuron.

    For each neuron:
        - Trials are split into two groups 
        - The mean response over time is computed for each group
        - Pearson correlation between the two averages is computed
        - The result is averaged across all possible splits
        
    This is a more computationally intensive versio of https://github.com/skriabineSop/waven Analysis_Utils.py repetability_trial3

    Parameters
    ----------
    resps_all : np.ndarray
        Array of shape (n_trials, n_timepoints, n_neurons)

    Returns
    -------
    respcorr : np.ndarray
        Array of shape (n_neurons,)
        Mean split-half correlation per neuron.
        Values range roughly from -1 to 1.
        NaN is returned if correlation cannot be computed (e.g. zero variance).

    Notes
    -----
    - Requires at least 2 trials.
    - Splits are generated using all combinations of size floor(n_trials / 2).
    """

    n_trials, n_t, n_neurons = resps_all.shape

    if n_trials < 2:
        raise ValueError("Need at least 2 trials to compute correlation.")

    k = n_trials // 2  # balanced split size
    splits = list(combinations(range(n_trials), k))

    respcorr = np.zeros(n_neurons)

    for ni in tqdm(range(n_neurons), desc="Computing split-half correlation per neuron"):
        corrs = []

        for groupA in splits:
            groupA = list(groupA)
            groupB = [i for i in range(n_trials) if i not in groupA]

            A = resps_all[groupA, :, ni].mean(axis=0)
            B = resps_all[groupB, :, ni].mean(axis=0)

            # Degeneracy check: zero variance → undefined correlation
            if np.std(A) == 0 or np.std(B) == 0:
                continue

            c = np.corrcoef(A, B)[0, 1]

            if not np.isnan(c):
                corrs.append(c)

        if len(corrs) == 0:
            respcorr[ni] = np.nan
        else:
            respcorr[ni] = np.mean(corrs)

    return respcorr

def FeatureSearch_correlation_batched(stim, resp, device="cuda", feature_batch_size=10_000  ):
    """
    Pearson correlation between WT stimulus features and neural responses. Uses GPU feature-batched.

    stim : np.ndarray
        Shape (n_timepoints, ...feature_dims)
    resp : np.ndarray
        Shape (n_timepoints, n_neurons)

    Returns
    -------
    rfs : np.ndarray
        Shape (n_neurons, ...feature_dims)
    """

    #types. float32 was the fastest
    dtype=torch.float32
    output_dtype=np.float32
    eps=1e-8
    
    #shapes
    stimshape = stim.shape
    n_timepoints = stim.shape[0]

    stim_flat = stim.reshape(n_timepoints, -1)
    n_features = stim_flat.shape[1]
    n_neurons = resp.shape[1]

    print(f"    stim_flat shape: {stim_flat.shape} (n_timepoints={n_timepoints}, n_features={n_features})")
    print(f"    resp shape: {resp.shape} (n_timepoints={resp.shape[0]}, n_neurons={n_neurons})")

    if resp.shape[0] != n_timepoints:
        raise ValueError(
            f"stim and resp must have same time dimension: "
            f"stim={n_timepoints}, resp={resp.shape[0]}"
        )

    device = handle_torch_device(device)

    # output is flat first, reshaped at end
    rfs_flat_out = np.empty((n_neurons, n_features), dtype=output_dtype)

    with torch.no_grad():
        # normalize response once
        R = torch.as_tensor(resp.T, dtype=dtype, device=device)  # (n_neurons, n_timepoints)
        R = R - R.mean(dim=1, keepdim=True)
        R = R / R.norm(dim=1, keepdim=True).clamp_min(eps)

        for f0 in tqdm( range(0, n_features, feature_batch_size), desc="Pearson RF feature batches" ):
            f1 = min(f0 + feature_batch_size, n_features)

            # S_chunk: (n_features_chunk, n_timepoints)
            S_chunk = torch.as_tensor(
                stim_flat[:, f0:f1].T,
                dtype=dtype,
                device=device
            )

            S_chunk = S_chunk - S_chunk.mean(dim=1, keepdim=True)
            S_chunk = S_chunk / S_chunk.norm(dim=1, keepdim=True).clamp_min(eps)

            # (n_neurons, n_timepoints) @ (n_timepoints, n_features_chunk)
            rfs_chunk = R @ S_chunk.T

            rfs_flat_out[:, f0:f1] = rfs_chunk.cpu().numpy()

    print_cuda_tensors_mem({"R": R, "S_chunk": S_chunk, "rfs_chunk": rfs_chunk})

    rfs_flat_out = np.nan_to_num(rfs_flat_out)
    rfs = rfs_flat_out.reshape((n_neurons, *stimshape[1:]))

    print(f"    output shape: {rfs.shape} (neurons={n_neurons}, feature_dims={stimshape[1:]})")

    del stim_flat, R, rfs_flat_out
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rfs


def dwt_amp_phase_torch_batched(dwt, device="cuda", batch_size=256, output_dtype=np.float32, ):
    """
    Compute _squared_ amplitude and phase from real/imag array using torch batching.

    Parameters
    ----------
    dwt : np.ndarray
        Shape (..., 2), last dim is [real, imag].
    device : str
        "cuda" or "cpu".
    batch_size : int
        Batch size along first axis.
    output_dtype : np.dtype
        Output dtype.

    Returns
    -------
    dwt_squared : np.ndarray
        Shape dwt.shape[:-1].
    dwt_phase : np.ndarray
        Shape dwt.shape[:-1].
    """

    device = handle_torch_device(device)

    out_shape = dwt.shape[:-1]

    dwt_squared = np.empty(out_shape, dtype=output_dtype)
    dwt_phase = np.empty(out_shape, dtype=output_dtype)

    n0 = dwt.shape[0]

    with torch.no_grad():
        for i0 in tqdm(range(0, n0, batch_size), desc="Calculating DWT amplitude and phase"):
            i1 = min(i0 + batch_size, n0)

            batch = torch.as_tensor(
                dwt[i0:i1],
                dtype=torch.float32,
                device=device
            )

            real = batch[..., 0]
            imag = batch[..., 1]

            amp2 = real * real + imag * imag
            phase = torch.atan2(imag, real)
            phase = phase + torch.pi

            dwt_squared[i0:i1] = amp2.cpu().numpy()
            dwt_phase[i0:i1] = phase.cpu().numpy()

        
        print_cuda_tensors_mem({"batch": batch, "real": real, "imag": imag, "amp2": amp2, "phase": phase})
        del batch, real, imag, amp2, phase

        if device.type == "cuda":
            torch.cuda.empty_cache()

    gc.collect()

    return dwt_squared, dwt_phase

def smooth_stimulus_signals(rho, phi, dphi, average_FWHM_samples):
    """
    Gaussian temporal smoothing of rho, phi, dphi.

    Phase is unwrapped before filtering and wrapped back to [0, 2π].
    FWHM is given in samples (sigma = FWHM / 2.355). No smoothing if ≤ 0.

    Returns: rho_smooth, phi_smooth, dphi_smooth
    """
    
    if average_FWHM_samples>0:
        rho=gaussian_filter1d(rho, sigma=average_FWHM_samples / 2.355) 
        phi=np.mod(gaussian_filter1d(np.unwrap(phi), sigma=average_FWHM_samples / 2.355), 2 * np.pi)
        dphi=gaussian_filter1d(dphi, sigma=average_FWHM_samples / 2.355) 
        
    return rho, phi, dphi


def fit_model(rho, phi, dphi, spks, hanning_window=4, ncut=20, smooth_stim_FWHM_samples=0):
    """
    Fit a 3D histogram-based model predicting spiking from (rho, phi, dphi). See Skriabine et al. 2026.

    Builds a weighted histogram, returns an interpolator for prediction along with the smoothed grid.
    
    Returns:
        interpolator
        Z - 3D histogram
        rho_grid, phi_grid, dphi_grid
    """
    
    def hanningconv3d(interp_grid, n):
        kern = np.hanning(n).reshape(-1, 1)
        kern = kern * kern.T
        kern=kern[:, :, np.newaxis]* kern.T
        kern /= kern.sum()  # normalize the kernel weights to sum to 1
        hanning = ndimage.convolve(interp_grid, kern)
        return hanning
    
    rho, phi, dphi = smooth_stimulus_signals(rho, phi, dphi, smooth_stim_FWHM_samples)
    
    # --- bin ranges
    rho_max = np.nanmax(np.abs(rho))
    if rho_max < 0.01:
        rho_max = 0.3
        
    dphi_max = np.nanmax(np.abs(dphi))
    if dphi_max <= 0.01:
        dphi_max = 1.0
    
    # --- phase into 0..2pi
    #phi = np.mod(phi, 2 * np.pi)

    E = [
        np.linspace(0, rho_max, ncut + 1),
        np.linspace(0, 2*np.pi, ncut + 1),
        np.linspace(-dphi_max, dphi_max, ncut + 1),
    ]

    # --- histogram input: shape (n_samples, 3)
    data = np.column_stack([rho, phi, dphi])
    
    # weighted histogram: sum of spikes per bin
    H_sum, edges = np.histogramdd(data, bins=E, density=False, weights=spks)
    # occupancy histogram: number of samples per bin
    H_count, _ = np.histogramdd(data, bins=E, density=False)
    
    # --- tile phase axis to handle circular boundary during convolution
    # axis 1 is phi: 0 and 2pi should be neighbors
    H_sum_tiled = np.concatenate([H_sum, H_sum, H_sum], axis=1)
    H_count_tiled = np.concatenate([H_count, H_count, H_count], axis=1)

    # --- smooth numerator and denominator
    num = hanningconv3d(H_sum_tiled, hanning_window)
    den = hanningconv3d(H_count_tiled, hanning_window)

    # --- avoid divide-by-zero
    Z_tiled = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)

    # --- crop back to one phase cycle, keep the center copy
    n_phi = H_sum.shape[1]
    Z = Z_tiled[:, n_phi:2*n_phi, :]
    
    rho_centers = (E[0][:-1] + E[0][1:]) / 2
    phi_centers = (E[1][:-1] + E[1][1:]) / 2
    dphi_centers = (E[2][:-1] + E[2][1:]) / 2
    
    #print(f"max rho: {np.max(rho_centers):.3f}, min rho: {np.min(rho_centers):.3f}")
    #print(f"max phi: {np.max(phi_centers):.3f}, min phi: {np.min(phi_centers):.3f}")
    #print(f"max dphi: {np.max(dphi_centers):.3f}, min dphi: {np.min(dphi_centers):.3f}")

    mask = np.where(np.isfinite(Z))
    interp = NearestNDInterpolator(np.transpose((rho_centers[mask[0]], phi_centers[mask[1]], dphi_centers[mask[2]])), Z[mask])
    
    return interp, Z, rho_centers, phi_centers, dphi_centers


def apply_model(interp, rho, phi, dphi, smooth_stim_FWHM_samples=0):
    """
    Apply fitted interpolator to (rho, phi, dphi) to predict response.

    Optionally smooths inputs (phase-aware Gaussian, FWHM in samples) before
    evaluation. Returns predicted time series.
    """
    rho, phi, dphi = smooth_stimulus_signals(rho, phi, dphi, smooth_stim_FWHM_samples)
    pred = interp(rho, phi, dphi)
    return  pred

def sine1x(x, constant, amplitude, orientation):
    """
    One-period sine on 0..1π : Used to fit angle tuning curve (phase = orientation). Orientation is in radians

    y = constant + amplitude * sin(x + orientation)
    """
    return constant + amplitude * np.cos( 2 * (x - orientation))


def fit_sine1x(x, y):
    """
    Fits sine1x to data (see sine1x). Used to fit angle tuning curve

    Parameters
    ----------
    x : array: phase/orientation
    y : array: response

    Returns
    -------
    params : dict
        {
            "constant": ...,
            "amplitude": ...,
            "orientation": ...
        }
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    x2=2*x
    X = np.column_stack([
        np.ones_like(x2),
        np.sin(x2),
        np.cos(x2),
    ])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    constant = coef[0]
    sin_coef = coef[1]
    cos_coef = coef[2]

    amplitude = np.sqrt(sin_coef**2 + cos_coef**2)
    orientation = np.mod(np.arctan2(sin_coef, cos_coef)/2,  np.pi)

    return {
        "constant": constant,
        "amplitude": amplitude,
        "orientation": orientation,
    }
