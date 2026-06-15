# This module implements the histogram-based model fitting and prediction as described in Skriabine et al. 2026.

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator


def smooth_stimulus_signals(rho, phi, dphi, average_FWHM_samples):
    """
    Gaussian temporal smoothing of rho, phi, dphi. See Skriabine et al. 2026.

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
    
    Parameters:
    - rho, phi, dphi: 1D arrays of stimulus features (shape: n_samples)
    - spks: 1D array of spike counts, "truth" (shape: n_samples). Note interpolator is linear in spks so average repeats beforehand
    - hanning_window: int, size of Hanning window for smoothing histogram (default: 4)
    - ncut: int, number of bins per dimension for histogram (default: 20)
    - smooth_stim_FWHM_samples: float, FWHM in samples for optional temporal smoothing of stimulus features before histogramming (default: 0, no smoothing)
    
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
    Apply fitted interpolator to (rho, phi, dphi) to predict response. See fit_model and Skriabine et al. 2026.

    Optionally smooths inputs (phase-aware Gaussian, FWHM in samples) before
    evaluation. Returns predicted time series.
    """
    rho, phi, dphi = smooth_stimulus_signals(rho, phi, dphi, smooth_stim_FWHM_samples)
    pred = interp(rho, phi, dphi)
    return  pred