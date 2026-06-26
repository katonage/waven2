import numpy as np
import torch
import gc
import cv2
from pathlib import Path
from tqdm import tqdm
try:
    from .torch_utils import handle_torch_device, print_cuda_tensors_mem
except ImportError:  # Support direct execution from the module directory.
    from torch_utils import handle_torch_device, print_cuda_tensors_mem
import torch.nn.functional as F
from itertools import combinations
from skimage.registration import optical_flow_tvl1
from scipy.ndimage import gaussian_filter


def downscale_binary_video(path, full_screen_coverage, visual_coverage, screen_x, screen_y=None, output_path=None, fps=30, force=False):
    """
    Crop and downscale a binary visual stimulus video.
    
    Adapted from: https://github.com/skriabineSop/waven WaveletGenerator.py downsample_video_binary

    Parameters
    ----------
    path : str or Path
        Input .mp4 file.
    full_screen_coverage : list/tuple
        [az_left, az_right, el_bottom, el_top] in visual degrees for the full video.
    visual_coverage : list/tuple
        [az_left, az_right, el_bottom, el_top] in visual degrees to keep.
    screen_x, screen_y : int
        Output frame size e.g. (100, 66).        
    output_path : str or Path, optional
        Output .npy path. If None, saves next to input.
    fps : float, optional
        Input video fps. Effective only if generate_optic_flow is True.
    force : bool, optional
        Overwrite existing .npy file. Otherwise, skips if already exists.
    Returns
    -------
    Path
        Path to saved .npy file.
    

    Saved array shape
    -----------------
    (n_frames, screen_x, screen_y), dtype bool
    """
    
    threshold=127 #Pixel threshold for binarization.
    
    full = np.asarray(full_screen_coverage, dtype=float)
    vis = np.asarray(visual_coverage, dtype=float)

    az_left, az_right, el_bottom, el_top = full
    v_az_left, v_az_right, v_el_bottom, v_el_top = vis
    
    if screen_y is None:
        # keep aspect ratio of visual coverage
        screen_y = int(screen_x * (v_el_top - v_el_bottom) / (v_az_right - v_az_left))
        print(f"Calculated screen_y={screen_y} to keep aspect ratio of visual coverage.")

    path = Path(path)
    
    if output_path is None:
        output_path = path.with_name(path.stem + f"_scaled{screen_x}x{screen_y}.npy")
    else:
        output_path = Path(output_path)

            
    print("Generating cropped and downsampled binary video...")
    if output_path.exists() and not force:
        print(f"Output file {output_path} already exists. Skipping generation.")
        return output_path

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if n_frames <= 0:
        raise ValueError("Could not determine number of frames from video.")

    # Degree -> pixel conversion
    x0 = round((v_az_left - az_left) / (az_right - az_left) * input_w)
    x1 = round((v_az_right - az_left) / (az_right - az_left) * input_w)

    # image y goes top -> bottom, elevation goes bottom -> top
    y0 = round((el_top - v_el_top) / (el_top - el_bottom) * input_h)
    y1 = round((el_top - v_el_bottom) / (el_top - el_bottom) * input_h)

    x0, x1 = sorted((max(0, x0), min(input_w, x1)))
    y0, y1 = sorted((max(0, y0), min(input_h, y1)))

    print(f"Input video: {n_frames} frames, {input_w} x {input_h}")
    print(f"Crop pixels: x={x0}:{x1}, y={y0}:{y1}")
    print(f"Output shape: ({n_frames}, {screen_x}, {screen_y})")
    
    deg_per_px_x = (v_az_right - v_az_left) / screen_x # output pixel dimensions
    deg_per_px_y = (v_el_top - v_el_bottom) / screen_y
    
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=bool,
        shape=(n_frames, screen_x, screen_y),
    )


    prev_gray = None
    for frame_idx in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        gray = frame[:, :, 0]
        
        cropped = gray[y0:y1, x0:x1]

        resized = cv2.resize(
            cropped,
            dsize=(screen_x, screen_y),
            interpolation=cv2.INTER_AREA,
        )
        # OpenCV frames are indexed from top to bottom. Store the NPY in
        # (x, y) order with y increasing from the physical bottom to the top.
        resized = np.flipud(resized).T

        binary = resized > threshold
        out[frame_idx] = binary
                

    cap.release()
    out.flush()

    print(f"Saved downsampled binary video: {output_path}")
    return output_path


def correctNeuronPos(neuron_pos, resolution=2.14):
    """
    converts neuron position in microns

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
        
    This is a more computationally intensive version of https://github.com/skriabineSop/waven Analysis_Utils.py repetability_trial3

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
        corrs = np.empty(len(splits))

        for i, groupA in enumerate(splits):
            groupA = list(groupA)
            groupB = [i for i in range(n_trials) if i not in groupA]

            A = resps_all[groupA, :, ni].mean(axis=0)
            B = resps_all[groupB, :, ni].mean(axis=0)

            # Degeneracy check: zero variance → undefined correlation
            if np.std(A) == 0 or np.std(B) == 0:
                continue

            c = np.corrcoef(A, B)[0, 1]

            if not np.isnan(c):
                corrs[i] = c

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


def dwt_amp_phase_torch_batched(dwt, phases=None, device="cuda", batch_size=8192, output_dtype=np.float32, calculate_phase=False):
    """
    Compute amplitude, and optionally phase, from a phase-sampled DWT.

    Parameters
    ----------
    dwt : np.ndarray
        Shape (..., n_phase). The last dimension contains DWT values sampled
        at each phase.
    phases : array-like, optional
        Shape (n_phase,). Phase values in radians. If None, phases are assumed
        to be evenly spaced over [0, 2*pi).
    device : str
        "cuda" or "cpu".
    batch_size : int
        Batch size along first axis.
    output_dtype : np.dtype
        Output dtype.
    calculate_phase : bool
        Calculate phase if True.

    Returns
    -------
    dwt_amplitude : np.ndarray
        Shape dwt.shape[:-1].
    dwt_phase : np.ndarray
        Shape dwt.shape[:-1]. If calculate_phase is True.
    """

    device = handle_torch_device(device)

    dwt = np.asarray(dwt)
    if dwt.ndim < 1:
        raise ValueError("dwt must have at least one dimension.")

    n_phases = dwt.shape[-1]
    if n_phases < 1:
        raise ValueError("dwt must contain at least one phase sample on the last axis.")

    if phases is None:
        phases = np.linspace(0.0, 1.0 * np.pi, n_phases, endpoint=False, dtype=np.float32)
        print(f"-- assuming evenly-spaced phases: {phases}")
    else:
        phases = np.asarray(phases, dtype=np.float32)

    if phases.shape != (n_phases,):
        raise ValueError(f"phases shape {phases.shape} must match dwt last dimension ({n_phases},)")
    
    simplify = (phases.size==2) and phases[0]==0 and phases[1]==np.pi/2 # that speeds by x1.5

    out_shape = dwt.shape[:-1]
    dwt_flat = dwt.reshape(-1, n_phases)
    n_items = dwt_flat.shape[0]

    dwt_amplitude_flat = np.empty(n_items, dtype=output_dtype)
    if calculate_phase:
        dwt_phase_flat = np.empty(n_items, dtype=output_dtype)

    with torch.no_grad():
        phase_tensor = torch.as_tensor(phases, dtype=torch.float32, device=device)
        scale = 2.0 / float(n_phases)
        cos_phi = torch.cos(phase_tensor)
        sin_phi = torch.sin(phase_tensor)

        for i0 in tqdm(range(0, n_items, batch_size), desc="Calculating DWT amplitude and phase"):
            i1 = min(i0 + batch_size, n_items)

            batch_np = dwt_flat[i0:i1]
            try:
                batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            except (TypeError, ValueError):
                batch = torch.as_tensor(batch_np.copy(), dtype=torch.float32, device=device)

            if simplify: # that speeds by x1.5
                real = batch[:, 0]
                imag = batch[:, 1]
            else:
                real = scale * torch.sum(batch * cos_phi, dim=-1)
                imag = scale * torch.sum(batch * sin_phi, dim=-1)
            amp = torch.sqrt(real * real + imag * imag)

            dwt_amplitude_flat[i0:i1] = amp.cpu().numpy()
            if calculate_phase:
                phase = torch.atan2(imag, real)
                dwt_phase_flat[i0:i1] = phase.cpu().numpy()

        
        if n_items > 0:
            if calculate_phase:
                print_cuda_tensors_mem({"batch": batch, "real": real, "imag": imag, "amp": amp, "phase": phase})
            else:
                print_cuda_tensors_mem({"batch": batch, "real": real, "imag": imag, "amp": amp})
            del batch_np, batch, real, imag, amp
            if calculate_phase: del phase
        del phase_tensor, cos_phi, sin_phi

        if device.type == "cuda":
            torch.cuda.empty_cache()

    gc.collect()

    dwt_amplitude = dwt_amplitude_flat.reshape(out_shape)
    if calculate_phase:
        dwt_phase = dwt_phase_flat.reshape(out_shape)
        return dwt_amplitude, dwt_phase
    else:
        return dwt_amplitude

def gaussian_filter1d_torch_axis0_chunked(x, sigma, chunk_size=20_000, device='cuda', dtype=torch.float32, return_dtype=None ):
    """
    Chunked Gaussian smoothing along axis=0 using torch. Replaces scipy.ndimage.gaussian_filter1d axis=0.

    x shape: (time, channels) or (time, ...)
    returns NumPy array with same shape.
    """

    if sigma <= 0:
        return x.copy()

    device = handle_torch_device(device)

    orig_shape = x.shape
    x2 = x.reshape(orig_shape[0], -1)
    n_time, n_ch = x2.shape

    if return_dtype is None:
        return_dtype = x.dtype

    y = np.empty_like(x2, dtype=return_dtype)

    truncate=4.0
    radius = int(truncate * sigma + 0.5)

    grid = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=dtype
    )

    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()

    for c0 in tqdm(range(0, n_ch, chunk_size), desc="Gaussian smoothing"):
        c1 = min(c0 + chunk_size, n_ch)

        # shape: (time, chunk)
        x_chunk = torch.as_tensor(
            x2[:, c0:c1],
            dtype=dtype,
            device=device
        )

        # conv1d expects: (batch, channels, time)
        x_chunk = x_chunk.T[None, :, :]

        k = kernel[None, None, :].repeat(x_chunk.shape[1], 1, 1)

        x_chunk = F.pad(x_chunk, (radius, radius), mode="reflect")

        y_chunk = F.conv1d(
            x_chunk,
            k,
            groups=x_chunk.shape[1]
        )

        # back to NumPy: (time, chunk)
        y[:, c0:c1] = y_chunk[0].T.detach().cpu().numpy().astype(return_dtype)

        del x_chunk, y_chunk, k

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return y.reshape(orig_shape)


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

def fit_quadratic(x, y, n_points=4):
    """
    Return refined x-position of the maximum using a local quadratic fit.

    Returns
    -------
    x_peak : float
        Refined x-position of the maximum.
        If fitting is not safe, this is the discrete maximum x.

    fit_info : tuple or None
        Minimal information needed to reproduce the fitted curve:
            (a, b, c, x_min_fit, x_max_fit)

        If None, no accepted fit was used.
    """

    def _local_peak_indices(x, y, imax, n_points=4):
        """
        Select local fit points around the maximum.

        For n_points=4:
        - maximum point
        - stronger immediate neighbor
        - one more point outside the pair on each side, if available
        """
        n = len(x)
        n_points = max(3, min(int(n_points), n))

        if n <= n_points:
            return np.arange(n)

        if n_points % 2 == 1:
            half = n_points // 2
            start = max(0, imax - half)
            stop = min(n, start + n_points)
            start = max(0, stop - n_points)
            return np.arange(start, stop)

        # Even n_points: use max + stronger immediate neighbor as core.
        # This function is called only for non-edge maxima.
        if y[imax - 1] >= y[imax + 1]:
            lo, hi = imax - 1, imax
        else:
            lo, hi = imax, imax + 1

        selected = set(range(lo, hi + 1))
        left = lo - 1
        right = hi + 1

        while len(selected) < n_points and (left >= 0 or right < n):
            if left >= 0 and len(selected) < n_points:
                selected.add(left)
                left -= 1
            if right < n and len(selected) < n_points:
                selected.add(right)
                right += 1

        return np.array(sorted(selected), dtype=int)

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan, None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    imax = int(np.argmax(y))
    x0 = float(x[imax])

    # Edge maximum: return the edge. Never extrapolate.
    if imax == 0 or imax == len(x) - 1:
        return x0, None

    if len(x) < 3:
        return x0, None

    idx = _local_peak_indices(x, y, imax, n_points=n_points)
    if len(idx) < 3:
        return x0, None

    xf = x[idx]
    yf = y[idx]

    a, b, c = np.polyfit(xf, yf, deg=2)

    # A maximum requires a downward-opening parabola.
    if not np.isfinite(a) or not np.isfinite(b) or a >= 0:
        return x0, None

    x_peak = float(-b / (2 * a))

    # No extrapolation: accept only inside fitted data range.
    if x_peak < xf.min() or x_peak > xf.max():
        return x0, None
    
    # further constrain to be within the neighbors of imax (imax already checked to be non-edge)
    if x_peak < x[imax - 1] or x_peak > x[imax + 1]:
        return x0, None

    fit_info = (float(a), float(b), float(c), float(xf.min()), float(xf.max()))
    return x_peak, fit_info

def restore_fit_quadratic(fit_info, n_points=100):
    if fit_info is None:
        raise ValueError("fit_info cannot be None")
    a, b, c, x_min_fit, x_max_fit = fit_info
    x_fit = np.linspace(x_min_fit, x_max_fit, n_points)
    y_fit = a * x_fit**2 + b * x_fit + c
    return x_fit, y_fit
