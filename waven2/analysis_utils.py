import numpy as np
import torch
import gc
import cv2
from pathlib import Path
from tqdm import tqdm
from torch_utils import handle_torch_device, print_cuda_tensors_mem
import torch.nn.functional as F
from itertools import combinations

def downscale_binary_video(path, full_screen_coverage, visual_coverage, screen_x, screen_y=None, output_path=None, force=False):
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
    
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=bool,
        shape=(n_frames, screen_x, screen_y),
    )


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

        binary = resized > threshold
        out[frame_idx] = binary.T

    cap.release()
    out.flush()
    del out

    print(f"Saved downsampled binary video: {output_path}")
    return output_path

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


def dwt_amp_phase_torch_batched(dwt, device="cuda", batch_size=256, output_dtype=np.float32, ):
    """
    Compute amplitude and phase from real/imag array using torch batching.

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
    dwt_amplitude : np.ndarray
        Shape dwt.shape[:-1].
    dwt_phase : np.ndarray
        Shape dwt.shape[:-1].
    """

    device = handle_torch_device(device)

    out_shape = dwt.shape[:-1]

    dwt_amplitude = np.empty(out_shape, dtype=output_dtype)
    dwt_phase = np.empty(out_shape, dtype=output_dtype)

    n0 = dwt.shape[0]

    with torch.no_grad():
        for i0 in tqdm(range(0, n0, batch_size), desc="Calculating DWT amplitude and phase"):
            i1 = min(i0 + batch_size, n0)

            batch = torch.as_tensor(
                dwt[i0:i1].copy(),
                dtype=torch.float32,
                device=device
            )

            real = batch[..., 0]
            imag = batch[..., 1]

            amp = torch.sqrt(real * real + imag * imag) 
            phase = torch.atan2(imag, real)
            phase = phase + torch.pi

            dwt_amplitude[i0:i1] = amp.cpu().numpy()
            dwt_phase[i0:i1] = phase.cpu().numpy()

        
        print_cuda_tensors_mem({"batch": batch, "real": real, "imag": imag, "amp": amp, "phase": phase})
        del batch, real, imag, amp, phase

        if device.type == "cuda":
            torch.cuda.empty_cache()

    gc.collect()

    return dwt_amplitude, dwt_phase

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
