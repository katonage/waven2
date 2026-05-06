import numpy as np
import torch
import gc
from tqdm import tqdm
from torch_utils import handle_torch_device, print_cuda_tensors_mem
from itertools import combinations

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