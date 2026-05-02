import numpy as np
import torch
import gc
from tqdm import tqdm
from torch_utils import handle_torch_device, print_cuda_tensors_mem

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

def repetability_trial3(resps_all):
    """
    Computes the repeatbility accross trials of the neuronal activity
    
    Adapted from: https://github.com/skriabineSop/waven Analysis_Utils.py repetability_trial3 

    Parameters:
        resps_all (array-like): shape (nb_trials, nb_timepoints, nb_neurons).
        neuron_pos (array-like): shape (nb_neurons, nb_dim(x, y,))

    Returns:
        response correation (array): shape (nb_neurons), pearsons correlation of the repeats
    """
    ## repetability across trial
    n_cell=resps_all.shape[2]

    respcorrs = np.zeros(n_cell)

    if resps_all.shape[0] == 2:
        for i in range(n_cell):
            meanresp1 = np.mean(resps_all[[0], :, i], axis=0)
            meanresp2 = np.mean(resps_all[[1], :, i], axis=0)
            respcorr = np.corrcoef(meanresp1, meanresp2)[0, 1]
            respcorrs[i] = respcorr

    elif resps_all.shape[0] == 3:
        for i in range(n_cell):
            meanresp1 = np.mean(resps_all[[0, 2], :, i], axis=0)
            meanresp2 = np.mean(resps_all[[1], :, i], axis=0)
            respcorr = np.corrcoef(meanresp1, meanresp2)[0, 1]
            respcorrs[i] = respcorr

    elif resps_all.shape[0] == 4:
        for i in range(n_cell):
            meanresp1 = np.mean(resps_all[[0, 2], :, i], axis=0)
            meanresp2 = np.mean(resps_all[[1, 3], :, i], axis=0)
            respcorr = np.corrcoef(meanresp1, meanresp2)[0, 1]
            respcorrs[i] = respcorr

    elif resps_all.shape[0] == 5:
        for i in range(n_cell):
            meanresp1 = np.mean(resps_all[[0, 2, 4], :, i], axis=0)
            meanresp2 = np.mean(resps_all[[1, 3], :, i], axis=0)
            respcorr = np.corrcoef(meanresp1, meanresp2)[0, 1]
            respcorrs[i] = respcorr

    else:
        raise ValueError(f"Unexpected number of trials: {resps_all.shape[0]}. Expected 2, 3, 4, or 5.")


    return respcorrs



def PearsonCorrelationRF_batched(stim, resp, device="cuda", feature_batch_size=10_000  ):
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