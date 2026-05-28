# Analysis test - modified correlation with the library
# Calculate best correlation match with (arithmetic modification) of Gabor filter library decomposed visual stimulation. 
# 
# !!! for metrics search
# Puts output in dedicated folder.


from turtle import delay

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

from wavelet_utils_vSpeed import loadFilterParamDict_vS, makeFilterParamDict_vS, saveFilterParamDict_vS, filename_fromFilterParam, compute_and_save_dwt_vS
from analysis_utils import compute_respcorr_split_half, FeatureSearch_correlation_batched, fit_sine1x, sine1x, dwt_amp_phase_torch_batched
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn.functional as F
from tqdm import tqdm


def gaussian_filter1d_torch_axis0_chunked(
        x,
        sigma,
        chunk_size=20_000,
        truncate=4.0,
        device=None,
        dtype=torch.float32,
        return_dtype=None,
    ):
    """
    Chunked Gaussian smoothing along axis=0.

    x shape: (time, channels) or (time, ...)
    returns NumPy array with same shape.
    """

    if sigma <= 0:
        return x.copy()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_shape = x.shape
    x2 = x.reshape(orig_shape[0], -1)
    n_time, n_ch = x2.shape

    if return_dtype is None:
        return_dtype = x.dtype

    y = np.empty_like(x2, dtype=return_dtype)

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


def full_analysis_vSpeed3(spks_path, downsampled_video_path,
                          full_screen_coverage = [-88, 0, -33, 33],
                          visual_coverage = [-88, 0, -33, 33],
                          screen_x = 100,
                          screen_t = 3,
                          nx = 6,
                          n_thetas = 8,
                          theta_max = np.pi,
                          size_min = 4,
                          size_max = 20,
                          n_sizes = 6,
                          freq_min = .025,
                          freq_max = .15,
                          n_freqs = 5,
                          n_phases = 4,
                          phase_max = np.pi*2,
                          driftmax=4,
                          driftnum=3,
                          smooth_fwhm_samples=3, # 0.1 sec at 30 fps
                          shift_samples=1, # +1 moves spks forward
                          comment="m3"):
    # runs full analysis:
    # Parameters
    # ----------
    # spks_path : str
    #     path to the spks file
    # downsampled_video_path : str
    #     path to the downsampled video .npy file to analyze
    # full_screen_coverage : list of 4 floats, optional
    #     [az_left, az_right, el_bottom, el_top] full screen position in visual degrees
    # visual_coverage : list of 4 floats, optional
    #     [az_left, az_right, el_bottom, el_top] screen coverage in visual degrees (seen by the animal)
    # screen_x : int, optional
    #     horizontal screen size in pixels for the Gabor filter generation and movie analysis
    # screen_t : int, optional
    #     time dimension of the screen in pixels
    # nx : int, optional
    #     number of Gabor filters in the horizontal direction (azimuth) (y will be generated)  40 is a good number. Use 5 for testing to make it fast
    # n_thetas : int, optional
    #     number of angles to generate for Gabor filters (default is 8)
    # theta_max : float, optional
    #     maximum angle to generate in radians (default is pi)
    # size_min : float, optional
    #     minimum size of Gabor filters in visual degrees (default is 4)
    # size_max : float, optional
    #     maximum size of Gabor filters in visual degrees (default is 20)
    # n_sizes : int, optional
    #     number of sizes to generate (default is 6)
    # freq_min : float, optional
    #     minimum frequency of Gabor filters in cycles per visual degree (default is 0.025)
    # freq_max : float, optional
    #     maximum frequency of Gabor filters in cycles per visual degree (default is 0.15)
    # n_freqs : int, optional
    #     number of frequencies to generate (default is 5)
    # n_phases : int, optional
    #     number of phases to generate (default is 2)
    # phase_max : float, optional
    #     maximum phase to generate in radians (default is pi)
    # driftmax : float, optional
    #     maximum drift speed to generate in degrees per frame (default is 4)
    # driftnum : int, optional
    #     number of drift speeds to generate 2n+1 (default is 3)
    # smooth_fwhm_samples : float, optional
    #     handles spks, full width half maximum of the Gaussian filter in samples (default is 3)
    # shift_samples : int, optional
    #     handles spks, forwardshift in samples (default is 1)

    if Path(spks_path).exists():   print(spks_path)
    else: print(f"File not found: {spks_path}")


    # %%
    #calculations
    az_left, az_right, el_bottom, el_top = visual_coverage

    screen_y = int(screen_x * (el_top - el_bottom) / (az_right - az_left))
    ny = int(nx * (el_top - el_bottom) / (az_right - az_left))

    # centers in visual degrees
    xs = np.linspace(az_left, az_right, nx, endpoint=False)+(az_right - az_left) / (2*nx)
    ys = np.linspace(el_bottom, el_top, ny, endpoint=False)+(el_top - el_bottom) / (2*ny)

    angles= np.linspace(0, theta_max, n_thetas, endpoint=False)
    sizes = np.logspace(np.log10(size_min), np.log10(size_max), n_sizes)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    phases = np.linspace(0,  phase_max, n_phases, endpoint=False)
    drifts=np.linspace(-driftmax, driftmax, driftnum*2+1)

    print(f"Screen size: {screen_x}x{screen_y} pixels")
    print(f"Full screen coverage: {full_screen_coverage} degrees")
    print(f"Visual coverage: {visual_coverage} degrees")
    #print(f"Center positions (x_deg): {np.round(xs, 1)} degrees")
    #print(f"Center positions (y_deg): {np.round(ys, 1)} degrees")
    print(f"Angles (degrees): {np.round(np.rad2deg(angles), 1)}")
    print(f"Sizes (degrees): {sizes}")
    print(f"Frequencies (cycles/degree): {freqs}")
    print(f"Phases (degrees): {np.rad2deg(phases)}")
    print(f"Drifts (degrees/frame): {drifts}")

    # %%
    total_n=len(sizes)*len(angles)*len(freqs)*len(drifts)*len(phases)*len(xs)*len(ys)
    print(f"Total number of Gabor filters to generate: {total_n}")

    # %%
    gabor_step=(az_right-az_left)/nx 
    print(f"Control: Gabor placement step in visual degrees (x): {gabor_step:.1f}, vs size_min: {size_min:.1f} degrees. {'OK' if (gabor_step < size_min) else 'WARNING!'}")
    gabor_step=(el_top-el_bottom)/ny
    print(f"Control: Gabor placement step in visual degrees (y): {gabor_step:.1f}, vs size_min: {size_min:.1f} degrees. {'OK' if (gabor_step < size_min) else 'WARNING!'}")
    visual_step_x=(az_right-az_left)/screen_x
    print(f"Control: Gabor resolution in visual degrees (x): {visual_step_x:.1f}, vs 1/freq_max: {1/freq_max:.1f} degrees. {'OK' if (visual_step_x < 1/freq_max/4) else 'WARNING!'}")
    visual_step_y=(el_top-el_bottom)/screen_y
    print(f"Control: Gabor resolution in visual degrees (y): {visual_step_y:.1f}, vs 1/freq_max: {1/freq_max:.1f} degrees. {'OK' if (visual_step_y < 1/freq_max/4) else 'WARNING!'}")

    wavelet_params=makeFilterParamDict_vS(screen_x, screen_y, screen_t, visual_coverage, full_screen_coverage, xs, ys, angles, sizes, freqs, drifts, phases)

    # %%
    # Video to analyze


    _, paramname = filename_fromFilterParam(wavelet_params)
    paramspath = downsampled_video_path.parent / paramname
    saveFilterParamDict_vS(wavelet_params, paramspath)
    print(f"Saved Gabor filter parameters to {paramspath}")


    # %% [markdown]
    # ## Calculate DWT if not already done

    # %%
    dwt_path=compute_and_save_dwt_vS(downsampled_video_path, wavelet_params,   device='cuda', force=False)
    if Path(dwt_path).exists():   print(dwt_path)
    else: print(f"File not found: {dwt_path}")

    # %%
    xs, ys, angles, sizes, freqs, drifts,  phases, visual_coverage, full_screen_coverage, screen_t, screen_x, screen_y = loadFilterParamDict_vS(paramspath)

    # %% [markdown]
    # ## Load spike data

    # %%
    spks=np.load(spks_path) # shape (n_trials, n_timepoints, n_neurons)
    if smooth_fwhm_samples > 0:
        spks = gaussian_filter1d(spks, sigma=smooth_fwhm_samples / 2.355,  axis=1)
    if shift_samples != 0:
        spks = np.roll(spks, -shift_samples, axis=1)
    print(f"spks shape: {spks.shape} (n_trials, n_timepoints, n_neurons), smoothed by {smooth_fwhm_samples} samples, shifted forward by {shift_samples} samples")
    n_trials=spks.shape[0]
    n_timepoints=spks.shape[1]
    n_neurons=spks.shape[2]
    comment=f"{comment}.s{smooth_fwhm_samples}.d{shift_samples}_"

    respcorr = compute_respcorr_split_half(spks)
    mean_spks = np.mean(spks[:, :, :], axis=0)

    working_dir = Path(spks_path).parent
    print(f"Working directory: {working_dir}")

    ## converts neuron position in microns
    #neuron_pos=np.load(working_dir / 'component_centers.npy')
    #neuron_pos=correctNeuronPos(neuron_pos, resolution)
    #print(f"neuron_pos shape: {neuron_pos.shape}")


    # ## Splitting data
    #defining train and test data parts
    train_split=0.85 #plit ratio in time if train_test
    train_split_index=int(n_timepoints*train_split)
    print(f"Train split at {train_split_index/n_timepoints*100:.1f}% of timepoints, test split at {((1-train_split_index/n_timepoints)*100):.1f}% of timepoints")
    
    output_folder = working_dir / (paramname.replace('.json', '')+f"_{comment}")
    output_folder.mkdir(exist_ok=True)
    saveFilterParamDict_vS(wavelet_params, output_folder / paramname)  # save a copy in the output folder 
    print(f"Saved Gabor filter parameters to {output_folder}")
    
    # %% [markdown]
    # ## Correlate neural data with decomposed stimulus

    # %%
    # load dwt
    dwt = np.load(dwt_path) # shape (n_timepoints, n_features) 
    r_trainMulti=[]
    r_testMulti=[]
    commentsMulti=[]
    
    print(f">>>>>>>>dwt shape: {dwt.shape}")

    for mode in range(13):
        in_comment=comment
        # modify dwt according to the selected mode
        if mode==0: # no modification
            in_comment+='orig'
            dwt_mod=dwt.copy()
        elif mode==1: # restrict dwt to >0
            in_comment+='Gt0'
            dwt_mod=dwt.copy()
            dwt_mod[dwt_mod < 0] = 0
        elif mode==2: # square >0
            in_comment+='Gt0^2'
            dwt_mod=dwt.copy()
            dwt_mod[dwt_mod < 0] = 0
            dwt_mod=dwt_mod**2
        elif mode==3: # sqrt >0
            in_comment+='Gt0_rt'
            dwt_mod=dwt.copy()
            dwt_mod[dwt_mod < 0] = 0
            dwt_mod=dwt_mod**0.5        
        elif mode==4: # take absolute value
            in_comment+='abs'
            dwt_mod=np.abs(dwt)
        elif mode==5: # square >0
            in_comment+='abs^2'
            dwt_mod=np.abs(dwt)
            dwt_mod=dwt_mod**2
        elif mode==6: # sqrt >0
            in_comment+='abs_rt'
            dwt_mod=np.abs(dwt)
            dwt_mod=dwt_mod**0.5  
        elif mode in [7, 8, 9]: # complex 
            if len(phases)!=2:
                print("For complex representation, n_phases must be 2., skipping complex modes.")
                continue
            #Calculating squared amplitude and phase from the two phase DWT  
            dwt_mod, _ =dwt_amp_phase_torch_batched(dwt) #drop phase for now
            # drops last feature dimension: phase
            if mode==7:
                in_comment+='cA'
            elif mode==8:
                in_comment+='cA^2'
                dwt_mod=dwt_mod**2 
            elif mode==9:
                in_comment+='cA_rt'
                dwt_mod=dwt_mod**0.5  
        elif mode==10: # restrict dwt to >0
            in_comment+='Lt0'
            dwt_mod=-dwt
            dwt_mod[dwt_mod < 0] = 0
        elif mode==11: # square >0
            in_comment+='Lt0^2'
            dwt_mod=-dwt
            dwt_mod[dwt_mod < 0] = 0
            dwt_mod=dwt_mod**2
        elif mode==12: # sqrt >0
            in_comment+='Lt0_rt'
            dwt_mod=-dwt
            dwt_mod[dwt_mod < 0] = 0
            dwt_mod=dwt_mod**0.5        
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
        feature_dim_number=len(dwt_mod.shape)-1
        print(f"Using dwt mode: {in_comment}, feature dimension number: {feature_dim_number}")

        #dwt_mod=np.float32(dwt_mod)
        if smooth_fwhm_samples > 0:
            sigma = smooth_fwhm_samples / 2.355
            #dwt_mod = gaussian_filter1d(dwt_mod, sigma=sigma,  axis=0)
            dwt_mod = gaussian_filter1d_torch_axis0_chunked(dwt_mod, sigma)

        ## runs correlation analysis
        rfs = FeatureSearch_correlation_batched(dwt_mod[:train_split_index], mean_spks[:train_split_index])


        # Analyze results
        # find maximum RF value for each neuron (train performance) and compute test correlation
        r_train = np.zeros(n_neurons) # correlation of train data at the best RF prediction
        r_test = np.zeros(n_neurons) # correlation of test data at the best RF prediction
        max_idxs = np.zeros((n_neurons, len(rfs.shape)-1), dtype=int) # store the RF index of the best RF for each neuron

        for i in range(n_neurons):
            myrfs = rfs[i]
            max_idx = np.unravel_index(np.argmax(myrfs), myrfs.shape)
            max_idx = tuple(int(ij) for ij in max_idx)
            max_idxs[i] = max_idx
            r_train[i] = myrfs[max_idx] # this equals to np.corrcoef(mean_spks[:train_split_index, i], dwt_squared[:train_split_index, *max_idx])[0, 1]
            r_test[i] = np.corrcoef(mean_spks[train_split_index:, i], dwt_mod[train_split_index:, *max_idx])[0, 1]

        r_trainMulti.append(r_train)
        r_testMulti.append(r_test)
        commentsMulti.append(in_comment)
    
        del dwt_mod
        del rfs
    
    
        #play: #index of largest 20 respcorr values
        largest_respcorr_indices = np.argsort(respcorr)[-20:]
        print(f"Mean respcorr of largest 20 neurons: {np.mean(respcorr[largest_respcorr_indices]):.3f}")
        print(f"Mean rfs_correlation of largest 20 neurons (train performance): {np.mean(r_train[largest_respcorr_indices]):.3f}")
        print(f"Mean rfs_correlation of largest 20 neurons (test performance): {np.mean(r_test[largest_respcorr_indices]):.3f}")
        stat={}
        stat['N_xs'] = len(xs)
        stat['N_ys'] = len(ys)
        stat['N_angles'] = len(angles)
        stat['N_sizes'] = len(sizes)
        stat['N_freqs'] = len(freqs)
        stat['N_drifts'] = len(drifts)
        stat['N_phases'] = len(phases)
        stat['mode'] = in_comment
        stat['mean_respcorr_top20'] = np.mean(respcorr[largest_respcorr_indices])
        stat['mean_r_train_top20'] = np.mean(r_train[largest_respcorr_indices])
        stat['mean_r_test_top20'] = np.mean(r_test[largest_respcorr_indices])
        

        # %%
        # plot neuron prediction correlation.

        fig, axes = plt.subplots( 1, 2,  figsize=(14, 6),  sharex=True,  sharey=True )

        # ---------- left: r_train ----------
        ax = axes[0]
        ax.scatter(respcorr, r_train, alpha=0.5)

        for i in largest_respcorr_indices:
            ax.annotate(f"{i}", (respcorr[i], r_train[i]))

        mask = np.isfinite(respcorr) & np.isfinite(r_train)
        a_train, b_train = np.polyfit(respcorr[mask], r_train[mask], 1)
        xx = np.linspace(np.nanmin(respcorr), np.nanmax(respcorr), 200)
        ax.plot(xx, a_train * xx + b_train, color='black')

        ax.text( 0.05, 0.95, f"y = {a_train:.3f}x + {b_train:.3f}",  transform=ax.transAxes,  va="top")

        ax.set_title("Train")
        ax.set_xlabel("Response correlation")
        ax.set_ylabel("Prediction correlation")
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        # ---------- right: r_test ----------
        ax = axes[1]
        ax.scatter(respcorr, r_test, alpha=0.5)

        for i in largest_respcorr_indices:
            ax.annotate(f"{i}", (respcorr[i], r_test[i]))

        mask = np.isfinite(respcorr) & np.isfinite(r_test)

        # constrained through origin
        a_test = np.sum(respcorr[mask] * r_test[mask]) / np.sum(respcorr[mask] ** 2)
        ax.plot(xx, a_test * xx, color='black')

        ax.text(0.05, 0.95, f"y = {a_test:.3f}x", transform=ax.transAxes,  va="top")

        ax.set_title("Test")
        ax.set_xlabel("Response correlation")
        ax.set_aspect('equal', adjustable='box')

        plt.suptitle(in_comment)
        plt.tight_layout()
        plt.savefig(output_folder / 'respcorr_vs_prediction_correlation_5.png', dpi=300)
        plt.show()

        print(f"train fit: y = {a_train:.6f} x + {b_train:.6f}")
        print(f"test fit : y = {a_test:.6f} x")

        # %%
        stat['train_fit_a'] = a_train
        stat['train_fit_b'] = b_train
        stat['test_fit_a'] = a_test

    print(">>>>>>>>>>>Finished all modes. ")
    # %%
    #save results into cell database
    input_pickle_path= working_dir / "cells_caiman.cellDB_pickle"
    if input_pickle_path.exists():
        df_cells=pd.read_pickle(open(input_pickle_path,"rb"))
    else:
        df_cells = pd.DataFrame()
        for _idx in range(n_neurons): #handling only good components
                record={}
                record['cell_id'] = _idx
                record['SeriesID'] = 'unknown'
                df_cells = pd.concat([df_cells, pd.DataFrame([record])], ignore_index=True)

    df_cells = df_cells.set_index("cell_id", drop=False)

    for _idx in tqdm(range(n_neurons)):
        df_cells.loc[_idx,'Repeatability'] = respcorr[_idx]
    
    for i in tqdm(range(len(r_trainMulti))):
        for _idx in range(n_neurons):
            
            df_cells.loc[_idx, 'r_train'+commentsMulti[i]] = r_trainMulti[i][_idx]
            df_cells.loc[_idx, 'r_test'+commentsMulti[i]] = r_testMulti[i][_idx]
            

    # Saving cell database to xls file. Omitting complicated data
    df_cells.drop(columns=['WL_transient_mod', 'WL_transient_phase', 'Cell_activity', 'contour', 'tun_xs', 'tun_ys', 'tun_angles', 'tun_sizes', 'tun_freqs', 'tun_drifts', 'tun_phases'], inplace=True, errors='ignore')
    df_cells.to_excel(output_folder / "cells_multi.xlsx", index=False)



    # %%
    #save stat to excel file
    stat_path = output_folder / "stat.xlsx"
    stat['path'] = str(stat_path)


    print(output_folder)
    return output_folder, stat

    