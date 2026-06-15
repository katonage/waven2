# Analysis test - modified correlation with the library
# Calculate best correlation match with (arithmetic modification) of Gabor filter library decomposed visual stimulation. 
# 
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



def full_analysis_vSpeed2(spks_path, downsampled_video_path,
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
                          mode=2, 
                          smooth_fwhm_samples=3, # 0.1 sec at 30 fps
                          shift_samples=1, # +1 moves spks forward
                          comment=""):
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
    # mode : int, optional
    #     mode of analysis 
    #               0: no modification, 
    #               1: restrict to >0, 
    #               2: absolute value, 
    #               3: square >0, 
    #               4: complex representation  (squared ampl)
    #               5: complex representation 
    #               6: absolute value squared 
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
    spks=np.load(spks_path)
    if smooth_fwhm_samples > 0:
        spks = gaussian_filter1d(spks, sigma=smooth_fwhm_samples / 2.355,  axis=1)
    if shift_samples != 0:
        spks = np.roll(spks, -shift_samples, axis=1)
    print(f"spks shape: {spks.shape} (n_trials, n_timepoints, n_neurons), smoothed by {smooth_fwhm_samples} samples, shifted forward by {shift_samples} samples")
    n_trials=spks.shape[0]
    n_timepoints=spks.shape[1]
    n_neurons=spks.shape[2]

    mean_spks = np.mean(spks[:, :, :], axis=0)

    working_dir = Path(spks_path).parent
    print(f"Working directory: {working_dir}")

    ## converts neuron position in microns
    #neuron_pos=np.load(working_dir / 'component_centers.npy')
    #neuron_pos=correctNeuronPos(neuron_pos, resolution)
    #print(f"neuron_pos shape: {neuron_pos.shape}")

    # %% [markdown]
    # ## Correlate neural data with decomposed stimulus

    # %%
    # load dwt
    dwt = np.load(dwt_path)
    print(f"dwt shape: {dwt.shape}")

    # %%
    # modify dwt according to the selected mode
    if mode==0: # no modification
        comment+='orig'
    elif mode==1: # restrict dwt to >0
        comment+='Gt0'
        dwt[dwt < 0] = 0
    elif mode==2: # square >0
        comment+='Gt0^2'
        dwt[dwt < 0] = 0
        dwt=dwt**2
    elif mode==3: # sqrt >0
        comment+='Gt0_rt'
        dwt[dwt < 0] = 0
        dwt=dwt**0.5        
    elif mode==4: # take absolute value
        comment+='abs'
        dwt=np.abs(dwt)
    elif mode==5: # square >0
        comment+='abs^2'
        dwt=np.abs(dwt)
        dwt=dwt**2
    elif mode==6: # sqrt >0
        comment+='abs_rt'
        dwt=np.abs(dwt)
        dwt=dwt**0.5  
    elif mode in [7, 8, 9]: # complex 
        if len(phases)!=2:
            raise ValueError("For complex representation, n_phases must be 2.")
        #Calculating squared amplitude and phase from the two phase DWT  
        dwt, _ =dwt_amp_phase_torch_batched(dwt) #drop phase for now
        # drops last feature dimension: phase
        if mode==7:
            comment+='cA'
        elif mode==8:
            comment+='cA^2'
            dwt=dwt**2 
        elif mode==9:
            comment+='cA_rt'
            dwt=dwt**0.5  
    elif mode==10: # restrict dwt to >0
        comment+='Lt0'
        dwt[dwt < 0] = 0
    elif mode==11: # square >0
        comment+='Lt0^2'
        dwt[dwt < 0] = 0
        dwt=dwt**2
    elif mode==12: # sqrt >0
        comment+='Gt0_rt'
        dwt[dwt < 0] = 0
        dwt=dwt**0.5   
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    feature_dim_number=len(dwt.shape)-1
    print(f"Using dwt mode: {comment}, feature dimension number: {feature_dim_number}")

    # %% [markdown]
    # ## Splitting data

    # %%
    #defining train and test data parts
    train_test=True #whether to split data in train and test set, if False use all data for training and no test evaluation
    train_split=0.85 #plit ratio in time if train_test

    if train_test:
        train_split_index=int(n_timepoints*train_split)
    else:
        train_split_index=n_timepoints

    print(f"Train split at {train_split_index/n_timepoints*100:.1f}% of timepoints, test split at {((1-train_split_index/n_timepoints)*100):.1f}% of timepoints")

    # %%
    ## runs correlation analysis

    rfs = FeatureSearch_correlation_batched(dwt[:train_split_index], mean_spks[:train_split_index])


    # Analyze results

    # %%
    output_folder = working_dir / (paramname.replace('.json', '')+f"_{comment}")
    output_folder.mkdir(exist_ok=True)
    saveFilterParamDict_vS(wavelet_params, output_folder / paramname)  # save a copy in the output folder 
    print(f"Saved Gabor filter parameters to {output_folder}")

    # %%
    respcorr = compute_respcorr_split_half(spks)

    # %%
    # fit angle tuning curve: sine
    tuning_angles_fit=[]
    for _idx in range(n_neurons):
        myrfs = rfs[_idx]
        max_idx = np.unravel_index(np.argmax(myrfs), myrfs.shape)
        max_idx = tuple(int(i) for i in max_idx)
        tuning_curve_raw = myrfs[max_idx[0], max_idx[1], :, *max_idx[3:feature_dim_number]]
        
        params = fit_sine1x(angles, tuning_curve_raw)

        # interpolated fit
        angles_interp = np.linspace(0, 1*np.pi, 100)
        tuning_curve_fit_interp = sine1x(angles_interp, **params)
        
        tuning_angles_fit.append((params, angles_interp, tuning_curve_fit_interp))
        

    # %%
    #play: # find absolut maximum RF value across all neurons and features
    max_idx = np.unravel_index(np.argmax(rfs), rfs.shape)
    max_idx = tuple(int(i) for i in max_idx)
    max_value = rfs[max_idx]
    print(f"max RF value: {max_value} at index {max_idx}")

    # %%
    #play: # find maximum RF value for a specific neuron
    my_neuron = 456
    # my_neuron = 1813 

    myrfs = rfs[my_neuron]
    max_idx = np.unravel_index(np.argmax(myrfs), myrfs.shape)
    max_idx = tuple(int(i) for i in max_idx)
    max_value = myrfs[max_idx]
    print(f"Neuron {my_neuron} max RF value: {max_value:.3f} at index {max_idx}, respcorr: {respcorr[my_neuron]:.3f}")

    # %%
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
        r_test[i] = np.corrcoef(mean_spks[train_split_index:, i], dwt[train_split_index:, *max_idx])[0, 1]


    # %%
    # save respcorr and max_values for each neuron to a csv file

    df = pd.DataFrame({'respcorr': respcorr, 'r_train': r_train, 'r_test': r_test})
    df.to_csv(output_folder / 'respcorr_max_values_vs4.csv', index=False)

    # %%
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
    stat['mode'] = comment
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


    plt.tight_layout()
    plt.savefig(output_folder / 'respcorr_vs_prediction_correlation_5.png', dpi=300)
    plt.show()

    print(f"train fit: y = {a_train:.6f} x + {b_train:.6f}")
    print(f"test fit : y = {a_test:.6f} x")

    # %%
    stat['train_fit_a'] = a_train
    stat['train_fit_b'] = b_train
    stat['test_fit_a'] = a_test

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

    df_cells["RF_indexes"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["WL_transient_mod"] = pd.Series([None] * len(df_cells), dtype="object")
    #df_cells["WL_transient_phase"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["Cell_activity"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["tun_xs"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["tun_ys"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["tun_angles"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["tun_sizes"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["tun_freqs"] = pd.Series([None] * len(df_cells), dtype="object")
    df_cells["tun_drifts"] = pd.Series([None] * len(df_cells), dtype="object")
    if feature_dim_number>6:
        df_cells["tun_phases"] = pd.Series([None] * len(df_cells), dtype="object")


    for _idx in tqdm(range(n_neurons)):
        df_cells.loc[_idx,'Repeatability'] = respcorr[_idx]
        
        max_idx = max_idxs[_idx] # use pre-stored max idx for consistency with interactive view
        
        df_cells.at[_idx, 'RF_indexes'] = [int(ij) for ij in max_idx] # store the RF index of the best RF for each neuron as a list
        df_cells.loc[_idx, 'r_train'] = r_train[_idx]
        df_cells.loc[_idx, 'r_test'] = r_test[_idx]
        
        df_cells.loc[_idx, 'Azimuth'] = xs[max_idx[0]]
        df_cells.loc[_idx, 'Elevation'] = ys[max_idx[1]]
        df_cells.loc[_idx, 'Angle'] = angles[max_idx[2]]
        df_cells.loc[_idx, 'Size'] = sizes[max_idx[3]]
        df_cells.loc[_idx, 'Frequency'] = freqs[max_idx[4]]
        df_cells.loc[_idx, 'Drift'] = drifts[max_idx[5]]
        if feature_dim_number>6:
            df_cells.loc[_idx, 'Phase'] = phases[max_idx[6]]


        transient_mod = dwt[:, *max_idx]
        #transient_phase = dwt_phase[:, *max_idx] 
        
        df_cells.at[_idx, 'WL_transient_mod'] = transient_mod
        #df_cells.at[_idx, 'WL_transient_phase'] = transient_phase
        df_cells.at[_idx, 'Cell_activity'] = spks[:, :, _idx]
        
        #store tuning curves
        myrfs = rfs[_idx]
        df_cells.at[_idx, 'tun_xs'] = myrfs[:, max_idx[1], max_idx[2], max_idx[3], *max_idx[4:feature_dim_number]]
        df_cells.at[_idx, 'tun_ys'] = myrfs[max_idx[0], :, max_idx[2], max_idx[3], *max_idx[4:feature_dim_number]]
        df_cells.at[_idx, 'tun_angles'] = myrfs[max_idx[0], max_idx[1], :, max_idx[3], *max_idx[4:feature_dim_number]]
        df_cells.at[_idx, 'tun_sizes'] = myrfs[max_idx[0], max_idx[1], max_idx[2], :, *max_idx[4:feature_dim_number]]
        df_cells.at[_idx, 'tun_freqs'] = myrfs[max_idx[0], max_idx[1], max_idx[2], max_idx[3], :, *max_idx[5:feature_dim_number]]
        if feature_dim_number>6:
            df_cells.at[_idx, 'tun_drifts'] = myrfs[max_idx[0], max_idx[1], max_idx[2], max_idx[3], max_idx[4], :, *max_idx[6:feature_dim_number]]
            df_cells.at[_idx, 'tun_phases'] = myrfs[max_idx[0], max_idx[1], max_idx[2], max_idx[3], max_idx[4], max_idx[5], :]
        else:
            df_cells.at[_idx, 'tun_drifts'] = myrfs[max_idx[0], max_idx[1], max_idx[2], max_idx[3], max_idx[4], :]
            
        #store angle fit
        params, x_fit, y_fit = tuning_angles_fit[_idx]
        df_cells.loc[_idx, 'Angle_fit_ori'] = params['orientation']
        df_cells.loc[_idx, 'Angle_fit_amplitude'] = params['amplitude']
        df_cells.loc[_idx, 'Angle_fit_constant'] = params['constant']
        df_cells.loc[_idx, 'Angle_fit_OSI'] = params['amplitude'] / (params['amplitude'] + params['constant'])

    del dwt # free memory
    picles_path=output_folder / "cells_waven1vs.cellDB_pickle"
    df_cells.to_pickle(picles_path)

    # Saving cell database to xls file. Omitting complicated data
    df_cells.drop(columns=['WL_transient_mod', 'WL_transient_phase', 'Cell_activity', 'contour', 'tun_xs', 'tun_ys', 'tun_angles', 'tun_sizes', 'tun_freqs', 'tun_drifts', 'tun_phases'], inplace=True, errors='ignore')
    df_cells.to_excel(output_folder / "cells_waven1.xlsx", index=False)

    print(f"Saved cell database to {picles_path}")


    # %%
    print(max_idx)
    print(rfs.shape)

    # %%
    #save stat to excel file
    stat_path = output_folder / "stat.xlsx"
    stat['path'] = str(stat_path)
    with pd.ExcelWriter(stat_path, engine='openpyxl') as writer:
        pd.DataFrame([stat]).to_excel(writer, index=False, sheet_name='stat')

    # %%

    print(output_folder)
    return output_folder, stat

    