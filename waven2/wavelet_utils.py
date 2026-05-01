
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import cv2
from skimage.filters import gabor_kernel
import gc
import torch


import os
import skimage
from scipy import ndimage
from skimage.measure import block_reduce
from skimage import transform

def makeGaborFilter2(i, j, angle, size, frequency, phase, screen_x=100, screen_y=75, plot=False):
    """
    Generate a localized 2D Gabor filter patch embedded in a zero-valued image.

    The function constructs a Gabor kernel using skimage, then places it at a specified
    spatial location (i, j) within a larger image (screen).

    Parameters
    ----------
    i, j : int
        Center coordinates (pixel indices) of the Gabor patch within the output image.
    angle : float
        Orientation of the Gabor filter in radians.
    size : float
        Gaussian envelope radius at half maximum.
    frequency : float
        Spatial frequency of the Gabor filter (cycles per pixel).
    phase : float
        Phase offset of the sinusoidal carrier (radians).
    screen_x, screen_y : int, optional
        Dimensions of the output image.
    plot : bool, optional
        If True, displays the generated filter using matplotlib.

    Returns
    -------
    np.ndarray
        2D array of shape (screen_y, screen_x), dtype float16.
        Contains the Gabor patch centered at (i, j), zero elsewhere.

    Notes
    -----
    - The function uses only the real part of the complex Gabor kernel.
    - The output is transposed before returning to match (x, y) vs (row, column) conventions..
    """
    sigma = size / (2 * np.sqrt(2 * np.log(2)))  

    gk = gabor_kernel(frequency=frequency, theta=angle, sigma_x=sigma, sigma_y=sigma, offset=phase, n_stds=4)

    backgrd = np.zeros((screen_x, screen_y))

    k = gk.shape[0]
    dp = k // 2

    x0 = max(0, i - dp)
    x1 = min(screen_x, i + dp + 1)
    y0 = max(0, j - dp)
    y1 = min(screen_y, j + dp + 1)

    kx0 = dp - (i - x0)
    kx1 = dp + (x1 - i)
    ky0 = dp - (j - y0)
    ky1 = dp + (y1 - j)

    backgrd[x0:x1, y0:y1] = gk.real[kx0:kx1, ky0:ky1]
    backgrd = backgrd.astype('float16')  # transpose 
        
    if plot:
        v = np.max(np.abs(backgrd))
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.imshow(backgrd.T, cmap='Greys', vmin=-v, vmax=v)
        plt.title(f'angle={angle:.2f}, size={size:.2f}, frequency={frequency:.2f}, size={size:.2f}, phase={phase:.2f}\n max={np.max(backgrd)}, min={np.min(backgrd)}')
        
    return backgrd

def makeGaborFilter_visual(i_deg, j_deg, angle, size_deg,  freq_deg, phase,  visual_coverage, screen_x=100, screen_y=None,  plot=False ):
    """
    Wrapper for makeGaborFilter2 using visual degrees instead of pixels.

    Parameters
    ----------
    i_deg : float
        Azimuth position (horizontal) in degrees
    j_deg : float
        Elevation position (vertical) in degrees
    size_deg : float
        Diameter of the Gabor patch in degrees
    freq_deg : float
        Spatial frequency in cycles per visual degree
    screen_x : int
        Width of the output image in pixels
    screen_y : int, optional
        Height of the output image in pixels. If None, it is set according to screen_x and the aspect ratio defined by visual_coverage.
    angle, phase : same as before
    visual_coverage : list
        [az_left, az_right, el_bottom, el_top] in degrees
    """

    az_left, az_right, el_bottom, el_top = visual_coverage
    
    if screen_y is None:
        screen_y = int(screen_x * (el_top - el_bottom) / (az_right - az_left))  # maintain aspect ratio

    # --- pixels per degree ---
    px_per_deg_x = screen_x / (az_right - az_left)
    px_per_deg_y = screen_y / (el_top - el_bottom)

    # --- convert position ---
    i_px = (i_deg - az_left) * px_per_deg_x
    j_px = (j_deg - el_bottom) * px_per_deg_y   

    # --- convert size ---
    size_px = size_deg * (px_per_deg_x + px_per_deg_y) / 2  # isotropic approx

    # --- convert frequency ---
    frequency = freq_deg * (px_per_deg_x + px_per_deg_y) / 2

    filt= makeGaborFilter2(
        int(round(i_px)),
        int(round(j_px)),
        angle=angle,
        size=size_px,
        frequency=frequency,
        phase=phase,
        screen_x=screen_x,
        screen_y=screen_y
    )
    
    if plot:
        v = np.max(np.abs(filt))
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.imshow(filt.T, cmap='Greys', vmin=-v, vmax=v, extent=visual_coverage,
            origin='lower',   # important for correct orientation
            aspect='auto')
        
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')
        plt.title(f'angle={angle:.2f}, size={size_px:.2f}, frequency={frequency:.2f}, phase={phase:.2f}\n max={np.max(filt)}, min={np.min(filt)}')
        plt.scatter(i_deg, j_deg, color='red', s=20)
        
    return filt


def makeFilterParamDict(screen_x, screen_y, visual_coverage, full_screen_coverage, xs, ys, angles, sigmas, frequencies, offsets):
    """
    Builds a dictionary containing the parameters used for Gabor filter generation.

    Parameters:
        see makeFilterLibrary() for details on each parameter.
    Returns:
        dict: A dictionary containing the parameters for Gabor filter generation.
    """
    paramsdict = {
        'xs': xs,
        'ys': ys,
        'angles': angles,
        'sigmas': sigmas,
        'frequencies': frequencies,
        'offsets': offsets,
        'screen_x': screen_x,
        'screen_y': screen_y,
        'visual_coverage': visual_coverage, 
        'full_screen_coverage': full_screen_coverage
    }
    return paramsdict

def loadFilterParamDict(json_path):
    """
    Loads the Gabor filter generation parameters from a JSON file.

    Parameters:
        json_path (str or Path): Path to the JSON file containing the parameters.
    Returns:
        tuple: A tuple containing the loaded parameters.
    """
    with open( json_path, 'r') as f:
        params = json.load(f)
    for k, v in params.items():
        if isinstance(v, list): # convert lists back to numpy arrays (only if they were arrays)
            params[k] = np.array(v)
            
    xs = params['xs']
    ys = params['ys']
    angles = params['angles']
    sizes = params['sigmas']
    freqs = params['frequencies']
    phases = params['offsets']
    visual_coverage = params['visual_coverage']
    full_screen_coverage = params['full_screen_coverage']
    screen_x = params['screen_x']
    screen_y = params['screen_y']
    return xs, ys, angles, sizes, freqs, phases, visual_coverage, full_screen_coverage, screen_x, screen_y

def makeFilterLibrary(paramsdict):
    """
    Builds a Gabor filter library.

    Parameter: paramsdict (dict): A dictionary containing the parameters for Gabor filter generation:
        screen_x, screen_y (int): Width and height of the screen in pixels.
        visual_coverage (float): Coverage of the visual field.
        full_screen_coverage (float): Full screen coverage in visual degrees.
        xs (array-like): Array of x positions (azimuth) in visual degrees.
        ys (array-like): Array of y positions (elevation) in visual degrees.
        angles (array-like): Orientations in radians (typically spanning 0 to π).
        sigmas (array-like): FWHM of the Gaussian envelope (in visual degrees).
        frequencies (array-like): Spatial frequencies (cycles per visual degree).
        offsets (array-like): Phase offsets (e.g., 0 and π/2).

    Returns:
        numpy.ndarray: Gabor filter library of shape
            (nx, ny, n_orientation, n_sigma, n_frequency, n_phase, nx * ny)
    """
    
    xs = paramsdict['xs']
    ys = paramsdict['ys']
    angles = paramsdict['angles']
    sigmas = paramsdict['sigmas']
    frequencies = paramsdict['frequencies']
    offsets = paramsdict['offsets']
    screen_x = paramsdict['screen_x']
    screen_y = paramsdict['screen_y']
    visual_coverage = paramsdict['visual_coverage']
    
    library=[]
    for x in tqdm(xs):
        for y in ys:
            for t in angles:
                for s in sigmas:
                    for f in frequencies:
                        for o in offsets:
                            library.append( makeGaborFilter_visual(            
                                                                    i_deg=x,
                                                                    j_deg=y,
                                                                    size_deg=s,
                                                                    angle=t,
                                                                    freq_deg=f,
                                                                    phase=o,
                                                                    visual_coverage=visual_coverage,
                                                                    screen_x=screen_x,
                                                                    screen_y=screen_y))

    library=np.array(library)
    library=library.reshape((len(xs), len(ys), len(angles), len(sigmas), len(frequencies), len(offsets), screen_x, screen_y))
    
    return library, paramsdict
    

def make_and_save_FilterLibrary(path, paramsdict, force=False):
    """Generates a Gabor filter library and saves it to disk.

    Parameters:
        path (str or Path): Directory where the library will be saved.
        paramsdict (dict): Dictionary containing the parameters for Gabor filter generation.
        force (bool): If True, the library will be generated even if it already exists.
    """
    
    def filename_from_params(dict):
        x = dict['xs']
        y = dict['ys']
        t = dict['angles']
        s = dict['sigmas']
        f = dict['frequencies']
        o = dict['offsets']
        name= f"gaborLibrary_{len(x)}_{len(y)}_{len(t)}_{len(s)}_{len(f)}_{len(o)}"
        return name + ".npy", name + ".json"
    
    Path(path).mkdir(parents=True, exist_ok=True)
    npy_filename, json_filename = filename_from_params(paramsdict)
    
    if not force and (Path(path) / npy_filename).exists():
        print("Gabor filter library file already exists. Skipping generation.")
        return (Path(path) / npy_filename, Path(path) / json_filename)  
    
    print ("Generating Gabor filter library...")
    library, paramsdict = makeFilterLibrary(paramsdict)
    print (f"Done. Library shape: {library.shape}")
    np.save(Path(path) / npy_filename, library)
    
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    paramsdict_str = {k: convert(v) for k, v in paramsdict.items()}
    with open(Path(path) / json_filename, 'w') as f:
        json.dump(paramsdict_str, f, indent=4)
        
    print(f"Library saved to {Path(path) / npy_filename} and {Path(path) / json_filename}")
    return (Path(path) / npy_filename, Path(path) / json_filename)    



def downscale_binary_video(path, full_screen_coverage, visual_coverage, screen_x, screen_y, output_path=None, force=False):
    """
    Crop and downscale a binary visual stimulus video.

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
    
    path = Path(path)

    if output_path is None:
        output_path = path.with_name(path.stem + "_downscaled.npy")
    else:
        output_path = Path(output_path)
    
    print("Generating cropped and downsampled binary video...")
    if output_path.exists() and not force:
        print(f"Output file {output_path} already exists. Skipping generation.")
        return output_path

    full = np.asarray(full_screen_coverage, dtype=float)
    vis = np.asarray(visual_coverage, dtype=float)

    az_left, az_right, el_bottom, el_top = full
    v_az_left, v_az_right, v_el_bottom, v_el_top = vis


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


def getWTfromVideo_batched(videodata, waveletLibrary,  device='cuda', batch_size=32):
    """
    Optimized wavelet transform using batch processing on GPU.
    improved from: LeonKremers/Waven_working getWTfromNPY_batched function
    
    Processes multiple frames at once instead of frame-by-frame,
    keeping the wavelet library in GPU memory throughout.
    
    Parameters:
        videodata: numpy array of shape (n_frames, height, width)
        waveletLibrary: numpy array of wavelet library (reshaped internally)
        device: CUDA device
        batch_size: number of frames to process at once (default 32)
    
    Returns:
        WT: numpy array of shape (n_frames, d1...d6) where d1...d6 are the dimensions of the wavelet library excluding the last two (height, width)
    """
        
    n_frames = videodata.shape[0]
    n_wavelets = int(np.prod(waveletLibrary.shape[:-2]))

    # Check that library and video have the same frame size
    Gabor_frame_size_library = waveletLibrary.shape[-1] * waveletLibrary.shape[-2]
    Gabor_frame_size_video = videodata.shape[-1] * videodata.shape[-2]
    if Gabor_frame_size_library != Gabor_frame_size_video or waveletLibrary.shape[-1] != videodata.shape[-1]:
        raise ValueError(f"Wavelet library frame size ({Gabor_frame_size_library}) does not match video frame size ({Gabor_frame_size_video}).")
    
    device = torch.device(device)
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"    Torch using: {device}, GPU name: {torch.cuda.get_device_name(idx)}, GPU index: {idx}")
    else:
        print(f"    Torch using: {device}")
    
    # Flatten and transfer library to GPU once and keep it there
    l_torch_flat = torch.Tensor(waveletLibrary.reshape(-1, Gabor_frame_size_library).T).to(device)
    print(f"    Frame batch shaped to [{batch_size}, {Gabor_frame_size_video}] to multiply by wavelet library reshaped to {l_torch_flat.shape} on device {device}")
    
    WT = np.empty((n_frames, n_wavelets), dtype=waveletLibrary.dtype)

    # Process frames in batches
    for batch_start in tqdm(range(0, n_frames, batch_size), desc=f"Wavelet transform batched"):
        batch_end = min(batch_start + batch_size, n_frames)
        batch_frames = videodata[batch_start:batch_end]
        # Flatten frames to shape (batch_size, H*W)
        batch_frames = batch_frames.reshape(batch_frames.shape[0], -1)
        #send batch to GPU
        frames_tensor = torch.as_tensor(batch_frames, dtype=l_torch_flat.dtype, device=l_torch_flat.device)
        
        # Vectorized matrix multiplication: (batch_size, H*W) @ (H*W, n_wavelets) -> (batch_size, n_wavelets)
        output = frames_tensor @ l_torch_flat
        
        # Transfer results back to CPU and store in WT
        batch_wt = output.cpu().numpy()
        WT[batch_start:batch_end] = batch_wt
    
    # Clean up GPU memory only after all batches are done
    del l_torch_flat
    del frames_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Reshape WT to match the wavelet library dimensions
    WT = WT.reshape((n_frames,) + waveletLibrary.shape[:-2])
    return WT

def compute_and_save_dwt(downsampled_video_path, libpath,  device='cuda', force=False):
    """
    Wrapper function to compute and save wavelet transform.
    
    Parameters:
        downsampled_video_path: Path to the downsampled video .npy file
        libpath: Path to the Gabor filter library .npy file
        device: CUDA device
        force: If True, overwrite existing DWT file
    Returns:
        Path to the saved wavelet transform .npy file
    """
    
    videodata=np.load(downsampled_video_path)
    print(f"Loaded downsampled video data from {downsampled_video_path} with shape {videodata.shape} and dtype {videodata.dtype}")
    
    library = np.load(libpath)
    print(f"Loaded Gabor filter library from {libpath} with shape {library.shape}")

    dwt_name= f"{downsampled_video_path.stem}_lib{"_".join([str(x) for x in library.shape])}dwt.npy"
    dwt_path= libpath.parent / dwt_name
    if dwt_path.exists() and not force:
        print(f"Wavelet transform file {dwt_path} already exists. Skipping computation.")
        return dwt_path

    WT=getWTfromVideo_batched(videodata, library, device=device)
    print(f"Computed wavelet transform with shape {WT.shape} ")


    np.save(dwt_path, WT)
    print(f"Saved wavelet transform to {dwt_path} with shape {WT.shape} and dtype {WT.dtype}")
    return dwt_path