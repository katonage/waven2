
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from skimage.filters import gabor_kernel
import hashlib
import gc
import torch
try:
    from .torch_utils import handle_torch_device, print_cuda_tensors_mem
except ImportError:  # Support direct execution from the module directory.
    from torch_utils import handle_torch_device, print_cuda_tensors_mem

def makeGaborFilter_vS(i, j, angle, size, frequency,  phase, screen_x=100, screen_y=75):
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

    Returns
    -------
    np.ndarray
        3D array of shape (screen_x, screen_y), dtype float16.
        Contains the Gabor patch centered at (i, j), zero elsewhere; 

    Notes
    -----
    - The function uses only the real part of the complex Gabor kernel.
    - The output is transposed before returning to match (x, y) vs (row, column) conventions..
    """
    sigma = size / (2 * np.sqrt(2 * np.log(2)))  

    gk = gabor_kernel(frequency=frequency, theta=-angle+np.pi/2, sigma_x=sigma, sigma_y=sigma, offset=phase, n_stds=4)
    gk=gk.real.astype('float16')  # keep only real part and convert to float16 for memory efficiency

    backgrd = np.zeros((screen_x, screen_y)).astype('float16')

    k = gk.shape[0]
    dp = k // 2

    # calculate the center position for this frame, applying drift perpendicular to angle
    center_x = int(np.round(i))
    center_y = int(np.round(j))

    x0 = min(screen_x, max(0, center_x - dp))
    x1 = min(screen_x, max(0, center_x + dp + 1))
    y0 = min(screen_y, max(0, center_y - dp))
    y1 = min(screen_y, max(0, center_y + dp + 1))

    kx0 = dp - (center_x - x0)
    kx1 = dp + (x1 - center_x)
    ky0 = dp - (center_y - y0)
    ky1 = dp + (y1 - center_y)

    backgrd[ x0:x1, y0:y1] = gk[kx0:kx1, ky0:ky1] # injecting Gabor patch into the frame
    
    backgrd = backgrd-np.mean(backgrd) # zero mean ! to make sense cutting response abovezero 
    
    #backgrd = backgrd  # no transpose, keep it as (x, y)
           
    return backgrd

def makeGaborFilter_visual_vS(i_deg, j_deg, angle, size_deg,  freq_deg, phase, visual_coverage, screen_x=100, screen_y=None ):
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

    filt= makeGaborFilter_vS(
        int(round(i_px)),
        int(round(j_px)),
        angle=angle,
        size=size_px,
        frequency=frequency,
        phase=phase,
        screen_x=screen_x,
        screen_y=screen_y
    )
        
    return filt


def makeFilterParamDict_vS(screen_x, screen_y,  visual_coverage, full_screen_coverage, xs, ys, angles, sigmas, frequencies,  offsets):
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
        'sizes': sigmas,
        'freqs': frequencies,
        'phases': offsets,
        'screen_x': screen_x,
        'screen_y': screen_y,
        'visual_coverage': visual_coverage, 
        'full_screen_coverage': full_screen_coverage
    }
    return paramsdict
    
def saveFilterParamDict_vS(paramsdict, pathstr):
    """
    Saves the parameters dictionary to a JSON file.
    """
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    paramsdict_str = {k: convert(v) for k, v in paramsdict.items()}
    with open(pathstr, 'w') as f:
        json.dump(paramsdict_str, f, indent=4)  
        
def loadFilterParamDict_vS(json_path):
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
    sizes = params['sizes']
    freqs = params['freqs']
    phases = params['phases']
    visual_coverage = params['visual_coverage']
    full_screen_coverage = params['full_screen_coverage']
    screen_x = params['screen_x']
    screen_y = params['screen_y']
    return xs, ys, angles, sizes, freqs,  phases, visual_coverage, full_screen_coverage, screen_x, screen_y

def makeFilterLibrary_vS(paramsdict):
    """
    Builds a Gabor filter library. We dont't use this as the library is too large to fit in memory. Generate filters on the fly. 

    Parameter: paramsdict (dict): A dictionary containing the parameters for Gabor filter generation:
        screen_x, screen_y (int): Width, height, and time dimension of the screen in pixels.
        visual_coverage (float): Coverage of the visual field.
        full_screen_coverage (float): Full screen coverage in visual degrees.
        xs (array-like): Array of x positions (azimuth) in visual degrees.
        ys (array-like): Array of y positions (elevation) in visual degrees.
        angles (array-like): Orientations in radians (typically spanning 0 to π).
        sizes (array-like): FWHM of the Gaussian envelope (in visual degrees).
        freqs (array-like): Spatial frequencies (cycles per visual degree).
        phases (array-like): Phase offsets (e.g., 0 and π/2).

    Returns:
        numpy.ndarray: Gabor filter library of shape
            (nx, ny, n_orientation, n_sigma, n_frequency,  n_phase, nx * ny)
    """
    
    xs = paramsdict['xs']
    ys = paramsdict['ys']
    angles = paramsdict['angles']
    sigmas = paramsdict['sizes']
    frequencies = paramsdict['freqs']
    offsets = paramsdict['phases']
    screen_x = paramsdict['screen_x']
    screen_y = paramsdict['screen_y']
    visual_coverage = paramsdict['visual_coverage']
    
    library = np.empty( (len(xs), len(ys), len(angles), len(sigmas), len(frequencies), len(drifts), len(offsets), screen_t, screen_x, screen_y), dtype=np.float16 )
    for xi, x in tqdm(enumerate(xs), total=len(xs)):
        for yi, y in enumerate(ys):
            for ti, t in enumerate(angles):
                for si, s in enumerate(sigmas):
                    for fi, f in enumerate(frequencies):
                            for oi, o in enumerate(offsets):
                                library[xi, yi, ti, si, fi, di, oi] = makeGaborFilter_visual_vS(            
                                                                        i_deg=x,
                                                                        j_deg=y,
                                                                        size_deg=s,
                                                                        angle=t,
                                                                        freq_deg=f,
                                                                        phase=o,
                                                                        visual_coverage=visual_coverage,
                                                                        screen_x=screen_x,
                                                                        screen_y=screen_y,
                                                                        )

    library=np.array(library)
    library=library.reshape((len(xs), len(ys), len(angles), len(sigmas), len(frequencies),  len(offsets),  screen_x, screen_y))
    
    return library, paramsdict
    

def get_filter_from_params(xi, yi, ai, si, fi,  oi, params):
    # retrieves a single Gabor filter based on the provided indices and parameters dictionary
    # xi, yi, ai, si, fi,  oi are indices corresponding to the parameter arrays in params
    # params is a dictionary containing the parameters for Gabor filter generation
    # returns the Gabor filter

    xs = params['xs']
    ys = params['ys']
    angles = params['angles']
    sizes = params['sizes']
    freqs = params['freqs']
    phases = params['phases']
    visual_coverage = params['visual_coverage']
    screen_x = params['screen_x']
    screen_y = params['screen_y']

    filt = makeGaborFilter_visual_vS(
        i_deg=xs[xi],
        j_deg=ys[yi],
        size_deg=sizes[si],
        angle=angles[ai],
        freq_deg=freqs[fi],
        phase=phases[oi],
        visual_coverage=visual_coverage,
        screen_x=screen_x,
        screen_y=screen_y,
    )
    return filt


def filename_fromFilterParam(indict):
        x = indict['xs']
        y = indict['ys']
        t = indict['angles']
        s = indict['sizes']
        f = indict['freqs']
        o = indict['phases']
        sx = indict['screen_x']
        sy = indict['screen_y']
        vc=indict['visual_coverage']
        fsc=indict['full_screen_coverage']
        
        
        # deterministic compact representation
        payload = np.concatenate([
            np.ravel(x),
            np.ravel(y),
            np.ravel(t),
            np.ravel(s),
            np.ravel(f),
            np.ravel(o),
            np.array([ sx, sy]),
            np.ravel(vc),
            np.ravel(fsc)
        ]).astype(np.float64)

        # short stable hash
        h = hashlib.sha1(payload.tobytes()).hexdigest()[:8]
        
        name= f"gaborLibrary_{len(x)}_{len(y)}_{len(t)}_{len(s)}_{len(f)}_{len(o)}.{h}"
        return name + ".npy", name + ".json"
  

def make_and_save_FilterLibrary_vS(path, paramsdict, force=False):
    """Generates a Gabor filter library and saves it to disk. We dont't use this as the library is too large to fit in memory. Generate filters on the fly.

    Parameters:
        path (str or Path): Directory where the library will be saved.
        paramsdict (dict): Dictionary containing the parameters for Gabor filter generation.
        force (bool): If True, the library will be generated even if it already exists.
    """
    
    Path(path).mkdir(parents=True, exist_ok=True)
    npy_filename, json_filename = filename_fromFilterParam(paramsdict)
    
    if not force and (Path(path) / npy_filename).exists():
        print("Gabor filter library file already exists. Skipping generation.")
        return (Path(path) / npy_filename, Path(path) / json_filename)  
    
    print ("Generating Gabor filter library...")
    library, paramsdict = makeFilterLibrary_vS(paramsdict)
    print (f"Done. Library shape: {library.shape}")
    np.save(Path(path) / npy_filename, library)
       
    saveFilterParamDict_vS(paramsdict, Path(path) / json_filename)
        
    print(f"Library saved to {Path(path) / npy_filename} and {Path(path) / json_filename}")
    return (Path(path) / npy_filename, Path(path) / json_filename)    





    
def getWTfromVideo_feature_batched_vS(videodata, paramsdict, device="cuda", feature_batch_size=10_00, output_dtype=None):
    """
    Compute the wavelet transform of a video using a precomputed filter library.

    Frames and filters are flattened and multiplied on GPU, batching over the feature dimension to limit memory use.
    Equivalent to applying each filter to each frame via dot product.
    
    Inspired by: https://github.com/LeonKremers/waven-working- WaveletGenerator.py getWTfromNPY_batched and https://github.com/skriabineSop/waven WaveletGenerator.py getWTfromNPY

    Parameters
    ----------
    videodata : np.ndarray
        Shape (n_frames, H, W).

    paramsdict : dict
        Dictionary containing wavelet parameters.

    device : str or torch.device, optional
        Compute device ("cuda" or "cpu").

    feature_batch_size : int, optional
        Number of features processed per batch (controls VRAM usage).

    Returns
    -------
    WT : np.ndarray
        Shape (n_frames, ...feature_dims).
    
    """
    from numpy.lib.stride_tricks import sliding_window_view

    device = handle_torch_device(device)

    n_frames = videodata.shape[0]
    
    xs = paramsdict['xs']
    ys = paramsdict['ys']
    angles = paramsdict['angles']
    sigmas = paramsdict['sizes']
    frequencies = paramsdict['freqs']
    offsets = paramsdict['phases']
    screen_x = paramsdict['screen_x']
    screen_y = paramsdict['screen_y']
    visual_coverage = paramsdict['visual_coverage']
    
    
    feature_shape = (len(xs), len(ys), len(angles), len(sigmas), len(frequencies),  len(offsets))
    n_wavelets = int(np.prod(feature_shape))

    frame_size_video = videodata.shape[-1] * videodata.shape[-2]
    frame_size_library = screen_x * screen_y

    if frame_size_video != frame_size_library:
        raise ValueError(
            f"Video frame size ({frame_size_video}) does not match "
            f"library frame size ({frame_size_library})."
        )

    output_dtype = np.float16

    torch_dtype = torch.float16 # else torch.float32

    print(f"    n_frames: {n_frames}")
    print(f"    n_wavelets: {n_wavelets}")
    print(f"    frame_size: {frame_size_video} ")
    print(f"    feature_batch_size: {feature_batch_size}")
    print(f"    output shape: ({n_frames}, {n_wavelets}) -> ({n_frames}, {feature_shape})")

    WT = np.empty((n_frames, n_wavelets), dtype=output_dtype)

    #### video_flat = videodata.reshape(n_frames, frame_size_video)
    # pad time axis at beginning/end

    # flatten to: (n_frames, size_t * size_x * size_y)
    video_flat = videodata.reshape(videodata.shape[0], -1)

    print(videodata.shape)
    print(video_flat.shape)

    
    with torch.no_grad():
        frames_tensor = torch.as_tensor( video_flat, dtype=torch_dtype, device=device,)

        for f0 in tqdm(range(0, n_wavelets, feature_batch_size), desc="Wavelet feature batches"):
            f1 = min(f0 + feature_batch_size, n_wavelets)
            feature_chunk_size = f1 - f0
            library_chunk = np.empty( (feature_chunk_size, screen_x, screen_y),  dtype=output_dtype )

            #library_flat = waveletLibrary.reshape(n_wavelets, frame_size_library)
            for k, flat_idx in enumerate(range(f0, f1)):
                xi, yi, ai, si, fi, oi = np.unravel_index(flat_idx, feature_shape)

                library_chunk[k] = makeGaborFilter_visual_vS(
                    i_deg=xs[xi],
                    j_deg=ys[yi],
                    size_deg=sigmas[si],
                    angle=angles[ai],
                    freq_deg=frequencies[fi],
                    phase=offsets[oi],
                    visual_coverage=visual_coverage,
                    screen_x=screen_x,
                    screen_y=screen_y,
                )

            library_flat_chunk = library_chunk.reshape(feature_chunk_size, -1)


            L_chunk = torch.as_tensor(library_flat_chunk.T, dtype=frames_tensor.dtype, device=device) #drops user warning about read-only memmapped input. can be suppressed with .copy() but it is slower

            output_chunk = frames_tensor @ L_chunk

            WT[:, f0:f1] = output_chunk.cpu().numpy()

        print_cuda_tensors_mem({"frames_tensor": frames_tensor, "L_chunk": L_chunk, "output_chunk": output_chunk})

        del frames_tensor, L_chunk, output_chunk

    WT = WT.reshape((n_frames,) + feature_shape)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return WT


def compute_and_save_dwt_vS(downsampled_video_path, params,  device='cuda', force=False):
    """
    Wrapper function to compute and save wavelet transform.
    
    Parameters:
        downsampled_video_path: Path to the downsampled video .npy file
        params: Dictionary containing wavelet parameters
        device: CUDA device
        force: If True, overwrite existing DWT file
    Returns:
        Path to the saved wavelet transform .npy file
    """
    
    videodata=np.load(downsampled_video_path)
    print(f"Loaded downsampled video data from {downsampled_video_path} with shape {videodata.shape} and dtype {videodata.dtype}")
    
    workpath = downsampled_video_path.parent
    _, json_filename = filename_fromFilterParam(params)
    saveFilterParamDict_vS(params, Path(workpath) / json_filename)
    
    json_filename=Path(json_filename)
    dwt_name= f"{downsampled_video_path.stem}_lib{"_".join(json_filename.stem.split("_")[1:])}dwt.npy"
    print(f"||Constructing: {dwt_name} ")
    dwt_path= downsampled_video_path.parent / dwt_name
    if dwt_path.exists() and not force:
        print(f"Wavelet transform file {dwt_path} already exists. Skipping computation.")
        return dwt_path

    WT=getWTfromVideo_feature_batched_vS(videodata, params, device=device)
    print(f"Computed wavelet transform with shape {WT.shape} ")

    np.save(dwt_path, WT)
    print(f"Saved wavelet transform to {dwt_path} with shape {WT.shape} and dtype {WT.dtype}")
    return dwt_path
