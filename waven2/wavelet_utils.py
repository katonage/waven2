
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

from skimage.filters import gabor_kernel

import os
import cv2
import gc
import torch
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


def makeFilterParamDict(screen_x, screen_y, visual_coverage, xs, ys, angles, sigmas, frequencies, offsets):
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
        'visual_coverage': visual_coverage
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
    screen_x = params['screen_x']
    screen_y = params['screen_y']
    return xs, ys, angles, sizes, freqs, phases, visual_coverage, screen_x, screen_y

def makeFilterLibrary(paramsdict):
    """
    Builds a Gabor filter library.

    Parameter: paramsdict (dict): A dictionary containing the parameters for Gabor filter generation:
        screen_x, screen_y (int): Width and height of the screen in pixels.
        visual_coverage (float): Coverage of the visual field.
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
        print("Library file already exists. Skipping generation.")
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