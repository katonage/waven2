import numpy as np
import matplotlib.pyplot as plt
import torch
from wavelet_utils import make_and_save_FilterLibrary, makeFilterParamDict, loadFilterParamDict, downscale_binary_video, compute_and_save_dwt


#paths:
temppath = r'D:\SynologyDriveSyncedDATA\PROCESSED\Waven'
videopath = r'D:\SynologyDriveSyncedDATA\PROCESSED\Waven\zebra_s0_d420.0_fps59.94_RESAMPLED13fps.mp4'

full_screen_coverage = [-90, 0, -45, 45] # [az_left, az_right, el_bottom, el_top] full screen position in visual degrees
visual_coverage = [-90, 0, -30, 30] # [az_left, az_right, el_bottom, el_top] screen coverage in visual degrees
screen_x = 100 # horizontal screen size in pixels for the Gabor filter generation and movie analysis

nx = 15 # number of Gabor filters in the horizontal direction (azimuth) (y will be generated)

n_thetas = 8 # number of angles to generate

size_min = 3 # minimum size in visual degrees
size_max = 14 # maximum size in visual degrees
n_sizes = 5   # number of sizes to generate

freq_min = .02 # minimum frequency in cycles per visual degree
freq_max = .1 # maximum frequency in cycles per visual degree
n_freqs = 4  # number of frequencies to generate

n_phases = 4  # number of phases to generate

#--------------------------------------------------

#calculations
az_left, az_right, el_bottom, el_top = visual_coverage

screen_y = int(screen_x * (el_top - el_bottom) / (az_right - az_left))
ny = int(nx * (el_top - el_bottom) / (az_right - az_left))

# centers in visual degrees
x_steps = np.linspace(az_left, az_right, nx, endpoint=False)+(az_right - az_left) / (2*nx)
y_steps = np.linspace(el_bottom, el_top, ny, endpoint=False)+(el_top - el_bottom) / (2*ny)

angles= np.linspace(0, np.pi, n_thetas, endpoint=False)
sizes = np.logspace(np.log10(size_min), np.log10(size_max), n_sizes)
freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

print(f"Screen size: {screen_x}x{screen_y} pixels")
print(f"Visual coverage: {visual_coverage} degrees")
print(f"Full screen coverage: {full_screen_coverage} degrees")
#print(f"Center positions (x_deg): {np.round(x_steps, 1)} degrees")
#print(f"Center positions (y_deg): {np.round(y_steps, 1)} degrees")
print(f"Angles (degrees): {np.round(np.rad2deg(angles), 1)}")
print(f"Sizes (degrees): {sizes}")
print(f"Frequencies (cycles/degree): {freqs}")
print(f"Phases (degrees): {np.rad2deg(phases)}")

total_n=len(sizes)*len(angles)*len(freqs)*len(phases)*len(x_steps)*len(y_steps)
print(f"Total number of Gabor filters to generate: {total_n}")

# some parameter checks
gabor_step=(az_right-az_left)/nx 
print(f"Control: Gabor placement step in visual degrees (x): {gabor_step:.1f}, vs size_min: {size_min:.1f} degrees. {'OK' if (gabor_step < size_min) else 'WARNING!'}")
gabor_step=(el_top-el_bottom)/ny
print(f"Control: Gabor placement step in visual degrees (y): {gabor_step:.1f}, vs size_min: {size_min:.1f} degrees. {'OK' if (gabor_step < size_min) else 'WARNING!'}")
visual_step_x=(az_right-az_left)/screen_x
print(f"Control: Gabor resolution in visual degrees (x): {visual_step_x:.1f}, vs 1/freq_max: {1/freq_max:.1f} degrees. {'OK' if (visual_step_x < 1/freq_max/4) else 'WARNING!'}")
visual_step_y=(el_top-el_bottom)/screen_y
print(f"Control: Gabor resolution in visual degrees (y): {visual_step_y:.1f}, vs 1/freq_max: {1/freq_max:.1f} degrees. {'OK' if (visual_step_y < 1/freq_max/4) else 'WARNING!'}")

#--------------------------------------------------
# pack parameters into a dictionary 
params=makeFilterParamDict(screen_x, screen_y, visual_coverage, full_screen_coverage, x_steps, y_steps, angles, sizes, freqs, phases)

# generate and save the filter library if it doesn't exist
lib_path, sidecar_path = make_and_save_FilterLibrary(temppath, params)

#-- load the library and parameters back 
library = np.load(lib_path)
print(f"Loaded Gabor filter library from {lib_path} with shape {library.shape}")

xs, ys, angles, sizes, freqs, phases, visual_coverage, full_screen_coverage, screen_x, screen_y = loadFilterParamDict(sidecar_path)

#------------------------------------------------
# downsample displayed video to match the Gabor filters resolution and visual coverage
downsampled_video_path=downscale_binary_video(videopath, full_screen_coverage, visual_coverage, screen_x, screen_y)

# video decomposition into Gabor filter responses   
dwt_path=compute_and_save_dwt(downsampled_video_path, lib_path)

print(f"Full analysis completed. DWT saved to {dwt_path}")