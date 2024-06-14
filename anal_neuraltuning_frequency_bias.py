import os
import matplotlib.pyplot as plt
import gc
import joblib
import time
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import zscore
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from utils import train_data_normalization, Timer, net_size_info, conv2_labels
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams.update({'font.size': 14, 'mathtext.fontset': 'stix'})

def convert_ang(angle):
    angle = 90 - angle
    angle[np.where(angle<-180)[0]] = angle[np.where(angle<-180)[0]] + 360
    return angle

# Visualization function
def plot_tuning_by_loc(x, y, voxel_tuning, performance, tuningname, unit_type, tuning_type):

    # Create a figure and a subplot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize the performance to scale the size of the circles
    performance_norm = Normalize(min(performance), max(performance))
    alphas = performance_norm(performance)  # Compute normalized performance for alpha values

    # Create a color wheel
    unique_tunings = np.unique(voxel_tuning)
    color_wheel = plt.cm.jet(np.linspace(0, 1, len(unique_tunings)))

    # Create a mapping from tuning value to color
    color_map = {tuning: color_wheel[i] for i, tuning in enumerate(unique_tunings)}

    # Plot each voxel's location with a circle
    for i in range(len(x)):
        circle_color = color_map[voxel_tuning[i]]
        ax.scatter(x[i], y[i], color=circle_color, s=35, alpha=alphas[i], edgecolors='none')

    # Create a colorbar with the color wheel
    sm = ScalarMappable(cmap=plt.cm.jet, norm=Normalize(vmin=min(unique_tunings), vmax=max(unique_tunings)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.036, pad=0.04)
    cbar.set_label('Frequency')

    # Set fixed ticks for the colorbar
    tick_values = np.arange(0.01, 0.40, 0.05)  # Use arange and append last value
    tick_values = np.append(tick_values, 0.39)  # Append the last tick manually
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(["{:.2f}".format(v) for v in tick_values])

    # Set the axis labels
    ax.set_xlabel('Horizontal position (deg)', size=18)
    ax.set_ylabel('Vertical position (deg)', size=18)

    # Set the axis limits
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([-8, 0, 8])
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.tick_params(axis='x', labelsize=12)  # Adjust the font
    ax.tick_params(axis='y', labelsize=12)  # Adjust the font

    # # Display the plot
    plt.savefig(pjoin(vispath, 'location', f'{tuningname}_frequency-location_unit-{unit_type}-{tuning_type}.png'))
    plt.close()


def plot_tuning_by_ecc(eccentricity, voxel_tuning, tuningname, unit_type, tuning_type):

    # Calculate exponential bin edges within 0 to 12 degrees
    num_bins = 20
    max_eccentricity = 12
    base = np.exp(np.log(max_eccentricity) / (num_bins - 1))
    bin_edges = np.array([base ** i - 1 for i in range(num_bins)])
    bin_edges[-1] = max_eccentricity  # Ensure the last bin edge exactly equals 12
    bin_edges[0] = 0  # Ensure the first bin edge starts at 0

    # Assign each eccentricity to a bin
    bin_indices = np.digitize(eccentricity, bin_edges) - 1  # Find which bin each eccentricity belongs to

    # Calculate mean and std deviation for each bin
    fre_tuning_means = [np.mean(voxel_tuning[bin_indices == i]) for i in range(num_bins)]
    fre_tuning_stds = [np.std(voxel_tuning[bin_indices == i]) for i in range(num_bins)]
    n_voxel = [(bin_indices == i).sum() for i in range(num_bins)]
    fre_tuning_error = [fre_tuning_stds[i] / (n_voxel[i] ** 0.5) for i in range(num_bins)]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(3.3, 2.8))

    # Plot the binned data
    ax.plot(bin_edges, fre_tuning_means, label='Radial Dev', color='#03AED2', linewidth=4.5)
    ax.fill_between(bin_edges, np.array(fre_tuning_means) - fre_tuning_error, 
                np.array(fre_tuning_means) + fre_tuning_error, color='#03AED2', alpha=0.5)

    # Set x and y axis labels
    # ax.set_xlabel('Eccentricity (deg)', size=18)
    # ax.set_ylabel('Frequency', size=18)

    # Set x axis limits
    ax.set_xlim(0, max_eccentricity)
    ax.set_xticks([0, 6, 12])
    ax.tick_params(axis='x', labelsize=16)  # Adjust the font
    ax.tick_params(axis='y', labelsize=16)  # Adjust the font

    # Show the plot
    plt.tight_layout() 
    # plt.savefig(pjoin(vispath, 'ecc', f'{tuningname}_frequency-ecc_unit-{unit_type}-{tuning_type}.png'))
    plt.savefig(pjoin(vispath, f'Ecc_{tuningname}_frequency-ecc_unit-{unit_type}-{tuning_type}.svg'), format='svg', bbox_inches='tight', pad_inches=0.01)

    plt.close()

def plot_tuning_by_angle(angle, voxel_tuning, tuningname, unit_type, tuning_type):

    # Define bins for the angle range
    num_bins = 20
    bins = np.linspace(0, 360, num_bins + 1)
    bin_indices = np.digitize(angle, bins) - 1

    # Calculate average deviations for each bin
    fre_tuning_means = [np.mean(voxel_tuning[bin_indices == i]) for i in range(num_bins)]

    # Ensure continuity by appending the first value at the end
    fre_tuning_means = np.append(fre_tuning_means, fre_tuning_means[0])

    # Convert bins to radians for plotting
    bins_radians = np.radians(bins)

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(2.8, 2.8))

    # Plot the data
    ax.plot(bins_radians, fre_tuning_means, color='#03AED2', linewidth=5)

    # Set the angle limits and labels
    ax.set_theta_zero_location('E')  # Theta=0 at the right
    ax.set_theta_direction(1)  # Angle increase counterclockwise
    
    # Define the ticks at the four angles and convert them to radians
    highlight_angles = [0, 45, 90, 135, 180, 215, 270, 315]  # Angles to highlight
    highlight_radians = np.radians(highlight_angles)
    ax.set_xticks(highlight_radians)  # Set ticks at these angles
    labels = []#['0��', '', '90��', '', '180��', '', '270��', '']
    ax.set_xticklabels(labels)  # Apply custom labels to the specified ticks
    # Reduce the number of radial ticks
    radial_ticks = [0.1, 0.2, 0.3]  # Minimal and maximal values
    ax.set_yticks(radial_ticks)  # Set specific radial ticks
    ax.tick_params(axis='y', labelsize=16)

    # Show the plot
    plt.tight_layout() 
    # plt.savefig(pjoin(vispath, 'angle', f'{tuningname}_frequency-angle_unit-{unit_type}-{tuning_type}.png'))
    plt.savefig(pjoin(vispath, f'Angle_{tuningname}_frequency-angle_unit-{unit_type}-{tuning_type}.svg'), format='svg', bbox_inches='tight', pad_inches=0.01)

    plt.close()

def compute_voxel_tuning(stimulus, response, tuning_type='argmax'):
    # define container
    voxel_tuning = np.zeros((response.shape[1]))
    # check tuning_type
    if tuning_type == 'argmax':
        for voxel_idx in range(response.shape[1]):
            voxel_tuning[voxel_idx] = stimulus[response[:, voxel_idx].argmax()]
    elif tuning_type == 'weighted':
        # Convert degrees to radians
        radians = np.radians(stimulus)

        # Calculate weighted average of sin and cos
        for voxel_idx in range(response.shape[1]):
            # Get the responses for this voxel
            responses = response[:, voxel_idx]
            
            # Base responses on minimum response to avoid negative weights
            adjusted_responses = responses - responses.min()
            
            # Normalize weights
            weights = adjusted_responses / adjusted_responses.sum()

            # Compute weighted average for sin and cos
            weighted_sin = np.sum(np.sin(radians) * weights)
            weighted_cos = np.sum(np.cos(radians) * weights)
            
            # Compute the angle from the average sin and cos
            mean_angle = np.arctan2(weighted_sin, weighted_cos)
            
            # Convert from radians to degrees
            voxel_tuning[voxel_idx] = int(np.degrees(mean_angle))
    return voxel_tuning


def plot_color_wheel(ori):
    # set axes
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'aspect': 'equal'})
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')  

    # ʹ��ori����ĳ�����������ɫ��
    color_wheel = plt.cm.hsv(np.linspace(0, 1, len(ori)))

    # ����Բ����ɫ��
    num_segments = len(ori)  # ����ori���鳤�Ⱦ����ֶ�����

    for i in range(num_segments):
        color = color_wheel[i]  # ѡ����ɫ
        # ���ƶ�Ӧ��0-180�ȵ�����
        wedge = patches.Wedge(center=(0, 0), r=1, theta1=ori[i], theta2=ori[i]+6, facecolor=color)
        ax.add_patch(wedge)
        # ���ƶ�Ӧ��180-360�ȵ�����
        if ori[i] + 180 < 360:
            wedge = patches.Wedge(center=(0, 0), r=1, theta1=ori[i]+180, theta2=ori[i]+186, facecolor=color)
            ax.add_patch(wedge)

    # ����ɫ��ע
    ax.text(0, -1.2, '270��', ha='center', va='center')
    ax.text(1.2, 0, '0��', ha='center', va='center')
    ax.text(0, 1.2, '90��', ha='center', va='center')
    ax.text(-1.2, 0, '180��', ha='center', va='center')
    plt.show()

# define params
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
voxelpath = pjoin(work_dir, 'prep/roi-concate/')
retinopath = pjoin(work_dir, 'anal/brainmap/masked_retinotopy')
output_path = pjoin(work_dir, 'anal/neural-selectivity/parameterspace')
voxelmodel_path = pjoin(work_dir, 'build/roi-voxelwisemodel')
vispath = pjoin(work_dir, 'vis/results_plots')# pjoin(work_dir, 'vis/frequency-bias')
layer = 'conv2'
unit_type = 'full'
tuning_type = 'argmax'

# define containers
all_x = np.array([])
all_y = np.array([])
all_tuning = np.array([], dtype=int)
all_lengths = np.array([])
all_retino = np.empty((2, 0))

for sub in [f'sub-0{_+1}' for _ in range(9)]:
    # load data
    tuningname = f'{sub}-conv2-V1-roi'
    voxel = np.load(pjoin(voxelpath, f'{sub}/{sub}_layer-googlenet-{layer}_V1-voxel.npy'))
    performance = np.load(pjoin(voxelmodel_path, f'{sub}/googlenet-{layer}/{sub}_V1-test-cor.npy'))
    retino_file = pjoin(retinopath,  f'{sub}/{sub}_masked-prior-prf.dscalar.nii')
    retino = nib.load(retino_file).get_fdata()
    retino = retino[:, voxel]
    # load gabor data
    gaborspace = np.load(pjoin(output_path, f'{tuningname}/{tuningname}_frequency-tuning_unit-{unit_type}.npy'), allow_pickle=True)
    frequency = gaborspace.item()['stim']
    response = gaborspace.item()['space']

    # filter voxel by selecting voxel performance
    voxel_loc = performance > 0
    retino = retino[:, voxel_loc]
    response = response[:, voxel_loc]
    performance = performance[voxel_loc]

    # concate voxel retino infos
    if 'prior' in retino_file:
        retino_regressor = np.c_[convert_ang(retino[0]), retino[1]]
    else:
        retino_regressor = np.c_[retino[1], retino[0]]
    # convert to XY cordinate
    x = np.cos(retino_regressor[:, 0]) * retino_regressor[:, 1]
    y = np.sin(retino_regressor[:, 0]) * retino_regressor[:, 1]

    # obtain voxel tuning 
    voxel_tuning = compute_voxel_tuning(frequency, response, tuning_type)

    # # plot tuning by location, eccentricity and angle
    # angle, eccentricity = retino[0], retino[1]
    # performance_dis = performance - performance.min()
    # plot_tuning_by_loc(x, y, voxel_tuning, performance_dis, tuningname, unit_type, tuning_type)
    # plot_tuning_by_ecc(eccentricity, voxel_tuning, tuningname, unit_type, tuning_type)
    # plot_tuning_by_angle(angle, voxel_tuning, tuningname, unit_type, tuning_type)

    # Store the collected data
    all_x = np.concatenate((all_x, x))
    all_y = np.concatenate((all_y, y))
    all_tuning = np.concatenate((all_tuning, voxel_tuning))
    all_lengths = np.concatenate((all_lengths, performance))
    all_retino = np.hstack((all_retino, retino[:2, :]))
    print(f'Finish ploting in {sub}: {performance.shape[0]} voxels selected')

# plot all deviation from orientation maps
angle, eccentricity = all_retino[0], all_retino[1]
tuningname = 'sub-all-conv2-V1-roi'
voxel_tuning = all_tuning
all_lengths = all_lengths - all_lengths.min()
# plot_tuning_by_loc(all_x, all_y, voxel_tuning, all_lengths, tuningname, unit_type, tuning_type)
plot_tuning_by_ecc(eccentricity, voxel_tuning, tuningname, unit_type, tuning_type)
plot_tuning_by_angle(angle, voxel_tuning, tuningname, unit_type, tuning_type)
