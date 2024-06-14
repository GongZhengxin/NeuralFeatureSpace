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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from utils import train_data_normalization, Timer, net_size_info, conv2_labels
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams.update({'font.size': 14})

def convert_ang(angle):
    angle = 90 - angle
    angle[np.where(angle<-180)[0]] = angle[np.where(angle<-180)[0]] + 360
    return angle

# Visualization function
def plot_tuning_by_loc(x, y, voxel_tuning, performance, tuningname, unit_type, tuning_type):

    # Create a figure and a subplot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a color wheel
    shifted_jet = shift_colormap(plt.cm.jet, rotation=0.5)
    color_wheel = shifted_jet(np.linspace(0, 1, 180))
    for i in range(len(x)):
        # Map the orientation preference to a color
        angle = voxel_tuning[i]
        line_length = performance[i]
        # Calculate the end point of the line
        end_x = x[i] + np.cos(np.deg2rad(angle)) * line_length * 3
        end_y = y[i] + np.sin(np.deg2rad(angle)) * line_length * 3
        # Plot a line between the start and end points
        ax.plot([x[i], end_x], [y[i], end_y], color=color_wheel[angle], linewidth=2)

    # Set the axis labels
    ax.set_xlabel('Horizontal position (deg)', size=18)
    ax.set_ylabel('Vertical position (deg)', size=18)

    # Set the axis limits
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([-8, 0, 8])
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.tick_params(axis='x', labelsize=16)  # Adjust the font
    ax.tick_params(axis='y', labelsize=16)  # Adjust the font

    # # Display the plot
    plt.savefig(pjoin(vispath, 'location', f'{tuningname}_orientation-location_unit-{unit_type}-{tuning_type}.png'))
    plt.close()


def plot_tuning_by_ecc(angle, eccentricity, voxel_tuning, tuningname, unit_type, tuning_type):

    # transform angle to 0 to 180
    angle_transformed = np.where(angle > 180, 360 - angle, angle)

    # Compute radial_dev
    radial_diff = np.abs(voxel_tuning - angle_transformed)
    radial_dev = np.radians(np.where(radial_diff > 90, 180 - radial_diff, radial_diff))

    # Compute vertical_dev
    vertical_dev = np.radians(np.abs(voxel_tuning - 90))

    # Compute cardinal dev
    y_angle = np.abs(angle_transformed - 90)
    cardinal_optimal = np.where(y_angle <= 45, 90, 0)
    cardinal_diff = np.abs(voxel_tuning - cardinal_optimal)
    cardinal_dev = np.radians(np.where(cardinal_diff > 90, 180 - cardinal_diff, cardinal_diff))

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
    radial_means = [np.mean(radial_dev[bin_indices == i]) for i in range(num_bins)]
    vertical_means = [np.mean(vertical_dev[bin_indices == i]) for i in range(num_bins)]
    cardinal_means = [np.mean(cardinal_dev[bin_indices == i]) for i in range(num_bins)]
    radial_stds = [np.std(radial_dev[bin_indices == i]) for i in range(num_bins)]
    vertical_stds = [np.std(vertical_dev[bin_indices == i]) for i in range(num_bins)]
    cardinal_stds = [np.std(cardinal_dev[bin_indices == i]) for i in range(num_bins)]
    n_voxel = [(bin_indices == i).sum() for i in range(num_bins)]
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(4.8, 2.8))

    # Plot the binned data
    radial_error = [radial_stds[i] / (n_voxel[i] ** 0.5) for i in range(num_bins)]
    vertical_error = [vertical_stds[i] / (n_voxel[i] ** 0.5) for i in range(num_bins)]
    cardinal_error = [cardinal_stds[i] / (n_voxel[i] ** 0.5) for i in range(num_bins)]
    ax.plot(bin_edges, radial_means, label='Radial', color='#FF8A08', linewidth=4.5)
    ax.plot(bin_edges, vertical_means, label='Vertical', color='#03AED2', linewidth=4.5)
    ax.plot(bin_edges, cardinal_means, label='Cardinal', color='#7ABA78', linewidth=4.5)
    ax.fill_between(bin_edges, np.array(radial_means) - radial_error, 
                np.array(radial_means) + radial_error, color='#FF8A08', alpha=0.5)
    ax.fill_between(bin_edges, np.array(vertical_means) - vertical_error, 
                np.array(vertical_means) + vertical_error, color='#03AED2', alpha=0.5)
    ax.fill_between(bin_edges, np.array(cardinal_means) - cardinal_error, 
                np.array(cardinal_means) + cardinal_error, color='#7ABA78', alpha=0.5)

    # Set x and y axis labels
    # ax.set_xlabel('Eccentricity (°)', size=16)
    # ax.set_ylabel('Deviation (rad)', size=16)

    # Set x axis limits
    ax.set_xlim(0, max_eccentricity)
    ax.set_xticks([0, 6, 12])
    ax.tick_params(axis='x', labelsize=16)  # Adjust the font
    ax.tick_params(axis='y', labelsize=16)  # Adjust the font

    # Show the plot
    ax.legend(bbox_to_anchor=(1., 0.5), loc='upper left', fontsize=14, frameon=False)#  
    plt.tight_layout() 
    # plt.savefig(pjoin(vispath, 'ecc', f'{tuningname}_orientation-eccentricity_unit-{unit_type}-{tuning_type}.png'))
    plt.savefig(pjoin(vispath, f'Ecc_{tuningname}_orientation-eccentricity_unit-{unit_type}-{tuning_type}.svg'), format='svg', bbox_inches='tight', pad_inches=0.01)

    plt.close()

def plot_tuning_by_angle(angle, voxel_tuning, tuningname, unit_type, tuning_type):

    # transform angle to 0 to 180
    angle_transformed = np.where(angle > 180, 360 - angle, angle)

    # Compute radial_dev
    radial_diff = np.abs(voxel_tuning - angle_transformed)
    radial_dev = np.radians(np.where(radial_diff > 90, 180 - radial_diff, radial_diff))

    # Compute vertical_dev
    vertical_dev = np.radians(np.abs(voxel_tuning - 90))

    # Compute cardinal dev
    y_angle = np.abs(angle_transformed - 90)
    cardinal_optimal = np.where(y_angle <= 45, 90, 0)
    cardinal_diff = np.abs(voxel_tuning - cardinal_optimal)
    cardinal_dev = np.radians(np.where(cardinal_diff > 90, 180 - cardinal_diff, cardinal_diff))

    # Define bins for the angle range
    num_bins = 20
    bins = np.linspace(0, 360, num_bins + 1)
    bin_indices = np.digitize(angle, bins) - 1

    # Calculate average deviations for each bin
    radial_means = np.array([np.mean(radial_dev[bin_indices == i]) for i in range(num_bins)])  # Include the first bin again at the end
    vertical_means = np.array([np.mean(vertical_dev[bin_indices == i]) for i in range(num_bins)])
    cardinal_means = np.array([np.mean(cardinal_dev[bin_indices == i]) for i in range(num_bins)])

    # Ensure continuity by appending the first value at the end
    radial_means = np.append(radial_means, radial_means[0])
    vertical_means = np.append(vertical_means, vertical_means[0])
    cardinal_means = np.append(cardinal_means, cardinal_means[0])

    # Convert bins to radians for plotting
    bins_radians = np.radians(bins)

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(2.8, 2.8))

    # Plot the data
    ax.plot(bins_radians, radial_means, label='Radial Dev', color='#FF8A08', linewidth=5)
    ax.plot(bins_radians, vertical_means, label='Vertical Dev', color='#03AED2', linewidth=5)
    ax.plot(bins_radians, cardinal_means, label='Cardinal Dev', color='#7ABA78', linewidth=5)

    # Set the angle limits and labels
    ax.set_theta_zero_location('E')  # Theta=0 at the right
    ax.set_theta_direction(1)  # Angle increase counterclockwise

    # Define the ticks at the four angles and convert them to radians
    highlight_angles = [0, 45, 90, 135, 180, 215, 270, 315]  # Angles to highlight
    highlight_radians = np.radians(highlight_angles)
    ax.set_xticks(highlight_radians) #  Set ticks at these angles
    # labels = ['0��', '', '90��', '', '180��', '', '270��', '']
    ax.set_xticklabels([])  # Apply custom labels to the specified ticks
    # Reduce the number of radial ticks
    radial_ticks = [0.3, 0.6, 0.9]  # Minimal and maximal values
    ax.set_yticks(radial_ticks)  # Set specific radial ticks
    ax.tick_params(axis='y', labelsize=16)
    ax.set_rlabel_position(45)  # Positions the radial labels at 90 degrees angle

    # # Add a legend to the plot
    # ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.5))

    # Show the plot
    plt.tight_layout() 
    # plt.savefig(pjoin(vispath, 'angle', f'{tuningname}_orientation-angle_unit-{unit_type}-{tuning_type}.png'))
    plt.savefig(pjoin(vispath, f'Angle_{tuningname}_orientation-angle_unit-{unit_type}-{tuning_type}.svg'), format='svg', bbox_inches='tight', pad_inches=0.01)
    plt.close()

def compute_voxel_tuning(stimulus, response, tuning_type='argmax'):
    # define container
    voxel_tuning = np.zeros((response.shape[1]), dtype=int)
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


def shift_colormap(cmap, start=0, midpoint=0.5, stop=1.0, rotation=0.25):
    """
    Function to shift the colors of a colormap.
    The rotation specifies how much to rotate the colormap (as a fraction of 1).
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    
    # Compute the rotation offset
    reg_index = np.linspace(start, stop, 257)
    shift_index = np.mod(reg_index - rotation, 1.0)
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(si)
        cdict['red'].append((ri, r, r))
        cdict['green'].append((ri, g, g))
        cdict['blue'].append((ri, b, b))
        cdict['alpha'].append((ri, a, a))
    
    new_cmap = LinearSegmentedColormap('shiftedcmap', cdict)
    return new_cmap


def plot_color_wheel(ori):
    # set axes
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'aspect': 'equal'})
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')  

    # ʹ��ori����ĳ�����������ɫ��
    jet = plt.cm.jet
    shifted_jet = shift_colormap(jet, rotation=0.5)
    color_wheel = shifted_jet(np.linspace(0, 1, len(ori)))

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
    # ax.text(0, -1.2, '270��', ha='center', va='center')
    # ax.text(1.2, 0, '0��', ha='center', va='center')
    # ax.text(0, 1.2, '90��', ha='center', va='center')
    # ax.text(-1.2, 0, '180��', ha='center', va='center')
    plt.show()

# define params
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
voxelpath = pjoin(work_dir, 'prep/roi-concate/')
retinopath = pjoin(work_dir, 'anal/brainmap/masked_retinotopy')
output_path = pjoin(work_dir, 'anal/neural-selectivity/parameterspace')
voxelmodel_path = pjoin(work_dir, 'build/roi-voxelwisemodel')
vispath = pjoin(work_dir, 'vis/results_plots')# pjoin(work_dir, 'vis/radial-bias')
layer = 'conv2'
unit_type = 'full'
tuning_type = 'argmax'

# define containers
all_x = np.array([])
all_y = np.array([])
all_tuning = np.array([], dtype=int)
all_lengths = np.array([])
all_retino = np.empty((2, 0))

draw_mean_only = True
for sub in [f'sub-0{_+1}' for _ in range(9)]:
    # load data
    tuningname = f'{sub}-conv2-V1-roi'
    voxel = np.load(pjoin(voxelpath, f'{sub}/{sub}_layer-googlenet-{layer}_V1-voxel.npy'))
    performance = np.load(pjoin(voxelmodel_path, f'{sub}/googlenet-{layer}/{sub}_V1-test-cor.npy'))
    retino_file = pjoin(retinopath,  f'{sub}/{sub}_masked-prior-prf.dscalar.nii')
    retino = nib.load(retino_file).get_fdata()
    retino = retino[:, voxel]
    # load gabor data
    gaborspace = np.load(pjoin(output_path, f'{tuningname}/{tuningname}_orientation-tuning_unit-{unit_type}.npy'), allow_pickle=True)
    ori = gaborspace.item()['ori']
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
    voxel_tuning = compute_voxel_tuning(ori, response, tuning_type)

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
plot_tuning_by_ecc(angle, eccentricity, voxel_tuning, tuningname, unit_type, tuning_type)
plot_tuning_by_angle(angle, voxel_tuning, tuningname, unit_type, tuning_type)

# # plot orientation map
# all_lengths = all_lengths - all_lengths.min()
# draw_orientation(all_x, all_y, all_tuning, all_lengths, tuningname, unit_type, tuning_type)

# # plot color wheel
# plot_color_wheel(ori)