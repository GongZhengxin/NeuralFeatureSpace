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

def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
    i, j = coords
    return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C

def adjust_RF(receptive_field):
    cur_receptive_field = receptive_field.copy()
    cur_receptive_field = cur_receptive_field + np.abs(np.min(cur_receptive_field, None)) + 1
    thres = np.min(cur_receptive_field) + 0.5*(np.max(cur_receptive_field) - np.min(cur_receptive_field)) #

    cur_receptive_field[cur_receptive_field < thres] = 0
    cur_receptive_field = cur_receptive_field / (cur_receptive_field.sum() + 1e-20)

    return cur_receptive_field

def construct_RF(fieldsize, A, x, y, sigma_x, sigma_y, C):
    # 0 创建假想体素感受野
    i = np.linspace(-8., 8., fieldsize)
    j = np.linspace(8., -8., fieldsize)
    i, j = np.meshgrid(i, j)
    return adjust_RF(gaussian_2d((i, j), A, x, y, sigma_x, sigma_y, C))

def normalize_data(input, mean, std):
    # 执行标准化
    normalized_data = (input - mean) / std
    return normalized_data

# path settings
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
activ_dir = pjoin(work_dir, 'prep/simplified_stimuli/activation')

mean_act_path = pjoin(work_dir, 'prep/roi-concate')
# save out path
output_path = pjoin(work_dir, 'anal/neural-selectivity/parameterspace')
os.makedirs(output_path, exist_ok=True)

layer = 'conv2'
sub_tunings = []
for roi in ['V1', 'V2', 'V3'] :
    tuning_file = pjoin(work_dir, f'anal/neural-selectivity/all-sub_{layer}-{roi}-tunings.npy')
    if not os.path.exists(tuning_file):
        subs_tuning = {}
        mean_tuning = np.zeros(63)
        subs = [f'sub-0{_+1}' for _ in range(9)]
        for sub in subs:
            vox = 'roi'
            tuning_dir = pjoin(work_dir, f'build/roi-concatemodel/{roi}')
            voxeltuning = joblib.load(pjoin(tuning_dir, f'{sub}_layer-googlenet-{layer}_{roi}-linear.pkl')).coef_
            mean_tuning += voxeltuning
            subs_tuning[f'{sub}-{layer}-{roi}-{vox}'] = voxeltuning
        subs_tuning[f'sub-mean-{layer}-{roi}-{vox}'] = mean_tuning / len(subs)
        np.save(tuning_file, subs_tuning)
        sub_tunings.append(subs_tuning)
    else:
        subs_tuning = np.load(tuning_file, allow_pickle=True).item()
        sub_tunings.append(subs_tuning)
neural_selectivity_path = pjoin(work_dir, 'anal/neural-selectivity')

axis_type = 'ica'
axes_tuning = np.load(pjoin(neural_selectivity_path, f'{axis_type}_tuning-dict.npy'), allow_pickle=True).item()
concateactiv_dir = pjoin(work_dir, 'prep/image_activations/concate-activs')
global_mean = np.load(pjoin(concateactiv_dir, 'all-sub_googlenet-conv2_63-activation_mean.npy'))
global_std = np.load(pjoin(concateactiv_dir, 'all-sub_googlenet-conv2_63-activation_std.npy'))

# ################################
# 1. 构建 simple attributes param space
# ################################
stim_type = ['180_gabor', '30_color', 'raw_shape', '36k_natural']
net_layer = 'Googlenet_conv2' 
layerlabel = net_layer.split('_')[1]
activ_files = sorted([ _ for _ in os.listdir(activ_dir) \
    if f'{net_layer}.npy' in _ and _.replace(f'_{net_layer}.npy', '') in stim_type])
activs = list(map(lambda x: np.load(x, mmap_mode='r'), [pjoin(activ_dir, _) for _ in activ_files]))
print(f'{activ_files}; 刺激实验数据')
for i in range(len(activ_files)):
    print(activs[i].shape)
voxelparams = (1,0,0,3.6,3.6,0)
num_channel = activs[0].shape[1]
fieldsize = activs[0].shape[-1]
rfmask = construct_RF(fieldsize, *voxelparams)

# spatial summation
ss_file = pjoin(work_dir, 'anal/neural-selectivity/spatial-summed_activs.npy')
if os.path.exists(ss_file):
    activs = np.load(ss_file, allow_pickle=True).tolist()
else:
    activs = list(map(lambda x: np.sum(activs[x] * rfmask, axis=(2,3)), list(range(len(stim_type)))))
    np.save(ss_file, activs)
investigate_tunings = axes_tuning #sub_tunings[2] #
for tuningname, voxeltuning in investigate_tunings.items():
    
    os.makedirs(pjoin(output_path, f'{axis_type}/{tuningname}'), exist_ok=True)
    weightsnum = voxeltuning.shape[-1]
    
    if axis_type == 'raw':
        _, subidx, layer, roi, _ = tuningname.split('-')
        sub_featuremean = np.load(pjoin(f"{mean_act_path}/sub-{subidx}/sub-{subidx}_layer-googlenet-{layer}_{roi}-mean-train-feature.npy")).mean(axis=0)
        sub_featurestd = np.sqrt((np.load(pjoin(f"{mean_act_path}/sub-{subidx}/sub-{subidx}_layer-googlenet-{layer}_{roi}-std-train-feature.npy"))**2).mean(axis=0))
        
        voxel_responses = list(map(lambda x: np.sum(normalize_data(x[:,0:weightsnum], sub_featuremean, sub_featurestd) * voxeltuning, axis=-1), activs))
    elif axis_type == 'pca':
        voxel_responses = list(map(lambda x: np.sum(normalize_data(x[:,0:weightsnum], global_mean, global_std) * voxeltuning, axis=-1), activs))
    elif axis_type == 'ica':
        voxel_responses = list(map(lambda x: np.sum(normalize_data(x[:,0:weightsnum], global_mean, 1) * voxeltuning, axis=-1), activs))
    # 载入刺激属性的信息
    stim_type = [ _.replace(f'_{net_layer}.npy', '') for _ in activ_files ]
    info_dir = pjoin(work_dir, 'prep/simplified_stimuli/stim/info')
    info_dfs = []
    for i, key in enumerate(stim_type):
        info_dfs.append(pd.read_csv(pjoin(info_dir, f'{key}.stim.csv')))

    # 创建参数空间矩阵
    # gabor
    stim_type = [ _.replace(f'_{net_layer}.npy', '').split('_')[-1] for _ in activ_files ]
    info_df = info_dfs[stim_type.index('gabor')]
    voxel_response = voxel_responses[stim_type.index('gabor')]
    params = np.c_[info_df['orientation'].values, info_df['frequency'].values]
    orientation_length = len(np.unique(info_df['orientation'].values))
    frequency_length = len(np.unique(info_df['frequency'].values))
    param_space = np.zeros((orientation_length, frequency_length))
    for i, orientation in enumerate(np.unique(info_df['orientation'].values)):
        for j, frequency in enumerate(np.unique(info_df['frequency'].values)):
            poses = np.intersect1d(np.where((params[:,0]==orientation))[0], np.where((params[:,1]==frequency))[0])
            param_space[i,j] = voxel_response[poses].mean(axis=0)
    orientation = np.unique(info_df['orientation'].values)
    frequency = np.unique(info_df['frequency'].values)
    gaborspace = {'ori': orientation, 'freq': frequency, 'space': param_space}
    # 1f save out
    np.save(pjoin(output_path, f'{axis_type}/{tuningname}/{tuningname}_gabor-space.npy'), gaborspace)

    # color
    info_df = info_dfs[stim_type.index('color')]
    voxel_response = voxel_responses[stim_type.index('color')]
    params = np.c_[info_df['hue'].values, info_df['lightness'].values, info_df['chroma'].values]
    hue_length = len(np.unique(info_df['hue'].values))
    light_length = len(np.unique(info_df['lightness'].values))
    chroma_length = len(np.unique(info_df['chroma'].values))
    param_space = np.zeros((hue_length, light_length, chroma_length))
    for i, hue in enumerate(np.unique(info_df['hue'].values)):
        for j, light in enumerate(np.unique(info_df['lightness'].values)):
            for k, chroma in enumerate(np.unique(info_df['chroma'].values)):
                poses = np.intersect1d(np.where((params[:,0]==hue))[0], np.where((params[:,1]==light))[0])
                poses = np.intersect1d(poses, np.where((params[:,2]==chroma))[0])
                param_space[i,j,k] = voxel_response[poses].mean(axis=0)
    hue = np.unique(info_df['hue'].values)
    light = np.unique(info_df['lightness'].values)
    chroma = np.unique(info_df['chroma'].values)
    colorspace = {'hue': hue, 'light': light, 'chroma': chroma , 'space': param_space}
    # 2d save out
    np.save(pjoin(output_path, f'{axis_type}/{tuningname}/{tuningname}_color-space.npy'), colorspace)

    # curv
    info_df = info_dfs[stim_type.index('shape')]
    voxel_response = voxel_responses[stim_type.index('shape')]
    params = np.c_[info_df['shape'].values, info_df['rotation'].values]
    shape_length = len(np.unique(info_df['shape'].values))
    rot_length = len(np.unique(info_df['rotation'].values))
    param_space = np.nan*np.zeros((shape_length, rot_length))
    for i, shape in enumerate(np.unique(info_df['shape'].values)):
        for j, rot in enumerate(np.unique(info_df['rotation'].values)):
            poses = np.intersect1d(np.where((params[:,0]==shape))[0], np.where((params[:,1]==rot))[0])
            if len(poses) >0: 
                param_space[i,j] = voxel_response[poses].mean(axis=0)
    shape = np.unique(info_df['shape'].values)
    rot = np.unique(info_df['rotation'].values)
    curvspace = {'shape': shape, 'rotation': rot, 'space': param_space}
    # 2d save out
    np.save(pjoin(output_path, f'{axis_type}/{tuningname}/{tuningname}_curv-space.npy'), curvspace)

    # natural
    info_df = info_dfs[stim_type.index('natural')]
    voxel_response = voxel_responses[stim_type.index('natural')]
    param_space = voxel_response
    params = [ _.split('/')[-1].replace('.JPEG', '') for _ in info_df['imagename'].values]
    naturalspace = {'patches': params, 'space': param_space}
    # 2d save out
    np.save(pjoin(output_path, f'{axis_type}/{tuningname}/{tuningname}_natural-space.npy'), naturalspace)