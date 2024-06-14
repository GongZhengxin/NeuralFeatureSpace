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
tuning_dir = pjoin(work_dir, 'build/roi-voxelwisemodel')
mean_act_path = pjoin(work_dir, 'prep/roi-concate')

# save out path
output_path = pjoin(work_dir, 'anal/neural-selectivity/parameterspace')
os.makedirs(output_path, exist_ok=True)

# ################################
# 1. 构建 simple attributes param space
# ################################
stim_type = ['180_gabor', '30_color', 'raw_shape']
net_layer = 'Googlenet_conv2' 
layerlabel = net_layer.split('_')[1]
activ_files = sorted([ _ for _ in os.listdir(activ_dir) \
    if f'{net_layer}.npy' in _ and _.replace(f'_{net_layer}.npy', '') in stim_type])
activs = list(map(lambda x: np.load(x, mmap_mode='r'), [pjoin(activ_dir, _) for _ in activ_files]))
print(f'{activ_files}; 刺激实验数据')
for i in range(len(activ_files)):
    print(activs[i].shape)
voxelparams = (1, 0, 0, 3.6, 3.6, 0)
num_channel = activs[0].shape[1]
fieldsize = activs[0].shape[-1]
rfmask = construct_RF(fieldsize, *voxelparams)

# spatial summation
activs = list(map(lambda x: np.sum(activs[x] * rfmask, axis=(2,3)), [0,1,2]))
# select unit which has good gaussian fit by human inspection
full_unit = np.arange(63)
select_unit = [0,2,4,5,8,9,10,11,12,13,14,16,18,19,25,27,28,29,32,33,34,
            37,39,40,41,42,43,46,47,49,50,51,53,54,55,56,57,60,61]
strict_unit = [0,4,5,8,10,12,13,14,16,18,19,27,28,33,37,39,41,42,43,50,53,54,55,56,60]

unit_type = 'full' #'strict'
for sub in [f'sub-0{_+1}' for _ in range(9)]:
    # define params
    roi = 'V1'
    unit_loc = eval(f'{unit_type}_unit')
    tuningname = f'{sub}-conv2-V1-roi'
    voxel_net_layer = net_layer.lower().replace('_', '-')
    coef_path = pjoin(tuning_dir, sub, voxel_net_layer, f'{sub}_{roi}-fullm-coef.npy')
    voxel_path = pjoin(mean_act_path, sub, f"{sub}_layer-{voxel_net_layer}_{roi}-voxel.npy")
    voxel_coefs = np.load(coef_path)
    voxels = np.load(voxel_path)
    vox_num, weights_num = voxel_coefs.shape

    # for idx, voxeltuning in enumerate(voxels):
    # os.makedirs(pjoin(output_path, f'{tuningname}'), exist_ok=True)
    # weightsnum = voxeltuning.shape[-1]
    # _, subidx, layer, roi, _ = tuningname.split('-')
    # sub_featuremean = np.load(pjoin(f"{mean_act_path}/sub-{subidx}/sub-{subidx}_layer-googlenet-{layer}_{roi}-mean-train-feature.npy")).mean(axis=0)
    # sub_featurestd = np.sqrt((np.load(pjoin(f"{mean_act_path}/sub-{subidx}/sub-{subidx}_layer-googlenet-{layer}_{roi}-std-train-feature.npy"))**2).mean(axis=0))
    # voxel_responses = list(map(lambda x: np.sum(normalize_data(x[:,0:weights_num], sub_featuremean, sub_featurestd) * voxeltuning, axis=-1), activs))

    # 载入刺激属性的信息
    stim_type = [ _.replace(f'_{net_layer}.npy', '') for _ in activ_files ]
    info_dir = pjoin(work_dir, 'prep/simplified_stimuli/stim/info')
    info_dfs = []
    for i, key in enumerate(stim_type):
        info_dfs.append(pd.read_csv(pjoin(info_dir, f'{key}.stim.csv')))

    # 创建参数空间矩阵
    # gabor
    stim_type = [ _.replace(f'_{net_layer}.npy', '').split('_')[-1] for _ in activ_files ]

    # get simluated voxel response on simplified stimulus set
    stimulus_dimension = 'frequency'
    stim_sum = ['orientation', 'frequency']
    act = activs[stim_type.index('gabor')][:, unit_loc]
    voxel_response = act @ voxel_coefs[:, unit_loc].transpose(1,0)
    info_df = info_dfs[stim_type.index('gabor')]
    params = np.c_[info_df[stim_sum[0]].values, info_df[stim_sum[1]].values]
    stimulus_unique = np.unique(info_df[stimulus_dimension].values)
    param_space = np.zeros((len(stimulus_unique), voxel_response.shape[1]))
    for i, stimulus in enumerate(stimulus_unique):
        poses = np.where((params[:,stim_sum.index(stimulus_dimension)]==stimulus))[0]
        param_space[i] = voxel_response[poses].mean(axis=0)
    stimulus = np.unique(info_df[stimulus_dimension].values)
    gaborspace = {'stim': stimulus, 'space': param_space}
    # 1f save out
    print(f'Finish {sub}')
    np.save(pjoin(output_path, f'{tuningname}/{tuningname}_{stimulus_dimension}-tuning_unit-{unit_type}.npy'), gaborspace)

# # color
# info_df = info_dfs[stim_type.index('color')]
# voxel_response = voxel_responses[stim_type.index('color')]
# params = np.c_[info_df['hue'].values, info_df['lightness'].values, info_df['chroma'].values]
# hue_length = len(np.unique(info_df['hue'].values))
# light_length = len(np.unique(info_df['lightness'].values))
# chroma_length = len(np.unique(info_df['chroma'].values))
# param_space = np.zeros((hue_length, light_length, chroma_length))
# for i, hue in enumerate(np.unique(info_df['hue'].values)):
#     for j, light in enumerate(np.unique(info_df['lightness'].values)):
#         for k, chroma in enumerate(np.unique(info_df['chroma'].values)):
#             poses = np.intersect1d(np.where((params[:,0]==hue))[0], np.where((params[:,1]==light))[0])
#             poses = np.intersect1d(poses, np.where((params[:,2]==chroma))[0])
#             param_space[i,j,k] = voxel_response[poses].mean(axis=0)
# hue = np.unique(info_df['hue'].values)
# light = np.unique(info_df['lightness'].values)
# chroma = np.unique(info_df['chroma'].values)
# colorspace = {'hue': hue, 'light': light, 'chroma': chroma , 'space': param_space}
# # 2d save out
# np.save(pjoin(output_path, f'{tuningname}/{tuningname}_color-space.npy'), colorspace)

# # curv
# info_df = info_dfs[stim_type.index('shape')]
# voxel_response = voxel_responses[stim_type.index('shape')]
# params = np.c_[info_df['shape'].values, info_df['rotation'].values]
# shape_length = len(np.unique(info_df['shape'].values))
# rot_length = len(np.unique(info_df['rotation'].values))
# param_space = np.nan*np.zeros((shape_length, rot_length))
# for i, shape in enumerate(np.unique(info_df['shape'].values)):
#     for j, rot in enumerate(np.unique(info_df['rotation'].values)):
#         poses = np.intersect1d(np.where((params[:,0]==shape))[0], np.where((params[:,1]==rot))[0])
#         if len(poses) >0: 
#             param_space[i,j] = voxel_response[poses].mean(axis=0)
# shape = np.unique(info_df['shape'].values)
# rot = np.unique(info_df['rotation'].values)
# curvspace = {'shape': shape, 'rotation': rot, 'space': param_space}
# # 2d save out
# np.save(pjoin(output_path, f'{tuningname}/{tuningname}_curv-space.npy'), curvspace)
