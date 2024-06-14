import os
from os.path import join as pjoin
import numpy as np
import pandas as pd

work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
# 准备好激活值
activ_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/simplified_stimuli/activation'
stim_type =  ['30_color', '180_gabor', 'shape', 'pinknoise'] #['color', 'gabor', 'shape']
net_layer = 'Googlenet_conv2'#'Googlenet_maxpool2' 
activ_files = sorted([ _ for _ in os.listdir(activ_dir) \
    if f'{net_layer}.npy' in _ and _.replace(f'_{net_layer}.npy', '') in stim_type])
activs = list(map(lambda x: np.load(x, mmap_mode='r'), [pjoin(activ_dir, _) for _ in activ_files]))
print(activ_files)

# Define the 2D Gaussian function
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

metric = 'rfmask'# 'max'
fieldsize = activs[0].shape[-1]
num_of_units =  activs[0].shape[1]

if metric == 'rfmask':
    A, x, y, sigma_x, sigma_y, C = 1, 0, 0, 1, 1, 0
    i = np.linspace(-8., 8., fieldsize)
    j = np.linspace(8., -8., fieldsize)
    i, j = np.meshgrid(i, j)
    rfmask = adjust_RF(gaussian_2d((i, j), A, x, y, sigma_x, sigma_y, C))

sampled_activs = {}
print('提取感受野采样的激活值')
stim_type = [ _.replace(f'_{net_layer}.npy','').split('_')[-1] for _ in activ_files ]
for i, key in enumerate(stim_type):
    if metric == 'rfmask':
        sampled_activs[key] = np.sum( activs[i] * rfmask, axis=(2,3))
    else:
        sampled_activs[key] = np.max( activs[i][:, :, fieldsize//4 : 3*fieldsize//4, fieldsize//4 : 3*fieldsize//4], axis=(2,3))
        # sampled_activs[key] = np.max( activs[i][:, :, fieldsize//6 : 5*fieldsize//6, fieldsize//6 : 5*fieldsize//6], axis=(2,3))
for i, key in enumerate(stim_type):
    print(key, sampled_activs[key].shape)

# 计算神经元的 d prime
def calc_dprime(f1,f2,f3):
    mean, std = np.mean(sampled_activs[f1], axis=0), np.std(sampled_activs[f1], axis=0)
    mean_other = np.mean(np.r_[sampled_activs[f2], sampled_activs[f3]], axis=0)
    std_other = np.std(np.r_[sampled_activs[f2], sampled_activs[f3]], axis=0)
    return (mean - mean_other) / np.sqrt(0.5*(std**2 + std_other**2))

def calc_dprimevsnoise(f1):
    mean, std = np.mean(sampled_activs[f1], axis=0), np.std(sampled_activs[f1], axis=0)
    mean_other = np.mean(sampled_activs['pinknoise'], axis=0)
    std_other = np.std(sampled_activs['pinknoise'], axis=0)
    return (mean - mean_other) / np.sqrt(0.5*(std**2 + std_other**2))

# dprimes
dprim_color = calc_dprime('color', 'gabor', 'shape')
dprim_shape = calc_dprime('shape', 'gabor', 'color')
dprim_gabor = calc_dprime('gabor', 'shape', 'color')

# d-prime versus pink noise
dprim_colorvp = calc_dprimevsnoise('color')
dprim_gaborvp = calc_dprimevsnoise('gabor')
dprim_shapevp = calc_dprimevsnoise('shape')

# 计算神经元的 selectivity
def calc_selectivity(f1,f2,f3):
    mean = np.mean(sampled_activs[f1], axis=0)
    mean_other = np.mean(np.r_[sampled_activs[f2], sampled_activs[f3]], axis=0)

    return (mean - mean_other) / (mean + mean_other) 

def calc_maxselectivity(f1,f2,f3):
    mean = np.max(sampled_activs[f1], axis=0)
    mean_other = np.mean(np.c_[np.max(sampled_activs[f2], axis=0), np.max(sampled_activs[f3], axis=0)], axis=1)

    return (mean - mean_other) / (mean + mean_other)

color_selectivity = calc_selectivity('color', 'gabor', 'shape')
shape_selectivity = calc_selectivity('shape', 'gabor', 'color')
gabor_selectivity = calc_selectivity('gabor', 'color', 'shape')

color_selectivityonmax = calc_maxselectivity('color', 'gabor', 'shape')
shape_selectivityonmax = calc_maxselectivity('shape', 'gabor', 'color')
gabor_selectivityonmax = calc_maxselectivity('gabor', 'color', 'shape')

layerlabel = net_layer.split('_')[1]
unitdata = {
    "unit" : np.arange(num_of_units),
    "color-d": dprim_color,
    "shape-d": dprim_shape,
    "gabor-d": dprim_gabor,
    'color-sls': color_selectivity,
    'shape-sls': shape_selectivity,
    'gabor-sls': gabor_selectivity,
    'color-sls-max': color_selectivityonmax,
    'shape-sls-max': shape_selectivityonmax,
    'gabor-sls-max': gabor_selectivityonmax,
    "color-vp-d": dprim_colorvp,
    "shape-vp-d": dprim_shapevp,
    "gabor-vp-d": dprim_gaborvp,
    'colormax': np.max(sampled_activs['color'], axis=0),
    'colormin': np.min(sampled_activs['color'], axis=0),
    'gabormax': np.max(sampled_activs['gabor'], axis=0),
    'gabormin': np.min(sampled_activs['gabor'], axis=0),
    'shapemax': np.max(sampled_activs['shape'], axis=0),
    'shapemin': np.min(sampled_activs['shape'], axis=0),
}
df = pd.DataFrame(unitdata)
if metric == 'rfmask':
    np.save(pjoin(work_dir, f'anal/unit-selectivity/unit-respond2stimset_rfmask-size{sigma_x}.npy'), sampled_activs)
    df.to_csv(pjoin(work_dir, f'anal/unit-selectivity/unit-info-{layerlabel}_rfmask-size{sigma_x}.csv'), index=False)
else:
    np.save(pjoin(work_dir, f'anal/unit-selectivity/unit-respond2stimset_{metric}.npy'), sampled_activs)
    df.to_csv(pjoin(work_dir, f'anal/unit-selectivity/unit-info-{layerlabel}_{metric}.csv'), index=False)