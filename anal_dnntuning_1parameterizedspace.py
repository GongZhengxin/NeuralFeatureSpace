from PIL import Image
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

def construct_RF(fieldsize):
    # 0 创建假想体素感受野
    A, x, y, sigma_x, sigma_y, C = 1, 0, 0, 2, 2, 0
    i = np.linspace(-8., 8., fieldsize)
    j = np.linspace(8., -8., fieldsize)
    i, j = np.meshgrid(i, j)
    return adjust_RF(gaussian_2d((i, j), A, x, y, sigma_x, sigma_y, C))

# path settings
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
# input path
activ_dir = pjoin(work_dir, 'prep/simplified_stimuli/activation')
# save out path
output_path = pjoin(work_dir, 'anal/unit-selectivity')
net_layer = 'Googlenet_conv2' 
# # ################################
# # 1. 构建 gabor param space
# # ################################

# # 1a 准备好gabor激活值, 高低空间频率都载入
stim_type = ['180_gabor'] #['gabor', 'low_gabor']
layerlabel = net_layer.split('_')[1]
activ_files = sorted([ _ for _ in os.listdir(activ_dir) \
    if f'{net_layer}.npy' in _ and _.replace(f'_{net_layer}.npy', '') in stim_type])
activs = list(map(lambda x: np.load(x, mmap_mode='r'), [pjoin(activ_dir, _) for _ in activ_files]))
print(f'{activ_files}; 刺激实验数据')
for i in range(len(activ_files)):
    print(activs[i].shape)  
num_channel = activs[0].shape[1]

fieldsize = activs[0].shape[-1]
activation = np.max(activs[0][:, :, fieldsize//4 : 3*fieldsize//4, fieldsize//4 : 3*fieldsize//4], axis=(2,3))

# rfmask = construct_RF(activs[0].shape[-1])
# activation = np.sum(activs[0] * rfmask, axis=(2,3))

if len(activ_files) >1:
    # 1b 将感受野内的激活进行加权并拼接
    sampled_activs = {}
    print('提取感受野采样的激活值')
    stim_type = [_.replace(f'_{net_layer}.npy', '') for _ in activ_files ]
    for i, key in enumerate(stim_type):
        sampled_activs[key] = np.sum( activs[i] * rfmask, axis=(2,3))
    for i, key in enumerate(stim_type):
        print(key, sampled_activs[key].shape)

    activation = np.r_[sampled_activs[stim_type[0]], sampled_activs[stim_type[1]]]

# 1c 载入刺激属性的信息
info_dir = pjoin(work_dir, 'prep/simplified_stimuli/stim/info')
info_df = []
for i, key in enumerate(stim_type):
    info_df.append(pd.read_csv(pjoin(info_dir, f'{key}.stim.csv')))
info_df = pd.concat(info_df, ignore_index=True)

# 1d 创建参数空间矩阵
params = np.c_[info_df['orientation'].values, info_df['frequency'].values]
orientation_length = len(np.unique(info_df['orientation'].values))
frequency_length = len(np.unique(info_df['frequency'].values))
param_space = np.zeros((orientation_length, frequency_length, num_channel))
for i, orientation in enumerate(np.unique(info_df['orientation'].values)):
    for j, frequency in enumerate(np.unique(info_df['frequency'].values)):
        poses = np.intersect1d(np.where((params[:,0]==orientation))[0], np.where((params[:,1]==frequency))[0])
        param_space[i,j,:] = activation[poses].mean(axis=0)
orientation = np.unique(info_df['orientation'].values)
frequency = np.unique(info_df['frequency'].values)
gaborspace = {'ori': orientation, 'freq': frequency, 'space': param_space}
# 1f save out
np.save(pjoin(output_path, f'{net_layer}_180gabor-space.npy'), gaborspace)

# ################################
# 2. 构建color space
# ################################

# 2a 载入并形成激活数据
stim_type = ['30_color']
net_layer = 'Googlenet_conv2' 
layerlabel = net_layer.split('_')[1]
activ_files = sorted([ _ for _ in os.listdir(activ_dir) \
    if f'{net_layer}.npy' in _ and _.replace(f'_{net_layer}.npy', '') in stim_type])
activs = list(map(lambda x: np.load(x, mmap_mode='r'), [pjoin(activ_dir, _) for _ in activ_files]))
print(f'{activ_files}; 刺激实验数据')
print(activs[0].shape)  
num_channel = activs[0].shape[1]

fieldsize = activs[0].shape[-1]
activation = np.max(activs[0][:, :, fieldsize//4 : 3*fieldsize//4, fieldsize//4 : 3*fieldsize//4], axis=(2,3))


# rfmask = construct_RF(activs[0].shape[-1])
# activation = np.sum(activs[0] * rfmask, axis=(2,3))

# 2b 载入刺激信息
info_dir = pjoin(work_dir, 'prep/simplified_stimuli/stim/info')
info_df = []
for i, key in enumerate(stim_type):
    info_df.append(pd.read_csv(pjoin(info_dir, f'{key}.stim.csv')))
info_df = pd.concat(info_df, ignore_index=True)

# 2c 创建参数空间矩阵
params = np.c_[info_df['hue'].values, info_df['lightness'].values, info_df['chroma'].values]
hue_length = len(np.unique(info_df['hue'].values))
light_length = len(np.unique(info_df['lightness'].values))
chroma_length = len(np.unique(info_df['chroma'].values))
param_space = np.zeros((hue_length, light_length, chroma_length, num_channel))
for i, hue in enumerate(np.unique(info_df['hue'].values)):
    for j, light in enumerate(np.unique(info_df['lightness'].values)):
        for k, chroma in enumerate(np.unique(info_df['chroma'].values)):
            poses = np.intersect1d(np.where((params[:,0]==hue))[0], np.where((params[:,1]==light))[0])
            poses = np.intersect1d(poses, np.where((params[:,2]==chroma))[0])
            param_space[i,j,k,:] = activation[poses].mean(axis=0)
hue = np.unique(info_df['hue'].values)
light = np.unique(info_df['lightness'].values)
chroma = np.unique(info_df['chroma'].values)
colorspace = {'hue': hue, 'light': light, 'chroma': chroma , 'space': param_space}
# 2d save out
filemarker = stim_type[0].replace('_', '')
np.save(pjoin(output_path, f'{net_layer}_{filemarker}-space.npy'), colorspace)


# ################################
# 3. 构建 shape param space
# ################################
activ_dir = pjoin(work_dir, 'prep/simplified_stimuli/activation')
# 3a 载入并形成激活数据
stim_type = ['shape']
net_layer = 'Googlenet_conv2' 
layerlabel = net_layer.split('_')[1]
activ_files = sorted([ _ for _ in os.listdir(activ_dir) \
    if f'{net_layer}.npy' in _ and _.replace(f'_{net_layer}.npy', '') in stim_type])
activs = list(map(lambda x: np.load(x, mmap_mode='r'), [pjoin(activ_dir, _) for _ in activ_files]))
print(f'{activ_files}; 刺激实验数据')
print(activs[0].shape)  
num_channel = activs[0].shape[1]

fieldsize = activs[0].shape[-1]
activation = np.max(activs[0][:, :, fieldsize//4 : 3*fieldsize//4, fieldsize//4 : 3*fieldsize//4], axis=(2,3))


# rfmask = construct_RF(activs[0].shape[-1])
# activation = np.sum(activs[0] * rfmask, axis=(2,3))

# 3b 载入刺激信息
info_dir = pjoin(work_dir, 'prep/simplified_stimuli/stim/info')
info_df = []
for i, key in enumerate(stim_type):
    info_df.append(pd.read_csv(pjoin(info_dir, f'{key}.stim.csv')))
info_df = pd.concat(info_df, ignore_index=True)

# 3c 创建参数空间矩阵
params = np.c_[info_df['shape'].values, info_df['rotation'].values]
shape_length = len(np.unique(info_df['shape'].values))
rot_length = len(np.unique(info_df['rotation'].values))
param_space = np.nan*np.zeros((shape_length, rot_length, num_channel))

for i, shape in enumerate(np.unique(info_df['shape'].values)):
    for j, rot in enumerate(np.unique(info_df['rotation'].values)):
        poses = np.intersect1d(np.where((params[:,0]==shape))[0], np.where((params[:,1]==rot))[0])
        if len(poses) >0: 
            param_space[i,j,:] = activation[poses].mean(axis=0)
shape = np.unique(info_df['shape'].values)
rot = np.unique(info_df['rotation'].values)

curvspace = {'shape': shape, 'rotation': rot, 'space': param_space}
# 2d save out
np.save(pjoin(output_path, f'{net_layer}_curv-space.npy'), curvspace)


# ################################
# 3. 构建 natural patch activations， 并取出 原始图像中的patch
# ################################
# activs = np.load(pjoin(work_dir, 'prep/image_activations/concate-activs/all-sub_googlenet-conv2_activation.npy'), mmap_mode='r')
preprocessed_path = pjoin(work_dir, 'prep/image_activations_preprocessed')
print("数据加载完成")
fieldsize = 57
rfmask = construct_RF(fieldsize)
imagesize = 227
imagemask = construct_RF(imagesize)
net_layer = 'Googlenet_conv2'
imagename_path = f'{work_dir}/prep/image_names'
files = sorted([f for f in os.listdir(imagename_path) if f.startswith('sub-') and f.endswith('imagenet.csv')])[:9]
 
subs = [f.split('-')[1].split('.')[0] for f in files]

units_activations = []
patches = []

for file, sub in zip(files, subs):
    file_path = os.path.join(imagename_path, file)
    data = pd.read_csv(file_path, header=None)
    image_paths = data[0].tolist()
    print(f"被试{sub}刺激数据读取完成")
    # subidx = int(sub[:2]) - 1
    # sub_activ = activs[4000*subidx:4000*(subidx+1)]
    sub_activ = np.load(pjoin(preprocessed_path, f'sub-{sub[:2]}_googlenet-conv2.npy'), mmap_mode='r')
    mask = rfmask > 0 
    sub_units_activations = np.max(sub_activ * mask, axis=(2,3))
    units_activations.append(sub_units_activations)
    print(f"被试{sub}单元激活存储完成, max:{np.max(sub_units_activations, axis=0)}")

    inputimages = []
    for image_path in image_paths:
        full_image_path = os.path.join(imagename_path, image_path)
        try:
            image = Image.open(full_image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            new_size = (227, 227)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            inputimages.append(np.array(image))
            # print(f"被试{sub}的输入刺激图片数据存在")
        except IOError:
            print(f"Error opening image {full_image_path}")

    if inputimages:
        mask = imagemask > 0
        inputimages = np.stack(inputimages)
        inputimages[:,mask==0,:] = [128, 128, 128]
        center_x, center_y = 227/2, 227/2
        radius = np.max(mask.sum(axis=0))/2
        rect_bound1 = int(center_x - radius) - 5
        rect_bound2 = int(center_x + radius) + 5
        sub_circle_patches = inputimages[:,rect_bound1:rect_bound2, rect_bound1:rect_bound2, :]
        patches.append(sub_circle_patches)
        print(f"被试{sub}的激活patches添加完成")

# save
np.save(pjoin(output_path, f'Mingz/{net_layer}_naturalpatches.npy'), np.concatenate(patches, axis=0))
print("patch存储完成")
np.save(pjoin(output_path, f'Mingz/{net_layer}_naturalpatches_units_activations.npy'), np.concatenate(units_activations, axis=0))
print("刺激单元激活存储完成")

# natural
info_dir = pjoin(work_dir, 'prep/simplified_stimuli/stim/info')
units_activations = np.load(pjoin(output_path, f'{net_layer}_naturalpatches_units_activations.npy'), mmap_mode='r')
info_df = pd.read_csv(pjoin(info_dir, 'natural.stim.csv'))

params = [ _.split('/')[-1].replace('.JPEG', '') for _ in info_df['imagename'].values]
params = np.array([int(param.split('_')[0].replace('n','')) for param in params])
param_space = units_activations
naturalspace = {'patches': params, 'space': param_space}
# 2d save out
np.save(pjoin(output_path, f'Mingz/{net_layer}_natural-space.npy'), naturalspace)
############
# back up plan
############
# rfmask = rfmask[None, None, :, :]  
# weighted_activs = activs * rfmask 
# print("权重激活计算完成")

# max_activation_values = np.max(weighted_activs, axis=(2, 3))
# indices = np.unravel_index(np.argmax(weighted_activs, axis=(2, 3)), (57, 57))
# scale_factor = 227 / fieldsize
# mapped_coordinates = np.stack(indices, axis=-1) * scale_factor
# print("映射坐标计算完成")
# patch_size = 5
# half_patch = patch_size // 2
# mapped_patches = []
# for i in range(mapped_coordinates.shape[0]):
#     y, x = int(mapped_coordinates[i, 0]), int(mapped_coordinates[i, 1])
#     y1 = max(y - half_patch, 0)
#     y2 = min(y + half_patch + 1, 227)
#     x1 = max(x - half_patch, 0)
#     x2 = min(x + half_patch + 1, 227)
#     patch = input_image[y1:y2, x1:x2, :]
#     mapped_patches.append(patch)
# mapped_patches = np.array(mapped_patches)
# print("映射patch计算完成")
# np.save(pjoin(output_path, f'{net_layer}_naturalpatches.npy'), mapped_patches)
 

# # ################
# # supplementary
# # ################
# activ_dir = pjoin(work_dir, 'prep/simplified_stimuli/activation')
# # 3a 载入并形成激活数据
# stim_type = ['raw_shape']
# net_layer = 'Googlenet_conv2' 
# layerlabel = net_layer.split('_')[1]
# activ_files = sorted([ _ for _ in os.listdir(activ_dir) \
#     if f'{net_layer}.npy' in _ and _.replace(f'_{net_layer}.npy', '') in stim_type])
# activs = list(map(lambda x: np.load(x, mmap_mode='r'), [pjoin(activ_dir, _) for _ in activ_files]))
# print(f'{activ_files}; 刺激实验数据')
# print(activs[0].shape)  
# num_channel = activs[0].shape[1]
# activation = np.max(activs[0], axis=(2,3))

# # 3b 载入刺激信息
# info_dir = pjoin(work_dir, 'prep/simplified_stimuli/stim/info')
# info_df = []
# for i, key in enumerate(stim_type):
#     info_df.append(pd.read_csv(pjoin(info_dir, f'{key}.stim.csv')))
# info_df = pd.concat(info_df, ignore_index=True)

# # 3c 创建参数空间矩阵
# params = np.c_[info_df['shape'].values, info_df['rotation'].values]
# shape_length = len(np.unique(info_df['shape'].values))
# rot_length = len(np.unique(info_df['rotation'].values))
# param_space = np.nan*np.zeros((shape_length, rot_length, num_channel))

# for i, shape in enumerate(np.unique(info_df['shape'].values)):
#     for j, rot in enumerate(np.unique(info_df['rotation'].values)):
#         poses = np.intersect1d(np.where((params[:,0]==shape))[0], np.where((params[:,1]==rot))[0])
#         if len(poses) >0: 
#             param_space[i,j,:] = activation[poses].mean(axis=0)
# shape = np.unique(info_df['shape'].values)
# rot = np.unique(info_df['rotation'].values)

# curvspace = {'shape': shape, 'rotation': rot, 'space': param_space}
# # 2d save out
# np.save(pjoin(output_path, f'{net_layer}_curv-space-maxact.npy'), curvspace)

# # plot 2d tuning
# valid2 = lambda x : np.round(x*100)/100
# unit = 0
# plt.style.use('default')
# plt.imshow(gaborspace['space'][0:15,:,unit])
# plt.xticks(np.arange(18), valid2(227 / 16 * gaborspace['freq']), rotation=45)
# plt.yticks(np.arange(15), gaborspace['ori'][0:15])
# plt.title(f'unit-{unit}')
# plt.colorbar(shrink=0.90)
# plt.show()