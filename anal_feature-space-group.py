import os
import glob
import joblib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.stats import zscore
from os.path import join as pjoin
from sklearn.decomposition import PCA
from utils import get_roi_data, save_ciftifile, conv2_labels, save2cifti

wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
corr_path = os.path.join(wk_dir, 'build/featurewise-corr')
voxel_mask_path = os.path.join(wk_dir, 'prep/voxel_masks/')

subs = [f'sub-0{i+1}' for i in list(range(9))]
bmaskname = 'subjvis'
layername = 'googlenet-conv2'
modelname = 'RFmodels'

mask_name = 'primaryvis-in-MMP'
voxel_mask_nii = nib.load(os.path.join(voxel_mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'))
voxel_mask = np.squeeze(voxel_mask_nii.get_fdata())
voxel_indices = np.where(voxel_mask==1)[0]

# generate specific ROI
v1_voxels = np.array([ _ for _ in np.where(get_roi_data(None, 'V1')==1)[0] if _ in voxel_indices])

v2_voxels = np.array([ _ for _ in  np.where(get_roi_data(None, 'V2')==1)[0] if _ in voxel_indices])

v3_voxels = np.array([ _ for _ in  np.where(get_roi_data(None, 'V3')==1)[0] if _ in voxel_indices])
v4_voxels = np.array([ _ for _ in  np.where(get_roi_data(None, 'V4')==1)[0] if _ in voxel_indices])

sharing_repeats = nib.load(os.path.join(voxel_mask_path, 'nod-voxmask_gmmret-in-subj-repeats.dscalar.nii')).get_fdata()
mask_gmm = nib.load(os.path.join(voxel_mask_path, 'nod-voxmask_gmmret-in-subj.dlabel.nii')).get_fdata()
mask_fix = nib.load(os.path.join(voxel_mask_path, 'nod-voxmask_fixret-in-subj.dlabel.nii')).get_fdata()
# 载入每个被试、每个feature的beta值
indexname = 'fullm-coef'
betas = []
pattern =os.path.join(corr_path, f"sub-*/*{indexname}.npy")
files = glob.glob(pattern)
files = sorted(files)
for file in files:
    beta = np.load(file)
    betas.append(beta)
    # print(file.split('/')[-1], ':', beta.shape)
    # print(file.split('/')[-1], ':', np.nanmin(beta), np.nanmax(beta))
    uniq_vox = np.unique(np.where(np.isnan(beta[:, voxel_indices])==1)[1])
    num_vox = len(np.where(np.isnan(beta[:, voxel_indices])==1)[1])
    average_nan = num_vox/(len(uniq_vox)+1e-18)
    # print('# nan voxel:',num_vox, '; include', uniq_vox, '; average nan per voxel', average_nan)
betas = np.stack(betas, axis=0)
print(f'tell me the shape of {indexname}:', betas.shape)
# 载入全模型显著性
statsmethod = 'FDR'
sigpath = pjoin(wk_dir, 'anal/brainmap/modelsigmap')
fullmodel_sig = nib.load(pjoin(sigpath, f'sub-all_ly-googlenet-conv2_modelSig-{statsmethod}.dlabel.nii')).get_fdata()
# 萃取各被试 roi 内显著的体素
sig_roi = {}
show_roi = v1_voxels
for isub, sub in enumerate(subs):
    sig_voxels = np.where(fullmodel_sig[isub, :]==1)[0]
    sig_roi[sub] = np.array(list(set(show_roi) & set(sig_voxels)))

# 提取出 feature p 值
indexname = 'fullm-p'
pvalues = []
pattern =os.path.join(corr_path, f"sub-*/*{indexname}.npy")
files = glob.glob(pattern)
files = sorted(files)
for file in files:
    pvalue = np.load(file)
    pvalues.append(pvalue)
    # print(file.split('/')[-1], ':', pvalue.shape)
    print(file.split('/')[-1], ':', np.nanmin(pvalue), np.nanmax(pvalue))
    uniq_vox = np.unique(np.where(np.isnan(pvalue[:, voxel_indices])==1)[1])
    num_vox = len(np.where(np.isnan(pvalue[:, voxel_indices])==1)[1])
    average_nan = num_vox/(len(uniq_vox)+1e-18)
    # print('# nan voxel:',num_vox, '; include', uniq_vox, '; average nan per voxel', average_nan)
    print('# significant ratio:',np.sum(pvalue[:, voxel_indices]<0.05)/ pvalue[:, voxel_indices].size)
pvalues = np.stack(pvalues, axis=0)

# feature beta scatter，对每一类特征，挑出不同被试双显著的体素， 查看显著β值的具体状况
subcolors = ['#404e67',  '#809bce',  '#95b8d1',  '#b8e0d4',  '#d6eadf',  '#eac4d5',  '#d1625c',  '#e89c81', '#ffd6a5']
label_dict = conv2_labels
setcolors = plt.cm.jet(np.linspace(0, 1, len(label_dict.keys())))
center_biases = np.linspace(-0.35, 0.35, len(subs))
# 排了个序
abs_beta_max = np.zeros(betas.shape[1])
for index in range(betas.shape[1]):
    y = []
    for isub, sub in enumerate(subs):
        feature_sig_voxel = np.array(list(set(sig_roi[sub]) & set(np.where(pvalues[isub, index, :] < 0.05)[0])))
        y.append(betas[isub, index, feature_sig_voxel])
    y = np.concatenate(y)
    abs_beta_max[index] = np.max(np.abs(y))
range_sorting = np.argsort(abs_beta_max)[::-1]
# 信息量太大，简单summary成 sub x feature，按 set 聚类
subcolors = ['#404e67',  '#809bce',  '#95b8d1',  '#b8e0d4',  '#d6eadf',  '#eac4d5',  '#d1625c',  '#e89c81', '#ffd6a5']
label_dict = conv2_labels
setcolors = plt.cm.jet(np.linspace(0, 1, len(label_dict.keys())))
draw_sorting = []
draw_colors = []
draw_ticks = []
draw_tickslabels = []
for key, indices in label_dict.items():
    iset = list(label_dict.keys()).index(key)
    set_color = setcolors[iset]
    draw_sorting.extend(list(indices))
    draw_colors.extend(len(indices)*[set_color])
    draw_ticks.append(len(indices))
    draw_tickslabels.append(key)
draw_ticks = [0] + list(np.cumsum(draw_ticks))
# 画个大图试试
fig, ax = plt.subplots(1, 1, figsize=(20,5))
sub_feature_mean = np.zeros((len(subs), len(draw_sorting)))
# collect data
for isub, sub in enumerate(subs):
    sub_means = []
    for index in draw_sorting:
        feature_sig_voxel = np.array(list(set(sig_roi[sub]) & set(np.where(pvalues[isub, index, :] < 0.05)[0])))
        sub_means.append(betas[isub, index, feature_sig_voxel].mean())
    sub_feature_mean[isub,:] = np.array(sub_means)

# plots
for isub, sub in enumerate(subs):
    ax.plot(sub_feature_mean[isub], color=subcolors[isub], lw=3, alpha=0.3)
    ax.scatter(np.arange(len(draw_sorting)), sub_feature_mean[isub], color=draw_colors, s=60, alpha=0.5, edgecolors='white', zorder=4)
ax.plot(sub_feature_mean.mean(axis=0), color='black', lw=5,zorder=4)
ax.scatter(np.arange(len(draw_sorting)), sub_feature_mean.mean(axis=0), color=draw_colors, s=60, edgecolors='black', zorder=5)
ax.set_xlim([-0.5, 62.5])
ax.axhline(y=0, color='gray', ls='--')

from scipy import stats
_, p_value = stats.ttest_1samp(sub_feature_mean, 0)
for pos in np.where(np.logical_and(p_value < 0.05, p_value >= 0.01)==1)[0]:
    infotxt = '*'
    ax.text(pos+0.08, sub_feature_mean.mean(axis=0)[pos]+0.01, infotxt, fontsize=16, color='red',
    horizontalalignment='left', verticalalignment='center', zorder=7)
for pos in np.where(np.logical_and(p_value < 0.01, p_value >= 0.001)==1)[0]:
    infotxt = '**'
    ax.text(pos+0.08, sub_feature_mean.mean(axis=0)[pos]+0.01, infotxt, fontsize=16, color='red',
    horizontalalignment='left', verticalalignment='center',zorder=7)
for pos in np.where(p_value < 0.001)[0]:
    infotxt = '***'
    ax.text(pos+0.08, sub_feature_mean.mean(axis=0)[pos]+0.01, infotxt, fontsize=12, color='red',
    horizontalalignment='left', verticalalignment='center',zorder=7)
ax.set_xticks(draw_ticks[0:-1])
ax.set_xticklabels(draw_tickslabels)

methodname = 'double-significance'
fig_path = pjoin(wk_dir, 'vis/group-feature-sig')
plt.savefig(pjoin(fig_path, f'Group_{methodname}-feature-space-component.tiff'), dpi=300, bbox_inches='tight')