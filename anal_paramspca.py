import joblib
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
from scipy import stats
from scipy.stats import zscore
from sklearn.decomposition import PCA
import nibabel as nib

def save_ciftifile(data, filename, template='./template.dtseries.nii'):
    ex_cii = nib.load(template)
    if data.ndim == 1:
      data = data[None,:]
    ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    nib.save(nib.Cifti2Image(data,ex_cii.header), filename)

wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'

sub = 'sub-03'
layername = 'features0'
maskname = 'primvis' #'subjvis' 
modelname = 'RF' #'RF' 
model_path = os.path.join(wk_dir, 'build/mappings') #'control/controlmappings'

# ==================================================
# this is the selection procedure via subject atlas
# ==================================================
opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
# getting retinotopic voxels
sub_prf_file = os.path.join(opendata_cifti_pth, sub, 'results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii')
retino_r2 = nib.load(sub_prf_file).get_fdata()[3, :]
r2_thres = 9
retino_voxels = np.where(retino_r2 > r2_thres)[0]
# getting floc voxels
sub_floc_path = os.path.join(opendata_cifti_pth, sub, 'results/ses-floc_task-floc')
altases_files = [f'floc-{_}.dlabel.nii' for _ in ['bodies', 'faces', 'places', 'words']]
floc_atlas = None
for atlas_file in altases_files:
    atlas = nib.load(os.path.join(sub_floc_path, atlas_file)).get_fdata()[0,:]
    if floc_atlas is None:
        floc_atlas = atlas
    else:
        floc_atlas += atlas
floc_voxels = np.where(floc_atlas != 0)[0]
# merging into one mask
voxel_indices = np.array([ _ for _ in range(59412) if (_ in floc_voxels) or (_ in retino_voxels)])


model_file = f'{sub}_bm-{maskname}_layer-{layername}_{modelname}models.pkl' #'sub-03_layer-features3_RFmodels.pkl' # 

model = joblib.load(os.path.join(model_path, sub, model_file))

voxels = [ _ for _ in list(model.keys()) if _ in voxel_indices]
param_mask = np.array([ _ in voxel_indices for _ in list(model.keys())])
# # aply mask
# r = np.load(os.path.join(perform_path, f'{sub}/{sub}_layer-{layername}_corrperformance-test.npy'))

params = np.array([_.coef_ for _ in model.values()])[np.where(param_mask==1)[0],:]

params = zscore(params)

n_comp = 10
pca = PCA(n_components=10)
pca.fit(params)
trans_params = pca.transform(params)

brainmap = np.nan * np.zeros((10, 91282))
brainmap[:, np.array(voxels)] = trans_params.transpose()

save_ciftifile(brainmap, f'./anal/brainmap/{sub}_layer-{layername}_params_vis-components.dtseries.nii')

# length = math.comb(4,2)
# feature_diff = np.zeros()
# pca.components_


# 设置雷达图的角度
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()
angles += angles[:1]
colors = ['#363062' , '#FF5B22' , '#0174BE','#C70039']
names = ['pc1', 'pc2', 'pc3', 'pc4']

# 多边形边数
n_sides = 12  # 例如六边形

# 计算多边形的角度
frame_angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False).tolist()
frame_angles += frame_angles[:1]

# 绘图
fig, ax = plt.subplots(4, 4, figsize=(16, 16), subplot_kw=dict(polar=True))

# 绘制正n边形外框

# 设置雷达图的角度
angles = np.linspace(0, 2 * np.pi, 192, endpoint=False).tolist()
angles += angles[:1]
colors = ['#363062' , '#FF5B22' , '#0174BE','#C70039']
names = ['pc1', 'pc2', 'pc3', 'pc4']

# 多边形边数
n_sides = 192  # 例如六边形

# 计算多边形的角度
frame_angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False).tolist()
frame_angles += frame_angles[:1]

# 绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制正n边形外框

# 绘制每个向量
for angle in frame_angles:
    ax.plot([angle, angle], [-0.2, 0.2], color='gray', linestyle='--', linewidth=1)
for data, color, label in zip([pca.components_[_,:] for _ in range(4)], colors, names):
    data = np.concatenate((data, [data[0]]))
    if label in ['pc2','pc4']:
         alpha = 0
    else:
        alpha = 1
    ax.plot(angles, data, color=color, linewidth=2, label=label, alpha=alpha, marker='o')
# 隐藏刻度
ax.set_xticklabels([])
ax.set_yticklabels([])

# # 添加图例
# ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
# for angle in frame_angles:
#           ax[i][j].plot([angle, angle], [-0.2, 0.2], color='gray', linestyle='--', linewidth=2)
# 展示图表
plt.show()# 绘制每个向量
for i in range(4):
    for j in range(4):
      for angle in frame_angles:
          ax[i][j].plot([angle, angle], [-0.2, 0.2], color='gray', linestyle='--', linewidth=2)
      for data, color, label in zip([pca.components_[_,(0+4*i+4*j):(12+4*i+4*j)] for _ in range(4)], colors, names):
          data = np.concatenate((data, [data[0]]))
          ax[i][j].plot(angles, data, color=color, linewidth=2, label=label, alpha=0.9, marker='o')
          # 隐藏刻度
          ax[i][j].set_xticklabels([])
          ax[i][j].set_yticklabels([])

# # 添加图例
# ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
# for angle in frame_angles:
#           ax[i][j].plot([angle, angle], [-0.2, 0.2], color='gray', linestyle='--', linewidth=2)
# 展示图表
plt.show()




maxmindiff = []
for i in range(192):
    maxmindiff.append(np.max(pca.components_[:,i]) - np.min((pca.components_[:,i])))