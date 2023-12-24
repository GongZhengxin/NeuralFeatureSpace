import joblib
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import nibabel as nib

from utils import get_roi_data

def save_ciftifile(data, filename, template='./template.dtseries.nii'):
    ex_cii = nib.load(template)
    if data.ndim == 1:
      data = data[None,:]
    ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    nib.save(nib.Cifti2Image(data,ex_cii.header), filename)

wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
sub = 'sub-03'
layername = 'features3'
model_path = os.path.join(wk_dir, 'build/mappings')
retino_path = os.path.join(wk_dir, 'build/retinoparams')

# get voxel indices
evc_pool = ['V1', 'V2', 'V3','V4']
# aggregate ROI vertices
roi_name = [__  for _ in [evc_pool] for __ in _]
# form ROI mask
selection_mask = np.sum([get_roi_data(None, _) for _ in roi_name], axis=0)
# trsnfer to indices in cifti space
wbvoxels = np.where(selection_mask==1)[0]

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

voxels = [ _ for _ in wbvoxels if _ in voxel_indices]
params_mask = np.array([ _ in voxel_indices for _ in wbvoxels])

fold = 'fold1'
# max position model
model_file = f'{sub}_layer-{layername}_models_{fold}.pkl'
model = joblib.load(os.path.join(model_path, sub, model_file))

# corr map
corr_path = os.path.join(wk_dir, 'retest_corrmap/sub-03')
corrmap = np.load(os.path.join(corr_path, f'sub-03_layer-features3_corrmap-test_{fold}.npy'))

find_pos = lambda x: (int(np.where(x==np.max(x))[0]), int(np.where(x==np.max(x))[1]))

poses = [find_pos(corrmap[:, :, _]) for _ in range(corrmap.shape[-1])]

params = np.array([model[_].coef_[idx, :] for idx, _ in enumerate(poses)])[np.where(params_mask==1)[0]]

params = zscore(params)

pca = PCA(n_components=10)
pca.fit(params)
trans_params = pca.transform(params)

brainmap = np.nan * np.zeros((10, 91282))
brainmap[:, np.array(voxels)] = trans_params.transpose()

save_ciftifile(brainmap, f'./anal/brainmap/{sub}_layer-{layername}_maxmodel{fold}-params_components.dtseries.nii')

#
# full_pca.fit(params)
# plt.plot(full_pca.explained_variance_ratio_[0:50], marker='o', ms=5, color='black')
# plt.xlabel('component')
# plt.ylabel('variance ratio')