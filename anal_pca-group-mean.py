import joblib
import numpy as np
from sklearn.decomposition import PCA
import os
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import zscore
from utils import get_roi_data, save_ciftifile
import matplotlib.pyplot as plt

wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
model_path = os.path.join(wk_dir, 'build/mappings')
subs = [f'sub-0{i+1}' for i in list(range(9))]#+1[3,4,5,8]
# subs.pop(2)
bmaskname = 'subjvis'
layername = 'googlenet-inception3a'
modelname = 'RFmodels'

model_params_inbrain = np.nan*np.zeros((len(subs), 59412, 256))
selected_counts = np.zeros(59412)

for i, sub in enumerate(subs):
    submask = np.zeros(59412)
    submodelpath = pjoin(model_path, f'{sub}/{sub}_bm-{bmaskname}_layer-{layername}_{modelname}.pkl')
    submodel = joblib.load(submodelpath)

    submask[np.array(list(submodel.keys()))] = 1
    selected_counts += submask
    
    submodel_params = np.stack([ model.coef_ for model in submodel.values() ], axis=0)
    model_params_inbrain[i, np.array(list(submodel.keys())), :] = submodel_params

ave_model_params_inbrain = np.nanmean(model_params_inbrain, axis=0)

count_treshold = 1
voxel_indices = np.where(selected_counts>=count_treshold)[0]

# select ROI names
evc_pool = ['V1']#, 'V2', 'V3','V4'
# vtc_pool = ['V8', 'PIT', 'FFC', 'VVC', 'VMV1', 'VMV2', 'VMV3']
# aggregate ROI vertices
roi_name = [__  for _ in [evc_pool] for __ in _]
# form ROI mask
selection_mask = np.sum([get_roi_data(None, _) for _ in roi_name], axis=0)
# trsnfer to indices in cifti space
voxel_indices = np.array([ _ for _ in np.where(selection_mask==1)[0] if _ in voxel_indices])

params = ave_model_params_inbrain[voxel_indices, :]

params = zscore(params)

n_comp = 10
pca = PCA(n_components=10)
pca.fit(params)
trans_params = pca.transform(params)

brainmap = np.nan * np.zeros((10, 91282))
brainmap[:, np.array(voxel_indices)] = trans_params.transpose()

mask_name = 'V1'
save_ciftifile(brainmap, f'./anal/brainmap/sub-average_layer-{layername}_bm-full-{mask_name}_params-components.dtseries.nii')

