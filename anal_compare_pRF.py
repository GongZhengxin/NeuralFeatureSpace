import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt 
from utils import get_roi_data

subs = []
sub = 'sub-02'
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
retinoparam_pth = os.path.join(work_dir, 'build/retinoparams')
mask_pth = os.path.join(work_dir, 'build/voxel_masks')

voxel_mask = nib.load(os.path.join(mask_pth, 'nod_voxel_selection.dlabel.nii')).get_fdata()
subject_mask = voxel_mask[1,:]
subject_voxels = np.where(subject_mask==1)[0]
# select ROI names
evc_pool = ['V1', 'V2', 'V3','V4']
# aggregate ROI vertices
roi_name = [__  for _ in [evc_pool] for __ in _]
# form ROI mask
selection_mask = np.sum([get_roi_data(None, _) for _ in roi_name], axis=0)
# trsnfer to indices in cifti space
voxel_indices = np.where(selection_mask==1)[0]
# dislogde the primary visual vertices
subject_voxels = [ _ for _ in voxel_indices if _ in subject_voxels]

# getting retinotopic voxels
sub_prf_file = os.path.join(opendata_cifti_pth, sub, 'results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii')

exp_prf = nib.load(sub_prf_file).get_fdata()
n_vertices = exp_prf.shape[-1]
prf_params = np.zeros((n_vertices, 3))
prf_params[:,0] = exp_prf[0, 0:n_vertices]
prf_params[:,1] = exp_prf[1, 0:n_vertices]*16/200
prf_params[:,2] = exp_prf[2, 0:n_vertices]*16/200
# we need to transfer the params into (x,y,size) model
prf_params_xy = np.zeros_like(prf_params)
prf_params_xy[:,0] = np.cos(prf_params[:,0]/180*np.pi)*prf_params[:,1]
prf_params_xy[:,1] = np.sin(prf_params[:,0]/180*np.pi)*prf_params[:,1]
prf_params_xy[:,2] = prf_params[:,2]


dnn_prf = np.load(os.path.join(retinoparam_pth, 'sub-02_layer-features3_params.npy'), allow_pickle=True)[0]
dprf_params = np.zeros((n_vertices, 3))
dprf_params[:,0] = 90 - dnn_prf['ang']
dprf_params[:,1] = dnn_prf['ecc']
dprf_params[:,2] = dnn_prf['rfsize']
# we need to transfer the params into (x,y,size) model
dprf_params_xy = np.zeros_like(dprf_params)
dprf_params_xy[:,0] = np.cos(dprf_params[:,0]/180*np.pi)*dprf_params[:,1]
dprf_params_xy[:,1] = np.sin(dprf_params[:,0]/180*np.pi)*dprf_params[:,1]
dprf_params_xy[:,2] = dprf_params[:,2]



plt.scatter(dprf_params_xy[subject_voxels,0], dprf_params_xy[subject_voxels,1])


