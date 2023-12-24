import glob
import os
from os.path import join as pjoin
import nibabel as nib
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import get_roi_data, save2cifti, solve_GMM_eq_point

wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'

mask_path = pjoin(wk_dir, 'prep/voxel_masks')

bfiletemplate = nib.load(pjoin(wk_dir, 'template.dtseries.nii'))
brain_models = bfiletemplate.header.get_index_map(1).brain_models
surface_brain_models = [bm for bm in brain_models if bm.model_type == 'CIFTI_MODEL_TYPE_SURFACE']
# ==================================================
# this is the selection procedure via standard atlas
# ==================================================
# select ROI names
evc_pool = ['V1', 'V2', 'V3','V4']
# vtc_pool = ['V8', 'PIT', 'FFC', 'VVC', 'VMV1', 'VMV2', 'VMV3']
# aggregate ROI vertices
roi_name = [__  for _ in [evc_pool] for __ in _]
# form ROI mask
selection_mask = np.sum([get_roi_data(None, _) for _ in roi_name], axis=0)
# trsnfer to indices in cifti space
voxel_indices = np.where(selection_mask==1)[0]

mask_name = 'primaryvis-in-MMP'

mapnames = []
label_tables = []
labels_name = {0: 'notselected', 1:'selected'}
labels_rgba = {0: (0,0,0,0),
          1: (.5,.5,.5,1)}
voxel_masks = np.zeros((1, 59412))
voxel_masks[-1,np.array(list(voxel_indices))] = 1
mapnames.append('primaryvis')
lbl_tb = nib.cifti2.Cifti2LabelTable()
for key in np.unique(voxel_masks):
    key = int(key)
    lbl_tb[key] = nib.cifti2.Cifti2Label(key, labels_name[key], *labels_rgba[key])
label_tables.append(lbl_tb)
save2cifti(pjoin(mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'),  
           voxel_masks, surface_brain_models, map_names=mapnames, label_tables=label_tables)


# ==================================================
# this is the selection procedure via subject atlas
# ==================================================

mask_name = 'gmmret-in-subj' #'fixret-in-subj'   'gmmretfloc-in-subj''fixretfloc-in-subj'

# def subj_selection_procedure(mask_name):

opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'

subs = [f'sub-0{i+1}' for i in list(range(9))]

subs_voxel_masks = np.zeros((len(subs), 59412))
sub_voxel_repeats = np.zeros((1,59412))
mapnames, label_tables = [], []
labels_name = {0: 'notselected', 1:'selected'}
labels_rgba = {0: (0,0,0,0),
        1: (.5,.5,.5,1)}
for i, sub in enumerate(subs):
    print(i, sub)
    # initialization
    retino_voxels, floc_voxels = None, None

    # 
    if 'ret' in mask_name:
        # ===========================================
        # getting retinotopic voxels
        # ===========================================
        sub_prf_file = os.path.join(opendata_cifti_pth, sub, 'results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii')
        retino_r2 = nib.load(sub_prf_file).get_fdata()[3, :]

        # getting the threshold
        if 'gmm' in mask_name:
            #-------------------------------------------
            # determine subject-specific threshold
            #-------------------------------------------
            gmm = GaussianMixture(n_components=2)
            gmm.fit(retino_r2.reshape((-1,1)))
            eqpoint = solve_GMM_eq_point(gmm.means_[0], gmm.means_[1], gmm.covariances_[0],gmm.covariances_[1])
            if np.min(eqpoint) < np.min(gmm.means_) and np.max(eqpoint) < np.max(gmm.means_):
                eqpoint = eqpoint[eqpoint>np.min(eqpoint)]
            elif np.min(eqpoint) > np.min(gmm.means_) and np.min(eqpoint) < np.max(gmm.means_):
                eqpoint = np.min(eqpoint)
            else:
                raise AssertionError(f'Equal point solution ${eqpoint}$ ha no qualified threshold with GMM means ${gmm.means_}$')

            r2_thres = eqpoint
        #-------------------------------------------
        # determine general threshold
        #-------------------------------------------
        elif 'fix' in mask_name:
            r2_thres = 9
        # 
        retino_voxels = np.where(retino_r2 > r2_thres)[0]
    if 'floc' in mask_name:
    # ===========================================
    # getting functional localizer voxels
    # ===========================================
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
    
    if 'ret' in mask_name and 'floc' in mask_name:
        # merging into one mask
        voxel_indices = np.array([ _ for _ in range(59412) if (_ in floc_voxels) or (_ in retino_voxels)])
    elif 'ret' in mask_name and ('floc' not in mask_name):
        voxel_indices = np.array([ _ for _ in range(59412) if (_ in retino_voxels)])
    elif 'floc' in mask_name and ('ret' not in mask_name):
        voxel_indices = np.array([ _ for _ in range(59412) if (_ in floc_voxels)])
    
    # construct 
    sub_voxel_repeats[0, voxel_indices] += 1
    subs_voxel_masks[i,voxel_indices] = 1
    mapnames.append(sub)

    lbl_tb = nib.cifti2.Cifti2LabelTable()
    for key in np.unique(subs_voxel_masks):
        key = int(key)
        lbl_tb[key] = nib.cifti2.Cifti2Label(key, labels_name[key], *labels_rgba[key])
    label_tables.append(lbl_tb)

save2cifti(pjoin(mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'),  
        subs_voxel_masks, surface_brain_models, map_names=mapnames, label_tables=label_tables)

save2cifti(pjoin(mask_path, f'nod-voxmask_{mask_name}-repeats.dscalar.nii'),  
        sub_voxel_repeats, surface_brain_models, map_names=['sharedvoxels'])


# subj_selection_procedure(mask_name)

