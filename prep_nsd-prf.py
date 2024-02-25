import os
import nibabel as nib
import numpy as np
from os.path import join as pjoin 
from utils import save2cifti

cifti_dir = '/nfs/z1/userhome/GongZhengXin/NSD/derivatives/ciftify'
gifti_dir = '/nfs/z1/userhome/GongZhengXin/NSD/nsddata/freesurfer/fsLRspace'

brain_models = list(nib.load('template.dtseries.nii').header.get_index_map(1).brain_models)
surface_models = [bm for bm in brain_models if bm.model_type == 'CIFTI_MODEL_TYPE_SURFACE']

for surface_model in surface_models:
    if 'LEFT' in surface_model.brain_structure:
        lh_vertices = np.array(list(surface_model.vertex_indices))
        lh_indexoffset = surface_model.index_offset
        lh_indexcount = surface_model.index_count
    if 'RIGHT' in surface_model.brain_structure:
        rh_vertices = np.array(list(surface_model.vertex_indices))
        rh_indexoffset = surface_model.index_offset
        rh_indexcount = surface_model.index_count


file_flags = ['ang', 'eccen', 'rfsize', 'R2']
n_vertices = 59412

subs = [f'subj0{i}' for i in range(1,9)]

for sub in subs:
    gifti_files = sorted([_ for _ in os.listdir(gifti_dir) if sub in _])

    for gifti_file in gifti_files:
        
        hemi = gifti_file.split('.')[1]
        index = gifti_file.split('.')[2]
        cifti_data = np.nan*np.zeros((len(file_flags), n_vertices))
        if any([flag in index for flag in file_flags]) and hemi=='lh':
            print(gifti_file)
            #
            lh_file = nib.load(pjoin(gifti_dir, gifti_file))
            #
            counterpart_file = gifti_file.replace('.lh.', '.rh.')
            if counterpart_file in  gifti_files:
                rh_file = nib.load(pjoin(gifti_dir, counterpart_file))
            else:
                print(f'====!!{gifti_file} has no corresponding {counterpart_file}!!======')
            #
            pos = np.where(np.array([flag in index for flag in file_flags])==True)[0]
            cifti_data[int(pos), lh_indexoffset : (lh_indexoffset + lh_indexcount)] = lh_file.agg_data()[lh_vertices]
            cifti_data[int(pos), rh_indexoffset : (rh_indexoffset + rh_indexcount)] = rh_file.agg_data()[rh_vertices]

    #
    save_dir = pjoin(cifti_dir, sub, 'MNINonLinear/Results/prf_session')
    os.makedirs(save_dir, exist_ok=True)
    cifti_filename = 'prf_session_params.dscalar.nii'
    save2cifti(pjoin(save_dir, cifti_filename), cifti_data, surface_models, map_names=file_flags)
    print(f'saved! {pjoin(save_dir, cifti_filename)}')
    txt_filename = 'prf_session_label.txt'
    with open(pjoin(save_dir, txt_filename), 'w') as f:
        f.writelines('\n'.join(file_flags))