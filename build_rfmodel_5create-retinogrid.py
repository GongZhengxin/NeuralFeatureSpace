import os
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
from os.path import join as pjoin
import time 
import pickle
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from utils import train_data_normalization, Timer, net_size_info, get_roi_data

work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
quater_names = [f'ang-quater{_+1}' for _ in range(8)]
angsplotlines = [0, 90, 180, 270, 360]
eccsplitlines = [0, 2, 4, 6, 8]
concate_path = pjoin(work_dir, 'prep/roi-concate')
layername = 'googlenet-conv2'
rois =  ['V1', 'V2', 'V3', 'V4']
subs = [f'sub-0{_+1}' for _ in range(9)]

for roiname in rois:
    all_subjects_data = pd.DataFrame()
    for sub in subs:
        map_dir = pjoin(work_dir, 'anal/brainmap/masked_retinotopy')
        ang = nib.load(pjoin(map_dir, sub, f'{sub}_masked-prior-prf.dscalar.nii')).get_fdata()[0,:]
        ecc = nib.load(pjoin(map_dir, sub, f'{sub}_masked-prior-prf.dscalar.nii')).get_fdata()[1,:]
        voxels = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roiname}-voxel.npy'))
        
        ecc_group = np.zeros_like(voxels)
        ang_group = np.zeros_like(voxels)

        for iline in range(len(eccsplitlines)-1):
            small_bound, large_bound = eccsplitlines[iline], eccsplitlines[iline+1]
            ecc_group[np.where((ecc[voxels] <= large_bound) & (ecc[voxels] > small_bound))] = iline
        ecc_group[np.where(ecc[voxels] > eccsplitlines[-1])] = iline + 1

        for iline in range(len(angsplotlines)-1):
            small_bound, large_bound = angsplotlines[iline], angsplotlines[iline+1]
            ang_group[np.where((ang[voxels] <= large_bound) & (ang[voxels] > small_bound))] = iline
        
        subject_data = pd.DataFrame({
            'subid': np.repeat([sub], len(voxels)),
            'voxidx': voxels,
            'ecc': ecc[voxels],
            'ang': ang[voxels],
            'eccgroup': ecc_group,
            'anggroup': ang_group
            })

        all_subjects_data = pd.concat([all_subjects_data, subject_data], ignore_index=True)
    print(pjoin(work_dir, f"prep/roi-concate/retino-grid/all-sub_{roiname}-retino-data.csv"))
    # all_subjects_data.to_csv(pjoin(work_dir, f"prep/roi-concate/retino-grid/all-sub_{roiname}-retino-data.csv"), index=False)

for roiname in rois:
    all_subjects_retinofile = pjoin(work_dir, f'prep/roi-concate/retino-grid/all-sub_{roiname}-retino-data.csv')
    all_subjects_retino = pd.read_csv(all_subjects_retinofile)
    os.makedirs(pjoin(work_dir, f'prep/roi-concate/retino-grid/'), exist_ok=True)
    for sub in subs: 
        subject_retino = all_subjects_retino[all_subjects_retino['subid']==sub]
        ecc_group =  subject_retino['eccgroup'].values
        ang_group =  subject_retino['anggroup'].values
        voxels =  subject_retino['voxidx'].values

        block_voxels, block_indx = {}, {}
        voxel_num = []
        for iecc in range(5):
            for iang in range(4):
                grid_voxels = voxels[np.where((ecc_group==iecc) & (ang_group==iang))]
                block_voxels[f'E{iecc}A{iang}'] = grid_voxels
                block_indx[f'E{iecc}A{iang}'] = np.array([voxels.tolist().index(_) for _ in grid_voxels])
                voxel_num.append(len(block_voxels[f'E{iecc}A{iang}']))
        print(pjoin(work_dir, f'prep/roi-concate/retino-grid/{sub}_{roiname}-retino-grids-voxels.npy'))
        print(pjoin(work_dir, f'prep/roi-concate/retino-grid/{sub}_{roiname}-retino-grids-idxs.npy'))
        # np.save(pjoin(work_dir, f'prep/roi-concate/retino-grid/{sub}_{roiname}-retino-grids-voxels.npy'), block_voxels)
        # np.save(pjoin(work_dir, f'prep/roi-concate/retino-grid/{sub}_{roiname}-retino-grids-idxs.npy'), block_indx)

