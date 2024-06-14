import os
import gc
import numpy as np
import nibabel as nib
from os.path import join as pjoin

from scipy.stats import zscore
from joblib import Parallel, delayed
import time 
import joblib
import matplotlib.pyplot as plt
import pickle
from utils import train_data_normalization, Timer, net_size_info, conv2_labels, get_roi_data


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

def concate_voxel_feature(idx, voxel):
    global X, X_test, i, j
    if not voxel in guassparams.keys():
        return []
    else:
        print(f'{sub} : {idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))# X= feature training set
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))# test set
        
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        X_voxel_test = zscore(X_voxel_test)# (X_voxel_test - X_voxel.mean(axis=0))/ X_voxel.std(axis=0)
        if np.isnan(X_voxel).any():
            train_nan = 1
            X_voxel = np.nan_to_num(X_voxel)
        if np.isnan(X_voxel_test).any():
            test_nan = 1
            X_voxel_test = np.nan_to_num(X_voxel_test)

        return {'train-feature': X_voxel, 'test-feature': X_voxel_test, 
                 'idx': idx, 'voxel': voxel}

def concate_voxel_test_feature(idx, voxel):
    global X_test, i, j
    if not voxel in guassparams.keys():
        return []
    else:
        print(f'{sub} : {idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))# test set
        
        # 特征标准化, 均值都已经减掉了
        X_voxel_test = zscore(X_voxel_test)# (X_voxel_test - X_voxel.mean(axis=0))/ X_voxel.std(axis=0)
        
        if np.isnan(X_voxel_test).any():
            test_nan = 1
            X_voxel_test = np.nan_to_num(X_voxel_test)

        return {'test-feature': X_voxel_test, 
                 'idx': idx, 'voxel': voxel}

layers = ['googlenet-conv2'] #'googlenet-maxpool1'[, 'googlenet-maxpool2','googlenet-inception3a'] # ,'googlenet-inception3a'
# 'googlenet-conv2', 


for inputlayername in layers:
    layer = {'name': inputlayername, 'size': net_size_info[inputlayername.replace('raw-', '')]}#alexnet_info[inputlayername]
    layername = layer['name']
    layername = layername.replace('.','')

    mask_name = 'primaryvis-in-MMP' #'fixretfloc-in-subj'
    test_set_name = 'coco'

    # path settings
    work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
    # input path
    resp_path = pjoin(work_dir, 'prep/brain_response')
    opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
    roi_mask = 'primaryvis-in-MMP'
    voxel_mask_path = pjoin(work_dir, 'prep/voxel_masks')
    image_activations_path = pjoin(work_dir, 'prep/image_activations')
    retino_path = pjoin(work_dir, f'build/retinoparams/{roi_mask}')
    guass_path = pjoin(work_dir, f'build/gaussianparams/{roi_mask}')
    # save out path
    concate_path = pjoin(work_dir, 'prep/roi-concate')
    subs = [f'sub-0{isub+1}' for isub in range(0, 9)]
    # axis
    axis_type = 'pca-shuffle'
    component_path = pjoin(work_dir,  f'prep/image_activations/{axis_type}')
    if axis_type=='pca':
        allsub_activations = np.load(pjoin(component_path, f'all-sub_{inputlayername}_{axis_type}-components.npy'), mmap_mode='r').transpose((0,3,1,2))
        coco_activations = np.load(pjoin(component_path, f'coco_{inputlayername}_{axis_type}-components.npy'), mmap_mode='r').transpose((0,3,1,2))
    if axis_type=='ica':
        coco_activations = np.load(pjoin(component_path, f'coco_{inputlayername}_{axis_type}-comps.npy'), mmap_mode='r')
    if axis_type=='pca-shuffle':
        coco_activations = np.load(pjoin(component_path, f'coco_{inputlayername}_{axis_type}-comps.npy'), mmap_mode='r')

    for sub in subs:
        with Timer() as t:
            print(sub, mask_name, layername)
            # save path
            if not os.path.exists(pjoin(concate_path, sub, axis_type)):
                os.makedirs(pjoin(concate_path, sub, axis_type))
                print('create:', pjoin(concate_path, sub, axis_type))
            
            # load
            if axis_type == 'pca':
                subidx = np.int(sub.split('-')[1])
                activations = allsub_activations[4000*(subidx-1):4000*subidx]
            elif axis_type == 'ica':
                activations = np.load(pjoin(component_path, f'{sub}_{inputlayername}_{axis_type}-comps.npy'))
            elif axis_type == 'pca-shuffle':
                activations = np.load(pjoin(component_path, f'{sub}_{inputlayername}_{axis_type}-comps.npy'))
            print(f'activations shape of {activations.shape}')
            guassparams = np.load(pjoin(guass_path, f'{sub}_weighted_Gauss.npy'), allow_pickle=True)[0]
              
            # load mask
            voxel_mask_nii = nib.load(pjoin(voxel_mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'))
            voxel_mask = voxel_mask_nii.get_fdata()
            named_maps = [named_map.map_name for named_map in voxel_mask_nii.header.get_index_map(0).named_maps]
            # determine the mask type
            if sub in named_maps:
                voxel_mask = voxel_mask[named_maps.index(sub),:]
            # squeeze into 1 dim
            voxel_mask = np.squeeze(np.array(voxel_mask))
            # transfer mask into indices
            mask_voxel_indices = np.where(voxel_mask==1)[0]
            
            prf_path = pjoin(opendata_cifti_pth, sub, 'results/ses-prf_task-prf')
            sub_prf_file = pjoin(prf_path, 'ses-prf_task-prf_params.dscalar.nii')

            # load and modify
            prf_data = nib.load(sub_prf_file).get_fdata()
            prf_r2 = prf_data[3,:]
            r2_thres = 10
            # make mask
            prf_voxel_mask = prf_r2 > r2_thres
            # transfer mask into indices
            voxel_indices = np.intersect1d(mask_voxel_indices, np.where(prf_voxel_mask==1)[0]) 
            num_voxel = voxel_indices.shape[-1]
            print('voxelnum', num_voxel)

            del voxel_mask
            gc.collect()

            # coordinate
            # Create grid data
            layer['size'] = activations.shape[-1]
            i = np.linspace(-8., 8., layer['size'])
            j = np.linspace(8., -8., layer['size'])
            i, j = np.meshgrid(i, j)

            if layername == 'googlenet-conv2':
                X = activations[:, 0:63, :, :]
                X_test = coco_activations[:, 0:63, :, :]
            else:
                X = activations
                X_test = coco_activations
            
            # concurrent computing
            voxels = voxel_indices.tolist()
            idxs = np.arange(num_voxel).tolist()
            # voxel_indices = voxel_indices
            results = Parallel(n_jobs=30)(delayed(concate_voxel_feature)(idx, voxel) for idx, voxel in zip(idxs, voxels))
            save_indexnames = [ 'train-feature', 'test-feature' ]#,'voxel'
            # dropout null
            results = [result for result in results if result]
            # saving
            voxels = np.array([ _['voxel'] for _ in results])
            for indexname in save_indexnames:
                index = np.array([ _[indexname] for _ in results])
                for roi in ['V1', 'V2', 'V3', 'V4']:
                    selection = np.array([ _ in np.where(get_roi_data(None, roi)==1)[0] for _ in voxels])
                    roi_index = index[selection]
                    np.save(pjoin(concate_path, sub, axis_type, f'{sub}_layer-{inputlayername}_{roi}-{indexname}.npy'), roi_index)
                    print(sub, indexname, inputlayername, roi_index.shape)
            # create response link
            for roi in ['V1', 'V2', 'V3', 'V4']:
                for indexname in [ 'train-resp', 'test-resp']:#,'voxel'
                    source_path = pjoin(concate_path, sub, f'{sub}_layer-{inputlayername}_{roi}-{indexname}.npy')
                    link_path = pjoin(concate_path, sub, axis_type, f'{sub}_layer-{inputlayername}_{roi}-{indexname}.npy')
                    try:
                        os.symlink(source_path, link_path)
                        print(f"软链接创建成功，链接 {link_path} 指向 {source_path}")
                    except OSError as error:
                        print(f"创建软链接时发生错误: {error}")
            #     
        print(f'{sub} consume : {t.interval} s')
