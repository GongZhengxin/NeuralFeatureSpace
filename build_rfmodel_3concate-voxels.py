import os
import gc
import torch
import torch.nn as nn
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


def concate_voxel_resp_and_feature(idx, voxel):
    global X, X_test, y, y_test, i, j, labels
    if not voxel in guassparams.keys():
        return []
    else:
        print(f'{sub} : {idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))
        
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        X_voxel_test = zscore(X_voxel_test)# (X_voxel_test - X_voxel.mean(axis=0))/ X_voxel.std(axis=0)
        if np.isnan(X_voxel).any():
            train_nan = 1
            X_voxel = np.nan_to_num(X_voxel)
        if np.isnan(X_voxel_test).any():
            test_nan = 1
            X_voxel_test = np.nan_to_num(X_voxel_test)
        # 取出当前体素的训练集和测试集神经活动
        y_voxel = zscore(y[:, idx], nan_policy='omit')
        y_voxel_test = zscore(y_test[:, idx], nan_policy='omit')
        
        return {'train-feature': X_voxel, 'test-feature': X_voxel_test, 
                'train-resp' : y_voxel, 'test-resp' : y_voxel_test, 'idx': idx, 'voxel': voxel}

layers = ['googlenet-conv2', 'googlenet-maxpool2','googlenet-inception3a'] # ,'googlenet-inception3a'
# 'googlenet-conv2', 
for inputlayername in layers:
    layer = {'name': inputlayername, 'size': net_size_info[inputlayername.replace('raw-', '')]}#alexnet_info[inputlayername]
    layername = layer['name']
    layername = layername.replace('.','')
    labels = conv2_labels
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


    for sub in subs:
        with Timer() as t:
            print(sub, mask_name, layername)
            # save path
            if not os.path.exists(pjoin(concate_path, sub)):
                os.makedirs(pjoin(concate_path, sub))

            # load
            brain_resp = np.load(pjoin(resp_path, f'{sub}_imagenet_beta.npy'), mmap_mode='r')
            activations = np.load(pjoin(image_activations_path, f'{sub}_{layername}.npy'), mmap_mode='r')
            coco_activations = np.load(pjoin(image_activations_path, f'{test_set_name}_{layername}.npy'), mmap_mode='r')
            print(f'activations shape of {activations.shape}')
            guassparams = np.load(pjoin(guass_path, f'{sub}_weighted_Gauss.npy'), allow_pickle=True)[0]
            
            # load, reshape and average the resp
            test_resp = np.load(pjoin(resp_path, f'{sub}_{test_set_name}_beta.npy'))
            num_trial = test_resp.shape[0]
            num_run = int(num_trial/120)
            test_resp = test_resp.reshape((num_run, 120, 59412))
            mean_test_resp = test_resp.mean(axis=0)
            
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
            # voxel_indices = np.array([ _ for _ in np.where(prf_voxel_mask==1)[0] if _ in mask_voxel_indices])
            
            voxel_indices = np.intersect1d(mask_voxel_indices, np.where(prf_voxel_mask==1)[0]) 
            print(voxel_indices)
            # collect resp in ROI
            brain_resp = brain_resp[:, voxel_indices]
            # test_resp = test_resp[:, :, voxel_indices]
            mean_test_resp = mean_test_resp[:, voxel_indices]


            # normalization
            norm_metric = 'session'
            brain_resp = train_data_normalization(brain_resp, metric=norm_metric)
            # mean_test_resp = zscore(test_resp.mean(axis=0))
            mean_test_resp = zscore(mean_test_resp, None)
            num_voxel = brain_resp.shape[-1]

            del test_resp, voxel_mask
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
            
            y = brain_resp
            y_test = mean_test_resp

            # concurrent computing
            voxels = voxel_indices.tolist()
            idxs = np.arange(num_voxel).tolist()
            # voxel_indices = voxel_indices
            results = Parallel(n_jobs=30)(delayed(concate_voxel_resp_and_feature)(idx, voxel) for idx, voxel in zip(idxs, voxels))
            results = [result for result in results if result]
            # 
            voxels = np.array([ _['voxel'] for _ in results])
            for indexname in ['train-feature', 'test-feature', 'train-resp', 'test-resp', 'voxel']:
                index = np.array([ _[indexname] for _ in results])
                for roi in ['V1', 'V2', 'V3', 'V4']:
                    selection = np.array([ _ in np.where(get_roi_data(None, roi)==1)[0] for _ in voxels])
                    roi_index = index[selection]
                    np.save(pjoin(concate_path, sub, f'{sub}_layer-{inputlayername}_{roi}-{indexname}.npy'), roi_index)
                    print(sub, indexname, inputlayername, roi_index.shape)
        print(f'{sub} consume : {t.interval} s')
