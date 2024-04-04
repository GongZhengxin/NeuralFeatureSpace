import os
import gc
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import zscore
from joblib import Parallel, delayed
import time 
from utils import train_data_normalization, Timer, net_size_info, conv2_labels


# Define the 2D Gaussian function
def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
    i, j = coords
    return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C

def adjust_RF(receptive_field):
    cur_receptive_field = receptive_field.copy()
    cur_receptive_field = cur_receptive_field + np.abs(np.min(cur_receptive_field, None)) + 1
    thres = np.min(cur_receptive_field) + 0.5*(np.max(cur_receptive_field) - np.min(cur_receptive_field)) # 
    # if np.mean(receptive_field < thres) < 0.1:
    #     thres = np.mean(receptive_field)
    cur_receptive_field[cur_receptive_field < thres] = 0
    cur_receptive_field = cur_receptive_field / (cur_receptive_field.sum() + 1e-20)
    # if (~np.isfinite(cur_receptive_field)).any():
    #     print('use full gaussian')
    #     cur_receptive_field = receptive_field + np.abs(np.min(receptive_field, None))
    #     cur_receptive_field = cur_receptive_field / (cur_receptive_field.sum() + 1e-20)
    return cur_receptive_field

subs = [f'sub-0{isub+1}' for isub in range(0, 9)]
# subs = ['sub-03'] # ,, 'sub-05' 'sub-04', 'sub-02','sub-06', 'sub-07''sub-04', 'sub-08', 'sub-02','sub-06', 'sub-07','sub-09'

# double sig
# feature_filter = np.array([1,3,17,21,28,29,35,37,45,47,49,57,58,60])

# # in every case sig
# feature_filter = np.array([1,47,60,23,49,28,29,37,2,5,21,42,45,3,35,40,17,57,51,58,25,52,61,18])

# # single sig
# feature_filter = np.array([1,47,60,49,28,29,37,21,42,45,3,35,17,57,58,25])
# time.sleep(1800)

# last 12 significant feature
feature_filter = np.array([ 1, 37, 28, 47, 49, 42, 45, 35,  3, 57, 58])

# pca
axis_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/stimaxis/googlenet-conv1_imagenet36k_pcspace.npy'
npc = 16
axis = np.load(axis_path)[0, :, 0:npc].astype(np.float32)
non_linear = False
for sub in subs:
    with Timer() as t:
        inputlayername = 'googlenet-conv2' 
        layer = {'name': inputlayername, 'size': net_size_info[inputlayername.replace('raw-', '')]}#alexnet_info[inputlayername]
        layername = layer['name']
        layername = layername.replace('.','')
        labels = conv2_labels
        mask_name = 'primaryvis-in-MMP' #'fixretfloc-in-subj'
        test_set_name = 'coco'
        print(sub, mask_name, layername)
        fold_indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
        # path settings
        work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
        opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
        # input path
        resp_path = pjoin(work_dir, 'prep/brain_response')
        voxel_mask_path = pjoin(work_dir, 'prep/voxel_masks')
        image_activations_path = pjoin(work_dir, 'prep/image_activations')
        retino_path = pjoin(work_dir, 'build/retinoparams')
        guass_path = pjoin(work_dir, 'build/gaussianparams')
        avgrf_path = pjoin(work_dir, 'prep/image_mask')
        prf_path = pjoin(opendata_cifti_pth, sub, 'results/ses-prf_task-prf')
        # save out path
        performance_path = pjoin(work_dir, 'build/featurewise-corr/roi-level')
        # save path
        if not os.path.exists(pjoin(performance_path, sub)):
            os.makedirs(pjoin(performance_path, sub))

        # getting retinotopic voxels
        sub_prf_file = pjoin(prf_path, 'ses-prf_task-prf_params.dscalar.nii')
        
        # load and modify 
        prf_data = nib.load(sub_prf_file).get_fdata()
        prf_r2 = prf_data[3,:]
        r2_thres = 10
        # make mask
        voxel_mask = prf_r2 > r2_thres
        # transfer mask into indices
        voxel_indices = np.where(voxel_mask==1)[0]

        # generate ROI in selected voxels
        v1_voxels = np.array([ _ for _ in np.where(get_roi_data(None, 'V1')==1)[0] if _ in voxel_indices])
        v2_voxels = np.array([ _ for _ in  np.where(get_roi_data(None, 'V2')==1)[0] if _ in voxel_indices])
        v3_voxels = np.array([ _ for _ in  np.where(get_roi_data(None, 'V3')==1)[0] if _ in voxel_indices])
        v4_voxels = np.array([ _ for _ in  np.where(get_roi_data(None, 'V4')==1)[0] if _ in voxel_indices])

        # voxel masks
        early_vis_rois = [v1_voxels, v2_voxels, v3_voxels, v4_voxels]

        # # we need to transfer the params into (x,y,size) model
        # n_vertices = prf_data.shape[-1]
        # trans_xys_params = np.zeros((n_vertices, 3))
        # trans_xys_params[:,0] = np.cos(prf_data[0, 0:n_vertices]/180*np.pi) * prf_data[1, 0:n_vertices]*16/200
        # trans_xys_params[:,1] = np.sin(prf_data[0, 0:n_vertices]/180*np.pi) * prf_data[1, 0:n_vertices]*16/200
        # trans_xys_params[:,2] = prf_data[2, 0:n_vertices] * 16/200
        avg_receptivefield = np.load(pjoin(avgrf_path, f'{sub}_average-receptivefield.npy'))

        # load
        brain_resp = np.load(pjoin(resp_path, f'{sub}_imagenet_beta.npy'))
        activations = np.load(pjoin(image_activations_path, f'{sub}_{layername}.npy'))
        coco_activations = np.load(pjoin(image_activations_path, f'{test_set_name}_{layername}.npy'))
        print(f'activations shape of {activations.shape}')
        if 'conv1' in layername:
            guass_layername = layername.replace('conv1', 'conv2')
            guassparams = np.load(pjoin(guass_path, f'{sub}_layer-{guass_layername}_Gauss.npy'), allow_pickle=True)[0]
            print(pjoin(guass_path, f'{sub}_layer-{guass_layername}_Gauss.npy'))
        else:
            guassparams = np.load(pjoin(guass_path, f'{sub}_layer-{layername}_Gauss.npy'), allow_pickle=True)[0]
        
        # load, reshape and average the resp
        test_resp = np.load(pjoin(resp_path, f'{sub}_{test_set_name}_beta.npy'))
        num_trial = test_resp.shape[0]
        num_run = int(num_trial/120)
        test_resp = test_resp.reshape((num_run, 120, 59412))
        mean_test_resp = test_resp.mean(axis=0)
        
        # collect resp and summed to ROI response
        roi_brain_resp = np.atleast_2d(brain_resp[:, v1_voxels].sum(axis=1)).T
        roi_mean_test_resp = np.atleast_2d(mean_test_resp[:, v1_voxels].sum(axis=1)).T

        # normalization
        norm_metric = 'session'
        roi_brain_resp = train_data_normalization(roi_brain_resp, metric=norm_metric)
        # mean_test_resp = zscore(test_resp.mean(axis=0))
        roi_mean_test_resp = zscore(roi_mean_test_resp, None)

        del test_resp, voxel_mask_nii, voxel_mask
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
        
        y = roi_brain_resp
        y_test = roi_mean_test_resp

        # filter X
        try:
            X = X[:, feature_filter, :, :]
            X_test = X_test[:, feature_filter, :, :]
            print('!!Note: feature filter will be carried out >>')
        except NameError:
            print('With feature shape:', X.shape)
        
        # 普通线性模型的预测
        X_avg = np.sum(X * avg_receptivefield, axis=(2,3))
        lr = LinearRegression(n_jobs=10).fit(X_avg, y)
        # 使用Lasso等办法进行特征选取


        # # shuffle X
        # shuffle_indices = [_ for _ in range(100)] + [_ for _ in range(200, 1000)] + [_ for _ in range(100,200)]
        # for isess in range(4):
        #     cur_shuffle = isess*1000 + np.array(shuffle_indices)
        #     X[isess*1000 : (isess+1)*1000] = X[cur_shuffle]
        
        # set_r2s_train = np.load(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_model-ctg-ev-train.npy'))
        # set_rs_train = np.load(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_model-ctg-r-train.npy'))
        # set_r2s_test = np.load(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_model-ctg-ev.npy'))
        # set_rs_test = np.load(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_model-ctg-r.npy'))

        # # loop for voxels
        # simple_cor = np.zeros((len(voxel_indices), X.shape[1]))
        # for idx, voxel in (zip(np.arange(num_voxel), voxel_indices)):
        #     print(100*idx/num_voxel,f"% - {sub}: voxel-{voxel}")
        #     if voxel in guassparams.keys():
        #         print(f'corr-({idx},{voxel})')
        #         # load receptive field
        #         receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        #         receptive_field = adjust_RF(receptive_field)
        #         # saptial summation
        #         X_voxel = np.sum(X * receptive_field, axis=(2,3))
        #         # assign to corr matrix
        #         simple_cor[idx,:] = np.corrcoef(X_voxel.transpose(), y[:, idx][np.newaxis, :])[:,-1][0 : X.shape[1]]
        #         # lr = LinearRegression(n_jobs=12).fit(X_voxel, y[:, idx])
        # all_performace = np.nan*np.zeros((X.shape[1], 59412))
        # all_performace[:, voxel_indices] = np.array(simple_cor).transpose()
        # np.save(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_corr.npy'), all_performace)     
        
        # concurrent computing