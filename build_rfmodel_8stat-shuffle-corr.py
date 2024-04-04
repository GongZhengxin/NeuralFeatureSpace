import os
import gc
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from os.path import join as pjoin
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import zscore
from joblib import Parallel, delayed
import time 
import joblib
import matplotlib.pyplot as plt
import pickle
from utils import train_data_normalization, Timer, net_size_info, get_roi_data



# path settings
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
# input path
concate_path = pjoin(work_dir, 'prep/roi-concate')
# save out path
performance_path = pjoin(work_dir, 'build/roi-concatemodel')
os.makedirs(performance_path, exist_ok=True)
inputlayername = 'googlenet-conv2' #'googlenet-maxpool2' # 'googlenet-inception3a' #
layer = {'name': inputlayername, 'size': net_size_info[inputlayername]}
layername = layer['name']
layername = layername.replace('.','')
rois = ['V1', 'V3', 'V2','V4']
mask_name = 'primaryvis-in-MMP' #
test_set_name = 'coco'
subs = [f'sub-0{isub+1}' for isub in range(0, 9)]

n_sample = 1000
# ==================================
# for retino grid models
# ==================================
all_shufle_perf = np.zeros((9, 5, 4, n_sample))

for roi_name in rois:
    for isub, sub in enumerate(subs):        
        print('loading test features ')
        os.makedirs(pjoin(performance_path, 'shuffleperformance'), exist_ok=True)
        # test feature and response
        test_feature = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-feature.npy'), mmap_mode='r')
        test_data = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-resp.npy'), mmap_mode='r')
        
        print('loading quater file')
        idxs_dict = np.load(pjoin(concate_path, 'retino-grid', f'{sub}_{roi_name}-retino-grids-idxs.npy'), allow_pickle=True).item()
        
        grid_models = np.load(pjoin(performance_path, 'retino-grid', roi_name, f'{sub}_layer-{layername}_grid-models.npy'), allow_pickle=True).item()
        grid_performance_cor = np.zeros((5, 4, n_sample))
        for grid, grid_voxel_idx in idxs_dict.items():
            grid_i, grid_j = int(grid[1]), int(grid[3])
            if len(grid_voxel_idx) == 0:
                grid_performance_cor[grid_i, grid_j] = np.nan
                
            else: 
                grid_test_feature = test_feature[grid_voxel_idx]
                grid_test_data = test_data[grid_voxel_idx]
                print('new shapes:', grid_test_feature.shape, grid_test_data.shape)
                
                # data reshape
                grid_test_feature = grid_test_feature.reshape((-1, test_feature.shape[-1]))
                grid_test_y = grid_test_data.reshape(-1)

                grid_model = grid_models[grid]
            
                y_pred = grid_model.predict(grid_test_feature)

                # 计算并记录性能指标
                y_shuffle = y_pred.copy()
                # 打乱数据计算并记录性能指标
                grid_corrs = []
                for _ in range(1000):
                    np.random.shuffle(y_shuffle)
                    grid_corrs.append(np.corrcoef(grid_test_y, y_shuffle)[0,1])
                # test_scores.append(lr.score(test_feature, test_y))
                grid_performance_cor[grid_i, grid_j] = np.array(grid_corrs)
                print(sub, grid, layername, np.percentile(grid_corrs, 95))
        
        all_shufle_perf[isub] = grid_performance_cor
        
        # np.save(pjoin(performance_path, roi_name, f'{sub}_layer-{layername}_grid-models.npy'), grid_models)
    filename = pjoin(performance_path, 'shuffleperformance', f'all-sub_model-{layername}_{roi_name}-grid-shuffle-cor.npy')
    np.save(filename, all_shufle_perf)






# ==================================
# for roi models
# ==================================
all_shufle_perf = np.zeros((9, n_sample))
for roi_name in ['V1', 'V3', 'V2','V4']:# , ,   
    val_scores, val_corrs = [], []
    test_scores, test_corrs = [], []
    # roi_name = 'V1'
    for sub in subs:
        sub_corrs = []
        t0 = time.time()
        os.makedirs(pjoin(performance_path, roi_name), exist_ok=True)
        # output dir
        os.makedirs(pjoin(performance_path, 'shuffleperformance'), exist_ok=True)
        
        print('loading test features ')
        # test feature and response
        test_feature = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-feature.npy'), mmap_mode='r')
        test_data = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-resp.npy'), mmap_mode='r')
        
        if roi_name == 'V4':
            voxels = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-voxel.npy')).tolist()
            voxel_selection = np.load(pjoin(concate_path, sub, f'{sub}_selection_V4-voxels.npy'))
            idxs_selection = np.array([voxels.index(_) for _ in voxel_selection])
            test_feature = test_feature[idxs_selection]
            test_data = test_data[idxs_selection]
            print('new shapes:', test_feature.shape, test_data.shape)
        # data reshape
        test_feature = test_feature.reshape((-1, test_feature.shape[-1]))
        test_y = test_data.reshape(-1)

        # training linear model
        lr = joblib.load(pjoin(performance_path, roi_name, f'{sub}_layer-{layername}_{roi_name}-linear.pkl'))
        y_pred = lr.predict(test_feature)
        print(sub, 'conv2', roi_name, np.max(y_pred), np.argmax(y_pred))
    #     y_shuffle = y_pred.copy()

    #     # 打乱数据计算并记录性能指标
    #     for _ in range(1000):
    #         np.random.shuffle(y_shuffle)
    #         sub_corrs.append(np.corrcoef(test_y, y_shuffle)[0,1])
    #     # test_scores.append(lr.score(test_feature, test_y))
    #     all_shufle_perf[subs.index(sub)] = np.array(sub_corrs)
        
    #     print(sub, f'finished', roi_name, np.percentile(sub_corrs, 95), np.mean(sub_corrs))#, val_corrs[-1], val_scores[-1]
    # np.save(pjoin(performance_path, 'shuffleperformance', f'all-sub_model-{layername}_{roi_name}_shuffle-cor.npy'), all_shufle_perf)




