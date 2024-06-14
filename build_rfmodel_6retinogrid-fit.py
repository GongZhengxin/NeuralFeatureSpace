# 在每个分隔开的 EA grid 体素上训练线性模型
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
from scipy import stats
from scipy.stats import zscore
from joblib import Parallel, delayed
import time 
import joblib
import matplotlib.pyplot as plt
import pickle
from utils import train_data_normalization, Timer, net_size_info, conv2_labels, get_roi_data

from sklearn.metrics import make_scorer
from scipy.stats import pearsonr

def pearson_correlation(y_true, y_pred):
    # 确保输入是NumPy数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 确保输入是一维的
    if y_true.ndim > 1:
        y_true = np.ravel(y_true)
    if y_pred.ndim > 1:
        y_pred = np.ravel(y_pred)
    
    # 移除NaN值
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # 计算并返回皮尔森相关系数
    return pearsonr(y_true, y_pred)[0]

work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
angsplotlines =  [0, 45, 90, 135, 180, 225, 270, 315, 360] #[0, 90, 180, 270, 360]
eccsplitlines = [0, 2, 4, 6, 8]
concate_path = pjoin(work_dir, 'prep/roi-concate')
grid_name = 'finer-retino-grid'
performance_path = pjoin(work_dir, f'build/roi-concatemodel/{grid_name}')
os.makedirs(performance_path, exist_ok=True)

rois =  [ 'V2','V3','V4'] # 'V1',
subs = [f'sub-0{isub+1}' for isub in range(0, 9)]
layername = 'googlenet-maxpool2' #'googlenet-maxpool2'
# roi_name = 'V2'
all_grid_performance_cor = np.zeros((9, len(eccsplitlines), len(angsplotlines)-1))
all_grid_performance_ev = np.zeros((9, len(eccsplitlines), len(angsplotlines)-1))
# print('sleeping')
# time.sleep(7200)
for roi_name in rois:
    for isub, sub in enumerate(subs):
        os.makedirs(pjoin(performance_path, roi_name), exist_ok=True)
        
        # training feature and response
        print('loading train features ')
        # training feature and response
        train_feature = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-train-feature.npy'), mmap_mode='r')
        train_data = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-train-resp.npy'), mmap_mode='r')
        
        print('loading test features ')
        # test feature and response
        test_feature = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-feature.npy'), mmap_mode='r')
        test_data = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-resp.npy'), mmap_mode='r')
        
        print('loading quater file')
        idxs_dict = np.load(pjoin(concate_path, grid_name, f'{sub}_{roi_name}-retino-grids-idxs.npy'), allow_pickle=True).item()
        
        grid_models = {}
        grid_performance_cor, grid_performance_ev = np.zeros((len(eccsplitlines), len(angsplotlines)-1)), np.zeros((len(eccsplitlines), len(angsplotlines)-1))
        for grid, grid_voxel_idx in idxs_dict.items():
            grid_i, grid_j = int(grid[1]), int(grid[3])
            if len(grid_voxel_idx) == 0:
                grid_models[grid] = None
                grid_performance_cor[grid_i, grid_j] = np.nan
                grid_performance_ev[grid_i, grid_j] = np.nan
            else:                
                grid_train_feature = train_feature[grid_voxel_idx]
                grid_train_data = train_data[grid_voxel_idx]
                grid_test_feature = test_feature[grid_voxel_idx]
                grid_test_data = test_data[grid_voxel_idx]
                print('new shapes:', grid_train_feature.shape, grid_train_data.shape, grid_test_feature.shape, grid_test_data.shape)
                
                # data reshape
                grid_train_feature = grid_train_feature.reshape((-1, train_feature.shape[-1]))
                grid_train_y = grid_train_data.reshape(-1)
                grid_test_feature = grid_test_feature.reshape((-1, test_feature.shape[-1]))
                grid_test_y = grid_test_data.reshape(-1)

                N = grid_train_feature.shape[0]
                # 生成分组标记，每4000个样本内部每1000个样本为一组
                # 这里我们假设N是4000的整数倍，每个大单元里面有4个子组
                groups = np.tile(np.repeat(np.arange(4), 1000), N // 4000)

                # training linear model
                lr = LinearRegression(n_jobs=20)
                lr.fit(grid_train_feature, grid_train_y)
                # joblib.dump(lr, pjoin(performance_path, roi_name, f'{sub}_layer-{layername}_{roi_name}-linear.pkl'))
                grid_models[grid] = lr

                y_pred = lr.predict(grid_test_feature)
                # 计算并记录性能指标
                grid_performance_cor[grid_i, grid_j] = np.corrcoef(grid_test_y, y_pred)[0,1]
                grid_performance_ev[grid_i, grid_j] = lr.score(grid_test_feature, grid_test_y)
                print(sub, grid, layername,grid_performance_cor[grid_i, grid_j], grid_performance_ev[grid_i, grid_j])
        all_grid_performance_cor[isub] = grid_performance_cor
        all_grid_performance_ev[isub] = grid_performance_ev
        np.save(pjoin(performance_path, roi_name, f'{sub}_layer-{layername}_grid-models.npy'), grid_models)
    np.save(pjoin(performance_path, roi_name, f'all-sub_layer-{layername}_grid-test-cor.npy'), all_grid_performance_cor)
    np.save(pjoin(performance_path, roi_name, f'all-sub_layer-{layername}_grid-test-expvar.npy'), all_grid_performance_ev)

