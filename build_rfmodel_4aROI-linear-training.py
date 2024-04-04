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

mask_name = 'primaryvis-in-MMP' #
test_set_name = 'coco'
subs = [f'sub-0{isub+1}' for isub in range(0, 9)]

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

for roi_name in ['V4']:# , , 'V3', 'V2', 'V1' 
    val_scores, val_corrs = [], []
    test_scores, test_corrs = [], []
    # roi_name = 'V1'
    for sub in subs:
        t0 = time.time()
        os.makedirs(pjoin(performance_path, roi_name), exist_ok=True)
        os.makedirs(pjoin(performance_path, 'performance'), exist_ok=True)
        print('loading train features ')
        # training feature and response
        train_feature = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-train-feature.npy'), mmap_mode='r')
        train_data = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-train-resp.npy'), mmap_mode='r')
        
        print('loading test features ')
        # test feature and response
        test_feature = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-feature.npy'), mmap_mode='r')
        test_data = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-test-resp.npy'), mmap_mode='r')
        
        if roi_name == 'V4':
            voxels = np.load(pjoin(concate_path, sub, f'{sub}_layer-{layername}_{roi_name}-voxel.npy')).tolist()
            voxel_selection = np.load(pjoin(concate_path, sub, f'{sub}_selection_V4-voxels.npy'))
            idxs_selection = np.array([voxels.index(_) for _ in voxel_selection])
            train_feature = train_feature[idxs_selection]
            train_data = train_data[idxs_selection]
            test_feature = test_feature[idxs_selection]
            test_data = test_data[idxs_selection]
            print('new shapes:', train_feature.shape, train_data.shape, test_feature.shape, test_data.shape)
        # data reshape
        train_feature = train_feature.reshape((-1, train_feature.shape[-1]))
        train_y = train_data.reshape(-1)
        test_feature = test_feature.reshape((-1, test_feature.shape[-1]))
        test_y = test_data.reshape(-1)

        N = train_feature.shape[0]
        # 生成分组标记，每4000个样本内部每1000个样本为一组
        # 这里我们假设N是4000的整数倍，每个大单元里面有4个子组
        groups = np.tile(np.repeat(np.arange(4), 1000), N // 4000)

        # training linear model
        lr = LinearRegression(n_jobs=20)
        lr.fit(train_feature, train_y)
        joblib.dump(lr, pjoin(performance_path, roi_name, f'{sub}_layer-{layername}_{roi_name}-linear.pkl'))
        y_pred = lr.predict(test_feature)
        # 计算并记录性能指标
        test_corrs.append(np.corrcoef(test_y, y_pred)[0,1])
        test_scores.append(lr.score(test_feature, test_y))

        # # 初始化GroupKFold，n_splits设置为4，因为我们想要进行4折交叉验证
        # gkf = GroupKFold(n_splits=4)
        # vallr = LinearRegression(n_jobs=10)

        # cv_scores = cross_val_score(vallr, train_feature, train_y, scoring='r2', cv=gkf, groups=groups)
        # val_scores.append(np.mean(cv_scores))

        # pearson_scorer = make_scorer(pearson_correlation, greater_is_better=True)
        # cv_scores = cross_val_score(vallr, train_feature, train_y, scoring=pearson_scorer,  cv=gkf, groups=groups)
        # val_corrs.append(np.mean(cv_scores))
   
        print(sub, f'finished {time.time() - t0} s', roi_name, test_corrs[-1], test_scores[-1])#, val_corrs[-1], val_scores[-1]
    # np.save(pjoin(performance_path,  'performance', f'all-sub_model-{layername}-linear_{roi_name}_validation-corr.npy'), np.array(val_corrs))
    # np.save(pjoin(performance_path,  'performance', f'all-sub_model-{layername}-linear_{roi_name}_validation-expvar.npy'), np.array(val_scores))
    np.save(pjoin(performance_path,  'performance', f'all-sub_model-{layername}-linear_{roi_name}_test-corr.npy'), np.array(test_corrs))
    np.save(pjoin(performance_path,  'performance', f'all-sub_model-{layername}-linear_{roi_name}_test-expvar.npy'), np.array(test_scores))
