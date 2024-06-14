import os
import gc
# import torch
# import torch.nn as nn
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

# path settings
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
# input path
concate_path = pjoin(work_dir, 'prep/roi-concate')
# save out path
axis_type =  'raw'#'ica'
performance_path = pjoin(work_dir, f'build/roi-concatemodel_feature-{axis_type}')
os.makedirs(performance_path, exist_ok=True)
inputlayername = 'googlenet-conv2' #'googlenet-maxpool2' # 'googlenet-inception3a' #'googlenet-maxpool1'  #
layer = {'name': inputlayername, 'size': net_size_info[inputlayername]}
layername = layer['name']
layername = layername.replace('.','')
#
mask_name = 'primaryvis-in-MMP' #
test_set_name = 'coco'
subs = [f'sub-0{isub+1}' for isub in range(0, 9)]
import time
# print('sleeping')
# time.sleep(7200)
for roi_name in [ 'V1', 'V2', 'V3' ]:# , ,'V4'
    val_scores, val_corrs = [], []
    test_scores, test_corrs = [], []
    # roi_name = 'V1'
    for sub in subs:
        t0 = time.time()
        if axis_type == 'raw':
            data_path = pjoin(concate_path, sub)
        else:
            data_path = pjoin(concate_path, sub, axis_type)
        model_path = pjoin(performance_path, roi_name)
        model = joblib.load(pjoin(model_path, f'{sub}_layer-{inputlayername}_{roi_name}-linear.pkl'))
        feature_sortedbyweights = np.argsort(np.abs(model.coef_))[::-1]
        print('loading train features ')
        # training feature and response
        train_feature = np.load(pjoin(data_path,  f'{sub}_layer-{layername}_{roi_name}-train-feature.npy'), mmap_mode='r')
        train_data = np.load(pjoin(data_path,  f'{sub}_layer-{layername}_{roi_name}-train-resp.npy'), mmap_mode='r')
        
        print('loading test features ')
        # test feature and response
        test_feature = np.load(pjoin(data_path,  f'{sub}_layer-{layername}_{roi_name}-test-feature.npy'), mmap_mode='r')
        test_data = np.load(pjoin(data_path,  f'{sub}_layer-{layername}_{roi_name}-test-resp.npy'), mmap_mode='r')
        
        if roi_name == 'V4':
            voxels = np.load(pjoin(data_path,  f'{sub}_layer-{layername}_{roi_name}-voxel.npy')).tolist()
            voxel_selection = np.load(pjoin(concate_path, sub,  f'{sub}_selection_V4-voxels.npy'))
            idxs_selection = np.array([voxels.index(_) for _ in voxel_selection])
            train_feature = train_feature[idxs_selection]
            train_data = train_data[idxs_selection]
            test_feature = test_feature[idxs_selection]
            test_data = test_data[idxs_selection]
            print('new shapes:', train_feature.shape, train_data.shape, test_feature.shape, test_data.shape)
        # data split and reshape
        train_feature1 = train_feature[:, 1::2,:].reshape((-1, train_feature.shape[-1]))
        train_y1 = train_data[:, 1::2].reshape(-1)
        train_feature2 = train_feature[:, 0::2,:].reshape((-1, train_feature.shape[-1]))
        train_y2 = train_data[:, 0::2].reshape(-1)
        split_half_performance = np.zeros((4, 63))
        for feature_num in range(63):
            lr = LinearRegression(n_jobs=20)
            sel_features = feature_sortedbyweights[0:(feature_num+1)]
            lr.fit(train_feature1[:, sel_features], train_y1)
            split_half_performance[0,feature_num] = np.corrcoef(lr.predict(train_feature1[:, sel_features]), train_y1)[0,1]
            split_half_performance[1,feature_num] = np.corrcoef(lr.predict(train_feature2[:, sel_features]), train_y2)[0,1]
            
            lr.fit(train_feature2[:, sel_features], train_y2)
            split_half_performance[2,feature_num] = np.corrcoef(lr.predict(train_feature2[:, sel_features]), train_y2)[0,1]
            split_half_performance[3,feature_num] = np.corrcoef(lr.predict(train_feature1[:, sel_features]), train_y1)[0,1]
            overfit = np.mean(split_half_performance[np.array([0,2]), feature_num])
            generaliztion = np.mean(split_half_performance[np.array([1,3]), feature_num])
            print(f'F{feature_num}: {overfit} , {generaliztion}')
        np.save(pjoin(performance_path, f'{sub}_model-{layername}-linear_{roi_name}_split-half-corr-v-featurenum.npy'), split_half_performance)
        del split_half_performance, train_feature1, train_y1, train_feature2, train_y2

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
        train_test_performance = np.zeros((2, 63))
        for feature_num in range(63):
            lr = LinearRegression(n_jobs=20)
            sel_features = feature_sortedbyweights[0:(feature_num+1)]
            lr.fit(train_feature[:, sel_features], train_y)
            # 计算并记录性能指标
            train_test_performance[0, feature_num] = np.corrcoef(train_y, lr.predict(train_feature[:, sel_features]))[0,1]
            train_test_performance[1, feature_num] = np.corrcoef(test_y, lr.predict(test_feature[:, sel_features]))[0,1]
            print(train_test_performance[1, feature_num], end=',')
        np.save(pjoin(performance_path, f'{sub}_model-{layername}-linear_{roi_name}_train-test-corr-v-featurenum.npy'), train_test_performance)

        print(sub, f'finished {time.time() - t0} s', roi_name)#, val_corrs[-1], val_scores[-1]
        print(train_test_performance[0])
        print(train_test_performance[1])