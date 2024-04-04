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
from utils import train_data_normalization, net_size_info

def save_result(result, indexname):
    global performance_path, roi_name, sub, layername
    os.makedirs(pjoin(performance_path, sub, layername), exist_ok=True)
    np.save(pjoin(performance_path, sub, layername, f'{sub}_{roi_name}-{indexname}.npy'), np.array(result))


def compute_model_performance_with_stats(idx):
    global train_feature, train_data, test_feature, test_data
    
    X_voxel = train_feature[idx]
    y_voxel = train_data[idx]
    X_voxel_test = test_feature[idx]
    y_voxel_test = test_data[idx]
    print(f'{sub} : {idx}')
    coefficients = np.nan*np.zeros(X_voxel.shape[1])
    standard_errors = np.nan*np.zeros(X_voxel.shape[1])
    p_values = np.nan*np.zeros(X_voxel.shape[1])
    # initialize the outputs
    full_r2_test, full_r2_train = np.nan, np.nan
    full_r_test, full_r_train = np.nan, np.nan
    fp_value = np.nan
    # check nan
    test_nan, train_nan = 0, 0
    
    lr = LinearRegression()
    lr.fit(X_voxel, y_voxel)

    full_r2_test = lr.score(X_voxel_test, y_voxel_test)
    full_r_test = np.corrcoef(lr.predict(X_voxel_test), y_voxel_test)[0,1]
    
    # 计算统计量 
    model = sm.OLS(y_voxel, X_voxel)
    model_summary = model.fit()
    # 提取所需的统计量：回归系数、标准误差、p 值
    coefficients = model_summary.params
    standard_errors = model_summary.bse
    p_values = model_summary.pvalues
    # f_statistic = model.fvalue
    fp_value = model_summary.f_pvalue
    return {'fullm-coef': coefficients, 'fullm-bse': standard_errors, 
            'fullm-p': p_values, 'fullm-f-pvalue':fp_value, 
            'test-cor': full_r_test, 'test-ev' : full_r2_test, 
            'testnan' : test_nan, 'trainnan' : train_nan}

# path settings
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
# input path
concate_path = pjoin(work_dir, 'prep/roi-concate')
# save out path
performance_path = pjoin(work_dir, 'build/roi-voxelwisemodel')
os.makedirs(performance_path, exist_ok=True)
inputlayernames = ['googlenet-conv2', 'googlenet-maxpool2', 'googlenet-inception3a'] #'googlenet-conv2' #'googlenet-maxpool2' # 'googlenet-inception3a' #
rois =  ['V1', 'V2', 'V3', 'V4']
test_set_name = 'coco'
subs = [f'sub-0{isub+1}' for isub in range(0, 9)]

for inputlayername in inputlayernames:
    layer = {'name': inputlayername, 'size': net_size_info[inputlayername]}
    layername = layer['name']
    layername = layername.replace('.','')

    for roi_name in rois:
        for sub in subs[1::]:
            t0 = time.time()
            print(sub, layername)
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
            # data 
            idxs = np.arange(train_feature.shape[0])

            results = Parallel(n_jobs=25)(delayed(compute_model_performance_with_stats)(idx) for idx in idxs)
            for indexname in ['fullm-coef', 'fullm-bse', 'fullm-p', 'fullm-f-pvalue',
                            'test-ev', 'test-cor', 'testnan', 'trainnan']:
                index = np.array([ _[indexname] for _ in results])
                save_result(index, indexname)
    
            print(sub, f'finished {time.time() - t0} s', roi_name, layername)
            