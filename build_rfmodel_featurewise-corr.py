import os
import gc
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

def calc_explained_var_and_corr(x, beta, y):
    return  1 - np.mean((x @ beta - y)**2) / np.var(y), np.corrcoef(x @ beta, y)[0, 1]

def adjust_RF(receptive_field):
    receptive_field[receptive_field < np.mean(receptive_field)] = 0
    receptive_field = receptive_field + np.abs(np.min(receptive_field, None))
    receptive_field = receptive_field / (receptive_field.sum() + 1e-20)
    return receptive_field

def save_result(result, indexname):
    global performance_path, voxel_indices, sub, mask_name, layername
    if result.ndim > 1:
        all_performace = np.nan * np.zeros((result.shape[1], 59412))
        all_performace[:, voxel_indices] = np.array(result).transpose()
    else:
        all_performace = np.nan * np.zeros((1, 59412))
        all_performace[:, voxel_indices] = np.array(result)
    np.save(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_{indexname}.npy'), all_performace)

def compute_fullmodel_stats(idx, voxel):
    global X, y, i, j, labels
    coefficients = np.nan*np.zeros(X.shape[1])
    standard_errors = np.nan*np.zeros(X.shape[1])
    p_values = np.nan*np.zeros(X.shape[1])
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        # 取出当前体素的训练集和测试集神经活动
        y_voxel = y[:, idx]
        model = sm.OLS(y_voxel, X_voxel)
        model_summary = model.fit()
        # 提取所需的统计量：回归系数、标准误差、p 值
        coefficients = model_summary.params
        standard_errors = model_summary.bse
        p_values = model_summary.pvalues
    return {'fullm-coef': coefficients, 'fullm-bse': standard_errors, 
            'fullm-p': p_values}   

def compute_fullmodel_expvar_debug(idx, voxel):
    global X, X_test, y, y_test, i, j, labels
    # initialize the outputs
    full_r2_test, full_r2_train = np.nan, np.nan
    full_r_test, full_r_train = np.nan, np.nan
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        X_voxel_test = (X_voxel_test - X_voxel.mean(axis=0))/ X_voxel.std(axis=0)
        # 取出当前体素的训练集和测试集神经活动
        y_voxel = y[:, idx]
        y_voxel_test = y_test[:, idx]
        
        lr = LinearRegression()
        lr.fit(X_voxel, y_voxel)
        full_r2_train = lr.score(X_voxel, y_voxel)
        full_r2_test = lr.score(X_voxel_test, y_voxel_test)
        full_r_train = np.corrcoef(lr.predict(X_voxel), y_voxel)[0,1]
        full_r_test = np.corrcoef(lr.predict(X_voxel_test), y_voxel_test)[0,1]

        # ============== 可能存在bug的区域 =================
        # # full model betas, no need for inception 'cause the mean is substracted 
        # beta_all = np.linalg.lstsq(X_voxel, y_voxel, rcond=None)[0]
        # # # calc the full model performance on train set
        # full_r2_train, full_r_train = calc_explained_var_and_corr(X_voxel, beta_all, y_voxel)
        # # calc the full model performance on test set
        # full_r2_test, full_r_test = calc_explained_var_and_corr(X_voxel_test, beta_all, y_voxel_test)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    return {'full-model-ev': full_r2_test, 'full-model-ev-train': full_r2_train, 
            'full-model-r': full_r_test, 'full-model-r-train': full_r_train}

def compute_fullmodel_expvar(idx, voxel):
    global X, X_test, y, y_test, i, j, labels
    # initialize the outputs
    full_r2_test, full_r2_train = np.nan, np.nan
    full_r_test, full_r_train = np.nan, np.nan
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        X_voxel_test = zscore(X_voxel_test)
        # 取出当前体素的训练集和测试集神经活动
        y_voxel = y[:, idx]
        y_voxel_test = y_test[:, idx]
        # full model betas, no need for inception 'cause the mean is substracted 
        beta_all = np.linalg.lstsq(X_voxel, y_voxel, rcond=None)[0]
        # # calc the full model performance on train set
        full_r2_train, full_r_train = calc_explained_var_and_corr(X_voxel, beta_all, y_voxel)
        # calc the full model performance on test set
        full_r2_test, full_r_test = calc_explained_var_and_corr(X_voxel_test, beta_all, y_voxel_test)
    return {'full-model-ev': full_r2_test, 'full-model-ev-train': full_r2_train, 
            'full-model-r': full_r_test, 'full-model-r-train': full_r_train}

def compute_feature_unique_var_insetmodel(idx, voxel):
    global X, X_test, y, y_test, i, j, labels, set_r2s_train, set_r2s_test, set_rs_train, set_rs_test
    # feature info
    n_features = X.shape[1]
    # initialize the outputs
    # unique indices
    unique_vars = np.nan * np.zeros(n_features)
    r_diff = np.nan * np.zeros(n_features)
    unique_vars_train = np.nan * np.zeros(n_features)
    r_diff_train = np.nan * np.zeros(n_features)

    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        X_voxel_test = zscore(X_voxel_test)
        # 取出当前体素的训练集和测试集神经活动
        y_voxel = y[:, idx]
        y_voxel_test = y_test[:, idx]

        # # calc the full model performance on train set
        voxel_set_r2s_train, voxel_set_rs_train = set_r2s_train[:, voxel], set_r2s_test[:, voxel]
        # calc the full model performance on test set
        voxel_set_r2s_test, voxel_set_rs_test = set_rs_train[:, voxel], set_rs_test[:, voxel]
        for ilabel, (key, value) in enumerate(labels.items()):
            set_r2_train, set_r_train = voxel_set_r2s_train[ilabel], voxel_set_rs_train[ilabel]
            set_r2_test, set_r_test = voxel_set_r2s_test[ilabel], voxel_set_rs_test[ilabel]
            if len(value) >1 :
                X_set = X_voxel[:, np.array(value)]
                X_set_test = X_voxel_test[:, np.array(value)]
                for ivalue, featureidx in enumerate(value):
                    # 取出除了当前大类其他所有特征
                    X_others = np.delete(X_set, ivalue, axis=1)
                    # 拟合方程参数
                    beta_others = np.linalg.lstsq(X_others, y_voxel, rcond=None)[0]
                    
                    # 计算其余大类模型预测能力
                    # 在训练集上
                    other_r2_train, other_r_train = calc_explained_var_and_corr(X_others, beta_others, y_voxel)
                    # 在测试集上
                    other_r2_test, other_r_test = calc_explained_var_and_corr(np.delete(X_set_test, ivalue, axis=1), beta_others, y_voxel_test)
                    
                    # 计算独立解释方差和相关差异
                    # 在训练集上
                    unique_vars_train[featureidx] = set_r2_train- other_r2_train
                    r_diff_train[featureidx] = set_r_train - other_r_train
                    # 在测试集上
                    unique_vars[featureidx] = set_r2_test - other_r2_test
                    r_diff[featureidx] = set_r_test - other_r_test
            else:
                # 在训练集上
                unique_vars_train[featureidx] = set_r2_train
                r_diff_train[featureidx] = set_r_train
                # 在测试集上
                unique_vars[featureidx] = set_r2_test
                r_diff[featureidx] = set_r_test
    
    return {'uv-winthin':unique_vars, 'rd-winthin':r_diff, 'uv-train-winthin':unique_vars_train, 'rd-train-winthin':r_diff_train}

def compute_feature_unique_var(idx, voxel):
    global X, X_test, y, y_test, i, j
    # feature info
    n_features = X.shape[1]
    # initialize the outputs
    # unique indices
    unique_vars = np.nan * np.zeros(n_features)
    r_diff = np.nan * np.zeros(n_features)
    unique_vars_train = np.nan * np.zeros(n_features)
    r_diff_train = np.nan * np.zeros(n_features)
    # simple indices
    feature_vars = np.nan * np.zeros(n_features)
    feature_r = np.nan * np.zeros(n_features)
    feature_vars_train = np.nan * np.zeros(n_features)
    feature_r_train = np.nan * np.zeros(n_features)
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        X_voxel_test = zscore(X_voxel_test)
        # 取出当前体素的训练集和测试集神经活动
        y_voxel = y[:, idx]
        y_voxel_test = y_test[:, idx]
        # full model betas, no need for inception 'cause the mean is substracted 
        beta_all = np.linalg.lstsq(X_voxel, y_voxel, rcond=None)[0]
        # # calc the full model performance on train set
        full_r2_train, full_r_train = calc_explained_var_and_corr(X_voxel, beta_all, y_voxel)
        # calc the full model performance on test set
        full_r2_test, full_r_test = calc_explained_var_and_corr(X_voxel_test, beta_all, y_voxel_test)
        for ifeature in range(n_features):
            # 取出当前大类的相关特征
            X_ft = X_voxel[:, np.array([ifeature])]
            # 取出除了当前大类其他所有特征
            X_others = np.delete(X_voxel, ifeature, axis=1)
            # 拟合方程参数
            beta_ft = np.linalg.lstsq(X_ft, y_voxel, rcond=None)[0]
            beta_others = np.linalg.lstsq(X_others, y_voxel, rcond=None)[0]
            
            # 计算当前大类模型预测能力
            # 在训练集上
            ft_r2_train, ft_r_train = calc_explained_var_and_corr(X_ft, beta_ft, y_voxel)
            # 在测试集上
            ft_r2_test, ft_r_test = calc_explained_var_and_corr(X_voxel_test[:, np.array([ifeature])], beta_ft, y_voxel_test)

            feature_vars[ifeature],  feature_r[ifeature] = ft_r2_test, ft_r_test
            feature_vars_train[ifeature],  feature_r_train[ifeature] = ft_r2_train, ft_r_train

            # 计算其余大类模型预测能力
            # 在训练集上
            other_r2_train, other_r_train = calc_explained_var_and_corr(X_others, beta_others, y_voxel)
            # 在测试集上
            other_r2_test, other_r_test = calc_explained_var_and_corr(np.delete(X_voxel_test, ifeature, axis=1), beta_others, y_voxel_test)
            
            # 计算独立解释方差和相关差异
            # 在训练集上
            unique_vars_train[ifeature] = full_r2_train - other_r2_train
            r_diff_train[ifeature] = full_r_train - other_r_train
            # 在测试集上
            unique_vars[ifeature] = full_r2_test - other_r2_test
            r_diff[ifeature] = full_r_test - other_r_test

    return {'ft-uv':unique_vars, 'ft-rd':r_diff, 'model-ft-ev':feature_vars, 'model-ft-r':feature_r, 
            'ft-uv-train':unique_vars_train, 'ft-rd-train':r_diff_train, 
            'model-ft-ev-train':feature_vars_train, 'model-ft-r-train':feature_r_train}

def compute_category_unique_var(idx, voxel):
    global X, X_test, y, y_test, i, j, labels
    # category info
    n_category = len(labels.keys())
    # initialize the outputs
    # unique indices
    unique_vars = np.nan * np.zeros(n_category)
    r_diff = np.nan * np.zeros(n_category)
    unique_vars_train = np.nan * np.zeros(n_category)
    r_diff_train = np.nan * np.zeros(n_category)
    # simple indices
    category_vars = np.nan * np.zeros(n_category)
    category_r = np.nan * np.zeros(n_category)
    category_vars_train = np.nan * np.zeros(n_category)
    category_r_train = np.nan * np.zeros(n_category)
    full_r2_test, full_r2_train = np.nan, np.nan
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx} == {voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        X_voxel_test = np.sum(X_test * receptive_field, axis=(2,3))
        # 特征标准化, 均值都已经减掉了
        X_voxel = zscore(X_voxel)
        X_voxel_test = zscore(X_voxel_test)
        # 取出当前体素的训练集和测试集神经活动
        y_voxel = y[:, idx]
        y_voxel_test = y_test[:, idx]
        # full model betas, no need for inception 'cause the mean is substracted 
        beta_all = np.linalg.lstsq(X_voxel, y_voxel, rcond=None)[0]
        # # calc the full model performance on train set
        full_r2_train, full_r_train = calc_explained_var_and_corr(X_voxel, beta_all, y_voxel)
        # calc the full model performance on test set
        full_r2_test, full_r_test = calc_explained_var_and_corr(X_voxel_test, beta_all, y_voxel_test)
        for ilabel, (key, value) in enumerate(labels.items()):
            # 取出当前大类的相关特征
            X_cate = X_voxel[:, np.array(value)]
            # 取出除了当前大类其他所有特征
            X_others = np.delete(X_voxel, value, axis=1)
            # 拟合方程参数
            beta_cate = np.linalg.lstsq(X_cate, y_voxel, rcond=None)[0]
            beta_others = np.linalg.lstsq(X_others, y_voxel, rcond=None)[0]
            
            # 计算当前大类模型预测能力
            # 在训练集上
            cate_r2_train, cate_r_train = calc_explained_var_and_corr(X_cate, beta_cate, y_voxel)
            # 在测试集上
            cate_r2_test, cate_r_test = calc_explained_var_and_corr(X_voxel_test[:, np.array(value)], beta_cate, y_voxel_test)

            category_vars[ilabel],  category_r[ilabel] = cate_r2_test, cate_r_test
            category_vars_train[ilabel],  category_r_train[ilabel] = cate_r2_train, cate_r_train

            # 计算其余大类模型预测能力
            # 在训练集上
            other_r2_train, other_r_train = calc_explained_var_and_corr(X_others, beta_others, y_voxel)
            # 在测试集上
            other_r2_test, other_r_test = calc_explained_var_and_corr(np.delete(X_voxel_test, value, axis=1), beta_others, y_voxel_test)
            
            # 计算独立解释方差和相关差异
            # 在训练集上
            unique_vars_train[ilabel] = full_r2_train - other_r2_train
            r_diff_train[ilabel] = full_r_train - other_r_train
            # 在测试集上
            unique_vars[ilabel] = full_r2_test - other_r2_test
            r_diff[ilabel] = full_r_test - other_r_test           
    return {'uv':unique_vars, 'rd':r_diff, 'model-ctg-ev':category_vars, 'model-ctg-r':category_r, 
            'uv-train':unique_vars_train, 'rd-train':r_diff_train, 
            'model-ctg-ev-train':category_vars_train, 'model-ctg-r-train':category_r_train, 'full-model-ev': full_r2_test}

def compute_partial_correlation_withincate(idx, voxel):

    global X, y, i, j, labels

    n_features = X.shape[1]
    partial_correlations = np.nan*np.zeros(n_features)
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx}={voxel}')
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        y_voxel = y[:, idx]
        for key, value in labels.items():
            # 取出当前大类的相关特征
            X_cate = X_voxel[:, np.array(value)] 
            if len(value) > 1:
                for ifeature, ichannel in enumerate(value):
                    # 选择当前特征和剩余特征
                    X_i = X_cate[:, ifeature]
                    X_rest = np.delete(X_cate, ifeature, axis=1)

                    # 计算残差
                    beta_i = np.linalg.lstsq(X_rest, X_i, rcond=None)[0]
                    beta_y = np.linalg.lstsq(X_rest, y_voxel, rcond=None)[0]

                    res_i = X_i - X_rest @ beta_i
                    res_y = y_voxel - X_rest @ beta_y

                    # 计算偏相关系数
                    r = np.corrcoef(res_i, res_y)[0,1]
                    partial_correlations[ichannel] = r
            else:
                partial_correlations[np.array(value)] = np.corrcoef(np.squeeze(X_cate), y_voxel)[0,1]
            
    return partial_correlations

def compute_partial_correlation(idx, voxel):

    global X, y, i, j

    n_features = X.shape[1]
    partial_correlations = np.nan*np.zeros(n_features)
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        print(f'{idx}={voxel}') 
        # load receptive field
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        y_voxel = y[:, idx]
        for ifeature in range(n_features):
            # 选择当前特征和剩余特征
            X_i = X_voxel[:, ifeature]
            X_rest = np.delete(X_voxel, ifeature, axis=1)

            # 计算残差
            beta_i = np.linalg.lstsq(X_rest, X_i, rcond=None)[0]
            beta_y = np.linalg.lstsq(X_rest, y_voxel, rcond=None)[0]

            res_i = X_i - X_rest @ beta_i
            res_y = y_voxel - X_rest @ beta_y

            # 计算偏相关系数
            r = np.corrcoef(res_i, res_y)[0,1]
            partial_correlations[ifeature] = r

    return partial_correlations

def compute_voxel_correlation(idx, voxel):
    global X, y, i, j
    
    # load receptive field
    if not voxel in guassparams.keys():
        pass
        simple_cor = np.nan*np.zeros((X.shape[1],))
    else:
        print(f'{idx}={voxel}')
        # load receptive field
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        receptive_field = adjust_RF(receptive_field)
        # saptial summation
        X_voxel = np.sum(X * receptive_field, axis=(2,3))
        # assign to corr matrix
        simple_cor = np.corrcoef(X_voxel.transpose(), y[:, idx][np.newaxis, :])[:,-1][0 : X.shape[1]]
                
    return simple_cor

# subs = ['sub-03'] # ,, 'sub-05' 'sub-04', 'sub-02','sub-06', 'sub-07''sub-04', 'sub-08', 'sub-02','sub-06', 'sub-07','sub-09'
# subs = [ 'sub-03']#
subs = [f'sub-0{isub+1}' for isub in range(9)] 
sub = subs[-1]
sleep_time = 5200*(subs.index(sub)+1)
check_file = f'/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/build/gaussianparams/{sub}_layer-googlenet-conv2_Gauss.npy'
print(f'wait {sub}; sleeping for {sleep_time} seconds')
# time.sleep(sleep_time)
while 1:
    if os.path.exists(check_file):
        print('file ready!')
        break
    else:
        print(f'checkfile not exists, wait again')
        time.sleep(180)
# subs = [sub]

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
        
        # input path
        resp_path = pjoin(work_dir, 'prep/brain_response')
        voxel_mask_path = pjoin(work_dir, 'prep/voxel_masks')
        image_activations_path = pjoin(work_dir, 'prep/image_activations')
        retino_path = pjoin(work_dir, 'build/retinoparams')
        guass_path = pjoin(work_dir, 'build/gaussianparams')
        # save out path
        performance_path = pjoin(work_dir, 'build/featurewise-corr')
        # save path
        if not os.path.exists(pjoin(performance_path, sub)):
            os.makedirs(pjoin(performance_path, sub))

        # load
        brain_resp = np.load(pjoin(resp_path, f'{sub}_imagenet_beta.npy'))
        activations = np.load(pjoin(image_activations_path, f'{sub}_{layername}.npy'))
        coco_activations = np.load(pjoin(image_activations_path, f'{test_set_name}_{layername}.npy'))
        print(f'activations shape of {activations.shape}')
        guassparams = np.load(pjoin(guass_path, f'{sub}_layer-{layername}_Gauss.npy'), allow_pickle=True)[0]
        
        # load, reshape and average the resp
        test_resp = np.load(pjoin(resp_path, f'{sub}_{test_set_name}_beta.npy'))
        num_trial = test_resp.shape[0]
        num_run = int(num_trial/120)
        test_resp = test_resp.reshape((num_run, 120, 59412))
        
        
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
        voxel_indices = np.where(voxel_mask==1)[0]

        del test_resp, voxel_mask_nii, voxel_mask
        gc.collect()
        # collect resp in ROI
        brain_resp = brain_resp[:, voxel_indices]
        test_resp = test_resp[:, :, voxel_indices]

        # normalization
        norm_metric = 'session'
        brain_resp = train_data_normalization(brain_resp, metric=norm_metric)
        mean_test_resp = zscore(test_resp.mean(axis=0))
        num_voxel = brain_resp.shape[-1]

        # coordinate
        # Create grid data
        layer['size'] = activations.shape[-1]
        i = np.linspace(-8., 8., layer['size'])
        j = np.linspace(8., -8., layer['size'])
        i, j = np.meshgrid(i, j)

        if layername == 'googlenet-conv2':
            X = activations[:, 0:63, :, :]
            X_test = coco_activations[:, 0:63, :, :]
        y = brain_resp
        y_test = mean_test_resp
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
        
        voxels = voxel_indices.tolist()
        idxs = np.arange(num_voxel).tolist()
        # voxel_indices = voxel_indices
        # # simple corr
        # results = Parallel(n_jobs=25)(delayed(compute_voxel_correlation)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # all_performace = np.nan*np.zeros((X.shape[1], 59412))
        # all_performace[:, voxel_indices] = np.array(results).transpose()
        # np.save(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_corr.npy'), all_performace)
        
        # # partial corr
        # results = Parallel(n_jobs=30)(delayed(compute_partial_correlation)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # all_performace = np.nan*np.zeros((X.shape[1], 59412))
        # all_performace[:, voxel_indices] = np.array(results).transpose()
        # np.save(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_parcorr.npy'), all_performace)
        
        # # partial corr wihin feature category
        # results = Parallel(n_jobs=30)(delayed(compute_partial_correlation_withincate)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # all_performace = np.nan*np.zeros((X.shape[1], 59412))
        # all_performace[:, voxel_indices] = np.array(results).transpose()
        # np.save(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_parcorrincate.npy'), all_performace)

        # # unique var of feature category
        # results = Parallel(n_jobs=20)(delayed(compute_category_unique_var)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # # extract the indices and save out
        # for indexname in ['uv', 'rd', 'model-ctg-ev', 'model-ctg-r',
        #                   'uv-train', 'rd-train', 'model-ctg-ev-train', 'model-ctg-r-train']:
        #     index = np.array([ _[indexname] for _ in results])
        #     save_result(index, indexname)

        # results = Parallel(n_jobs=20)(delayed(compute_feature_unique_var_insetmodel)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # for indexname in ['uv-winthin', 'rd-winthin', 'uv-train-winthin', 'rd-train-winthin']:
        #     index = np.array([ _[indexname] for _ in results])
        #     save_result(index, indexname)

        # # 
        # results = Parallel(n_jobs=20)(delayed(compute_feature_unique_var)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # for indexname in ['ft-uv', 'ft-rd', 'model-ft-ev', 'model-ft-r',
        #                   'ft-uv-train', 'ft-rd-train', 'model-ft-ev-train', 'model-ft-r-train']:
        #     index = np.array([ _[indexname] for _ in results])
        #     save_result(index, indexname)

        results = Parallel(n_jobs=15)(delayed(compute_fullmodel_expvar_debug)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        for indexname in ['full-model-ev', 'full-model-ev-train', 'full-model-r', 'full-model-r-train']:
            index = np.array([ _[indexname] for _ in results])
            save_result(index, indexname)
            
        # unique_var_on_coco = np.array([ _['uv'] for _ in results])
        # r_diff_on_coco = np.array([ _['rd'] for _ in results])
        # cate_ev_on_coco = np.array([ _['model-ctg-ev'] for _ in results])
        # cate_r_on_coco = np.array([ _['model-ctg-r'] for _ in results])
        
        # unique_var_on_im = np.array([ _['uv-train'] for _ in results])
        # r_diff_on_im = np.array([ _['rd-train'] for _ in results])
        # cate_ev_on_im = np.array([ _['model-ctg-ev-train'] for _ in results])
        # cate_r_on_im = np.array([ _['model-ctg-r-train'] for _ in results])
            
        # results = Parallel(n_jobs=10)(delayed(compute_fullmodel_stats)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # for indexname in ['fullm-coef', 'fullm-bse', 'fullm-p']:
        #     index = np.array([ _[indexname] for _ in results])
        #     save_result(index, indexname)
        
    print(f'{sub} consume : {t.interval} s')
