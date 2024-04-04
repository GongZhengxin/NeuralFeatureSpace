import os
import gc
import torch
import time
import torch.nn as nn
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from os.path import join as pjoin
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from scipy.stats import zscore
import joblib
from scipy.optimize import curve_fit

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso
from utils import train_data_normalization, Timer, net_size_info, conv2_labels, get_roi_data, ReceptiveFieldProcessor

# 定义生成数列的函数
def generate_pairs(n, step=120):
    return [(i*step, (i+1)*step) for i in range(n)]

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

subs = [f'sub-0{isub+1}' for isub in range(0, 9)]
inputlayername = 'googlenet-conv2' 
layer = {'name': inputlayername, 'size': net_size_info[inputlayername.replace('raw-', '')]}
layername = layer['name']
labels = conv2_labels
mask_name = 'primaryvis-in-MMP' #'fixretfloc-in-subj'
test_set_name = 'coco'

# model_types = ['priorwRF-crsub-ls-wnsub-ls-N', 
#                 'priorwRF-crsub-ls-wnsub-ls-U', 
#                 'priorRF-crsub-ls-wnsub-ls-N',
#                 'priorRF-crsub-ls-wnsub-ls-U',
#                 'globalmean-crsub-ls-wnsub-ls-N',
#                 'globalmean-crsub-ls-wnsub-ls-U' ]
model_types = ['priorRF', 'globalmean', 'priorwRF'] #'priorwRF-crsub-ls-wnsub-lr-N' #'globalmean' #'linearmean' # 'priorwRF' # #!

for model_type in model_types:
    # path settings
    work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
    opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
    # input path
    resp_path = pjoin(work_dir, 'prep/brain_response')
    voxel_mask_path = pjoin(work_dir, 'prep/voxel_masks')
    image_activations_path = pjoin(work_dir, 'prep/image_activations')
    concate_feature_path = pjoin(work_dir, 'prep/concate_features')
    retino_path = pjoin(work_dir, 'build/retinoparams')
    guass_path = pjoin(work_dir, 'build/gaussianparams')
    avgrf_path = pjoin(work_dir, 'prep/image_mask')
    filter_path = pjoin(work_dir, 'prep/feature_filters')
    # save out path
    performance_path = pjoin(work_dir, 'build/lasso-feature-selection/')
    if not os.path.exists(pjoin(performance_path, model_type)):
        os.makedirs(pjoin(performance_path, model_type))
    # coordinate
    # Create grid data
    i = np.linspace(-8., 8., layer['size'])
    j = np.linspace(8., -8., layer['size'])
    i, j = np.meshgrid(i, j)

    # concatenate features and brain responses
    concate_X, concate_X_test = [], []
    concate_y, concate_y_test = [], []


    # for sub in subs:
    #     rf_file = f'{sub}_prior-weighted-receptivefield-V2.npy' 
    #     if os.path.exists(pjoin(avgrf_path, rf_file)):
    #         continue
    #     # getting retinotopic voxels
    #     sub_prf_file = os.path.join(opendata_cifti_pth, sub, 'results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii')
    #     prf_r2 = nib.load(sub_prf_file).get_fdata()[3, :]
    #     r2_thres = 10
    #     voxel_mask = np.where(prf_r2 > r2_thres)[0]

    #     # select ROI names
    #     evc_pool = ['V2'] 
    #     # aggregate ROI vertices
    #     roi_name = [__  for _ in [evc_pool] for __ in _]
    #     # form ROI mask
    #     selection_mask = np.sum([get_roi_data(None, _) for _ in roi_name], axis=0)
    #     # trsnfer to indices in cifti space
    #     voxel_indices = [_ for _ in np.where(selection_mask==1)[0] if _ in voxel_mask]

    #     retino_info = nib.load(sub_prf_file).get_fdata()
    #     n_vertices = retino_info.shape[-1]
    #     # we need to transfer the params into (x,y,size) model
    #     trans_retinotopy_params = np.zeros((n_vertices, 3))
    #     trans_retinotopy_params[:,0] = np.cos(retino_info[0, 0:n_vertices]/180*np.pi)*retino_info[1, 0:n_vertices]*16/200
    #     trans_retinotopy_params[:,1] = np.sin(retino_info[0, 0:n_vertices]/180*np.pi)*retino_info[1, 0:n_vertices]*16/200
    #     trans_retinotopy_params[:,2] = retino_info[2, 0:n_vertices]*16/200

    #     pos2vf = lambda x : (x-int(layer['size']/2)) * 16 / layer['size']
    #     fwhm2sigma = lambda x : x / 2.354282

    #     receptive_field = np.zeros((57, 57))
    #     for voxel in voxel_indices:
    #         params = [1,*trans_retinotopy_params[voxel],trans_retinotopy_params[voxel,2], 0]
    #         # receptive_field += adjust_RF(gaussian_2d((i, j), *params))
    #         receptive_field += 0.01 * prf_r2[voxel] * adjust_RF(gaussian_2d((i, j), *params))

    #     receptive_field = receptive_field / receptive_field.sum()
    #     np.save(pjoin(avgrf_path, rf_file), receptive_field)
    feature_type = model_type.split('-')[0]
    check_file = pjoin(concate_feature_path, f'cross-sub_{feature_type.lower()}_imn_conv2-V3.npy') #!
    if not os.path.exists(check_file):
        for sub in subs:
            print(sub, mask_name, layername)
            # sub dir
            prf_path = pjoin(opendata_cifti_pth, sub, 'results/ses-prf_task-prf')
            sub_prf_file = pjoin(prf_path, 'ses-prf_task-prf_params.dscalar.nii')
            # save path
            if not os.path.exists(pjoin(performance_path, model_type)):
                os.makedirs(pjoin(performance_path, model_type))

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
            # average receptive field
            pri_receptivefield = np.load(pjoin(avgrf_path, f'{sub}_prior-average-receptivefield-V3.npy')) #!
            pri_weightedrf =  np.load(pjoin(avgrf_path, f'{sub}_prior-weighted-receptivefield-V3.npy')) #!
            # load
            brain_resp = np.load(pjoin(resp_path, f'{sub}_imagenet_beta.npy'))
            activations = np.load(pjoin(image_activations_path, f'{sub}_{layername}.npy'))
            coco_activations = np.load(pjoin(image_activations_path, f'{test_set_name}_{layername}.npy'))
            print(f'activations shape of {activations.shape}')

            # load, reshape and average the resp
            test_resp = np.load(pjoin(resp_path, f'{sub}_{test_set_name}_beta.npy'))
            num_trial = test_resp.shape[0]
            num_run = int(num_trial/120)
            test_resp = test_resp.reshape((num_run, 120, 59412))
            mean_test_resp = test_resp.mean(axis=0)

            # collect resp and summed to ROI response
            roi_brain_resp = np.atleast_2d(brain_resp[:, v3_voxels].sum(axis=1)).T #!
            roi_mean_test_resp = np.atleast_2d(mean_test_resp[:, v3_voxels].sum(axis=1)).T #!

            # normalization
            norm_metric = 'session'
            roi_brain_resp = train_data_normalization(roi_brain_resp, metric=norm_metric)
            # mean_test_resp = zscore(test_resp.mean(axis=0))
            roi_mean_test_resp = zscore(roi_mean_test_resp, None)

            del test_resp, voxel_mask, brain_resp
            gc.collect()

            if layername == 'googlenet-conv2':
                X = activations[:, 0:63, :, :]
                X_test = coco_activations[:, 0:63, :, :]
            else:
                X = activations
                X_test = coco_activations

            # X_avg = zscore(np.sum(X * avg_receptivefield, axis=(2,3)))
            # X_test_avg = zscore(np.sum(X_test * avg_receptivefield, axis=(2,3)))

            if model_type == 'globalmean':
                X_avg = zscore(np.mean(X, axis=(2,3)))
                X_test_avg = zscore(np.mean(X_test, axis=(2,3)))
            if model_type == 'priorRF':
                X_avg = zscore(np.mean(X * pri_receptivefield, axis=(2,3)))
                X_test_avg = zscore(np.mean(X_test * pri_receptivefield, axis=(2,3)))
            if model_type == 'priorwRF':
                X_avg = zscore(np.mean(X * pri_weightedrf, axis=(2,3)))
                X_test_avg = zscore(np.mean(X_test * pri_weightedrf, axis=(2,3)))        
            
            y = roi_brain_resp
            y_test = roi_mean_test_resp

            concate_X.append(X_avg) 
            concate_X_test.append(X_test_avg)
            concate_y.append(y) 
            concate_y_test.append(y_test)

        np.save(pjoin(concate_feature_path, f'cross-sub_{model_type.lower()}_imn_conv2-V3.npy'), np.vstack(concate_X)) #!
        np.save(pjoin(concate_feature_path, f'cross-sub_{model_type.lower()}_coco_conv2-V3.npy'), np.vstack(concate_X_test)) #!
        if model_type=='priorRF':
            np.save(pjoin(concate_feature_path, 'cross-sub_imn_averaged-V3.npy'), np.vstack(concate_y)) #!
            np.save(pjoin(concate_feature_path, 'cross-sub_coco_averaged-V3.npy'), np.vstack(concate_y_test)) #!

    # concate_X = np.load(pjoin(concate_feature_path, f'cross-sub_{feature_type.lower()}_imn_conv2-V2.npy'))
    # concate_X_test = np.load(pjoin(concate_feature_path, f'cross-sub_{feature_type.lower()}_coco_conv2-V2.npy'))
    # concate_y = np.load(pjoin(concate_feature_path, 'cross-sub_imn_averaged-V2.npy'))
    # concate_y_test = np.load(pjoin(concate_feature_path, 'cross-sub_coco_averaged-V2.npy'))

#     print('finished')
#     # # ALL TOGETHER FIT 
#     # concate_X = np.vstack(concate_X)
#     # concate_X_test = np.vstack(concate_X_test)
#     # concate_y = np.vstack(concate_y)
#     # concate_y_test = np.vstack(concate_y_test)

#     # # 1. 先由 lasso 找到被试间泛化性有用的特征
#     # # 定义要测试的alpha值的范围
#     # alphas = np.logspace(-3, -1, 50)  # 创建一个覆盖广泛范围的alpha值
#     # scores = []  # 用于存储每个alpha对应的平均交叉验证分数
#     # # 自定义9-fold交叉验证
#     # cv = KFold(n_splits=9, shuffle=False)
#     # # 对每个alpha进行评估
#     # for alpha in alphas:
#     #     lasso = Lasso(alpha=alpha)
#     #     cv_scores = cross_val_score(lasso, concate_X, concate_y, scoring='r2', cv=cv)
#     #     scores.append(np.mean(cv_scores))
#     # # 创建Lasso模型实例
#     # max_alpha = alphas[np.argmax(scores)]
#     # lasso = Lasso(alpha=max_alpha)
#     # print('max_alpha:', max_alpha, 'at where', np.argmax(scores))

#     # # 初始化9折交叉验证，保持样本顺序
#     # kf = KFold(n_splits=9, shuffle=False)

#     # # 用于存储每次的性能指标
#     # accuracy_scores, explained_scores = [], []
#     # non_zero_features = []
#     # # 进行9折交叉验证
#     # print('lasso validating')
#     # for train_index, test_index in kf.split(concate_X):
#     #     print('.', end=' ')
#     #     # 分割数据
#     #     X_train, X_test = concate_X[train_index], concate_X[test_index]
#     #     y_train, y_test = concate_y[train_index], concate_y[test_index]
#     #     # 训练模型
#     #     lasso.fit(X_train, y_train)
#     #     #
#     #     non_zero_features.append(np.where(lasso.coef_!=0)[0])
#     #     # 预测测试集
#     #     y_pred = lasso.predict(X_test)
#     #     # 计算并记录性能指标
#     #     accuracy_scores.append(np.corrcoef(y_test[:,0], y_pred)[0,1])
#     #     explained_scores.append(lasso.score(X_test, y_test))

#     # print('saving validation performance')
#     # np.save(pjoin(performance_path, model_type, f'nod-cross-sub_model-{model_type}_validation-corr.npy'), np.array(accuracy_scores))
#     # np.save(pjoin(performance_path, model_type, f'nod-cross-sub_model-{model_type}_validation-expvar.npy'), np.array(explained_scores))

#     # # 找到对所有被试间泛化都有用的特征交集
#     # select_feature_intersection = set(list(range(64)))
#     # for non_zero_feature in non_zero_features:
#     #     select_feature_intersection = select_feature_intersection & set(non_zero_feature)

#     # select_feature_intersection = sorted(list(select_feature_intersection))
#     # print(select_feature_intersection)
#     # # 找到对被试间泛化都有用的特征并集
#     # select_feature_union = set([])
#     # for non_zero_feature in non_zero_features:
#     #     select_feature_union = select_feature_union | set(non_zero_feature)

#     # select_feature_union = sorted(list(select_feature_union))
#     # print(select_feature_union)

#     # print('saving features')
#     # np.save(pjoin(performance_path, model_type, f'nod-cross-sub_model-{model_type}_features-intersection.npy'), np.array(select_feature_intersection))
#     # np.save(pjoin(performance_path, model_type, f'nod-cross-sub_model-{model_type}_features-union.npy'), np.array(select_feature_union))


#     # 2. 再到被试内用特征训练被试内预测的模型
#     # # filtered feature fit
#     # lasso = Lasso(alpha=max_alpha)
#     # lasso.fit(concate_X[:, np.array(select_feature_intersection)], concate_y)

#     # test performance
#     train_pairs = generate_pairs(9, step=4000)
#     test_pairs = generate_pairs(9)
#     final_test_corr, final_test_ev = [], []
#     final_train_corr, final_train_ev = [], []
#     final_val_corr, final_val_ev = [], []
#     if model_type.split('-')[-1] == 'U':
#         feature_filter = np.load(pjoin(filter_path, f'{feature_type}-union.npy'))
#     elif model_type.split('-')[-1] == 'N':
#         feature_filter = np.load(pjoin(filter_path, f'{feature_type}-inter.npy'))
    
#     # 对每个被试单独做lasso模型，挑选刺激并得到表现
#     for ipair, train_pair in enumerate(train_pairs):
#         X_train = concate_X[train_pair[0]:train_pair[1], feature_filter]
#         y_train = concate_y[train_pair[0]:train_pair[1]]
#         print('train data shape:', X_train.shape, y_train.shape)
        
#         sessionKf = KFold(n_splits=4, shuffle=False)
#         alphas = np.logspace(-3, -1, 20)  # 创建一个覆盖广泛范围的alpha值
#         scores = []  # 用于存储每个alpha对应的平均交叉验证分数    
#         # 对每个alpha进行评估
#         for alpha in alphas:
#             lasso = Lasso(alpha=alpha)
#             cv_scores = cross_val_score(lasso, X_train, y_train, scoring='r2', cv=sessionKf)
#             scores.append(np.mean(cv_scores))
#         # 创建Lasso模型实例
#         # print(scores)
#         max_alpha = alphas[np.argmax(scores)]
#         lasso = Lasso(alpha=max_alpha)
#         print('sub-', (ipair+1), ' max_alpha:', max_alpha, 'at where', np.argmax(scores))

#         # 存出validation的表现
#         final_val_ev.append(np.mean(cross_val_score(lasso, X_train, y_train, scoring='r2', cv=sessionKf)))
#         pearson_scorer = make_scorer(pearson_correlation, greater_is_better=True)
#         final_val_corr.append(np.mean(cross_val_score(lasso, X_train, y_train, scoring=pearson_scorer, cv=sessionKf)))
#         # 存出训练好的模型
#         lasso.fit(X_train, y_train)
#         joblib.dump(lasso, pjoin(performance_path, model_type, f'nod-within-sub-{ipair+1}_model-{model_type}_lasso.pkl'))
#         # 存出 train 和 test 的表现
#         y_pred = lasso.predict(X_train)
#         final_train_corr.append(np.corrcoef(y_pred, y_train[:,0])[0, 1]) 
#         final_train_ev.append(lasso.score(X_train, y_train)) 

#         X_test = concate_X_test[test_pairs[ipair][0]:test_pairs[ipair][1], feature_filter]
#         y_test = concate_y_test[test_pairs[ipair][0]:test_pairs[ipair][1]]
#         y_pred = lasso.predict(X_test)
#         final_test_corr.append(np.corrcoef(y_pred, y_test[:,0])[0, 1])
#         final_test_ev.append(lasso.score(X_test, y_test))

#         # # 线性模型的cv
#         # lr = LinearRegression(n_jobs=8)
#         # cv_scores = cross_val_score(lr, X_train, y_train, scoring='r2', cv=sessionKf)
#         # final_val_ev.append(np.mean(cv_scores))
#         # # corr index
#         # pearson_scorer = make_scorer(pearson_correlation, greater_is_better=True)
#         # cv_scores = cross_val_score(lr, X_train, y_train, scoring=pearson_scorer, cv=sessionKf)
#         # final_val_corr.append(np.mean(cv_scores))

#         # lr = LinearRegression(n_jobs=8)
#         # lr.fit(X_train, y_train)
#         # joblib.dump(lr, pjoin(performance_path, model_type, f'nod-within-sub-{ipair+1}_model-{model_type}_linear.pkl'))
#         # y_pred = lr.predict(X_train)
#         # final_train_corr.append(np.corrcoef(y_pred[:,0], y_train[:,0])[0, 1]) 
#         # final_train_ev.append(lr.score(X_train, y_train)) 

#         # X_test = concate_X_test[test_pairs[ipair][0]:test_pairs[ipair][1], feature_filter]
#         # y_test = concate_y_test[test_pairs[ipair][0]:test_pairs[ipair][1]]

#         # y_pred = lr.predict(X_test)
#         # final_test_corr.append(np.corrcoef(y_pred[:,0], y_test[:,0])[0, 1])
#         # final_test_ev.append(lr.score(X_test, y_test))

#     print('saving feature model and test performance')
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_train-corr.npy'), np.array(final_train_corr))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_train-expvar.npy'), np.array(final_train_ev))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_test-corr.npy'), np.array(final_test_corr))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_test-expvar.npy'), np.array(final_test_ev))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_val-corr.npy'), np.array(final_val_corr))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_val-expvar.npy'), np.array(final_val_ev))

# =======================================================
# 
# =======================================================

# work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
# opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
# # input path
# resp_path = pjoin(work_dir, 'prep/brain_response')
# voxel_mask_path = pjoin(work_dir, 'prep/voxel_masks')
# image_activations_path = pjoin(work_dir, 'prep/image_activations')
# concate_feature_path = pjoin(work_dir, 'prep/concate_features')
# retino_path = pjoin(work_dir, 'build/retinoparams')
# guass_path = pjoin(work_dir, 'build/gaussianparams')
# avgrf_path = pjoin(work_dir, 'prep/image_mask')
# performance_path = pjoin(work_dir, 'build/lasso-feature-selection/')

# concate_y = np.load(pjoin(concate_feature_path, 'cross-sub_imn_averaged-V1.npy'))
# concate_y_test = np.load(pjoin(concate_feature_path, 'cross-sub_coco_averaged-V1.npy'))

# print('loading training data')
# feature_types = ['priorwRF', 'globalmean', 'priorRF']
# traindata = {}
# testdata = {}
# for _feature in feature_types:
#     traindata[_feature] = np.load(pjoin(concate_feature_path, f'cross-sub_{_feature.lower()}_imn_conv2.npy'))
#     testdata[_feature] = np.load(pjoin(concate_feature_path, f'cross-sub_{_feature.lower()}_coco_conv2.npy'))

# for feature_type in  feature_types:

#     model_type = f'lasso-{feature_type}'
#     if not os.path.exists(pjoin(performance_path, model_type)):
#         os.makedirs(pjoin(performance_path, model_type))

#     # # scores = []  # 用于存储每个alpha对应的平均交叉验证分数
#     # # # 自定义9-fold交叉验证
#     # # cv = KFold(n_splits=4, shuffle=False)
#     # # # 对每个alpha进行评估
#     # # for feature_type, feature in traindata.items():
#     # #     lr = LinearRegression(n_jobs=10)
#     # #     concate_X = feature
#     # #     cv_scores = cross_val_score(lr, concate_X, concate_y, scoring='r2', cv=cv)
#     # #     scores.append(np.mean(cv_scores))
#     # # feature_type = feature_types[np.argmax(scores)]


#     # LinearRegression
#     print('best feature type', feature_type)
#     concate_X = traindata[feature_type]
#     # lr = LinearRegression(n_jobs=10)
#     # # 初始化9折交叉验证，保持样本顺序
#     # kf = KFold(n_splits=9, shuffle=False)
#     # # 用于存储每次的性能指标
#     # accuracy_scores, explained_scores = [], []
#     # # 进行9折交叉验证
#     # for train_index, test_index in kf.split(concate_X):
#     #     # 分割数据
#     #     X_train, X_test = concate_X[train_index], concate_X[test_index]
#     #     y_train, y_test = concate_y[train_index], concate_y[test_index]
#     #     # 训练模型
#     #     lr.fit(X_train, y_train)
#     #     # 预测测试集
#     #     y_pred = lr.predict(X_test)
#     #     # 计算并记录性能指标
#     #     accuracy_scores.append(np.corrcoef(y_test[:,0], y_pred[:,0])[0, 1])
#     #     explained_scores.append(lr.score(X_test, y_test))

#     # print('saving validation performance')
#     # np.save(pjoin(performance_path, model_type, f'nod-cross-sub_model-{model_type}_validation-corr.npy'), np.array(accuracy_scores))
#     # np.save(pjoin(performance_path, model_type, f'nod-cross-sub_model-{model_type}_validation-expvar.npy'), np.array(explained_scores))

#     # test performance
#     concate_X_test = testdata[feature_type]
#     train_pairs = generate_pairs(9, step=4000)
#     test_pairs = generate_pairs(9)
#     final_test_corr, final_test_ev = [], []
#     final_val_corr, final_val_ev = [], []
#     final_train_corr, final_train_ev = [], []
#     for ipair, train_pair in enumerate(train_pairs):

#         X_train = concate_X[train_pair[0]:train_pair[1]]
#         y_train = concate_y[train_pair[0]:train_pair[1]]
#         print('train data shape:', X_train.shape, y_train.shape)

#         sessionKf = KFold(n_splits=4, shuffle=False)

#         alphas = np.logspace(-2, 1, 20)  # 创建一个覆盖广泛范围的alpha值
#         scores = []  # 用于存储每个alpha对应的平均交叉验证分数    
#         # 对每个alpha进行评估
#         for alpha in alphas:
#             lasso = Lasso(alpha=alpha)
#             cv_scores = cross_val_score(lasso, X_train, y_train, scoring='r2', cv=sessionKf)
#             scores.append(np.mean(cv_scores))
#         # 创建Lasso模型实例
#         # print(scores)
#         max_alpha = alphas[np.argmax(scores)]
#         lasso = Lasso(alpha=max_alpha)
#         print('sub-', (ipair+1), ' max_alpha:', max_alpha, 'at where', np.argmax(scores))

#         # 存出validation的表现
#         final_val_ev.append(np.mean(cross_val_score(lasso, X_train, y_train, scoring='r2', cv=sessionKf)))
#         pearson_scorer = make_scorer(pearson_correlation, greater_is_better=True)
#         final_val_corr.append(np.mean(cross_val_score(lasso, X_train, y_train, scoring=pearson_scorer, cv=sessionKf)))
#         # 存出训练好的模型
#         lasso.fit(X_train, y_train)
#         joblib.dump(lasso, pjoin(performance_path, model_type, f'nod-within-sub-{ipair+1}_model-{model_type}_lasso.pkl'))
#         # 存出 train 和 test 的表现
#         y_pred = lasso.predict(X_train)
#         final_train_corr.append(np.corrcoef(y_pred, y_train[:,0])[0, 1]) 
#         final_train_ev.append(lasso.score(X_train, y_train)) 

#         X_test = concate_X_test[test_pairs[ipair][0]:test_pairs[ipair][1]]
#         y_test = concate_y_test[test_pairs[ipair][0]:test_pairs[ipair][1]]
#         y_pred = lasso.predict(X_test)
#         final_test_corr.append(np.corrcoef(y_pred, y_test[:,0])[0, 1])
#         final_test_ev.append(lasso.score(X_test, y_test))        

#         # # 线性模型的CV
#         # lr = LinearRegression(n_jobs=8)
#         # cv_scores = cross_val_score(lr, X_train, y_train, scoring='r2', cv=sessionKf)
#         # final_val_ev.append(np.mean(cv_scores))
#         # # corr index
#         # pearson_scorer = make_scorer(pearson_correlation, greater_is_better=True)
#         # cv_scores = cross_val_score(lr, X_train, y_train, scoring=pearson_scorer, cv=sessionKf)
#         # final_val_corr.append(np.mean(cv_scores))

#         # lr.fit(X_train, y_train)
#         # joblib.dump(lr, pjoin(performance_path, model_type, f'nod-cross-sub-{ipair+1}_model-{model_type}_linear.pkl'))
#         # X_test = concate_X_test[test_pairs[ipair][0]:test_pairs[ipair][1]]
#         # y_test = concate_y_test[test_pairs[ipair][0]:test_pairs[ipair][1]]
        
#         # y_pred = lr.predict(X_test)
#         # final_test_corr.append(np.corrcoef(y_pred[:,0], y_test[:,0])[0, 1])
#         # final_test_ev.append(lr.score(X_test, y_test))

#         # y_pred = lr.predict(X_train)
#         # final_train_corr.append(np.corrcoef(y_pred[:,0], y_train[:,0])[0, 1])
#         # final_train_ev.append(lr.score(X_train, y_train))

#     print('saving feature model and test performance')
#     # 
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_test-corr.npy'), np.array(final_test_corr))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_test-expvar.npy'), np.array(final_test_ev))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_train-corr.npy'), np.array(final_train_corr))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_train-expvar.npy'), np.array(final_train_ev))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_val-corr.npy'), np.array(final_val_corr))
#     np.save(pjoin(performance_path, model_type, f'nod-within-sub_model-{model_type}_val-expvar.npy'), np.array(final_val_ev))
