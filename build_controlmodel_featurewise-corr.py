import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression
import joblib
from joblib import Parallel, delayed
import time 
from scipy.stats import zscore
from utils import train_data_normalization, Timer, net_size_info, conv2_labels

def calc_explained_var(y_pred, y):
    return  (1 - np.mean((y_pred - y)**2, axis=0)/ np.var(y, axis=0))

def calc_corr(y_pred, y):
    return np.diagonal( np.corrcoef(np.transpose(y_pred), np.transpose(y))[num_voxel::, 0:num_voxel])

def save_result(result, indexname):
    global performance_path, voxel_indices, sub, mask_name, layername
    if result.ndim > 1:
        all_performace = np.nan * np.zeros((result.shape[0], 59412))
        all_performace[:, voxel_indices] = np.array(result)
    else:
        all_performace = np.nan * np.zeros((1, 59412))
        all_performace[:, voxel_indices] = np.array(result)
    np.save(pjoin(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_{indexname}.npy'), all_performace)

subs = ['sub-09', 'sub-01', 'sub-02', 'sub-06', 'sub-07', 'sub-03', 'sub-04',  'sub-05', 'sub-08'] # 'sub-01', 'sub-02', 'sub-06', 'sub-07', 'sub-03', 'sub-04',  'sub-05', 'sub-08'
# subs = [] # 
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
labels = conv2_labels

for sub in subs:
    with Timer() as t:
        inputlayername = 'googlenet-conv2' 
        layer = {'name': inputlayername, 'size': net_size_info[inputlayername.replace('raw-', '')]}#alexnet_info[inputlayername]
        layername = layer['name']
        layername = layername.replace('.','')
        test_set_name = 'coco'
        mask_name = 'primaryvis-in-MMP' #'fixretfloc-in-subj'
        print(sub, mask_name, layername)
        fold_indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
        # path settings
        work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
        
        # input path
        resp_path = os.path.join(work_dir, 'prep/brain_response')
        voxel_mask_path = os.path.join(work_dir, 'prep/voxel_masks')
        image_activations_path = os.path.join(work_dir, 'prep/image_activations')

        # save out path
        performance_path = os.path.join(work_dir, 'build/control/featurewise-corr')
        # save path
        if not os.path.exists(os.path.join(performance_path, sub)):
            os.mkdir(os.path.join(performance_path, sub))

        # load
        brain_resp = np.load(os.path.join(resp_path, f'{sub}_imagenet_beta.npy'))
        activations = np.load(os.path.join(image_activations_path, f'{sub}_{layername}.npy'))
        coco_activations = np.load(pjoin(image_activations_path, f'{test_set_name}_{layername}.npy'))

        print(f'activations shape of {activations.shape}')


        # load, reshape and average the resp
        test_resp = np.load(pjoin(resp_path, f'{sub}_{test_set_name}_beta.npy'))
        num_trial = test_resp.shape[0]
        num_run = int(num_trial/120)
        test_resp = test_resp.reshape((num_run, 120, 59412))
        mean_test_resp = test_resp.mean(axis=0)

        voxel_mask_nii = nib.load(os.path.join(voxel_mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'))
        voxel_mask = voxel_mask_nii.get_fdata()
        named_maps = [named_map.map_name for named_map in voxel_mask_nii.header.get_index_map(0).named_maps]
        
        # determine the mask type
        if sub in named_maps:
            voxel_mask = voxel_mask[named_maps.index(sub),:]
        # squeeze into 1 dim
        voxel_mask = np.squeeze(np.array(voxel_mask))
        # transfer mask into indices
        voxel_indices = np.where(voxel_mask==1)[0]

        # collect resp in ROI
        brain_resp = brain_resp[:, voxel_indices]
        mean_test_resp = mean_test_resp[:, voxel_indices]

        # normalization
        norm_metric = 'session'
        brain_resp = train_data_normalization(brain_resp, metric=norm_metric)
        mean_test_resp = zscore(mean_test_resp)
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
            X_mean = np.mean(X, axis=(2,3))
            X_mean = zscore(X_mean)
            X_test_mean = zscore(np.mean(X_test, axis=(2,3)))
        
        y = brain_resp 
        y_test = mean_test_resp

        # full model betas, no need for inception 'cause the mean is substracted 
        full_model = LinearRegression(n_jobs=10)
        full_model.fit(X_mean, y)
        # # calc the full model performance on train set
        full_r2_train = calc_explained_var(full_model.predict(X_mean), y)
        full_r_train = calc_corr(full_model.predict(X_mean), y)
        
        # calc the full model performance on test set
        full_r2_test =calc_explained_var(full_model.predict(X_test_mean), y_test)
        full_r_test = calc_corr(full_model.predict(X_test_mean), y_test)

        # n_category = len(labels.keys())
        # unique_vars_train, r_diff_train = np.nan * np.zeros((n_category, num_voxel)), np.nan * np.zeros((n_category, num_voxel))
        # unique_vars, r_diff = np.nan * np.zeros((n_category, num_voxel)), np.nan * np.zeros((n_category, num_voxel))
        # set_vars, set_r = np.nan * np.zeros((n_category, num_voxel)), np.nan * np.zeros((n_category, num_voxel))
        # set_vars_train, set_r_train = np.nan * np.zeros((n_category, num_voxel)), np.nan * np.zeros((n_category, num_voxel))
        
        # for ilabel, (key, value) in enumerate(labels.items()):
        #     print(key)
        #     # 取出当前大类的相关特征
        #     X_cate = X_mean[:, np.array(value)]
        #     X_test_cate = X_test_mean[:, np.array(value)]
        #     # 取出除了当前大类其他所有特征
        #     X_others = np.delete(X_mean, value, axis=1)
        #     X_test_others = np.delete(X_test_mean, value, axis=1)
        #     # 拟合方程参数
        #     other_model = LinearRegression(n_jobs=10)
        #     set_model = LinearRegression(n_jobs=10)
            
        #     other_model.fit(X_others, y)
        #     set_model.fit(X_cate, y)

        #     # 计算当前大类模型预测能力
        #     # 在训练集上
        #     set_r2_train = calc_explained_var(set_model.predict(X_cate), y)
        #     set_r_train = calc_corr(set_model.predict(X_cate), y) 
        #     # 在测试集上
        #     set_r2_test = calc_explained_var(set_model.predict(X_test_cate), y_test)
        #     set_r_test = calc_corr(set_model.predict(X_test_cate), y_test)

        #     # 计算其余大类模型预测能力
        #     # 在训练集上
        #     other_r2_train = calc_explained_var(other_model.predict(X_others), y)
        #     other_r_train = calc_corr(other_model.predict(X_others), y) 
        #     # 在测试集上
        #     other_r2_test = calc_explained_var(other_model.predict(X_test_others), y_test)
        #     other_r_test = calc_corr(other_model.predict(X_test_others), y_test) 
            
        #     # 计算独立解释方差和相关差异
        #     # 在训练集上
        #     unique_vars_train[ilabel, :] = full_r2_train - other_r2_train
        #     r_diff_train[ilabel, :] = full_r_train - other_r_train
        #     # 在测试集上
        #     unique_vars[ilabel, :] = full_r2_test - other_r2_test
        #     r_diff[ilabel, :] = full_r_test - other_r_test
        # computing 
        # all_performace = np.nan*np.zeros((X_mean.shape[-1], 59412))
        # all_performace[:, voxel_indices] = np.corrcoef(y.transpose(), X_mean.transpose())[-X_mean.shape[-1]::, 0:len(voxel_indices)]
        # np.save(os.path.join(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_corr.npy'), all_performace)
        # results = {'uv':unique_vars, 'rd': r_diff, 
        #            'model-ctg-ev': set_r2_test, 'model-ctg-r': set_r_test, 
        #            'uv-train': unique_vars_train,'rd-train': r_diff_train, 
        #            'model-ctg-ev-train': set_r2_train, 'model-ctg-r-train': set_r_train, 
        #            'full-model-ev': full_r2_test, 'full-model-ev-train': full_r2_train,
        #            'full-model-r': full_r_test, 'full-model-r-train': full_r_train} 
        results = {'full-model-r': full_r_test, 'full-model-r-train': full_r_train}
        for index_name, result in results.items():
            save_result(result, index_name)
        
    print(f'{sub} consume : {t.interval} s')
