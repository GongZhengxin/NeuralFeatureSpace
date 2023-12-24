import glob
import os
import time
from PIL import Image

import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from scipy.stats import zscore

from utils import get_roi_data, train_data_normalization, Timer, alexnet_info


with Timer() as t:
    sub = 'sub-03'
    inputlayername = 'features.3'
    layer = {'name': inputlayername, 'size':alexnet_info[inputlayername]}
    layername = layer['name']
    layername = layername.replace('.','')
    # 
    image_activations_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/image_activations'
    work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
    resp_path = os.path.join(work_dir, 'brain_response')

    model_path = os.path.join(work_dir, 'mappings')
    re_corrmap_path = os.path.join(work_dir, 'retest_corrmap')
    # load
    brain_resp = np.load(os.path.join(resp_path, f'{sub}_imagenet_beta.npy'))
    activations = np.load(os.path.join(image_activations_path, f'{sub}_{layername}.npy'))

    # select ROI names
    evc_pool = ['V1', 'V2', 'V3','V4']
    # aggregate ROI vertices
    roi_name = [__  for _ in [evc_pool] for __ in _]
    # form ROI mask
    selection_mask = np.sum([get_roi_data(None, _) for _ in roi_name], axis=0)
    # trsnfer to indices in cifti space
    voxel_indices = np.where(selection_mask==1)[0]

    # collect resp in ROI
    brain_resp = brain_resp[:, voxel_indices]

    # normalization
    norm_metric = 'session'
    brain_resp = train_data_normalization(brain_resp, metric=norm_metric)
    num_voxel = brain_resp.shape[-1]


    X = activations
    y = brain_resp
    fold_indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
    print('kfold start')
    for fold, (start, end) in enumerate(fold_indices):
        if not os.path.exists(os.path.join(re_corrmap_path, sub, f'{sub}_layer-{layername}_corrmap-test_fold{fold}.npy')):
            model_name = f'{sub}_layer-{layername}_models_fold{fold}.pkl'
            models = joblib.load(os.path.join(model_path, sub, model_name))

            X_val = X[start:end]
            y_val = y[start:end]
            
            all_predictions = np.zeros((X.shape[-2], X.shape[-1], end-start,  num_voxel))
            correlation_matrix = np.zeros((X.shape[-2], X.shape[-1], num_voxel))
            
            for i in range(layer['size']):
                for j in range(layer['size']):
                    print(f'f-{fold}({i},{j})')
                    lr = models[(i,j)]
                    y_pred = lr.predict(X_val[:, :, i, j])
                    all_predictions[i, j, :, :] = y_pred
                    
            all_predictions = zscore(all_predictions, axis=-2)
            y_val = zscore(y_val, axis=0)
            correlation_matrix = np.mean(all_predictions*y_val, axis=-2)
            # save out
            if not os.path.exists(os.path.join(re_corrmap_path, sub)):
                os.mkdir(os.path.join(re_corrmap_path, sub))
            # 保存每个 fold 的相关性矩阵
            if not os.path.exists(os.path.join(re_corrmap_path, sub, f'{sub}_layer-{layername}_corrmap-test_fold{fold}.npy')):
                np.save(os.path.join(re_corrmap_path, sub, f'{sub}_layer-{layername}_corrmap-test_fold{fold}.npy'), correlation_matrix)
    
    for fold in range(4):
        file = f'{sub}_layer-{layername}_corrmap-test_fold{fold}.npy'
        if fold==0:
            all_data = np.load(os.path.join(re_corrmap_path, sub, file))
        else:
            all_data += np.load(os.path.join(re_corrmap_path, sub, file))

    mean_data = all_data/4
    file = f'{sub}_layer-{layername}_corrmap-test.npy'
    np.save(os.path.join(re_corrmap_path, file), mean_data)

print(f'consume : {t.interval} s')

