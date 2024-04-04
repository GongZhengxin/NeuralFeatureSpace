import os, gc
import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import time
from scipy.stats import zscore
from scipy.ndimage import zoom
from utils import train_data_normalization, Timer, net_size_info
from joblib import Parallel, delayed
from scipy.ndimage import median_filter


subs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-08', 'sub-06', 'sub-07', 'sub-09'] #[]#
#[f'sub-{_+1}' for _ in range(10,29)] #
print(subs)
def compute_corr_matrix( i, j):
    global X_train, y_train, sub, fold
    print(sub, '-', fold, ':',i, j)
    lr = LinearRegression().fit(X_train[:, :, i, j], y_train)
    y_pred = lr.predict(X_val[:, :, i, j])
    y_pred = zscore(y_pred, axis=0)
    return i, j, lr, y_pred

for sub in subs[2 : ]:
    with Timer() as t:
        print(sub)
        inputlayername = 'googlenet-maxpool2' #'googlenet-inception3a'#conv2
        layer = {'name': inputlayername, 'size':net_size_info[inputlayername.replace('raw-','')]}
        layername = layer['name']
        layername = layername.replace('.','')
        mask_name = 'primaryvis-in-MMP' #''
        downsample = True

        # path setting
        work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
        # input path
        resp_path = os.path.join(work_dir, 'prep/brain_response')
        image_activations_path = os.path.join(work_dir, 'prep/image_activations')
        voxel_mask_path = os.path.join(work_dir, 'prep/voxel_masks')
        # save out path
        model_path = os.path.join(work_dir, 'build/mappings')
        corrmap_path = os.path.join(work_dir, 'build/corrmap')
        ciftiy_path = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
        # load
        brain_resp = np.load(os.path.join(resp_path, f'{sub}_imagenet_beta.npy'))
        if layer['size'] > 28 and downsample:
            activations = np.load(os.path.join(image_activations_path, f'{sub}_{layername}_ds.npy'))
            layer['size'] = 28
        else:
            activations = np.load(os.path.join(image_activations_path, f'{sub}_{layername}.npy'))
        print(f'shape of activations : {activations.shape}')
        voxel_mask_nii = nib.load(os.path.join(voxel_mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'))
        voxel_mask = voxel_mask_nii.get_fdata()
        named_maps = [named_map.map_name for named_map in voxel_mask_nii.header.get_index_map(0).named_maps]      

        # determine the mask type
        if sub in named_maps:
            voxel_mask = voxel_mask[named_maps.index(sub),:]
        # squeeze into 1 dim
        voxel_mask = np.squeeze(np.array(voxel_mask))

        #读入prf文件
        prf_data = nib.load(os.path.join(ciftiy_path, f'{sub}/results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii')).get_fdata()
        
        #选取R2大于10的体素
        R2_values = prf_data[3, :]
        valid_R2_indices = np.where(R2_values >= 10)[0]

        # determine the mask type
        if sub in named_maps:
            voxel_mask = voxel_mask[named_maps.index(sub),:]
        # squeeze into 1 dim
        mmp_voxel_mask = np.squeeze(np.array(voxel_mask))
        # 确定最终的mask indices
        mmp_voxel_indices = np.where(mmp_voxel_mask==1)[0]
        voxel_indices = np.intersect1d(mmp_voxel_indices, valid_R2_indices) #!

        # collect resp in ROI
        brain_resp = brain_resp[:, voxel_indices]

        # normalization
        norm_metric = 'session' 
        brain_resp = train_data_normalization(brain_resp, metric=norm_metric)
        num_voxel = brain_resp.shape[-1]

        X = activations
        y = brain_resp
        fold_indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
        # fold_indices = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)] # for last 21 subjects
        print('kfold start')
        for fold, (start, end) in enumerate(fold_indices):
            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])

            X_val = X[start:end]
            y_val = y[start:end]
            t_norm = time.time()
            print('normalizing')
            y_val = zscore(y_val, axis=0)
            print('finished normalization at ', time.time()-t_norm, ' s')
            models = {}  # 存储每个位置的线性回归模型
            #============
            # 秀儿写法
            #============
            # 预先准备矩阵，该矩阵先用于存储标准化过的预测激活            
            correlation_matrix = np.zeros((layer['size'], layer['size'], end-start,  num_voxel))
            results = Parallel(n_jobs=28)(delayed(compute_corr_matrix)(i, j) for i in range(layer['size']) for j in range(layer['size']))
            n_pos = len(results)
            for i_pos in range(n_pos):
                if i_pos % 100 ==0:
                    print(f'{i_pos}', end=' ')
                i, j, lr, y_pred = results.pop(0)
                models[(i, j)] = lr
                correlation_matrix[i, j, :, :] = y_pred
                gc.collect()
            correlation_matrix = np.mean(correlation_matrix*y_val, axis=-2)
            # correlation_matrix = median_filter(correlation_matrix, size=(3, 3, 1))
            #============
            # 老掉牙的写法
            #============
            # all_predictions = np.zeros((X.shape[-2], X.shape[-1], end-start,  num_voxel))
            # correlation_matrix = np.zeros((X.shape[-2], X.shape[-1], num_voxel))
            
            # for i in range(layer['size']):
            #     print(f'f-{sub}-{fold}({i},-)')
            #     for j in range(layer['size']):
            #         lr = LinearRegression(n_jobs=10).fit(X_train[:, :, i, j], y_train)
            #         y_pred = lr.predict(X_val[:, :, i, j])
            #         models[(i, j)] = lr
            #         all_predictions[i, j, :, :] = y_pred
                    
            # all_predictions = zscore(all_predictions, axis=-2)
            # y_val = zscore(y_val, axis=0)
            # correlation_matrix = np.mean(all_predictions*y_val, axis=-2)
            # save out
            if not os.path.exists(os.path.join(model_path, sub)):
                os.mkdir(os.path.join(model_path, sub))
            if not os.path.exists(os.path.join(corrmap_path, sub)):
                os.mkdir(os.path.join(corrmap_path, sub))
            if not downsample:
                # 保存每个 fold 的模型
                joblib.dump(models, os.path.join(model_path, sub, f'{sub}_bm-{mask_name}_layer-raw-{layername}_models_fold{fold}.pkl'))
                # 保存每个 fold 的相关性矩阵
                np.save(os.path.join(corrmap_path, sub, f'{sub}_bm-{mask_name}_layer-raw-{layername}_corrmap-test_fold{fold}.npy'), correlation_matrix)
            else:
                # 保存每个 fold 的模型
                joblib.dump(models, os.path.join(model_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_models_fold{fold}.pkl'))
                # 保存每个 fold 的相关性矩阵
                np.save(os.path.join(corrmap_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_corrmap-test_fold{fold}.npy'), correlation_matrix)
        del models, X_train, X_val, y_train, y_val
        gc.collect()

        if downsample:
            for fold in range(len(fold_indices)):
                file = f'{sub}_bm-{mask_name}_layer-{layername}_corrmap-test_fold{fold}.npy'
                if fold==0:
                    all_data = np.load(os.path.join(corrmap_path, sub, file))
                else:
                    all_data += np.load(os.path.join(corrmap_path, sub, file))
            file = f'{sub}_bm-{mask_name}_layer-{layername}_corrmap-test.npy'
        else:
            for fold in range(len(fold_indices)):
                file = f'{sub}_bm-{mask_name}_layer-raw-{layername}_corrmap-test_fold{fold}.npy'
                if fold==0:
                    all_data = np.load(os.path.join(corrmap_path, sub, file))
                else:
                    all_data += np.load(os.path.join(corrmap_path, sub, file))
            file = f'{sub}_bm-{mask_name}_layer-raw-{layername}_corrmap-test.npy'
        mean_data = all_data/len(fold_indices)
        np.save(os.path.join(corrmap_path, file), mean_data)

    print(f'{sub} consume : {t.interval} s')



