import os, gc
import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from scipy.stats import zscore
from scipy.ndimage import zoom
from utils import train_data_normalization, Timer, net_size_info

subs = ['sub-01', 'sub-02', 'sub-03', 'sub-04'] #['sub-05', 'sub-08', 'sub-06', 'sub-07']#, 'sub-09'
for sub in subs:
    with Timer() as t:
        print(sub)
        inputlayername = 'googlenet-conv2'#inception3a
        layer = {'name': inputlayername, 'size':net_size_info[inputlayername.replace('raw-','')]}
        layername = layer['name']
        layername = layername.replace('.','')
        mask_name = 'primaryvis-in-MMP'
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
        # transfer mask into indices
        voxel_indices = np.where(voxel_mask==1)[0]

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
            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])

            X_val = X[start:end]
            y_val = y[start:end]
            
            models = {}  # 存储每个位置的线性回归模型
            
            all_predictions = np.zeros((X.shape[-2], X.shape[-1], end-start,  num_voxel))
            correlation_matrix = np.zeros((X.shape[-2], X.shape[-1], num_voxel))
            
            for i in range(layer['size']):
                print(f'f-{sub}-{fold}({i},-)')
                for j in range(layer['size']):
                    lr = LinearRegression(n_jobs=10).fit(X_train[:, :, i, j], y_train)
                    y_pred = lr.predict(X_val[:, :, i, j])
                    models[(i, j)] = lr
                    all_predictions[i, j, :, :] = y_pred
                    
            all_predictions = zscore(all_predictions, axis=-2)
            y_val = zscore(y_val, axis=0)
            correlation_matrix = np.mean(all_predictions*y_val, axis=-2)
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
        del models, all_predictions, X_train, X_val, y_train, y_val
        gc.collect()

        if downsample:
            for fold in range(4):
                file = f'{sub}_bm-{mask_name}_layer-{layername}_corrmap-test_fold{fold}.npy'
                if fold==0:
                    all_data = np.load(os.path.join(corrmap_path, sub, file))
                else:
                    all_data += np.load(os.path.join(corrmap_path, sub, file))
            file = f'{sub}_bm-{mask_name}_layer-{layername}_corrmap-test.npy'
        else:
            for fold in range(4):
                file = f'{sub}_bm-{mask_name}_layer-raw-{layername}_corrmap-test_fold{fold}.npy'
                if fold==0:
                    all_data = np.load(os.path.join(corrmap_path, sub, file))
                else:
                    all_data += np.load(os.path.join(corrmap_path, sub, file))
            file = f'{sub}_bm-{mask_name}_layer-raw-{layername}_corrmap-test.npy'
        mean_data = all_data/4
        np.save(os.path.join(corrmap_path, file), mean_data)

    print(f'{sub} consume : {t.interval} s')



