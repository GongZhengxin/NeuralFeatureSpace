import os
import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
import joblib
from joblib import Parallel, delayed
import time 
from utils import train_data_normalization, Timer, net_size_info

# Define the 2D Gaussian function
def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
    i, j = coords
    return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C

def compute_voxel_performance(idx, voxel):
    global X, y, eqpoint, retinoR2, fold_indices
     #
    performance = np.nan * np.zeros(len(fold_indices)+1, )
    # load receptive field
    if not voxel in guassparams.keys():
        pass
    else:
        receptive_field = gaussian_2d((i, j), *guassparams[voxel])
        # receptive_field[receptive_field < 0.5*np.max(receptive_field)] = 0
        receptive_field = receptive_field / (receptive_field.sum() + 1e-20)
        if retinoR2[voxel] >= eqpoint:
            print(f'{idx}={voxel}')
            for fold, (start, end) in enumerate(fold_indices):
                X_train = np.concatenate([X[:start], X[end:]])
                y_train = np.concatenate([y[:start], y[end:]])
                
                X_val = X[start:end]
                y_val = y[start:end, idx]
                # saptial summation
                X_voxel_train = np.sum(X_train * receptive_field, axis=(2,3))
                X_voxel_val = np.sum(X_val * receptive_field, axis=(2,3))
                lr = LinearRegression().fit(X_voxel_train, y_train[:, idx])
                y_pred = lr.predict(X_voxel_val)
                performance[fold] = np.corrcoef(y_val, y_pred)[0,1]
            performance[-1] = np.nanmean(performance[0:fold])
    return performance

# subs = ['sub-03', 'sub-04','sub-06', 'sub-07']'sub-01', 'sub-02',
subs = [ 'sub-05', 'sub-08']
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
        inputlayername = 'googlenet-conv2' # 'features.0'
        downsample = True
        if 'raw' in inputlayername:
            downsample = False
        layer = {'name': inputlayername, 'size': net_size_info[inputlayername.replace('raw-', '')]}#alexnet_info[inputlayername]
        layername = layer['name']
        layername = layername.replace('.','')
        mask_name = 'primaryvis-in-MMP' #'fixretfloc-in-subj'
        print(sub, mask_name, layername)
        fold_indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)] #[(0, 2000), (2000, 4000)]
        # path settings
        work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
        
        # input path
        resp_path = os.path.join(work_dir, 'prep/brain_response')
        voxel_mask_path = os.path.join(work_dir, 'prep/voxel_masks')
        image_activations_path = os.path.join(work_dir, 'prep/image_activations')
        retino_path = os.path.join(work_dir, 'build/retinoparams')
        guass_path = os.path.join(work_dir, 'build/gaussianparams')
        # save out path
        model_path = os.path.join(work_dir, 'build/mappings')
        performance_path = os.path.join(work_dir, 'build/retrainperformance')
        # save path
        if not os.path.exists(os.path.join(model_path, sub)):
            os.mkdir(os.path.join(model_path, sub))
        if not os.path.exists(os.path.join(performance_path, sub)):
            os.mkdir(os.path.join(performance_path, sub))

        # load
        brain_resp = np.load(os.path.join(resp_path, f'{sub}_imagenet_beta.npy'))
        if downsample:
            lname = layername.replace('raw-', '')
            activations = np.load(os.path.join(image_activations_path, f'{sub}_{lname}_ds.npy'))
        else:
            lname = layername.replace('raw-', '')
            activations = np.load(os.path.join(image_activations_path, f'{sub}_{lname}.npy'))
        print(f'activations shape of {activations.shape}')
        retinoR2 = np.load(os.path.join(retino_path, f'{sub}_layer-{layername}_params.npy'), allow_pickle=True)[0]['R2']
        guassparams = np.load(os.path.join(guass_path, f'{sub}_layer-{layername}_Gauss.npy'), allow_pickle=True)[0]
        
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

        # coordinate
        # Create grid data
        layer['size'] = activations.shape[-1]
        i = np.linspace(-8., 8., layer['size'])
        j = np.linspace(8., -8., layer['size'])
        i, j = np.meshgrid(i, j)

        X = activations
        y = brain_resp
        # final train
        model_name = f'{sub}_bm-{mask_name}_layer-{layername}_RFmodels.pkl'.replace(mask_name,'subjvis')
        model_savepath = os.path.join(model_path, sub, model_name)
        if os.path.exists(model_savepath):
            models = joblib.load(model_savepath)
        else:
            models = {}
        # loop for voxels
        for idx, voxel in (zip(np.arange(num_voxel), voxel_indices)):
            print(100*idx/num_voxel,f"% - {sub}: voxel-{voxel}")
            if voxel in guassparams.keys():
                if voxel in models.keys():
                    print(f'voxel-{voxel} has already exists')
                else:
                    print(f'final-({idx},{voxel})')
                    # load receptive field
                    receptive_field = gaussian_2d((i, j), *guassparams[voxel])
                    # receptive_field[receptive_field < 0.5*np.max(receptive_field)] = 0
                    receptive_field = receptive_field / (receptive_field.sum() + 1e-20)
                    # saptial summation
                    X_voxel = np.sum(X * receptive_field, axis=(2,3))
                    lr = LinearRegression(n_jobs=12).fit(X_voxel, y[:, idx])
                    models[voxel] = lr
        joblib.dump(models, model_savepath)

        # voxels = voxel_indices.tolist()
        # idxs = np.arange(num_voxel).tolist()
        # results = Parallel(n_jobs=25)(delayed(compute_voxel_performance)(idx, voxel) for idx, voxel in zip(idxs, voxels))
        # all_performace = np.nan*np.zeros((len(fold_indices)+1, len(retinoR2)))
        # all_performace[:, voxel_indices] = np.array(results).transpose()
        # np.save(os.path.join(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_corrperformance-test.npy'), all_performace)
        
    print(f'{sub} consume : {t.interval} s')
