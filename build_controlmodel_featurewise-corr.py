import os
import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
import joblib
from joblib import Parallel, delayed
import time 
from utils import train_data_normalization, Timer, net_size_info


subs = ['sub-09'] # 'sub-01', 'sub-02', 'sub-06', 'sub-07', 'sub-03', 'sub-04',  'sub-05', 'sub-08'
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

for sub in subs:
    with Timer() as t:
        inputlayername = 'googlenet-conv2' 
        layer = {'name': inputlayername, 'size': net_size_info[inputlayername.replace('raw-', '')]}#alexnet_info[inputlayername]
        layername = layer['name']
        layername = layername.replace('.','')
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
        print(f'activations shape of {activations.shape}')
 
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

        if layername == 'googlenet-conv2':
            X = activations[:, 0:63, :, :]
            X_mean = np.mean(X, axis=(2,3))
        y = brain_resp

        # computing 
        all_performace = np.nan*np.zeros((X_mean.shape[-1], 59412))
        all_performace[:, voxel_indices] = np.corrcoef(y.transpose(), X_mean.transpose())[-X_mean.shape[-1]::, 0:len(voxel_indices)]
        np.save(os.path.join(performance_path, sub, f'{sub}_bm-{mask_name}_layer-{layername}_corr.npy'), all_performace)
        
    print(f'{sub} consume : {t.interval} s')
