import joblib
import os
import nibabel as nib
import numpy as np
from os.path import join as pjoin
from utils import save2cifti
import gc
import time

wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
model_path = pjoin(wk_dir, 'build/mappings/')
activ_path = pjoin(wk_dir, 'prep/image_activations/')
resp_path = pjoin(wk_dir, 'prep/brain_response/')
gauss_path = pjoin(wk_dir, 'build/gaussianparams')
mask_dir = pjoin(wk_dir,'prep/voxel_masks/')

save_dir = pjoin(wk_dir,'anal/test_performance')


mask_name = 'primaryvis-in-MMP'
primvis_mask = np.squeeze(nib.load(pjoin(mask_dir, f'nod-voxmask_{mask_name}.dlabel.nii')).get_fdata())
primivis_idx = np.where(primvis_mask == 1)[0]

test_set_name = 'coco'

def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
    i, j = coords
    return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C

model_name = 'RFmodels'
mask_name = 'subjvis'
net_name = 'googlenet'
layer_names = ['conv2', 'inception3a']
subs = [f'sub-0{_+1}' for _ in range(9)]


load_model = lambda x : joblib.load(pjoin(model_path, sub, x))
load_activ = lambda x : np.load(pjoin(activ_path, f'{test_set_name}_{net_name}-{x}.npy'))
load_retino = lambda x, y :np.load(pjoin(gauss_path, f'{x}_layer-{net_name}-{y}_Gauss.npy'), allow_pickle=True)[0]

all_performance = np.zeros((len(subs), len(layer_names), 59412))
for i, sub in enumerate(subs):
    # load, reshape and average the resp
    test_resp = np.load(pjoin(resp_path, f'{sub}_{test_set_name}_beta.npy'))
    num_trial = test_resp.shape[0]
    num_run = int(num_trial/120)
    test_resp = test_resp.reshape((num_run, 120, 59412))
    mean_test_resp = test_resp.mean(axis=0)
    for j, layer_name in enumerate(layer_names):
        print(sub, '-', layer_name)
        # load model
        fullmodelname = f'{sub}_bm-{mask_name}_layer-{net_name}-{layer_name}_{model_name}.pkl'
        model = load_model(fullmodelname)
        # determine the shared voxel
        shared_voxel = [ _ for _ in primivis_idx if _ in list(model.keys())]
        # load feature
        test_features = load_activ(layer_name)
        # generate grids
        ii = np.linspace(-8., 8., test_features.shape[-1])
        jj = np.linspace(8., -8., test_features.shape[-1])
        ii, jj = np.meshgrid(ii, jj)
        # load guassian params
        gaussparams = load_retino(sub, layer_name)
        # initialize performance
        performance = []
        # make model prediction 
        for voxel in shared_voxel:
            # print(f'{voxel},{sub},{layer_name}')
            rf = gaussian_2d((ii, jj), *gaussparams[voxel])
            rf = rf / (rf.sum() + 1e-20)
            voxel_test_feature = np.sum(test_features * rf, axis=(2,3)) 
            prediction = model[voxel].predict(voxel_test_feature)
            performance.append(np.corrcoef(prediction, mean_test_resp[:, voxel])[0, 1])
        
        del model, test_features, gaussparams
        gc.collect()

        all_performance[i, j, np.array(shared_voxel)] = np.array(performance)
        print(f'finish {sub} {layer_name}')
# saveout the performance matrix
np.save(pjoin(save_dir, f'sub-all_ly-0{layer_names[0]}-1{layer_names[1]}_performance.npy'), all_performance)
print(f'saved')