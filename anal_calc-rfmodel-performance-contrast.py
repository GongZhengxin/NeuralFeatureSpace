import joblib
import nibabel as nib
import numpy as np
from os.path import join as pjoin

# path assignment
wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
model_path = pjoin(wk_dir, 'build/mappings/')
activ_path = pjoin(wk_dir, 'prep/image_activations/')
resp_path = pjoin(wk_dir, 'prep/brain_response/')
gauss_path = pjoin(wk_dir, 'build/gaussianparams')

subs = ['sub-03','sub-04','sub-05','sub-08', 'sub-01', 'sub-02', 'sub-06', 'sub-07', 'sub-09' ]#
mask_name = 'subjvis'

alexlayer = 'features3'
inceplayer = 'googlenet-inception3a'
model_name = 'RFmodels'

for sub in subs:
    print('================', sub)
    alexnetmodelname = f'{sub}_bm-{mask_name}_layer-{alexlayer}_{model_name}.pkl'
    googlemodelname = f'{sub}_bm-{mask_name}_layer-{inceplayer}_{model_name}.pkl'

    # load models
    load_model = lambda x : joblib.load(pjoin(model_path, sub, x))
    alexnetmodel = load_model(alexnetmodelname)
    googlemodel = load_model(googlemodelname)
    print('voxel number in models:', len(alexnetmodel.keys()), len(googlemodel.keys()))

    shared_voxel = list(set(list(googlemodel.keys())) & set(list(alexnetmodel.keys())))

    import nibabel as nib
    mask_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/voxel_masks/'
    roi_mask_name = 'primaryvis-in-MMP'
    primvis_mask = np.squeeze(nib.load(pjoin(mask_dir, f'nod-voxmask_{roi_mask_name}.dlabel.nii')).get_fdata())
    primivis_idx = np.where(primvis_mask == 1)[0]

    selected_voxels =[ _ for _ in primivis_idx if _ in shared_voxel]# list(googlemodel.keys()) 
    print('voxel number in selection:', len(selected_voxels))

    test_set_name = 'coco'
    test_resp = np.load(pjoin(resp_path, f'{sub}_{test_set_name}_beta.npy'))
    num_trial = test_resp.shape[0]
    num_run = int(num_trial/120)
    test_resp = test_resp.reshape((num_run, 120, 59412))
    mean_test_resp = test_resp.mean(axis=0)
    print('test resp shape:', mean_test_resp.shape)

    # load features
    load_activ = lambda x : np.load(pjoin(activ_path, f'{test_set_name}_{x}.npy'))
    test_alexfeature = load_activ(alexlayer)
    test_gglfeature = load_activ(inceplayer)

    def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
        i, j = coords
        return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C
    # load guassian mask
    alex_gaussparams = np.load(pjoin(gauss_path, f'{sub}_layer-{alexlayer}_Gauss.npy'), allow_pickle=True)[0]
    ggl_gaussparams = np.load(pjoin(gauss_path, f'{sub}_layer-{inceplayer}_Gauss.npy'), allow_pickle=True)[0]

    i_alex = np.linspace(-8., 8., 27)
    j_alex = np.linspace(8., -8., 27)
    i_alex, j_alex = np.meshgrid(i_alex, j_alex)

    i_ggl = np.linspace(-8., 8., 28)
    j_ggl = np.linspace(8., -8., 28)
    i_ggl, j_ggl = np.meshgrid(i_ggl, j_ggl)

    # make predictions
    alex_performance, ggl_performance = [], []

    for voxel in selected_voxels:
        alex_rf = gaussian_2d((i_alex, j_alex), *alex_gaussparams[voxel])
        alex_rf = alex_rf / (alex_rf.sum() + 1e-20)
        voxel_test_alexfeature = np.sum(test_alexfeature * alex_rf, axis=(2,3)) 
        alex_prediction = alexnetmodel[voxel].predict(voxel_test_alexfeature)
        alex_performance.append(np.corrcoef(alex_prediction, mean_test_resp[:, voxel])[0, 1])

        ggl_rf = gaussian_2d((i_ggl, j_ggl), *ggl_gaussparams[voxel])
        ggl_rf = ggl_rf / (ggl_rf.sum() + 1e-20)
        voxel_test_gglfeature = np.sum(test_gglfeature * ggl_rf, axis=(2,3)) 
        ggl_prediction = googlemodel[voxel].predict(voxel_test_gglfeature)
        ggl_performance.append(np.corrcoef(ggl_prediction, mean_test_resp[:, voxel])[0, 1])

    print('mean performance of models:', np.mean(alex_performance), np.mean(ggl_performance))
    save_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/anal/test_performance'
    np.save(pjoin(save_dir, f'{sub}_alexnet_performance.npy'), alex_performance)
    np.save(pjoin(save_dir, f'{sub}_gglnet_performance.npy'), ggl_performance)