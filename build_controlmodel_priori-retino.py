import os
import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.mixture import GaussianMixture
from utils import get_roi_data, ReceptiveFieldProcessor, Timer, alexnet_info, train_data_normalization

subs = ['sub-04', 'sub-03', 'sub-01', 'sub-02', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

for sub in subs:
    inputlayername = 'features.3'
    layer = {'name': inputlayername, 'size':alexnet_info[inputlayername]}
    layername = layer['name']
    layername = layername.replace('.','')
    fold_indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)] #[(0, 2000), (2000, 4000)]
    # 
    work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
    resp_path = os.path.join(work_dir, 'prep/brain_response')
    image_activations_path = os.path.join(work_dir, 'prep/image_activations')
    retino_path = os.path.join(work_dir, 'build/retinoparams')
    model_path = os.path.join(work_dir, 'build/control/controlmappings')
    performance_path = os.path.join(work_dir, 'build/control/controlperformance')
    # load
    brain_resp = np.load(os.path.join(resp_path, f'{sub}_imagenet_beta.npy'))
    activations = np.load(os.path.join(image_activations_path, f'{sub}_{layername}.npy'))
    # save path
    if not os.path.exists(os.path.join(model_path, sub)):
        os.makedirs(os.path.join(model_path, sub))
    if not os.path.exists(os.path.join(performance_path, sub)):
        os.makedirs(os.path.join(performance_path, sub))

    # ==================================================
    # this is the selection procedure via subject atlas
    # ==================================================
    opendata_cifti_pth = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
    # getting retinotopic voxels
    sub_prf_file = os.path.join(opendata_cifti_pth, sub, 'results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii')
    retino_r2 = nib.load(sub_prf_file).get_fdata()[3, :]
    r2_thres = 9
    retino_voxels = np.where(retino_r2 > r2_thres)[0]
    # getting floc voxels
    sub_floc_path = os.path.join(opendata_cifti_pth, sub, 'results/ses-floc_task-floc')
    altases_files = [f'floc-{_}.dlabel.nii' for _ in ['bodies', 'faces', 'places', 'words']]
    floc_atlas = None
    for atlas_file in altases_files:
        atlas = nib.load(os.path.join(sub_floc_path, atlas_file)).get_fdata()[0,:]
        if floc_atlas is None:
            floc_atlas = atlas
        else:
            floc_atlas += atlas
    floc_voxels = np.where(floc_atlas != 0)[0]
    # merging into one mask
    # voxel_indices = np.array([ _ for _ in range(59412) if (_ in floc_voxels) or (_ in retino_voxels)])
    voxel_mask = np.array([ _ for _ in range(59412) if (_ in floc_voxels) or (_ in retino_voxels)])

    # ==================================================
    # this is the selection procedure via standard atlas
    # ==================================================
    # select ROI names
    evc_pool = ['V1', 'V2', 'V3','V4']
    # aggregate ROI vertices
    roi_name = [__  for _ in [evc_pool] for __ in _]
    # form ROI mask
    selection_mask = np.sum([get_roi_data(None, _) for _ in roi_name], axis=0)
    # trsnfer to indices in cifti space
    voxel_indices = [_ for _ in np.where(selection_mask==1)[0] if _ in voxel_mask]

    # collect resp in ROI
    brain_resp = brain_resp[:, voxel_indices]

    # normalization
    norm_metric = 'session'
    brain_resp = train_data_normalization(brain_resp, metric=norm_metric)
    num_voxel = brain_resp.shape[-1]

    X = activations
    y = brain_resp
    # final train
    models = {}
    retino_info = nib.load(sub_prf_file).get_fdata()
    n_vertices = retino_info.shape[-1]
    retinotopy_params = np.zeros((n_vertices, 3))
    retinotopy_params[:,0] = retino_info[0, 0:n_vertices]
    retinotopy_params[:,1] = retino_info[1, 0:n_vertices]*16/200
    retinotopy_params[:,2] = retino_info[2, 0:n_vertices]*16/200
    # we need to transfer the params into (x,y,size) model
    trans_retinotopy_params = np.zeros_like(retinotopy_params)
    trans_retinotopy_params[:,0] = np.cos(retinotopy_params[:,0]/180*np.pi)*retinotopy_params[:,1]
    trans_retinotopy_params[:,1] = np.sin(retinotopy_params[:,0]/180*np.pi)*retinotopy_params[:,1]
    trans_retinotopy_params[:,2] = retinotopy_params[:,2]
    # 
    corr_performace = np.nan * np.zeros((n_vertices,))
    for idx, voxel in (zip(np.arange(num_voxel), voxel_indices)):
        print(100*idx/num_voxel,f"% - {sub}")
        
        # print(f'final-({idx},{voxel})')
        # load receptive field
        receptive_field = ReceptiveFieldProcessor(16, *trans_retinotopy_params[voxel]).get_spatial_kernel(X.shape[-1])
        if not np.isnan(receptive_field).any():
            # saptial summation
            X_voxel = np.sum(X * receptive_field, axis=(2,3))
            lr = LinearRegression(n_jobs=8).fit(X_voxel, y[:, idx])
            corr_performace[voxel] = np.corrcoef(lr.predict(X_voxel), y[:, idx])[0,1]
            models[voxel] = lr

    joblib.dump(models, os.path.join(model_path, sub, f'{sub}_bm-primvis_layer-{layername}_controlRFmodels.pkl'))
    np.save(os.path.join(performance_path, sub, f'{sub}_bm-primvis_layer-{layername}_corrperformance-train.npy'), corr_performace)

