import os
import numpy as np
import nibabel as nib
from scipy.optimize import curve_fit
import time
from utils import net_size_info

# Define the 2D Gaussian function
def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
    i, j = coords
    return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C

def compute_theta(x, y):
    # Calculate angle using arctan2
    theta_rad = np.arctan2(y, x)
    
    # Rotate by 90 degrees to align with y-axis
    theta_rad = np.pi/2 - theta_rad
    
    # Convert to degrees and handle the range
    theta_deg = np.degrees(theta_rad)
    if theta_deg > 180:
        theta_deg -= 360

    return theta_deg

# ========================================================
subs = ['sub-03', 'sub-04','sub-06', 'sub-07']
# subs = ['sub-05', 'sub-08'] #'sub-01', 'sub-02', 
print(subs)
sub = subs[-1]
check_file = f'/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/build/corrmap/{sub}_bm-primaryvis-in-MMP_layer-googlenet-conv2_corrmap-test.npy'
sleep_time = 180 #2600*(subs.index(sub)+1)
print(f'wait {sub}; sleeping for {sleep_time} seconds')
time.sleep(sleep_time)
while 1:
    if os.path.exists(check_file):
        print('file ready!')
        break
    else:
        print(f'checkfile not exists, wait again')
        time.sleep(180)
# subs = [sub]

for sub in subs:
    print(f'!! {sub}')
    
    inputlayername = 'googlenet-conv2' #f'googlenet-inception{k}a'
    layername = inputlayername.replace('.', '')
    dataset = 'test'
    mask_name = 'primaryvis-in-MMP' #'fixretfloc-in-subj'

    wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
    # input path
    corrmap_dir = os.path.join(wk_dir, 'build/corrmap')
    voxel_mask_path = os.path.join(wk_dir, 'prep/voxel_masks')
    # output path
    retino_path = os.path.join(wk_dir, 'build/retinoparams')
    guass_path = os.path.join(wk_dir, 'build/gaussianparams')
    
    # load
    corrmap_file = f'{sub}_bm-{mask_name}_layer-{layername}_corrmap-{dataset}.npy' #.replace(mask_name,'subjvis')
    corr_data = np.load(os.path.join(corrmap_dir, corrmap_file))
    voxel_mask_nii = nib.load(os.path.join(voxel_mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'))
    voxel_mask = np.squeeze(voxel_mask_nii.get_fdata())
    named_maps = [named_map.map_name for named_map in voxel_mask_nii.header.get_index_map(0).named_maps]
    
    # determine the mask type
    if sub in named_maps:
        voxel_mask = voxel_mask[named_maps.index(sub),:]
    # squeeze into 1 dim
    voxel_mask = np.squeeze(np.array(voxel_mask))
    # transfer mask into indices
    voxel_indices = np.where(voxel_mask==1)[0]

    # Create grid data
    layersize = corr_data.shape[0]
    i = np.linspace(-8., 8., layersize)
    j = np.linspace(8., -8., layersize)
    i, j = np.meshgrid(i, j)

    pos2vf = lambda x : (x-int(layersize/2)) * 16 / layersize
    fwhm2sigma = lambda x : x / 2.354282

    retino_dict = {'ecc':np.nan*np.zeros((59412,)), 'ang':np.nan*np.zeros((59412,)), 
                'rfsize':np.nan*np.zeros((59412,)), 'R2':np.nan*np.zeros((59412,))}
    gauss_dict = {}
    failed_corrmaps = []
    for idx in range(corr_data.shape[-1]):
        data = corr_data[:, :, idx]
        if np.sum(np.isnan(data)) > 0:
            # print(f'ratio of NaN is {np.sum(np.isnan(data))/(data.shape[0]*data.shape[1])}')
            data = data[0:layersize, 0:layersize]
        max_pos = np.unravel_index(np.argmax(data, axis=None), data.shape)
        half_max = 0.5*np.max(data, axis=None)
        # initialize settings
        A_init, C_init = 1, np.min(data, axis=None)
        # x_init, y_init = pos2vf(max_pos[0]), pos2vf(max_pos[1])
        x_poses, y_poses = np.where(data > np.percentile(data, 90))
        x_init, y_init = pos2vf(np.mean(x_poses)), pos2vf(np.mean(y_poses))

        xsigma_init, ysigma_init = fwhm2sigma(np.sum(data[:, max_pos[1]]>=half_max)), fwhm2sigma(np.sum(data[max_pos[0], :]>=half_max))
        
        param_initial = [A_init, x_init, y_init, xsigma_init, ysigma_init, C_init]
        try:
            # Fit the mock data
            bound =([0, -20, -20, 0, 0, -np.inf], [np.inf, 20, 20, np.inf, np.inf, np.inf])
            params, covariance = curve_fit(gaussian_2d, (i.ravel(), j.ravel()), data.ravel(), p0=param_initial, bounds=bound)
            A, x_0, y_0, xsigma, ysigma, C = params

            fitted_data = gaussian_2d((i, j), A, x_0, y_0, xsigma, ysigma, C)

            retino_dict['ecc'][voxel_indices[idx]] = np.sqrt(x_0 ** 2 + y_0 ** 2)
            retino_dict['ang'][voxel_indices[idx]] = compute_theta(x_0, y_0)
            retino_dict['rfsize'][voxel_indices[idx]] = np.sqrt(xsigma*ysigma)
            retino_dict['R2'][voxel_indices[idx]] = np.corrcoef(fitted_data.ravel(), data.ravel())[0,1]
            print(f'{idx}: voxel{voxel_indices[idx]}')

            gauss_dict[voxel_indices[idx]] = (A, x_0, y_0, xsigma, ysigma, C)
        except RuntimeError: 
            failed_corrmaps.append(idx)
            print(f'======>>>>>>>Failed {idx}: voxel{voxel_indices[idx]}')
    
    np.save(os.path.join(retino_path, f'{sub}_layer-{layername}_params.npy'), np.array([retino_dict]), allow_pickle=True)
    np.save(os.path.join(guass_path, f'{sub}_layer-{layername}_Gauss.npy'), np.array([gauss_dict]), allow_pickle=True)
    