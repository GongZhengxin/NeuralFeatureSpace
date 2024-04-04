import os
from os.path import join
import numpy as np
import nibabel as nib
from scipy.optimize import curve_fit
import time
from utils import net_size_info
from scipy.ndimage import uniform_filter, median_filter

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
subs = [f'sub-{i:02d}' for i in range(1, 10)]
# subs=['sub-30']
# subs = ['sub-05', 'sub-08'] #'sub-01', 'sub-02', 
print(subs)
# sub = subs[-1]
# check_file = f'/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/build/corrmap/{sub}_bm-primaryvis-in-MMP_layer-raw-googlenet-conv2_corrmap-test.npy'
# sleep_time = 180 #2600*(subs.index(sub)+1)
# print(f'wait {sub}; sleeping for {sleep_time} seconds')
# # time.sleep(sleep_time)
# while 1:
#     if os.path.exists(check_file):
#         print('file ready!')
#         break
#     else:
#         print(f'checkfile not exists, wait again')
#         time.sleep(180)
# # subs = [sub]'googlenet-inception3a', 'googlenet-conv2'
# time.sleep(180)
for cur_layer in ['raw-googlenet-concate']:#'googlenet-maxpool2','googlenet-inception3a', 'googlenet-conv2'
    for sub in subs[0: 2]:
        t0 = time.time()
        print(f'!! {sub}')
        # maskname = 'primar-in-MMP' # 
        inputlayername = cur_layer #f''raw-googlenet-conv2maxpool2
        layername = inputlayername.replace('.', '')
        dataset = 'test'
        mask_name = 'primaryvis-in-MMP' #'loandit-in-MMP' #
        ciftiy_path = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
        wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
        # input path
        corrmap_dir = os.path.join(wk_dir, 'build/corrmap')
        voxel_mask_path = os.path.join(wk_dir, 'prep/voxel_masks')
        # output path
        retino_path = os.path.join(wk_dir, f'build/retinoparams/{mask_name}')
        guass_path = os.path.join(wk_dir, f'build/gaussianparams/{mask_name}')
        
        # load
        corrmap_file = f'{sub}_bm-{mask_name}_layer-{layername}_corrmap-{dataset}.npy' #.replace(mask_name,'subjvis')
        corr_data = np.load(os.path.join(corrmap_dir, corrmap_file))
        voxel_mask_nii = nib.load(os.path.join(voxel_mask_path, f'nod-voxmask_{mask_name}.dlabel.nii'))
        voxel_mask = np.squeeze(voxel_mask_nii.get_fdata())
        named_maps = [named_map.map_name for named_map in voxel_mask_nii.header.get_index_map(0).named_maps]
        
        if 'conv2' in cur_layer:
            # # determine the mask type
            if sub in named_maps:
                voxel_mask = voxel_mask[named_maps.index(sub),:]
            # squeeze into 1 dim
            mmp_voxel_mask = np.squeeze(np.array(voxel_mask))
            voxel_indices = np.where(mmp_voxel_mask==1)[0] # only for conv2 !! 只为conv2
        else:
            #读入prf文件
            prf_data = nib.load(os.path.join(ciftiy_path, f'{sub}/results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii')).get_fdata()
            
            #选取R2大于10的体素
            R2_values = prf_data[3, :]
            valid_R2_indices = np.where(R2_values >= 10)[0]

            # # determine the mask type
            if sub in named_maps:
                voxel_mask = voxel_mask[named_maps.index(sub),:]
            # squeeze into 1 dim
            mmp_voxel_mask = np.squeeze(np.array(voxel_mask))
            # 确定最终的mask indices
            mmp_voxel_indices = np.where(mmp_voxel_mask==1)[0]
            voxel_indices = np.intersect1d(mmp_voxel_indices, valid_R2_indices) #!
        
        
        print('vox num',len(voxel_indices))
        # Create grid data
        layersize = corr_data.shape[0]
        i = np.linspace(-8., 8., layersize)
        j = np.linspace(8., -8., layersize)
        i, j = np.meshgrid(i, j)

        xpos2vf = lambda x : (x-int(layersize/2)) * 16 / layersize
        ypos2vf = lambda y : (int(layersize/2) - y) * 16 / layersize

        fwhm2sigma = lambda x : x / 2.354282

        retino_dict = {'ecc':np.nan*np.zeros((59412,)), 'ang':np.nan*np.zeros((59412,)), 
                    'rfsize':np.nan*np.zeros((59412,)), 'R2':np.nan*np.zeros((59412,))}
        gauss_dict = {}
        failed_corrmaps = []
        for idx in range(corr_data.shape[-1]):
            data = corr_data[:, :, idx]
            data  = median_filter(data, size=3)
            if np.sum(np.isnan(data)) > 0:
                # print(f'ratio of NaN is {np.sum(np.isnan(data))/(data.shape[0]*data.shape[1])}')
                data = data[0:layersize, 0:layersize]
            max_pos = np.unravel_index(np.argmax(data, axis=None), data.shape)
            half_max = 0.5*np.max(data, axis=None)

            i = np.linspace(-8., 8., layersize)
            j = np.linspace(8., -8., layersize)
            i, j = np.meshgrid(i, j)

            # initialize settings
            A_init, C_init = 1, np.min(data, axis=None)
            # x_init, y_init = pos2vf(max_pos[0]), pos2vf(max_pos[1])
            y_poses, x_poses = np.where(data > np.percentile(data, 95))
            x_init, y_init = xpos2vf(np.mean(x_poses)), ypos2vf(np.mean(y_poses))

            xsigma_init, ysigma_init = fwhm2sigma(np.sum(data[:, max_pos[1]]>=half_max)), fwhm2sigma(np.sum(data[max_pos[0], :]>=half_max))
            
            param_initial = [A_init, x_init, y_init, xsigma_init, ysigma_init, C_init]
            # if np.sqrt(x_init**2 + y_init**2) < 2:
            good_thres = np.sort(data, axis=None)[-200]
            good_pos = np.where(data >= good_thres)
            i, j = i[good_pos], j[good_pos]
            data = data[good_pos]
            try:
                # Fit the mock data
                bound =([0, -10, -10, 0, 0, -np.inf], [np.inf, 10, 10, np.inf, np.inf, np.inf])
                params, covariance = curve_fit(gaussian_2d, (i.ravel(), j.ravel()), data.ravel(), p0=param_initial, bounds=bound)
                A, x_0, y_0, xsigma, ysigma, C = params

                fitted_data = gaussian_2d((i, j), A, x_0, y_0, xsigma, ysigma, C)

                retino_dict['ecc'][voxel_indices[idx]] = np.sqrt(x_0 ** 2 + y_0 ** 2)
                retino_dict['ang'][voxel_indices[idx]] = compute_theta(x_0, y_0)
                retino_dict['rfsize'][voxel_indices[idx]] = np.sqrt(xsigma*ysigma)
                retino_dict['R2'][voxel_indices[idx]] = np.nanmax(data)#np.corrcoef(fitted_data.ravel(), data.ravel())[0,1]
                print(f'{idx}: voxel{voxel_indices[idx]}')

                gauss_dict[voxel_indices[idx]] = (A, x_0, y_0, xsigma, ysigma, C)
            except RuntimeError: 
                failed_corrmaps.append(voxel_indices[idx])
                print(f'======>>>>>>>Failed {idx}: voxel{voxel_indices[idx]}')
        # with open()
        # np.save(os.path.join(retino_path, f'{sub}_layer-{layername}_fialedvoxels.npy'), np.array(failed_corrmaps))
        print(f'{sub} consume: {time.time() - t0}s')
        np.save(os.path.join(retino_path, f'{sub}_layer-{layername}_params.npy'), np.array([retino_dict]), allow_pickle=True)
        np.save(os.path.join(guass_path, f'{sub}_layer-{layername}_Gauss.npy'), np.array([gauss_dict]), allow_pickle=True)
        print(os.path.join(retino_path, f'{sub}_layer-{layername}_params.npy'))