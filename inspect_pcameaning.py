import numpy as np
from glob import glob
from os.path import join as pjoin
import joblib

# Define the 2D Gaussian function
def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
    i, j = coords
    return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C

sub = 'sub-03'
layername = 'features3'
wk_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'

features = np.load(pjoin(wk_dir, f'build/image_activations/{sub}_features3.npy'))
comps = np.load(pjoin(wk_dir, f'anal/pca-axis/{sub}_pcacomp.npy'))
r = np.load(pjoin(wk_dir, f'build/retrainperformance/{sub}/{sub}_layer-features3_corrperformance-test.npy'))

models = joblib.load(pjoin(wk_dir, 'build/mappings/sub-03/sub-03_layer-features3_RFmodels.pkl'))

# co-ordinate
i = np.linspace(-8., 8., 27)
j = np.linspace(8., -8., 27)
i, j = np.meshgrid(i, j)

# average feature
average_feature = features.mean(axis=(2,3))
num_pics = 4000
cor_wrt_pc1 = [np.corrcoef(average_feature[_,:], comps[0,:])[0,1] for _ in range(num_pics)]
cor_wrt_pc2 = [np.corrcoef(average_feature[_,:], comps[1,:])[0,1] for _ in range(num_pics)]
cor_wrt_pc3 = [np.corrcoef(average_feature[_,:], comps[2,:])[0,1] for _ in range(num_pics)]
cor_wrt_pc4 = [np.corrcoef(average_feature[_,:], comps[3,:])[0,1] for _ in range(num_pics)]

#
guass_path = pjoin(wk_dir, 'build/gaussianparams')
guassparams = np.load(pjoin(guass_path, f'{sub}_layer-{layername}_Gauss.npy'), allow_pickle=True)[0]
voxel_features = []

voxels = [ _ for _ in list(models.keys()) if _ in np.where(r[-1,:]>0.12)[0]]
for voxel in voxels:
    # load receptive field
    receptive_field = gaussian_2d((i, j), *guassparams[voxel])
    # receptive_field[receptive_field < 0.5*np.max(receptive_field)] = 0
    receptive_field = receptive_field / (receptive_field.sum() + 1e-20)
    # saptial summation
    voxel_features.append(np.sum(features * receptive_field, axis=(2,3)))
full_features = np.stack(voxel_features, axis=0)

