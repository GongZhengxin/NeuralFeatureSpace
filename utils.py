import os
import time
import numpy as np
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio 
import nibabel as nib
from scipy.stats import zscore
import nibabel as nib
from nibabel import cifti2
import scipy.io as sio
# from dnnbrain.dnn.core import Activation

conv2_human_labels = {
    'color' : [2, 48, 36, 46, 35, 26, 16, 17],
    'colorcontrast': [53, 50, 42, 45, 21, 12, 9, 6, 10, 20, 5, 56, 57, 8],
    'gabor': [0,1,4,13,19, 22,23,27,32,34, 39,43,44,47,54, 60,61,62,63,25, 29,24,18,49,38],
    'hatch': [14, 30, 33, 41],
    'texture': [3,40,7,11,37,15,28,51,52,55,58,59,31],
}

conv2_labels = {
    'lowfreq':[1,13,27,47,56,60,23,49,0,43,28,29, 34,8,37,19,15],
    'color':[2,48,36,46],
    'colorcontrast':[5,9,10,12,20,21,42,45,50,53],
    'multicolor':[3,33,35,40,17,16,57,26,31],
    'complxgabor':[51,58,30,25,52,54,22,61,55],
    'gabor': [4,6,32,38,41,7,11,18,24,39, 63], #: all zeros,
    'other':[44,62,59],
    'hatch':[14]
}

incep3a_labels = {'LinEnd':[133,96], 
 'ThinLin':[140,78,89],
 'Cross':[91,185,64,118],
 'Eye':[174,168,79,125,175],
 'Fur':[46,47,26,63,80,23,16],
 'Ang':[188,94,164,107,77,157,149,100],
 'BW-C': [214,208,201,223,210,197,222,204,220],
 'Curv':[81,104,92,145,95,163,171,71,147,189,137],
 'Rep':[237,31,17,20,39,126,124,156,98,105,230,228],
 'Other':[38,43,58,67,190,109,122,128,142,143,155,170,179,184],
 'Lin':[227,75,146,69,169,57,154,187,27,134,150,240,101,176],
 'Linmsc':[191, 121,116,14,24,0,159,152,165,83,173,87,90,82],
 'Compc-s':[178,181,161,166,172,68,130,49,52,114,115,120,144,37],
 'Cocontr':[195, 84, 85, 123, 203, 217, 199, 211, 205,212,202,200, 138,32],
 'Bri':[216,127,22,182,162,25,249,15,28,59,29,196,206,18,247],
 'HLfrq':[110,180,153,106,112,186,132,136,117,113,108,70,86,88,160],
 'Coc-s':[119,34,167,76,19,30,131,251,226,13,7,50,1,4,41,192,36,40,103,213,10,35,221,193,158,73,74,177,97,141],
 'Tex':[246,242,253,232,233,209,139,65,44,51,194,207,111,218,224,225,215,198, 62, 21,254, 255,61,2,3,8,12,53, 56, 102, 148, 244, 250,11, 238,248,9,219, 234, 252, 236, 5,183, 241, 229, 93,243, 99,45,33,135,231,60, 235,48,55,42,151,54,72,6,239,66,129,245]
}



net_size_info = {'features.0':55, 'features.1':55,
                'features.3':27, 'features.4':27,
                'features.6':13, 'features.7':13,
                'features.8':13, 'features.9':13,
                'features.10':13, 'features.11':13,
                'googlenet-conv1' : 114,
                'googlenet-conv2' : 57,
                'googlenet-conv3' : 57,
                'googlenet-maxpool2' : 28,
                'googlenet-inception3a' : 28,
                'googlenet-concate' : 28}

def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_tables=None):
    """
    Save data as a cifti file
    If you just want to simply save pure data without extra information,
    you can just supply the first three parameters.
    NOTE!!!!!!
        The result is a Nifti2Image instead of Cifti2Image, when nibabel-2.2.1 is used.
        Nibabel-2.3.0 can support for Cifti2Image indeed.
        And the header will be regard as Nifti2Header when loading cifti file by nibabel earlier than 2.3.0.
    Parameters:
    ----------
    file_path: str
        the output filename
    data: numpy array
        An array with shape (maps, values), each row is a map.
    brain_models: sequence of Cifti2BrainModel
        Each brain model is a specification of a part of the data.
        We can always get them from another cifti file header.
    map_names: sequence of str
        The sequence's indices correspond to data's row indices and label_tables.
        And its elements are maps' names.
    volume: Cifti2Volume
        The volume contains some information about subcortical voxels,
        such as volume dimensions and transformation matrix.
        If your data doesn't contain any subcortical voxel, set the parameter as None.
    label_tables: sequence of Cifti2LableTable
        Cifti2LableTable is a mapper to map label number to Cifti2Label.
        Cifti2Lable is a specification of the label, including rgba, label name and label number.
        If your data is a label data, it would be useful.
    """
    if file_path.endswith('.dlabel.nii'):
        assert label_tables is not None
        idx_type0 = 'CIFTI_INDEX_TYPE_LABELS'
    elif file_path.endswith('.dscalar.nii'):
        idx_type0 = 'CIFTI_INDEX_TYPE_SCALARS'
    else:
        raise TypeError('Unsupported File Format')

    if map_names is None:
        map_names = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

    if label_tables is None:
        label_tables = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(label_tables), "Label_tables are mismatched with the data"

    # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
    # and this index_map always contains some scalar information, such as named_maps.
    # We can get label_table and map_name and metadata from named_map.
    mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], idx_type0)
    for mn, lbt in zip(map_names, label_tables):
        named_map = cifti2.Cifti2NamedMap(mn, label_table=lbt)
        mat_idx_map0.append(named_map)

    # CIFTI_INDEX_TYPE_BRAIN_MODELS always corresponds to Cifti2Image.header.get_index_map(1),
    # and this index_map always contains some brain_structure information, such as brain_models and volume.
    mat_idx_map1 = cifti2.Cifti2MatrixIndicesMap([1], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
    for bm in brain_models:
        mat_idx_map1.append(bm)
    if volume is not None:
        mat_idx_map1.append(volume)

    matrix = cifti2.Cifti2Matrix()
    matrix.append(mat_idx_map0)
    matrix.append(mat_idx_map1)
    header = cifti2.Cifti2Header(matrix)
    img = cifti2.Cifti2Image(data, header)
    cifti2.save(img, file_path)

from sklearn.base import BaseEstimator, TransformerMixin
class ReceptiveFieldProcessor(BaseEstimator, TransformerMixin):
    """
    The method is to process feature map data.
    This is a transformer based on sk-learn base.

    parameter
    ---------
    center_x : float / array
    center_y : float / array
    size : float / array

    return
    ------
    feature_vector :
    """

    def __init__(self, vf_size, center_x, center_y, rf_size):
        self.vf_size = vf_size
        self.center_x = center_x
        self.center_y = center_y
        self.rf_size = rf_size

    def get_spatial_kernel(self, kernel_size):
        """
        For an image stimuli cover the visual field of **vf_size**(unit of deg.) with
        height=width=**max_resolution**, this method generate the spatial receptive
        field kernel in a gaussian manner with center at (**center_x**, **center_y**),
        and sigma **rf_size**(unit of deg.).

        parameters
        ----------
        kernel_size : int
            Usually the origin stimuli resolution.

        return
        ------
        spatial_kernel : np.ndarray

        """
        # t3 = time.time()
        # prepare parameter for np.meshgrid
        low_bound = - int(self.vf_size / 2)
        up_bound = int(self.vf_size / 2)
        # center at (0,0)
        x = np.linspace(low_bound, up_bound, kernel_size)
        y = np.linspace(low_bound, up_bound, kernel_size)
        y = -y  # adjust orientation
        # generate grid
        xx, yy = np.meshgrid(x, y)
        # prepare for spatial_kernel
        ind = -((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2) / (2 * self.rf_size ** 2)  # gaussian index

        spatial_kernel = np.exp(ind)  # initial spatial_kernel
        # normalize
        spatial_kernel = spatial_kernel / (np.sum(spatial_kernel)+ 1e-8)
        k = 0
        while 1:
            if len(np.unique(spatial_kernel)) == 1:
                k = k+1
                ind = -((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2) / (2 * ((2**k)*self.rf_size) ** 2)
                spatial_kernel = np.exp(ind)
                spatial_kernel = spatial_kernel / (np.sum(spatial_kernel)+ 1e-8)
            else:
                break
        # t4 = time.time()
        # print('get_spatial_kernel() consumed {} min'.format((t4-t3)/60))
        return spatial_kernel

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, ):
        """

        """
        # initialize
        # t0 = time.time()
        feature_vectors = np.array([])
        if not ((type(X) == list) ^ (type(X) == np.ndarray)):
            raise AssertionError('Data type of X is not supported, '
                                 'Please check only list or numpy.ndarray')
        elif type(X) == np.ndarray:
            # input array height
            map_size = X.shape[-1]
            kernel = self.get_spatial_kernel(map_size)
            feature_vectors = np.sum(X * kernel, axis=(2, 3))
        # t1 = time.time()
        # print('transform comsumed {} min'.format((t1-t0)/60))
        return feature_vectors

# save nifti
def save_ciftifile(data, filename, template='./template.dtseries.nii'):
    ex_cii = nib.load(template)
    if data.ndim == 1:
      data = data[None,:]
    ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    nib.save(nib.Cifti2Image(data,ex_cii.header), filename)

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def train_data_normalization(data, metric='run',runperses=10,trlperrun=100):
    if data.ndim != 2:
        raise AssertionError('check data shape into (n-total-trail,n-brain-size)')
        return 0
    if metric == 'run':
        nrun = data.shape[0] / trlperrun
        for i in range(int(nrun)):
            # run normalization is to demean the run effect 
            data[i*trlperrun:(i+1)*trlperrun,:] = zscore(data[i*trlperrun:(i+1)*trlperrun,:], None)
    elif metric =='session':
        nrun = data.shape[0] / trlperrun
        nses = nrun/runperses
        for i in range(int(nses)):
            data[i*trlperrun*runperses:(i+1)*trlperrun*runperses,:] = zscore(data[i*trlperrun*runperses:(i+1)*trlperrun*runperses,:], None)
    elif metric=='trial':
        data = zscore(data, axis=1)
    return data


def get_roi_data(data, roi_name, hemi=False):
    roi_info = pd.read_csv('/nfs/z1/zhenlab/BrainImageNet//Analysis_results/roilbl_mmp.csv',sep=',')
    roi_list = list(map(lambda x: x.split('_')[1], roi_info.iloc[:,0].values))
    roi_brain = sio.loadmat('/nfs/z1/zhenlab/BrainImageNet/Analysis_results/MMP_mpmLR32k.mat')['glasser_MMP'].reshape(-1)
    if data is not None:
      if data.shape[1] == roi_brain.size:
        if not hemi:
            return np.hstack((data[:, roi_brain==(1+roi_list.index(roi_name))], data[:, roi_brain==(181+roi_list.index(roi_name))]))
        elif hemi == 'L':
            return data[:, roi_brain==(1+roi_list.index(roi_name))]
        elif hemi == 'R':
            return data[:, roi_brain==(181+roi_list.index(roi_name))]
      else:
        roi_brain = np.pad(roi_brain, (0, data.shape[1]-roi_brain.size), 'constant')
        if not hemi:
            return np.hstack((data[:, roi_brain==(1+roi_list.index(roi_name))], data[:, roi_brain==(181+roi_list.index(roi_name))]))
        elif  hemi == 'L':
            return data[:, roi_brain==(1+roi_list.index(roi_name))]
        elif hemi == 'R':
            return data[:, roi_brain==(181+roi_list.index(roi_name))]
    else:
      roi_brain = np.pad(roi_brain, (0, 91282-roi_brain.size), 'constant')
      if type(roi_name)==list:
        return np.sum([get_roi_data(None, _,hemi) for _ in roi_name], axis=0)
      else:
        if not hemi:
            return (roi_brain==(1+roi_list.index(roi_name))) +(roi_brain==(181+roi_list.index(roi_name)))
        elif  hemi == 'L':
            return roi_brain==(1+roi_list.index(roi_name))
        elif hemi == 'R':
            return roi_brain==(181+roi_list.index(roi_name))

def solve_GMM_eq_point(m1,m2,std1,std2):
    a = np.squeeze(1/(2*std1**2) - 1/(2*std2**2))
    b = np.squeeze(m2/(std2**2) - m1/(std1**2))
    c = np.squeeze(m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1))
    return np.roots([a,b,c])



def trash():
    # use kfold to estimate the performance
    fold_indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
    print('kfold start')
    for fold, (start, end) in enumerate(fold_indices):
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        
        X_val = X[start:end]
        y_val = y[start:end]
        
        models = {}  # 存储每个位置的线性回归模型
        
        performance = np.nan * np.zeros((len(fold_indices)+1, len(retinoR2)))
        
        for idx, voxel in tqdm(zip(np.arange(num_voxel), voxel_indices)):
            # print(f'f-{fold}({idx},{voxel})')
            if retinoR2[voxel] >= eqpoint:
                # load receptive field
                receptive_field = gaussian_2d((i, j), *guassparams[voxel])
                receptive_field = receptive_field / receptive_field.sum()
                # saptial summation
                X_voxel_train = np.sum(X_train * receptive_field, axis=(2,3))
                X_voxel_val = np.sum(X_val * receptive_field, axis=(2,3))
                lr = LinearRegression(n_jobs=8).fit(X_voxel_train, y_train[:, idx])
                y_pred = lr.predict(X_voxel_val)
                performance[fold, voxel] = np.corrcoef(y_val[:, idx], y_pred)[0,1]
                models[voxel] = lr
        
        # 保存每个 fold 的模型
        # joblib.dump(models, os.path.join(model_path, sub, f'{sub}_layer-{layername}_RFmodels_fold{fold}.pkl'))
    # 保存 performance
    performance[-1, :] = np.nanmean(performance[0:fold, :], axis=0)