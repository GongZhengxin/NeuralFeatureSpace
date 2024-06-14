import glob
import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# 读取NSD —— googlenet-conv2 激活
workdir =  '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
directory_path = pjoin(workdir, 'prep/image_activations')
pattern = f'{directory_path}/sub-0[1-9]_googlenet-conv2.npy'
file_paths = glob.glob(pattern)

file_paths.sort()

def compute_correlation_matrix(idx):
    global batch_search_data, step
    print(idx)
    image_feature = batch_search_data[idx : idx + step, :, :, :].copy()
    image_feature = image_feature.transpose(0,2,3,1).reshape((-1, 64))
    # 检查是否有全为常数的特征列
    constant_cols = np.where(np.all(image_feature == image_feature[0, :], axis=0)==True)[0]
    return {'corrlations':np.nan_to_num(np.corrcoef(image_feature.T)), 'constant':constant_cols, 'idx':idx}

results = []
for subj, fpath in enumerate(file_paths):
    print(f'读取第{subj+1}号被试！')
    batch_search_data = np.load(fpath, mmap_mode='r')
    n_pics = batch_search_data.shape[0]
    print(f"{subj+1}号被试数据加载成功！有{n_pics}张图片")

    # 【选取不同量的图片进行测试】选取不同图片数量进行处理（batch无关）
    step =  4000
    print('开始计算相关矩阵')
    sum_of_matrices = np.zeros((64, 64))

    result = Parallel(n_jobs=1)(delayed(compute_correlation_matrix)(idx) for idx in range(0, n_pics, step))
    results.append(result)

all_sub_mean = []
for subj, result in enumerate(results):
    sub_mean_corr = []
    for datadict in result:
        sub_mean_corr.append(datadict['corrlations'])
    
    all_sub_mean.extend(sub_mean_corr)
    print(f"开始保存{subj+1}号被试的平均相关矩阵")
    sub_mean_corr = np.array(sub_mean_corr).mean(axis=0)
    
    np.save(pjoin(workdir, f'prep/image_activations/pca/subj-{subj+1}_conv2_avg-corrmatrix_step-4000.npy'), sub_mean_corr)
    
    print(f"{subj+1}号被试的平均相关矩阵保存成功！{sub_mean_corr.shape}")

all_sub_mean = np.array(all_sub_mean).mean(axis=0)

np.save(pjoin(workdir, f'prep/image_activations/pca/subj-all_avg-corrmatrix_googlenet_conv2_step-4000.npy'), all_sub_mean)
print(f"全体被试的平均相关矩阵保存成功！{all_sub_mean.shape}")

# import pickle
# with open(pjoin(workdir, 'prep/image_activations/pca/conv2_results.pkl'), 'wb') as f:
#     pickle.dump(results, f)

# # 读取—— googlenet-conv2 maxpool2 inception3a 激活
# workdir =  '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
# directory_path = pjoin(workdir, 'prep/image_activations')
# pattern = f'{directory_path}/step_0[1-3]_2d.npy'
# file_paths = sorted(glob.glob(pattern))

# step1, step2, step3 = list(map(lambda x: np.load(x, mmap_mode='r'), file_paths))

# def compute_concate_cormat(idx):
#     global step1, step2, step3
#     print(idx)
#     image_feature = np.c_[step1[idx : idx + step, :].copy(), step2[idx : idx + step, :].copy(), step3[idx : idx + step, :].copy()] 
#     # 检查是否有全为常数的特征列
#     constant_cols = np.where(np.all(image_feature == image_feature[0, :], axis=0)==True)[0]
#     return {'corrlations':np.nan_to_num(np.corrcoef(image_feature.T)), 'constant':constant_cols, 'idx':idx}

# n_pics = step1.shape[0]
# print(f"加载成功！有{n_pics} patch")

# # 【选取不同量的图片进行测试】选取不同图片数量进行处理（batch无关）
# step =  28 * 28
# print('开始计算相关矩阵')

# result = Parallel(n_jobs=20)(delayed(compute_concate_cormat)(idx) for idx in range(0, n_pics, step))

# all_mean_corr = []
# for datadict in result:
#     all_mean_corr.append(datadict['corrlations'])

# all_mean_corr = np.array(all_mean_corr).mean(axis=0)

# np.save(pjoin(workdir, f'prep/image_activations/all-images_concate512features-SS_avg-corrmatrix.npy'), all_mean_corr)

# print(f"平均相关矩阵保存成功！{all_mean_corr.shape}")

# import pickle
# with open(pjoin(workdir, 'prep/image_activations/concate512-SS-results.pkl'), 'wb') as f:
#     pickle.dump(results, f)