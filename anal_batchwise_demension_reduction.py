import numpy as np
import os
import glob
print('start')
directory_path = '/nfs/z1/userhome/zzl-liuyaoze/BrainImageNet/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/image_activations'
# pattern = f'{directory_path}/subj*_avg-corrmatrix.npy'
# file_paths = glob.glob(pattern)
# file_paths.sort()

# sum_matrix = None
# for file in file_paths:
#     mat = np.load(file)
#     if sum_matrix is None:
#         sum_matrix = mat
#     else:
#         sum_matrix += mat

# corr_matrix = sum_matrix / len(file_paths)
# print('loaded corr_matrix')
layername = 'googlenet_conv2'
corr_matrix = np.load(f"{directory_path}/subj-all_avg-corrmatrix_{layername}.npy",mmap_mode='r')
print('loaded corr_matrix', corr_matrix.shape )
# 使用numpy计算特征值
eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix[:63, :63])
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# np.save(f'{directory_path}/nodstimeigenvectors-256.npy', sorted_eigenvectors)
np.save(f'{directory_path}/nodstimeigenvalues-64.npy', sorted_eigenvalues)

print('sorted_eigenvectors saved')
# selected_eigenvectors = sorted_eigenvectors[:, :6]
# # 使用特征值进行降维
# print('reducing data')
# reduced_data = np.dot(dataconcat, selected_eigenvectors)
# print('done')
# print('saving')
# np.save('/nfs/z1/userhome/zzl-liuyaoze/BrainImageNet/Analysis_results/pcaalexnetsctivation/reduce_data.npy', reduced_data)
# print('data_saved')