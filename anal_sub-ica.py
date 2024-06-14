import numpy as np
import pandas as pd
import glob
from os.path import join as pjoin
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# path setting
work_dir =  '/nfs/z1/userhome/zzl-liuyaoze/BrainImageNet/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
activ_dir = pjoin(work_dir, 'prep/image_activations')
input_path = pjoin(activ_dir, 'concate-activs')
save_path = pjoin(activ_dir, 'ica')
trans_path = pjoin(activ_dir, 'ica-transform')
icapath = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/anal/feature-ica'
# file
file = 'all-sub_googlenet-conv2_activation.npy'
activ_file = pjoin(input_path, file)
activations = np.load(activ_file, mmap_mode='r')

for sub in ['sub-01','sub-08']:
    subidx = int(sub.split('-')[-1])
    # for sess in range(2):
    subactiv = activations[4000*(subidx-1) : 4000*(subidx)]
    print(f" {file} 数据加载成功")
    if subactiv.ndim ==4:
        print('check find:', subactiv.shape)
        subactiv = subactiv.transpose((0,2,3,1))[:, :, :, :].reshape((-1, 64))
    if 'conv2' in file and subactiv.shape[-1] == 64:
        # subactiv = subactiv[:, ~np.all(conv2_activations == 0, axis=0)]
        subactiv = subactiv[:, 0:63]
    print(f"数据整理完成，数据形状为{subactiv.shape}")

    # Step1 中心化
    gm = np.load(f"{input_path}/all-sub_googlenet-conv2_63-activation_mean.npy")
    subactiv_c = subactiv - gm
    # Step2 白化
    wm = np.load(f"{icapath}/gzxica_whitening-matrix.npy")
    subactiv_w = subactiv_c.dot(wm)
    for ses in range(4):
        # Step3 独立成分
        sesactive_w = subactiv_w[ int(ses*1000*57*57) : int((ses+1)*1000*57*57) ]
        try:
            ica = FastICA(n_components=2, algorithm='parallel', fun='logcosh', max_iter=200, tol=1e-4, whiten=False)
            sesactiv_icacomp = ica.fit_transform(sesactive_w)
            np.save(f'{icapath}/{sub}_ses-{ses+1}_ica-comp.npy', sesactiv_icacomp)
            joblib.dump(ica, f"{icapath}/{sub}_ses-{ses+1}_ica_model.pkl")
        except Exception as e:
            print(f"ICA拟合过程中的出现错误“{e}")

# ###################################
# train whitening & ica
# ###################################
# # 对数据进行预处理，也即标准化
# scaler = StandardScaler()
# scaler.fit(subactiv)

# # Step 1: 数据中心化
# subactiv_c = subactiv -  scaler.mean_

# # Step 2: 计算协方差矩阵
# cov = np.cov(subactiv_c, rowvar=False)

# # Step 3: 计算协方差矩阵的特征值和特征向量
# eigen_values, eigen_vectors = np.linalg.eigh(cov)


# # Step 4: 对特征值进行排序（降序）
# sort_idx = np.argsort(eigen_values)[::-1]
# eigen_values_sorted = eigen_values[sort_idx]
# eigen_vectors_sorted = eigen_vectors[:, sort_idx]
# np.save(f'{icapath}/sub-2_eigen_values.npy', eigen_values_sorted)
# np.save(f'{icapath}/sub-2_eigen_vectors.npy', eigen_vectors_sorted)

# # Step 5: 形成根据特征值缩放的特征向量矩阵
# sqrt_eigen_values = np.sqrt(eigen_values_sorted)
# scaling_matrix = np.diag(1.0 / sqrt_eigen_values)

# # Step 6: 计算白化矩阵
# whitening_matrix = eigen_vectors_sorted.dot(scaling_matrix).dot(eigen_vectors_sorted.T)
# np.save(f'{icapath}/sub-2_whiten_matrix.npy', whitening_matrix)
# # Step 7: 白化数据
# subactiv_w = subactiv_c.dot(whitening_matrix)

# try:
#     ica = FastICA(n_components=2, algorithm='parallel', fun='logcosh', max_iter=300, tol=1e-4, whiten=False)
#     subactiv_icacomp = ica.fit_transform(subactiv_w)
#     np.save(f'{icapath}/sub-2_ica-comp.npy', subactiv_icacomp)
#     joblib.dump(ica, f"{icapath}/sub-2_ica_model.pkl")
# except Exception as e:
#     print(f"ICA拟合过程中的出现错误“{e}")
# print("ICA成分 降维后的数据 存储完成")



# wm = np.load(f'{icapath}/sub-1_whiten_matrix.npy')
# ica = joblib.load(f'{icapath}/sub-1_ica_model.pkl')
# gm = np.load(f"{input_path}/all-sub_googlenet-conv2_63-activation_mean.npy")
# subactiv_c = subactiv - gm
# subactiv_w = subactiv_c.dot(wm)
# subactiv_icacomp = ica.transform(subactiv_w)

# wm2 = np.load(f'{icapath}/sub-2_whiten_matrix.npy')
# ica2 = joblib.load(f'{icapath}/sub-2_ica_model.pkl')
# subactiv_w2 = subactiv_c.dot(wm2)
# subactiv_icacomp2 = ica2.transform(subactiv_w2)
