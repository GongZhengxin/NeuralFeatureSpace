import numpy as np
import pandas as pd
import glob
from os.path import join as pjoin
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
layername = 'googlenet-conv2'
for sub in ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']: #'sub-01',
    subidx = int(sub.split('-')[-1])
    # for sess in range(2):
    subactiv = activations[4000*(subidx-1) : 4000*(subidx)]
    print(f" {file} 数据加载成功")
    if subactiv.ndim ==4:
        print('check find:', subactiv.shape)
        subactiv = subactiv.transpose((0,2,3,1))[:, :, :, :].reshape((-1, 64))
    if 'conv2' in file and subactiv.shape[-1] == 64:
        subactiv = subactiv[:, 0:63]
    print(f"数据整理完成，数据形状为{subactiv.shape}")

    # Step1 中心化
    gm = np.load(f"{input_path}/all-sub_googlenet-conv2_63-activation_mean.npy")
    subactiv_c = subactiv - gm
    # Step2 白化
    wm = np.load(f"{icapath}/gzxica_whitening-matrix.npy")
    subactiv_w = subactiv_c.dot(wm)
    # Step3 独立成分
    comp = np.load(f"{icapath}/ica-unmixing_kmeans-centers.npy")
    subactiv_icacomp = subactiv_w.dot(comp.T)
    np.save(f'{save_path}/{sub}_{layername}_ica-comps.npy', subactiv_icacomp.reshape((4000, 57, 57, 63)).transpose((0, 3, 1, 2)))

# file = 'coco_googlenet-conv2.npy'
# subactiv = np.load(f'{activ_dir}/{file}')
# if subactiv.ndim ==4:
#     print('check find:', subactiv.shape)
#     subactiv = subactiv.transpose((0,2,3,1))[:, :, :, :].reshape((-1, 64))
# if 'conv2' in file and subactiv.shape[-1] == 64:
#     subactiv = subactiv[:, 0:63]
# print(f"数据整理完成，数据形状为{subactiv.shape}")
# # Step1 中心化
# gm = np.load(f"{input_path}/all-sub_googlenet-conv2_63-activation_mean.npy")
# subactiv_c = subactiv - gm
# # Step2 白化
# wm = np.load(f"{icapath}/gzxica_whitening-matrix.npy")
# subactiv_w = subactiv_c.dot(wm)
# # Step3 独立成分
# comp = np.load(f"{icapath}/ica-unmixing_kmeans-centers.npy")
# subactiv_icacomp = subactiv_w.dot(comp.T)
# np.save(f'{save_path}/coco_ica-comps.npy', subactiv_icacomp.reshape((120, 57, 57, 63)).transpose((0, 3, 1, 2)))
