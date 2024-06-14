import os
import numpy as np
from os.path import join as pjoin
import torch
import torch.optim as optim

def logcosh(x):
    return torch.log(torch.cosh(x))

def ica(X, iterations=1000, lr=0.01, device='cuda'):
    # 检查CUDA设备
    if not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA is not available. Using CPU instead.")
    else:
        print(f"Using {device}")

    # 将数据和模型参数移至设备（CPU/GPU）
    X = X.to(device)
    
    # 初始化权重，并移至相应设备
    W = torch.randn(X.shape[1], X.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    
    # 优化器
    optimizer = optim.SGD([W], lr=lr)
    print('优化循环')
    # 优化循环
    for _ in range(iterations):
        optimizer.zero_grad()
        
        # 使用权重计算ICA的投影
        Y = torch.mm(X, W)
        
        # 计算损失，这里使用-logcosh作为非高斯性的代理
        loss = -torch.mean(logcosh(Y))
        if _ % 2 == 1:
            print(f'iter{_+1}, loss:', loss.item())
        # 反向传播
        loss.backward()
        
        # 权重更新
        optimizer.step()
        
        # 可选：对权重进行对称正交化处理
        with torch.no_grad():
            W /= W.norm(dim=0, keepdim=True)

    # 非混合矩阵
    unmixing_matrix = W.detach()
    # ICA成分
    ica_components = torch.mm(X, unmixing_matrix)
    # 混合矩阵（计算非混合矩阵的逆）
    mixing_matrix = torch.linalg.pinv(unmixing_matrix)

    return ica_components.cpu(), mixing_matrix.cpu(), unmixing_matrix.cpu()

# path
data_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/image_activations/concate-activs'
saveout_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/anal/feature-ica'
# 数据
whiten_data_file = pjoin(saveout_path, 'all-sub_whitened-activ.npy')
if os.path.exists(pjoin(saveout_path, 'all-sub_whitened-activ.npy')):
    data_white = np.load(whiten_data_file) #, mmap_mode='r'
    train_data = torch.from_numpy(data_white[0:int(data_white.shape[0]/2)])
else:
    epsilon = 1e-5
    data = np.load(f"{data_path}/all-sub_googlenet-conv2_63-activation_dim-2.npy") #, mmap_mode='r'
    data = torch.from_numpy(data)
    # 数据中心化和白化
    data -= data.mean(dim=0)
    cov = data.T @ data / data.shape[0]
    U, S, V = torch.linalg.svd(cov)
    S_inv_root = torch.diag(1.0 / torch.sqrt(S + epsilon))
    whiten_matrix = U @ S_inv_root @ U.T
    # whiten_matrix = torch.linalg.inv(torch.linalg.cholesky(cov))
    data_white = data @ whiten_matrix
    np.save(pjoin(saveout_path, 'all-sub_whitened-activ.npy'), data_white.numpy())

    train_data = data_white[0:int(data_white.shape[0]/4)]
print('train data shap:', train_data.shape)
# 使用GPU进行ICA
ica_components, mixing_matrix, unmixing_matrix = ica(train_data, device='cpu')

np.save(pjoin(saveout_path, 'all-sub_half-1_ica-comp.npy'), ica_components.numpy())
np.save(pjoin(saveout_path, 'all-sub_half-1_mix-mat.npy'), mixing_matrix.numpy())
np.save(pjoin(saveout_path, 'all-sub_half-1_unmix-mat.npy'), unmixing_matrix.numpy())

# print("ICA成分：\n", ica_components)
# print("混合矩阵：\n", mixing_matrix)
# print("非混合矩阵：\n", unmixing_matrix)
