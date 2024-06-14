import glob
import os
import time
from PIL import Image
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from scipy.stats import zscore
from scipy.ndimage import zoom
from utils import net_size_info

t_begin = time.time()
# ===============================================================
# settings
netname = 'Googlenet'#'alexnet'
pretrained_weights = {'alexnet':  '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/alexnet-owt-4df8aa71.pth',
                      'Googlenet': '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/googlenet-1378be20.pth'}
pretrained_models = {'alexnet': models.alexnet(),
                     'Googlenet': models.googlenet()} 
inputlayername = 'maxpool1' #'conv2' # 'conv3' #'inception3a'#'features.3' input('enter layer name:')

# imagenet image dir
im_image_dir = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/stimuli/imagenet'
cifti_dir = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
# experiment info
stims =  ['180_gabor']#  'shape', 'pinknoise'## , , , '30_color', "raw_shape"
print(stims, netname)

for stim in stims:
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained net weights
    pretrained_weights_path = pretrained_weights[netname]

    # training params
    layer = {'name': inputlayername}#, 'size': alexnet_info[inputlayername]

    # output paths
    work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
    image_name_path = os.path.join(work_dir, 'prep/image_names')
    image_activations_path = os.path.join(work_dir, 'prep/simplified_stimuli/activation')

    # ===============================================================
    # prep images tensor

    if os.path.exists(os.path.join(image_name_path, f'{stim}.stim.csv')):
        with open(os.path.join(image_name_path, f'{stim}.stim.csv'), 'r') as f:
            image_paths = [_.replace('\n', '').split(',')[0] for _ in f.readlines()] 
            image_paths = [pjoin(image_paths[0], image_paths[_+1]) for _ in range(len(image_paths)-1) ]

    # intial preprocess
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])

    # 读取和转换图片
    images = [Image.open(p).convert('RGB') for p in image_paths]
    images = [transform(img) for img in images]

    # 初始化数据和模型
    num_images = len(images)
    x_data = torch.stack(images).to(device)

    # ===============================================================
    # model loading and initialization

    # initialize structure
    net = pretrained_models[netname]
    # load parameters
    net.load_state_dict(torch.load(pretrained_weights_path))
    net.to(device)
    net.eval()

    # 用于存储激活值的列表
    activations = []

    # 钩子函数
    def hook(module, input, output):
        activations.append(output.cpu().detach().numpy())
        print('.',end=' ')
    # 用 named_modules 获取所有层的名称和模块，并在特定层上注册钩子
    for name, module in net.named_modules():
        if name == layer['name']:  # 这是第二个卷积层（Conv2）的名称
            handle = module.register_forward_hook(hook)

    # make dataset
    dataset = TensorDataset(x_data)
    loader = DataLoader(dataset, batch_size=40, shuffle=False)

    for batch_idx, (inputs,) in enumerate(loader):
        if batch_idx % 25 == 0:
            print(f'extract activation batch {batch_idx}')
        inputs = inputs.to(device)

        with torch.no_grad():
            net(inputs)

    activations_arr = np.concatenate(activations)
    print(f'save {stim} image activation data shape of : {activations_arr.shape}')
    layername = layer['name']
    layername = layername.replace('.','')
    np.save(os.path.join(image_activations_path, f'{stim}_{netname}_{layername}.npy'), activations_arr)

    activations.clear()
    print(f'at prep image tensor : {time.time() - t_begin} s')