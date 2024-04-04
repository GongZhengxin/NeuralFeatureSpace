import glob
import os
import time
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from scipy.stats import zscore
from scipy.ndimage import zoom
from utils import net_size_info

def downsample_4d_efficient(data, new_size):
    """
    对给定的四维数据的最后两维进行降采样，使用 NumPy 高效操作。
    :param data: 原始数据，四维 numpy 数组。
    :param new_size: 新的尺寸，应该是一个二元组。
    :return: 降采样后的数据。
    """
    # 使用 np.newaxis 来增加维度，然后使用 NumPy 的广播机制
    zoom_factors = [1, 1, new_size[0]/data.shape[2], new_size[1]/data.shape[3]]
    return zoom(data, zoom_factors, order=3)

t_begin = time.time()
# ===============================================================
# settings
netname = 'googlenet'#'alexnet'
pretrained_weights = {'alexnet':  '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/alexnet-owt-4df8aa71.pth',
                      'googlenet': '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/googlenet-1378be20.pth'}
pretrained_models = {'alexnet': models.alexnet(),
                     'googlenet': models.googlenet()} 
inputlayername = 'maxpool2' #'conv3' #'conv2' # 'inception3a'#'features.3' input('enter layer name:')

# imagenet image dir
im_image_dir = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/stimuli/imagenet'
cifti_dir = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'
# experiment info
subs = ['coco'] #'sub-01', 'sub-09''sub-09''sub-07', 'sub-01', 'sub-02', 'sub-03', 'sub-04''sub-05', 'sub-06', 'sub-07', 'sub-09'
print(subs, netname)
# time.sleep(480*2)
downsample = True
for sub in subs:
    sessions = ['imagenet01']#,'imagenet02','imagenet03','imagenet04'
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained net weights
    pretrained_weights_path = pretrained_weights[netname]

    # training params
    layer = {'name': inputlayername}#, 'size': alexnet_info[inputlayername]

    # output paths
    work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
    image_name_path = os.path.join(work_dir, 'prep/image_names')
    image_activations_path = os.path.join(work_dir, 'prep/image_activations')

    # ===============================================================
    # prep images tensor

    if os.path.exists(os.path.join(image_name_path, f'{sub}_imagenet.csv')):
        with open(os.path.join(image_name_path, f'{sub}_imagenet.csv'), 'r') as f:
            image_paths = [_.replace('\n', '') for _ in f.readlines()] 
    else:
        # collect run folder
        folders = []
        for session in sessions:
            for run_folder in glob.glob(os.path.join(cifti_dir, sub, "results/", f"*{session}*")):
                folders.append(run_folder)
        folders = sorted(folders)

        # collect image names
        image_names_files = []
        for folder in folders:
            for img_label_txt in glob.glob(os.path.join(folder, "*_label.txt")): 
                image_names_files.append(img_label_txt) # 顺序为 1 10 2 3 ...

        # compose image paths
        image_paths = []
        for image_name_file in image_names_files:
            with open(image_name_file, 'r') as f:
                image_names = [ _.split('/')[-1].replace('\n', '') for _ in f.readlines() ]
                image_path = [ os.path.join(im_image_dir, _.split('_')[0], _) for _ in image_names]
            image_paths.extend(image_path)

        with open(os.path.join(image_name_path, f'{sub}_imagenet.csv'), 'w') as f:
            f.writelines('\n'.join(image_paths))


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
    print(f'save {sub} image activation data shape of : {activations_arr.shape}')
    layername = layer['name']
    layername = layername.replace('.','')
    np.save(os.path.join(image_activations_path, f'{sub}_{netname}-{layername}.npy'), activations_arr)
    # if activations_arr.shape[-1] > 28 and downsample:
    #     activations_arr = downsample_4d_efficient(activations_arr, (28,28))
    #     np.save(os.path.join(image_activations_path, f'{sub}_{netname}-{layername}_ds.npy'), activations_arr)
    activations.clear()
    print(f'at prep image tensor : {time.time() - t_begin} s')