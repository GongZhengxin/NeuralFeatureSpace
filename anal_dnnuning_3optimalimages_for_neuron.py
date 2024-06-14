import torch
from torch.optim import Adam
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join as pjoin
import random
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

netname = 'googlenet'
pretrained_weights = {'alexnet':  '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/alexnet-owt-4df8aa71.pth',
                      'googlenet': '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/googlenet-1378be20.pth'}
pretrained_models = {'alexnet': models.alexnet(),
                     'googlenet': models.googlenet()} 
inputlayername = 'conv2'

# pretrained net weights
pretrained_weights_path = pretrained_weights[netname]
# initialize structure
model = pretrained_models[netname]
# load parameters
model.load_state_dict(torch.load(pretrained_weights_path))
model.eval()
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 设定目标层
target_layer = model.conv2

# 定义一个钩子来捕获目标层的激活
activations = []
def hook_fn(module, input, output):
    activations.append(output)
target_layer.register_forward_hook(hook_fn)


import numpy as np
import random
import torchvision.transforms as transforms

# 定义预处理操作
preprocess = transforms.Compose([
    transforms.Resize(227),              # 调整图片大小
    transforms.ToTensor(),               # 转换为张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

tensor_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def total_variation(image):
  # 计算水平方向上的差异
  tv_h = torch.pow(image[:, :-1, :] - image[:, 1:, :], 2).sum()

  # 计算垂直方向上的差异
  tv_w = torch.pow(image[:, :, :-1] - image[:, :, 1:], 2).sum()

  # 计算总的TV并返回
  tv = torch.sqrt(tv_h + tv_w)

  return tv

# Define the 2D Gaussian function
def gaussian_2d(coords, A, x_0, y_0, sigma_x, sigma_y,C):
    i, j = coords
    return A * np.exp(-((i - x_0)**2 / (2 * sigma_x**2) + (j - y_0)**2 / (2 * sigma_y**2))) + C

def adjust_RF(receptive_field):
    cur_receptive_field = receptive_field.copy()
    cur_receptive_field = cur_receptive_field + np.abs(np.min(cur_receptive_field, None)) + 1
    thres = np.min(cur_receptive_field) + 0.5*(np.max(cur_receptive_field) - np.min(cur_receptive_field)) #

    cur_receptive_field[cur_receptive_field < thres] = 0
    cur_receptive_field = cur_receptive_field / (cur_receptive_field.sum() + 1e-20)

    return cur_receptive_field

def construct_RF(fieldsize, A=1, x=0, y=0, sigma_x=3.6, sigma_y=3.6, C=0):
    """
    # 创建假想体素感受野
    # default params
    # A, x, y, sigma_x, sigma_y, C = 1, 0, 0, 3.6, 3.6, 0
    """
    i = np.linspace(-8., 8., fieldsize)
    j = np.linspace(8., -8., fieldsize)
    i, j = np.meshgrid(i, j)
    return adjust_RF(gaussian_2d((i, j), A, x, y, sigma_x, sigma_y, C))


def set_random_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 内置的random模块
    random.seed(seed)
    # NumPy
    np.random.seed(seed)

def deprocess_image(img):
    return img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

def crop_center(img, crop_width, crop_height):
    img_width, img_height = img.size
    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2
    return img.crop((left, top, right, bottom))

def align_datatype(input_tensor, mean, std):
    """
    标准化输入的张量，确保 mean 和 std 与 input_tensor 在同一设备上，且数据类型一致。
    """
    # 确保 mean 和 std 是 torch.Tensor 类型
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    # 转换 mean 和 std 到 input_tensor 的数据类型和设备
    mean = mean.to(dtype=input_tensor.dtype, device=input_tensor.device)
    std = std.to(dtype=input_tensor.dtype, device=input_tensor.device)
    return mean, std

def normalize_data(input_tensor, mean, std):
    """
    参数:
        input_tensor (torch.Tensor): 待标准化的数据，形状为 (npic, 63)。
        mean (torch.Tensor or array-like): 数据的均值，形状为 (63,)。
        std (torch.Tensor or array-like): 数据的标准差，形状为 (63,)。
    
    返回:
        torch.Tensor: 标准化后的张量。
    """
    # 执行标准化
    normalized_tensor = (input_tensor - mean) / std
    return normalized_tensor

rfmask = torch.from_numpy(construct_RF(57)).to(device)
hugemask = torch.from_numpy(construct_RF(227)).to(device)
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
optimal_dir = 'maximum'
image_savepath = f'{work_dir}/vis/optimalimages'
mean_act_path = f'{work_dir}/prep/roi-concate'
target_brightness = 0.5

# channel activ in simple 
stimsets = ['30color', '180gabor', 'curv', 'natural']
activs_path = pjoin(work_dir, 'anal/unit-selectivity')
max_activs = []
for stimset in stimsets:
    activ = np.load(pjoin(activs_path, f'Googlenet_conv2_{stimset}-space.npy'), allow_pickle=True).item()
    max_activs.append(np.nanmax(activ['space'].reshape(-1, 64), axis=0))
channel_max_activs = np.nanmax(np.array(max_activs), axis=0)
retrain = True
for target_resp in [np.nan]:
    # 设置随机种子,1,2,45
    seed = 42

    for channel in range(64):#[ 28,29,60, 5, 33, 34,62]

        if retrain:
            target_flag = f'unit-{channel}'
            picname = [ _ for _ in os.listdir(f"{image_savepath}/{optimal_dir}/dnntuning/") if target_flag in _ ][0]
            # img = Image.open(f"{image_savepath}/{optimal_dir}/dnntuning/unit-39-T0.7_1.0.png")
            img = Image.open(f'{work_dir}/vis/optimalimages/showinfig/unit-61-T4.7_5.0.png')
            # 预处理图片
            img_t = preprocess(img)
            # # 添加batch维度
            img_t = img_t.unsqueeze(0)

        # 重置activations
        activations.clear()
        # 指定的像素值，分别对应RGB通道
        pixel_values = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1) #[0.5, 0.5, 0.5] #[1,1,1]#
        # 图像大小参数
        height, width = 227, 227
        # 生成图像
        n_pic = 1

        input_img = torch.zeros(n_pic, 3, height, width)  # 创建一个3x227x227的零张量
        if retrain:
            input_img[:, :, :, :] = img_t
        else: 
            input_img[:, 0, :, :] =  pixel_values[0, 0, 0]  # 设置R通道
            input_img[:, 1, :, :] = pixel_values[0, 1, 0]  # 设置G通道
            input_img[:, 2, :, :] = pixel_values[0, 2, 0]    # 设置B通道
            input_img[:, :, 14:214, 14:214] = torch.randn(n_pic, 3, 200, 200) #1 #
        input_img[:, :, hugemask == 0] = pixel_values
        input_img = input_img.to(device)
        # input_img = tensor_normalize(input_img)
        # input_img = (input_img * hugemask).float()
        input_img.requires_grad_(True) # = torch.tensor(input_img, requires_grad=True)

        #torch.randn(1, 3, 227, 227, requires_grad=True)
        optimizer = Adam([input_img], lr=0.01)
        # 每隔10个epoch，学习率乘以0.1
        scheduler_step = StepLR(optimizer, step_size=200, gamma=0.8)
        # # 当指标停止改进时降低学习率
        scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)
        accumulated_loss = 0.0
        update_frequency = 30 
        scheduler_losses = []
        progressing_images = []
        progressing_losses = []
        for iteration in range(20000):
            # 进行30次迭代
            print(iteration, end='\r')
            activations.clear()
            optimizer.zero_grad()
            model(input_img)
            # print(activations[0].shape)
            # 假设我们想最大化第一个神经元的激活
            target_activation = activations[0][:, channel]  # 调整索引以选择不同的神经元
            # rf_summation = torch.sum(rfmask * target_activation)
            mask = rfmask >= rfmask.max()
            rf_summation = torch.sum(mask * target_activation) / torch.sum(mask)
            # rf_summation = torch.max(mask * target_activation)
            if np.isnan(target_resp):
                # 尽可能大
                loss_act = - rf_summation
            else:
                # 特定激活值
                loss_act = torch.mean(target_resp - rf_summation)**2
            brightness_loss = 0.01 * (((hugemask *input_img).sum() - target_brightness) ** 2)
            loss_tv = 0.01 * total_variation(hugemask * input_img) #+ \
                #(torch.sum(hugemask * input_img) - (3 * hugemask.sum() * 0.3)) ** 2 / hugemask.sum()
            reg_loss = 0.01 * (((input_img - 1).clamp(min=0) ** 2).sum() + \
                ((input_img - 0).clamp(max=0) ** 2).sum())
            
            loss = loss_act + loss_tv #+ reg_loss + brightness_loss# 最大化激活，即最小化负激活
            # if iteration % 400 == 399:
            #     print(iteration, loss_act.item(), loss_tv.item(), 0.01 * reg_loss.item())
            if iteration % 1000 == 999:
                print(iteration, loss_act.item(), loss_tv.item(), 0.01 * reg_loss.item())
                progressing_images.append(input_img.detach().squeeze().cpu().clone())
                progressing_losses.append(-loss_act.item())
            # 如果设定了特定激活值
            if loss_act <1e-8 and (not np.isnan(target_resp)):
                progressing_images.append(input_img.detach().squeeze().cpu().clone())
                progressing_losses.append(-loss_act.item())
                break
            
            if  -loss_act.item() >= 5 * channel_max_activs[channel] and -loss_act.item() >= 10 :
                progressing_images.append(input_img.detach().squeeze().cpu().clone())
                progressing_losses.append(-loss_act.item())
                # break
            if iteration in [0,1,2]:
                print(iteration, loss_act.item(), loss_tv.item(), 0.01 * reg_loss.item())
            # print(loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            input_img.data = torch.clamp(input_img.data, 0, 1)
            
            # 每30次迭代更新一次
            accumulated_loss += loss.item()
            if (iteration + 1) % update_frequency == 0:
                average_loss = accumulated_loss / update_frequency
                scheduler_plateau.step(average_loss)
                scheduler_losses.append(average_loss)
                accumulated_loss = 0.0

        # 转换并保存图片
        cropped_images = []
        n_pic = len(progressing_images)
        activations.clear()
        voxel_response_predictions = []
        for i in range(n_pic):
            # optimized_img = deprocess_image(progressing_images[i])#input_img.detach().squeeze()#
            optimized_img = progressing_images[i]
            optimized_img = optimized_img.clamp(0, 1)
            optimized_img_np = optimized_img.permute(1, 2, 0).numpy()
            optimized_img_pil = Image.fromarray((optimized_img_np * 255).astype(np.uint8))
            # 使用函数进行裁剪
            cropped_image = crop_center(optimized_img_pil, 227, 227)
            cropped_images.append(cropped_image)
            act_value = np.round(progressing_losses[i], decimals=1)
            # cropped_image.save(f"/content/drive/MyDrive/ROI/{roi_name}_init{initial_pic}_pic{i}.jpg")
            
            # 读取和转换图片
            images = [[preprocess(img) for img in cropped_images][-1]]

            # 初始化数据和模型
            num_images = len(images)
            print(len(activations), num_images)
            x_data = torch.stack(images).to(device)
            model(x_data)
            actmap = np.squeeze(activations[0].detach().cpu().numpy())
            mask = rfmask.cpu().numpy() > 0
            normed_activ = (np.sum(actmap * mask, axis=(1,2))/mask.sum())[channel]
            # normed_activ = np.max(actmap * mask, axis=(1,2))[channel]
            activations.clear()
            print(f'unit-{channel}-T{act_value}', 'resp:', normed_activ, 'maxinset', channel_max_activs[channel])
            true_savedir = f'{image_savepath}/{optimal_dir}/dnntuning-neuron/'
            normed_activ = np.round(normed_activ)
            os.makedirs(true_savedir, exist_ok=True)
            if i == n_pic -1:
                cropped_image.save(f"{true_savedir}/unit-{channel}-T{act_value}_{normed_activ}.png", 'PNG', optimize=True)

# %%
# mean_act_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/roi-concate'
# layer = 'conv2'
# for roi in ['V1', 'V2', 'V3', 'V4']: 
#     for sub in subs:
#         submean = np.load(pjoin(f"{mean_act_path}/{sub}/{sub}_layer-googlenet-{layer}_{roi}-mean-train-feature.npy")).mean(axis=0)
#         substd = np.sqrt((np.load(pjoin(f"{mean_act_path}/{sub}/{sub}_layer-googlenet-{layer}_{roi}-std-train-feature.npy"))**2).mean(axis=0))
#         allsub_mean.append(submean)
#         allsub_std.append(substd)
#     sub = 'sub-mean'
#     np.save(pjoin(f"{mean_act_path}/{sub}/{sub}_layer-googlenet-{layer}_{roi}-mean-train-feature.npy"), np.array(allsub_mean).mean(axis=0))    
#     np.save(pjoin(f"{mean_act_path}/{sub}/{sub}_layer-googlenet-{layer}_{roi}-std-train-feature.npy"), np.sqrt((np.array(allsub_std)**2).mean(axis=0)))


