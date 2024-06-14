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
import numpy as np
import random
import torchvision.transforms as transforms


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

def construct_GRF(fieldsize, A=1, x=0, y=0, sigma_x=3.6, sigma_y=3.6, C=0):
    """
    # 创建假想体素感受野
    # default params
    # A, x, y, sigma_x, sigma_y, C = 1, 0, 0, 3.6, 3.6, 0
    """
    i = np.linspace(-8., 8., fieldsize)
    j = np.linspace(8., -8., fieldsize)
    i, j = np.meshgrid(i, j)
    return gaussian_2d((i, j), A, x, y, sigma_x, sigma_y, C)

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

def normalization(input_tensor, mean, std):
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

def demean(input_tensor, mean, std):
    """
    参数:
        input_tensor (torch.Tensor): 待标准化的数据，形状为 (npic, 63)。
        mean (torch.Tensor or array-like): 数据的均值，形状为 (63,)。
        std (torch.Tensor or array-like): 数据的标准差，形状为 (63,)。
    
    返回:
        torch.Tensor: 标准化后的张量。
    """
    # 执行标准化
    normalized_tensor = (input_tensor - mean)
    return normalized_tensor

netname = 'googlenet'
pretrained_weights = {'alexnet':  '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/alexnet-owt-4df8aa71.pth',
                      'googlenet': '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/googlenet-1378be20.pth'}
pretrained_models = {'alexnet': models.alexnet(),
                     'googlenet': models.googlenet()} 
inputlayername = 'conv2'
preprocess_mode = False
concernmeanfeature_mode = False
gaussianmask_mode = False
optimal_dir = 'targetvalue'

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

# 定义预处理操作
if preprocess_mode == True: 
    preprocess = transforms.Compose([
        transforms.Resize(227),              # 调整图片大小
        transforms.ToTensor(),               # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
else:
    preprocess = transforms.Compose([
        transforms.Resize(227),              # 调整图片大小
        transforms.ToTensor(),               # 转换为张量
    ])


# prep mask for activation & for image
if gaussianmask_mode:
    rfmask = torch.from_numpy(construct_GRF(57)).to(device)
    hugemask = torch.from_numpy(construct_GRF(227)).to(device)
else:
    rfmask = torch.from_numpy(construct_RF(57)).to(device)
    hugemask = torch.from_numpy(construct_RF(227)).to(device)

# define paths
work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline'
concateactiv_dir = pjoin(work_dir, 'prep/image_activations/concate-activs')


fit_type = 'hierarchicalnewtuning' #'subtuning'# 'axes'
axis_type = 'pca'#'raw'#

if fit_type == 'axes':
    # load axes
    if axis_type == 'ica':
        feature_path = pjoin(work_dir, 'anal/feature-ica')
        save_tuning_file = pjoin(feature_path, 'ica_tuning-dict.npy')
        if os.path.exists(save_tuning_file):
            tunings = np.load(save_tuning_file, allow_pickle=True).item()
        else:
            tunings = {}
            whiten_m = np.load(pjoin(feature_path, 'gzxica_whitening-matrix.npy'))
            unmix_m = np.load(pjoin(feature_path, 'ica-unmixing_kmeans-centers.npy'))
            tuning_m = whiten_m.dot(unmix_m.T)
            for icomp in range(tuning_m.shape[-1]):
                tunings[f'ic-{icomp}'] = tuning_m[:, icomp]
            np.save(save_tuning_file, tunings)
    if axis_type == 'pca':
        save_tuning_file = pjoin(feature_path, 'pca_tuning-dict.npy')
        if os.path.exists(save_tuning_file):
            tunings = np.load(save_tuning_file, allow_pickle=True).item()
        else:
            tunings = {}
            feature_path = pjoin(work_dir, 'prep/image_activations/pca/vectors')
            pcavectors = np.squeeze(np.load(pjoin(feature_path, 'nodstimeigenvectors-64.npy')))
            for icomp in range(pcavectors.shape[-1]):
                tunings[f'pc-{icomp}'] = pcavectors[:, icomp]
            np.save(save_tuning_file, tunings)

if fit_type != 'axes':
    save_tuning_path = pjoin(work_dir, f'build/roi-concatemodel_feature-{axis_type}/{fit_type}')
    roi = 'V1'
    if fit_type == 'subtuning':
        save_tuning_file = pjoin(save_tuning_path, f'allsub_{roi}-tuning.npy')
        subtunings = np.load(save_tuning_file, allow_pickle=True).item()
        tunings = subtunings
    if fit_type == 'hierarchicalnewtuning':
        save_tuning_file = pjoin(save_tuning_path, 'submean-hierarchy-others-46.npy')
        subtunings = np.load(save_tuning_file, allow_pickle=True).item()
    if axis_type == 'pca':
        feature_path = pjoin(work_dir, 'prep/image_activations/pca/vectors')
        pcavectors = np.squeeze(np.load(pjoin(feature_path, 'nodstimeigenvectors-64.npy')))
        tunings = {}
        for tuningname, subtuning in subtunings.items():
            if "newtuning" in fit_type:
                subtuning = subtuning / np.sqrt(np.sum(subtuning**2))
                featuretype = tuningname.split('-')[-2]
                featurenum = int(tuningname.split('-')[-1])
                if featuretype == 'all':
                    newpcavectors = pcavectors[:,0:featurenum]
                if featuretype == 'last':
                    newpcavectors = pcavectors[:,(63-featurenum)::]
                else:
                    featureidxs = np.load(pjoin(save_tuning_path, f'submean-hierarchy-{featuretype}-0.npy'))
                    newpcavectors = pcavectors[:,featureidxs]
                tunings[tuningname] = np.squeeze(np.dot(newpcavectors, subtuning[:,None]))
            else:
                tunings[tuningname] = np.squeeze(np.dot(pcavectors, subtuning[:,None]))
# 
image_savepath = f'{work_dir}/vis/optimalimages'
mean_act_path = f'{work_dir}/prep/roi-concate'
target_brightness = 0.8
for target_resp in [0,-2,2, -4, 4, -6, 6]: #2, -2, 6, -6,  0, -2, 
    # 设置随机种子,1,2,45[-3]: #0,
    seed = 42
    allsub_mean, allsub_std = [], []
    for tuningname, tuning in tunings.items():
        # 指定生成数量
        n_pic = 1
        # 指定的像素值，分别对应RGB通道
        pixel_values = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1) #[0.5, 0.5, 0.5] ##[0.485, 0.456, 0.406]
        # 图像大小参数
        height, width = 227, 227
        # 轴号
        if fit_type == 'axes':
            idx = int(tuningname.split('-')[-1])
            if idx >= 99:
                continue
        else:
            if 'sub-mean' not in tuningname:
                continue
            featuretype = tuningname.split('-')[-2]
            featurenum = int(tuningname.split('-')[-1])
        # 读取图片
        if np.abs(target_resp) > 0:
            target_flag = f'T{target_resp - 2 * np.sign(target_resp)}'
            picname = [ _ for _ in os.listdir(f"{image_savepath}/{optimal_dir}/{axis_type}_axes") if target_flag in _ and tuningname in _][0]
            if os.path.exists(f"{image_savepath}/{optimal_dir}/{picname}"):
                img = Image.open(f"{image_savepath}/{optimal_dir}/{picname}")
                # 预处理图片
                img_t = preprocess(img)
                # # 添加batch维度
                img_t = img_t.unsqueeze(0)
                # img_t = img_t * hugemask
                # img_t = img_t.double()
            else:
                img_t = torch.randn(n_pic, 3, height, width) 

        # 假设我们的目标层是inception的第一个卷积层的输出
        target_layer = model.conv2
        # 重置activations
        activations.clear()

        # 生成初始图像
        input_img = torch.zeros(n_pic, 3, height, width)  # 创建一个3x227x227的零张量
        if np.abs(target_resp) > 2 :
            input_img[:, :, :, :] = img_t
        else:
            input_img[:, :, 14:214, 14:214] = torch.randn(n_pic, 3, 200, 200) # 1 #
        if gaussianmask_mode:
            input_img[:, :, hugemask <= hugemask.mean()] = pixel_values
        else:
            input_img[:, :, hugemask == 0] = pixel_values
        input_img = input_img.to(device)
        input_img.requires_grad_(True)

        # load 被试的 conv2 平均特征
        global_mean = np.load(pjoin(concateactiv_dir, 'all-sub_googlenet-conv2_63-activation_mean.npy'))
        global_std = np.load(pjoin(concateactiv_dir, 'all-sub_googlenet-conv2_63-activation_std.npy'))
        submean_torch, substd_torch = align_datatype(input_img, global_mean, global_std)
        if axis_type == 'ica':
            normalize_data = demean
        if axis_type == 'pca':
            normalize_data = normalization
        optimizer = Adam([input_img], lr=3)

        # 每隔n个epoch，学习率乘以0.1
        scheduler_step = StepLR(optimizer, step_size=400, gamma=0.9)
        # # 当指标停止改进时降低学习率
        # scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)
        roi_name = 'V1'
        params = tuning
        params = torch.from_numpy(params)
        params = params.to(device)

        progressing_images = []
        progressing_losses = []
        for iteration in range(11000):
            # 进行迭代
            print(iteration, end='\r')
            activations.clear()
            optimizer.zero_grad()
            model(input_img)
            # print(activations[0].shape)
            # 假设我们想最大化第一个神经元的激活
            target_activation = activations[0][:, 0:63]  # 调整索引以选择不同的神经元
            # if 'newtuning' in fit_type:
            #     if featuretype == 'all':
            #         target_activation = activations[0][:, 0:featurenum]
            #         submean_torch, substd_torch = submean_torch[0:featurenum], substd_torch[0:featurenum]
            #     if featuretype == 'last':
            #         target_activation = activations[0][:, 0:63][:, int(63-featurenum)::]
            #         submean_torch, substd_torch = submean_torch[int(63-featurenum)::], substd_torch[int(63-featurenum)::]
            rf_summation = torch.sum(rfmask * target_activation, axis=(2,3))
            
            # # 对空间平均做标准化
            if concernmeanfeature_mode:
                normed_summation = normalize_data(rf_summation, submean_torch, substd_torch)
            else:
                normed_summation = rf_summation

            if optimal_dir == 'targetvalue':
                # 特定激活值
                loss_act = torch.mean(target_resp - torch.sum( normed_summation * params, axis=-1).mean())**2
            else:
                # 尽可能大
                loss_act = - torch.sum( normed_summation * params, axis=-1).mean()
            brightness_loss = 10 * (((hugemask *input_img).mean() - target_brightness) ** 2)
            loss_tv = 0.1 * total_variation(hugemask * input_img) #+ \
                #(torch.sum(hugemask * input_img) - (3 * hugemask.sum() * 0.3)) ** 2 / hugemask.sum()
            reg_loss = 0.1 * (((input_img - 1).clamp(min=0) ** 2).sum() + \
                ((input_img - 0).clamp(max=0) ** 2).sum())
            
            loss = loss_act + loss_tv + reg_loss + brightness_loss# 最大化激活，即最小化负激活
            if iteration % 5000 == 4999:
                print(iteration, loss_act.item(), loss_tv.item(), 0.01 * reg_loss.item())
                progressing_images.append(input_img.detach().squeeze().cpu().clone())
                progressing_losses.append(-loss_act.item())
            # 如果设定了特定激活值
            if loss_act <1e-8 and target_resp is not None:
                if preprocess_mode:
                    progressing_images.append(deprocess_image(input_img.detach().squeeze().cpu().clone()))
                else:
                    progressing_images.append(input_img.detach().squeeze().cpu().clone())
                progressing_losses.append(-loss_act.item())
                break
            
            # if  -loss_act.item() >= 7:
            #     progressing_images.append(input_img.detach().squeeze().cpu().clone())
            #     progressing_losses.append(-loss_act.item())
            #     break
            if iteration in [0,1,2]:
                print(iteration, loss_act.item(), loss_tv.item(), 0.01 * reg_loss.item())
            # print(loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            input_img.data = torch.clamp(input_img.data, 0, 1)
        # scheduler_step.step()

        # 转换并保存图片
        cropped_images = []
        n_pic = len(progressing_images)
        activations.clear()
        voxel_response_predictions = []
        for i in range(n_pic):
            cropped_images = []
            # optimized_img = deprocess_image(progressing_images[i])#input_img.detach().squeeze()#
            optimized_img = progressing_images[i]
            optimized_img = optimized_img.clamp(0, 1)
            optimized_img_np = optimized_img.permute(1, 2, 0).numpy()
            optimized_img_pil = Image.fromarray((optimized_img_np * 255).astype(np.uint8))
            # 使用函数进行裁剪
            cropped_image = crop_center(optimized_img_pil, 227, 227)
            cropped_images.append(cropped_image)
            # intial preprocess
            images = [preprocess(img) for img in cropped_images]
            act_value = np.round(progressing_losses[i], decimals=1)

            # 初始化数据和模型
            num_images = len(images)
            print(len(activations), num_images)
            x_data = torch.stack(images).to(device)
            model(x_data)
            actmap = np.squeeze(activations[0].detach().cpu().numpy())
            # if 'newtuning' in fit_type:
            #     if featuretype == 'all':
            #         target_activation = activations[0][:, 0:featurenum]
            #         global_mean, global_std = global_mean[0:featurenum], global_std[0:featurenum]
            #     if featuretype == 'last':
            #         target_activation = activations[0][:, 0:63][:, int(63-featurenum)::]
            #         global_mean, global_std = global_mean[int(63-featurenum)::], global_std[int(63-featurenum)::]
            
            if axis_type == 'pca':
                normed_activ = (np.sum(actmap * rfmask.cpu().numpy(), axis=(1,2))[0:63] - global_mean) / global_std
            elif axis_type == 'ica':
                normed_activ = (np.sum(actmap * rfmask.cpu().numpy(), axis=(1,2))[0:63] - global_mean)
            else:
                normed_activ = np.sum(actmap * rfmask.cpu().numpy(), axis=(1,2))[0:63]
            voxel_resp = np.dot(normed_activ, tuning) 
            voxel_response_predictions.append(voxel_resp)
            activations.clear()
            print(f'{tuningname}-T{target_resp}_{voxel_resp}', 'resp:', voxel_resp, 'loss:', act_value)
            # cropped_image.save(f"/content/drive/MyDrive/ROI/{roi_name}_init{initial_pic}_pic{i}.jpg")
            true_save_dir = pjoin(image_savepath, optimal_dir, f"{axis_type}_axes")
            os.makedirs(true_save_dir, exist_ok=True)
            cropped_image.save(f"{true_save_dir}/{tuningname}-T{target_resp}_{np.round(voxel_resp, decimals=1)}.png", 'PNG', optimize=True)

# # %%
# image_paths = [f"{image_savepath}/{optimal_dir}/{tuningname}-T{target_resp}.png"]

# # 读取和转换图片
# images = [Image.open(p).convert('RGB') for p in image_paths]
# images = [transform(img) for img in images]

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


