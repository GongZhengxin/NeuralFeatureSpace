import numpy as np
import pandas as pd
from scipy.stats import vonmises
import scipy.special as sps
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os.path import join as pjoin
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from PIL import Image
import os
import math


def crop_center(img, crop_width, crop_height):
    img_width, img_height = img.size
    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2
    return img.crop((left, top, right, bottom))

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

tuningparameters = {'voxel' : [],
    'freq_fit': [], 'light_fit': [], 'chroma_fit': [], 'hue_fit': [], 'ori_fit': [],
    'freq_gain': [], 'light_gain': [], 'chroma_gain': [], 'hue_gain': [], 'ori_gain': [],
    'freq_center': [], 'light_center': [], 'chroma_center': [], 'hue_mu': [], 'ori_mu': [],
    'freq_sigma': [], 'light_sigma': [], 'chroma_sigma': [], 'hue_kappa': [], 'ori_kappa': [],
    'curv_sort': [],
    'curv_max': [], 'gabor_max': [], 'color_max': [], 'shape_max': [], 'natural_max':[],
    'curv_min': [], 'gabor_min': [], 'color_min': [], 'shape_min': [], 'natural_min':[],
    'color_median' : [], 'gabor_median' : [], 'curv_median' : [], 'natural_median' : [],
    'color_mean' : [], 'gabor_mean' : [], 'curv_mean' : [], 'natural_mean' : [],
}

work_dir = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/'
# save out path
neuralsels_path =  pjoin(work_dir, 'anal/neural-selectivity')
paramspace_dir = pjoin(neuralsels_path, 'parameterspace')
output_path = pjoin(neuralsels_path, 'tuningsummary')
vis_path =  pjoin(work_dir, f'vis/neural-selectivity-plots')
os.makedirs(output_path, exist_ok=True)
os.makedirs(vis_path, exist_ok=True)

# subs = [f'sub-0{_+1}' for _ in range(9)]# ['sub-01']
# layers = ['conv2']
# rois = ['V1']
# voxs = ['roi']
# modeltuning_names = []
# for sub in subs:
#     for layer in layers:
#         for roi in rois:
#             for vox in voxs:
#                 modeltuning_names.append(f'{sub}-{layer}-{roi}-{vox}')

# subtuning
net_roi = 'conv2-V1' 
modeltuning_names = list(np.load(pjoin(neuralsels_path, f'all-sub_{net_roi}-tunings.npy'), allow_pickle=True).item().keys())

# # axis tuning
# net_roi = 'conv2'
# axis_type = 'pca'
# modeltuning_names = list(np.load(pjoin(neuralsels_path, f'{axis_type}_tuning-dict.npy'), allow_pickle=True).item().keys())
# paramspace_dir = pjoin(paramspace_dir, axis_type)
# vis_path = pjoin(vis_path, axis_type)
# os.makedirs(vis_path, exist_ok=True)

norm_f = lambda x: (x - np.min(x) )/ np.std(x) #(np.max(x) - np.min(x)) # x #

for modeltuning_name in modeltuning_names:
    
    tuningparameters['voxel'].append(modeltuning_name)
    figsave_dir = pjoin(vis_path, f'{modeltuning_name}-vertical')
    os.makedirs(figsave_dir, exist_ok=True)
    # #####################
    # gabor preference
    # #####################
    gaborspace = np.load(pjoin(paramspace_dir, f'{modeltuning_name}/{modeltuning_name}_gabor-space.npy'), allow_pickle=True).item()
    freq = gaborspace['freq']  * 227 / 16 # 转换成 cycle/degree
    ori = gaborspace['ori']
    freq_tuning = gaborspace['space'].mean(axis=0)
    ori_tuning = gaborspace['space'].mean(axis=1)
    tuningparameters['gabor_max'].append(np.nanmax(gaborspace['space']))
    tuningparameters['gabor_mean'].append(np.nanmean(gaborspace['space']))
    tuningparameters['gabor_median'].append(np.nanmedian(gaborspace['space']))
    tuningparameters['gabor_min'].append(np.nanmin(gaborspace['space']))
    # #####################
    # color preference
    # #####################
    colorspace = np.load(pjoin(paramspace_dir, f'{modeltuning_name}/{modeltuning_name}_color-space.npy'), allow_pickle=True).item()
    hue = colorspace['hue']
    hue_tuning = colorspace['space'].mean(axis=(1,2))
    light = colorspace['light']
    light_tuning = colorspace['space'].mean(axis=(0,2))
    chroma = colorspace['chroma']
    chroma_tuning = colorspace['space'].mean(axis=(0,1))
    tuningparameters['color_max'].append(np.nanmax(colorspace['space']))
    tuningparameters['color_mean'].append(np.nanmean(colorspace['space']))
    tuningparameters['color_median'].append(np.nanmedian(colorspace['space']))
    tuningparameters['color_min'].append(np.nanmin(colorspace['space']))
    # #####################
    # curv preference
    # #####################
    curvspace = np.load(pjoin(paramspace_dir, f'{modeltuning_name}/{modeltuning_name}_curv-space.npy'), allow_pickle=True).item()
    curv = curvspace['shape']
    tuningparameters['curv_max'] = np.nanmax(curvspace['space'])
    tuningparameters['curv_mean'] = np.nanmean(curvspace['space'])
    tuningparameters['curv_median'] = np.nanmedian(curvspace['space'])
    tuningparameters['curv_min'] = np.nanmin(curvspace['space'])

    curv_tuning = np.nanmean(curvspace['space'], axis=1)
    tuningparameters['shape_max'].append(np.max(curv_tuning))
    tuningparameters['shape_min'].append(np.min(curv_tuning))
    # #####################
    # natural texture preference
    # #####################
    naturalspace = np.load(pjoin(paramspace_dir, f'{modeltuning_name}/{modeltuning_name}_natural-space.npy'), allow_pickle=True).item()
    pacthes = np.arange(len(naturalspace['patches']))
    tuningparameters['natural_max'].append(np.nanmax(naturalspace['space']))
    tuningparameters['natural_mean'].append(np.nanmean(naturalspace['space']))
    tuningparameters['natural_median'].append(np.nanmedian(naturalspace['space']))
    tuningparameters['natural_min'].append(np.nanmin(naturalspace['space']))
    

    # #####################
    # 提取参数化信息
    # #####################

    # 定义高斯函数
    def gaussian(x, gain, center, sigma):
        return gain * np.exp(-(x - center)**2 / (2 * sigma**2 + 1e-8))

    # freq & light & chroma 都是线性的
    linear_tunings = [freq_tuning, light_tuning, chroma_tuning]
    linear_axes = [freq, light, chroma]
    axis_names = ['freq', 'light', 'chroma']
    param_names = ['gain', 'center', 'sigma']
    for tuning, axis, axisname in zip(linear_tunings, linear_axes, axis_names):
        
        x_data = axis
        y_data = norm_f(tuning)
        try:
            p0 = [np.max(y_data), x_data[np.argmax(y_data)], 1]
            params, covariance = curve_fit(gaussian, x_data, y_data, p0)

            for param, param_name in zip(params, param_names):
                tuningparameters[f'{axisname}_{param_name}'].append(param)
            tuningparameters[f'{axisname}_fit'].append(1)
            interplot_f = ius(x_data, y_data, k=5)
        except RuntimeError:
            tuningparameters[f'{axisname}_fit'].append(0)
            # 得到一款插值函数
            interplot_f = ius(x_data, y_data, k=5)
            # 强行按高斯估计
            max_tuning = np.max(y_data)
            left_part = np.where(x_data<x_data[np.argmax(y_data)])[0]
            right_part = np.where(x_data>x_data[np.argmax(y_data)])[0]
            if len(left_part) > 0 and len(right_part) >0:
                left_HM_point = left_part[np.argmin((y_data[left_part] - 0.5 * max_tuning)**2)]
                right_HM_point = right_part[np.argmin((y_data[right_part] - 0.5 * max_tuning)**2)]
                hist_FW = x_data[right_HM_point] - x_data[left_HM_point]
            elif len(left_part) > 0:
                left_HM_point = left_part[np.argmin((y_data[left_part] - 0.5 * max_tuning)**2)]
                hist_FW = 2 * (x_data[np.argmax(y_data)] - x_data[left_HM_point])
            elif len(right_part) > 0:
                right_HM_point = right_part[np.argmin((y_data[right_part] - 0.5 * max_tuning)**2)]
                hist_FW = 2 * (x_data[right_HM_point] - x_data[np.argmax(y_data)])
            sigma = hist_FW / 2.35
            params = [np.max(y_data), x_data[np.argmax(y_data)], sigma]

            for param, param_name in zip(params, param_names):
                tuningparameters[f'{axisname}_{param_name}'].append(param)
        # 绘图
        plt.figure(figsize=(2, 2))
        plt.scatter(x_data, y_data, label='Data', marker='+', color='black', zorder=5, s=30)
        x = np.linspace(x_data.min(), x_data.max(), 100)
        if tuningparameters[f'{axisname}_fit'][-1] >= 0:
            # plt.plot(x, gaussian(x, *params), label='Fitted tuning', color='red', alpha=0.6, lw=3)
            plt.plot(x, interplot_f(x), label='interplot', color='blue', alpha=0.6, lw=3)
        else:
            plt.plot(x, interplot_f(x), label='interplot', color='blue', alpha=0.6, lw=3)
        # plt.legend(fontsize=9)
        # plt.xlabel(axisname,fontsize=10)
        # plt.ylabel(f'unit-{unit} Resp.',fontsize=10)
        plt.xticks(fontsize=16)  # 调整x轴标签大小
        plt.yticks(fontsize=16)  # 调整y轴标签大小
        # plt.title(f'{axisname}', fontsize=16)
        ax = plt.gca()
        ax_right = ax.secondary_yaxis('right')
        ax_right.set_yticks([])
        ax_right.set_ylabel(f'Resp. to {axisname}', fontsize=16)
        plt.savefig(pjoin(figsave_dir, f'{axisname}tuning.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()

    # hue & ori 是周期的

    # 将角度转换为弧度
    def degrees_to_radians(degrees):
        return degrees * np.pi / 180

    # 将角度翻倍来适应180度周期
    def double_angles(degrees):
        return (degrees * 2) % 360

    # 定义von Mises分布函数来进行拟合
    def vonmises_pdf(x, gain, kappa, mu):
        return gain*vonmises.pdf(x, kappa, loc=mu)

    param_names = ['gain', 'kappa', 'mu']
    # 拟合 ori
    axisname = 'ori'
    angle_data = degrees_to_radians(double_angles(ori))  # 翻倍并转换为弧度
    initial_ori = ori[np.argmax(ori_tuning)]
    mu = degrees_to_radians(double_angles(initial_ori))  # 设置两倍，适应周期
    kappa = 2  # 设置浓度参数
    y_data = norm_f(ori_tuning)
    # 拟合von Mises分布
    try:
        params, covariance = curve_fit(vonmises_pdf, angle_data, y_data, p0=[1, 2, mu])
        for param, param_name in zip(params, param_names):
            tuningparameters[f'{axisname}_{param_name}'].append(param)
        tuningparameters[f'{axisname}_fit'].append(1)
        interplot_f = ius(angle_data, y_data, k=5)
    except RuntimeError:
        tuningparameters[f'{axisname}_fit'].append(0)
        # 得到一款插值函数
        interplot_f = ius(angle_data, y_data, k=5)
        # interplot_f = interp1d(angle_data, y_data, kind='cubic')
        # 强行按高斯估计
        cos_x, sin_x = np.cos(angle_data), np.sin(angle_data)
        mean_cos_x = np.sum(cos_x * y_data) / y_data.sum()
        mean_sin_x = np.sum(sin_x * y_data) / y_data.sum()
        r = np.sqrt(mean_cos_x ** 2 + mean_sin_x ** 2)
        kappa = r / (1 - r**2)
        i0_value = sps.i0(kappa)
        gain = 2* np.pi * i0_value * np.max(y_data) / np.exp(kappa) 
        params = [gain, axis[np.argmax(y_data)], kappa]
        for param, param_name in zip(params, param_names):
            tuningparameters[f'{axisname}_{param_name}'].append(param)

    # ori 绘图
    plt.figure(figsize=(2, 2))
    plt.scatter(ori, y_data, label='Data', marker='+', color='black', zorder=5, s=30)
    x_ori = np.linspace(ori.min(), ori.max(), 100)
    x_plot = degrees_to_radians(double_angles(x_ori))
    if tuningparameters[f'{axisname}_fit'][-1] >= 0:
        # plt.plot(x_ori, vonmises_pdf(x_plot, *params), label='Fitted tuning', color='red', alpha=0.6, lw=3)
        plt.plot(x_ori, interplot_f(x_plot), label='interplot', color='blue', alpha=0.6, lw=3)
    else:
        plt.plot(x_ori, interplot_f(x_plot), label='interplot', color='blue', alpha=0.6, lw=3)
    # plt.legend(fontsize=9)
    # plt.xlabel('ori.',fontsize=10)
    # plt.ylabel(f'Resp.',fontsize=16)
    plt.xticks(fontsize=16)  # 调整x轴标签大小
    plt.yticks(fontsize=16)  # 调整y轴标签大小
    # plt.title('ori.', fontsize=16)
    ax = plt.gca()
    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks([])
    ax_right.set_ylabel(f'Resp. to ori.', fontsize=16)
    plt.savefig(pjoin(figsave_dir, 'orituning.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()


    # 拟合 hue
    axisname = 'hue'
    angle_data = degrees_to_radians(hue)
    initial_hue = hue[np.argmax(hue_tuning)]
    mu = degrees_to_radians(initial_hue)
    kappa = 2  # 设置浓度参数
    y_data = norm_f(hue_tuning)
    # 拟合von Mises分布
    try:
        params, covariance = curve_fit(vonmises_pdf, angle_data, y_data, p0=[1, 2, mu])
        for param, param_name in zip(params, param_names):
            tuningparameters[f'{axisname}_{param_name}'].append(param)
        tuningparameters[f'{axisname}_fit'].append(1)
        interplot_f = ius(angle_data, y_data, k=5)
    except RuntimeError:
        tuningparameters[f'{axisname}_fit'].append(0)
        # 得到一款插值函数
        # interplot_f = interp1d(angle_data, y_data, kind='cubic')
        interplot_f = ius(angle_data, y_data, k=5)
        # 强行按高斯估计
        cos_x, sin_x = np.cos(angle_data), np.sin(angle_data)
        mean_cos_x = np.sum(cos_x * y_data) / y_data.sum()
        mean_sin_x = np.sum(sin_x * y_data) / y_data.sum()
        r = np.sqrt(mean_cos_x ** 2 + mean_sin_x ** 2)
        kappa = r / (1 - r**2)
        i0_value = sps.i0(kappa)
        gain = 2* np.pi * i0_value * np.max(y_data) / np.exp(kappa) 
        params = [gain, axis[np.argmax(y_data)], kappa]
        for param, param_name in zip(params, param_names):
            tuningparameters[f'{axisname}_{param_name}'].append(param)

    # hue 绘图
    plt.figure(figsize=(2, 2))
    plt.scatter(hue, y_data, label='Data', marker='+', color='black', zorder=5, s=30)
    x_hue = np.linspace(hue.min(), hue.max(), 100)
    x_plot = degrees_to_radians(x_hue)
    
    if tuningparameters[f'{axisname}_fit'][-1] >= 0:
        # plt.plot(x_hue, vonmises_pdf(x_plot, *params), label='Fitted tuning', color='red', alpha=0.6, lw=3)
        plt.plot(x_hue, interplot_f(x_plot), label='interplot', color='blue', alpha=0.6, lw=3)
    else:
        plt.plot(x_hue, interplot_f(x_plot), label='Fitted tuning', color='red', alpha=0.6, lw=3)
    # plt.legend(fontsize=9)
    # plt.xlabel('hue',fontsize=10)
    # plt.ylabel(f'unit-{unit} Resp.',fontsize=10)
    plt.xticks(fontsize=16)  # 调整x轴标签大小
    plt.yticks(fontsize=16)  # 调整y轴标签大小
    # plt.title('hue', fontsize=16)
    ax = plt.gca()
    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks([])
    ax_right.set_ylabel(f'Resp. to hue', fontsize=16)
    plt.savefig(pjoin(figsave_dir, f'huetuning.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # curv 是按形状进行sorting

    def encode_to_51_base(num):
        # 为十进制51以上的每个数字定义一个字符
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop"
        if num == 0:
            return "0"
        base51 = ""
        while num:
            remainder = num % 51
            base51 = symbols[remainder] + base51
            num //= 51
        return base51

    def decode_from_51_base(str_num):
        # 为十进制51以上的每个数字定义一个字符
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop"
        num = 0
        for char in str_num:
            num = num * 51 + symbols.index(char)
        return num

    tuningparameters['curv_sort'].append(''.join([encode_to_51_base(_) for _ in  np.argsort(curv_tuning)[::-1]]))

    # 
    y_data = norm_f(curv_tuning)
    norm = Normalize(vmin=np.min(y_data), vmax=np.max(y_data))
    sort_tuning = np.sort(y_data)[::-1]
    normalized_data = norm(sort_tuning)

    # 颜色和透明度
    cmap = plt.cm.cool  # 选择颜色映射，例如'viridis'
    colors = cmap(normalized_data)  # 映射颜色
    # 创建一个图像和轴，但仅用于显示colorbar
    fig, ax = plt.subplots(figsize=(2, 0.35))  # 宽度和高度可以根据需要调整
    fig.subplots_adjust(bottom=0.3)  # 调整布局以适应colorbar
    # 创建一个ScalarMappable并使用之前的归一化和颜色映射
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 只是为了避免警告
    # 绘制colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal')#vertical
    # cbar.set_label('Resp.', fontsize=5)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(['{:.2f}'.format(norm.vmin), '{:.2f}'.format(norm.vmax)])
    cbar.ax.tick_params(labelsize=8)
    # cbar.ax.yaxis.set_ticks_position('left')
    # 隐藏原始轴
    ax.remove()
    # 保存图像
    plt.savefig(pjoin(figsave_dir, f'curv_olorbar.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 设置图片路径和新图像尺寸
    image_folder = pjoin(work_dir, 'prep/simplified_stimuli/stim/raw_shape_patch')
    image_files = np.array([ f'Shape_{_}_Rotation_1.png' for _ in curv ])
    image_files = image_files[np.argsort(curv_tuning)[::-1]]
    num_rows = 5 #2 #3
    num_cols = 2 #5 #17
    # image_files = image_files[0:5].tolist() + image_files[-5::].tolist()
    # idxes = np.arange(51)[0:5].tolist() + np.arange(51)[-5::].tolist()
    image_files_top5, image_files_last5 = image_files[0:5].tolist(), image_files[-5::].tolist()
    idxes_top5, idxes_last5 = np.arange(51)[0:5].tolist(), np.arange(51)[-5::].tolist() 

    image_files = [None] * (len(image_files_top5) + len(image_files_last5))
    idxes = [None] * (len(idxes_top5) + len(idxes_last5))
    # 遍历listA和listB，将元素放在listC的奇数和偶数位置
    for i in range(len(image_files_top5)):
        image_files[2 * i] = image_files_top5[i]   #
        image_files[2 * i + 1] = image_files_last5[i]  #
        idxes[2 * i] = idxes_top5[i]   #
        idxes[2 * i + 1] = idxes_last5[i]  #
    new_image_size = (60, 60)  # 新的图片大小为50x50

    # 创建一个新的空白图像
    output_image = Image.new('RGB', (num_cols * new_image_size[0], num_rows * new_image_size[1]))

    # 定义灰色的目标值和容差
    target_gray = (128, 128, 128)
    tolerance = 20

    # 加载图片并按照指定的行列顺序排列
    for i, filename in enumerate(image_files):
        if i >= 51:  # 只需要前51张图片
            break
        img = Image.open(os.path.join(image_folder, filename))
        img = img.resize(new_image_size, Image.Resampling.LANCZOS)  # 调整图片大小为50x50

        # 将非黑色像素变为白色
        pixels = img.load()
        for x in range(img.width):
            for y in range(img.height):
                current_pixel = pixels[x, y]
                if all(abs(current_pixel[i] - target_gray[i]) <= tolerance for i in range(3)):
                    pixels[x, y] = tuple((255*colors[idxes[i]][0:3]).astype(np.int))

        # 计算当前图片的位置
        x_pos = (i % num_cols) * new_image_size[0]
        y_pos = (i // num_cols) * new_image_size[1]
        output_image.paste(img, (x_pos, y_pos))

    # 保存新图像
    output_image.save(pjoin(figsave_dir, f'curvtuning.png'))
    output_image.close()

    #################################################
    # natural patch 是按形状进行sorting

    # 设置图片路径和新图像尺寸
    images = np.load(pjoin(work_dir, 'prep/simplified_stimuli/stim/naturalpatches/Googlenet_conv2_naturalpatches.npy'), mmap_mode='r')
    naturaltop5 = np.argsort(naturalspace['space'])[::-1][0:5].tolist()
    naturallast5 = np.argsort(naturalspace['space'])[::-1][-5::].tolist()
    # selection = naturaltop5 + naturallast5
    selection = [None] * (len(naturaltop5) + len(naturallast5))
    icolor = [None] * (len(naturaltop5) + len(naturallast5))
    # 遍历listA和listB，将元素放在listC的奇数和偶数位置
    for i in range(len(naturaltop5)):
        selection[2 * i] = naturaltop5[i]   #
        selection[2 * i + 1] = naturallast5[i]  #
        icolor[2 * i] = 0
        icolor[2 * i + 1] = 6
    images = images[selection]
    num_rows = 5#2
    num_cols = 2#5
    new_image_size = (60, 60)  # 新的图片大小为50x50

    # 创建一个新的空白图像
    output_image = Image.new('RGB', (num_cols * new_image_size[0], num_rows * new_image_size[1]))

    # 定义灰色的目标值和容差
    target_gray = (128, 128, 128)
    tolerance = 20

    # 加载图片并按照指定的行列顺序排列
    for i in range(len(images)):
        if i >= 50:  # 只需要前51张图片
            break
        img = Image.fromarray(images[i])
        # img = crop_center(img, 50, 50)
        mask = construct_RF(img.width, sigma_x=5, sigma_y=5,) > 0 
        # 将非黑色像素变为白色
        pixels = img.load()
        for x in range(img.width):
            for y in range(img.height):
                if mask[x,y] == 0 and icolor[i] < 5:
                    pixels[x, y] = tuple((255*np.array([0.8,0.8,0.8])).astype(np.int))
                if mask[x,y] == 0 and icolor[i]  >= 5:
                    pixels[x, y] = tuple((255*np.array([0.2,0.2,0.2])).astype(np.int))
        img = img.resize(new_image_size, Image.Resampling.LANCZOS)  # 调整图片大小为50x50
        # 计算当前图片的位置
        x_pos = (i % num_cols) * new_image_size[0]
        y_pos = (i // num_cols) * new_image_size[1]
        output_image.paste(img, (x_pos, y_pos))

    # 保存新图像
    output_image.save(pjoin(figsave_dir, f'naturaltuning.png'))
    output_image.close()
    print(f'voxelmodel-{modeltuning_name} over!')


# df = pd.DataFrame(tuningparameters)
# df['ori_center'] = np.degrees(df['ori_mu'])/2
# hue_mu = df['hue_mu'].values
# hue_mu[hue_mu < 0] = hue_mu[hue_mu < 0] + 2 * np.pi   
# df['hue_center'] = np.degrees(df['hue_mu'])

# df.to_csv(pjoin(output_path, f'{axis_type}-tuning-{net_roi}.csv'), index=False)

# df.to_csv(pjoin(output_path, f'neural-model-tuning-{net_roi}.csv'), index=False)
