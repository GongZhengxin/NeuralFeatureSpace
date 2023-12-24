import numpy as np
import os
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker

# 路径定义
# input dirs
wk_dir = "/nfs/z1/userhome/zzl-liuyaoze/BrainImageNet/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/"
activ_dir = os.path.join(wk_dir, 'prep/image_activations/')
pcspace_dir = os.path.join(wk_dir, 'anal/pca-axis/')

# saveout dirs
coco_score_dir = os.path.join(wk_dir, 'anal/coco-patch-in-pcspace')
ldmrk_activ_dir = os.path.join(wk_dir, 'anal/landmark-activ')
ldmrk_in_space_dir = os.path.join(wk_dir, 'anal/landmark-in-pcspace')
fig_ldmark_path = os.path.join(wk_dir, 'vis/landmarks-in-spaces')

# for activation 确定分析的 net 和 layer
net = 'googlenet'
layer = 'inception3a'
# for pc space 还需确定 sub 和 brianmask
sub = 'avg'
for bm_name in ['shrdv1', 'shrdv2', 'shrdv3', 'shrdv4', 'shrdvis']:
    bm = bm_name #'shrdv1',

    # 加载 coco 激活值数据 == 用于挑选landmark patches
    activation_data = np.load(os.path.join(activ_dir, f'coco_{net}-{layer}.npy'))
    check_activation_shape = activation_data.shape
    print(f"原始激活数据的形状为：{check_activation_shape}")
    # 对数据进行变形
    transform_data = activation_data.transpose((0, 2, 3, 1)).reshape((-1, 256))
    check_transform_data_shape = transform_data.shape
    print(f"转换后的数据形状为：{check_transform_data_shape}")

    # 加载 pca space
    pc_space = np.load(os.path.join(pcspace_dir, f'sub-{sub}_ly-{layer}_bm-{bm}_axis.npy'))

    # ======================================
    # 以 GRID 方式选出 landmark 代表整个 space
    # ======================================
    # 1. 得到 pc space 的整体 并存出
    coco_score_in_pcspace = np.dot(transform_data, pc_space.T)
    save_out_name = f'sub-{sub}_ly-{layer}_bm-{bm}_coco-scores.npy'
    # np.save(os.path.join(coco_score_dir, save_out_name), coco_score_in_pcspace)
    print(f'saved {save_out_name} : data shape of {coco_score_in_pcspace.shape}')

    # 2. 基于coco score 的分布，按 GRID 选出 一批 landmarks，以代表整个分布：
    # 2.1 确定各轴极值
    # 确定要看哪两个 PC 
    x, y = 0, 1
    pcx_min = np.min(coco_score_in_pcspace[:, x])
    pcx_max = np.max(coco_score_in_pcspace[:, x])
    pcy_min = np.min(coco_score_in_pcspace[:, y])
    pcy_max = np.max(coco_score_in_pcspace[:, y])
    # 2.2 定义相交函数，检查是否可用于grid方式选出landmark
    def get_intersection(range1, range2):
        start1, end1 = range1
        start2, end2 = range2
        start = max(start1, start2)
        end = min(end1, end2)
        if start < end:
            return (start, end)
        else:
            return None
    # 2.3 找到 patch 索引
    # 网格大小
    n_points = 12  # 为了生成 100 个点
    # 生成网格点
    square_max = np.max([pcx_max, pcy_max])
    square_min = np.min([pcx_min, pcy_min])
    x = np.linspace(square_min, square_max, n_points)
    y = np.linspace(square_min, square_max, n_points)
    grid_points = np.array([(xi, yi) for xi in x for yi in y])
    # 计算搜索区域的大小
    search_radius = (square_max - square_min) / n_points / 2  # 例如，网格间距的一半
    def find_nearest_point_index(grid_point, data, radius):
        # 计算数据点与网格点之间的距离
        distances = np.sqrt((data[:, 0] - grid_point[0])**2 + (data[:, 1] - grid_point[1])**2)
        
        # 找到距离最近的点，只考虑在搜索半径内的点
        within_radius = distances < radius
        if np.any(within_radius):
            # 获取原始数据中的索引
            nearest_index_within_radius = np.argmin(distances[within_radius])
            nearest_index = np.where(within_radius)[0][nearest_index_within_radius]
            return nearest_index
        else:
            return None
    # 对于每个网格点，找到最近的数据点的索引
    nearest_point_indices = np.array([find_nearest_point_index(point, coco_score_in_pcspace, search_radius) for point in grid_points])
    # 过滤掉 None 值（即没有找到的点）
    nearest_point_indices = np.array([index for index in nearest_point_indices if index is not None])
    # 2.4. 存储 landmarks 的原始激活
    landmark_activ = transform_data[nearest_point_indices,:]
    # 确保目录存在
    if not os.path.exists(ldmrk_activ_dir):
        os.makedirs(ldmrk_activ_dir)
    np.save(os.path.join(ldmrk_activ_dir, f'sub-{sub}_ly-{layer}_bm-{bm}_landmarks.npy'), landmark_activ)


    # ======================================
    # 将 landmarks 映射到不同的 space
    # ======================================
    for target_bm_name in ['shrdv1', 'shrdv2', 'shrdv3', 'shrdv4', 'shrdvis']:# ['fullv1', 'fullv2', 'fullv3', 'fullv4', 'fullvis']:#
        # 需确定target space 的 sub 和 brianmask
        target_sub = 'avg'
        target_bm = target_bm_name #'shrdvis'
        # 加载 target space
        target_space = np.load(os.path.join(pcspace_dir, f'sub-{target_sub}_ly-{layer}_bm-{target_bm}_axis.npy'))

        # 加载 landmark activ
        landmark_activ = np.load(os.path.join(ldmrk_activ_dir, f'sub-{sub}_ly-{layer}_bm-{bm}_landmarks.npy'))

        # 将 landmark 投影到 targe space中
        landmark_proj = np.dot(landmark_activ, target_space.T)
        color = np.arange(len(landmark_proj)) 
        # 使用前两个主成分作为二维坐标
        x, y = 0, 1
        pcx = landmark_proj[:, x]
        pcy = landmark_proj[:, y]

        fig, axs = plt.subplots(1, 1, figsize=(3, 3))
        # 设置坐标轴标签
        # axs.set_xlabel('PC1')
        # axs.set_ylabel('PC2')
        # 移动坐标轴到中心
        axs.spines['left'].set_position('zero')
        axs.spines['bottom'].set_position('zero')

        # 隐藏右边和上面的边框
        axs.spines['right'].set_color('none')
        axs.spines['top'].set_color('none')

        # 显示刻度但隐藏刻度值
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        
        # 设置刻度线使其位于轴线中心
        axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axs.yaxis.set_major_locator(ticker.MultipleLocator(1))
        axs.tick_params(axis='x', direction='inout', length=3)
        axs.tick_params(axis='y', direction='inout', length=3)
        
        # 添加标题
        titlename = f'{bm} landmarks in {target_bm} space'
        # axs.set_title(titlename)
        axs.scatter(pcx, pcy, c=color, cmap='gist_rainbow', edgecolors='gray', linewidth=0.25, s=60, zorder=3)
        # # 将 coco 全集作为背景
        # coco_proj =  np.dot(transform_data, target_space.T)
        # axs.scatter(coco_proj[:,x], coco_proj[:,y], color='gray', s=20, alpha=0.05, zorder=2)
        
        axs.set_xlim([-4, 4])
        axs.set_ylim([-4, 4])
        axs.set_axisbelow(True)
        titlename = titlename.replace(' ', '_')
        plt.savefig(os.path.join(fig_ldmark_path, f'{layer}/{titlename}.png'), dpi=72*6, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()


# ======================================================

