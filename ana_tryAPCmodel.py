
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# 辅助函数：计算给定角度的单位方向向量
def unit_vector_at_angle(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    return np.array([np.cos(angle_radians), np.sin(angle_radians)])

# 辅助函数：计算曲率
def curvature(s, x, y):
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    return (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

# 加载图像并转换为灰度
image_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/simplified_stimuli/stim/raw_shape/Shape_50_Rotation_3.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 应用边缘检测找到形状轮廓
edges = cv2.Canny(image, 100, 200)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest_contour = max(contours, key=cv2.contourArea)
largest_contour = largest_contour[:, 0, :]  # 去掉冗余的维度

# 确定形状的中心
M = cv2.moments(largest_contour)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
center = np.array([cx, cy])

# 指定角度
angles = [0, 45, 90, 135, 180, 225, 270, 315]

# 对每个角度计算曲率
curvatures = []
for angle in angles:
    # 创建射线方向的单位向量
    direction = unit_vector_at_angle(angle)
    
    # 查找轮廓上与射线相交的点
    dots = np.dot(largest_contour - center, direction)
    max_dot = np.argmax(dots)
    point_on_contour = largest_contour[max_dot]
    
    # 使用UnivariateSpline拟合轮廓附近的点
    window = 30  # 选择一个窗口，用于曲率拟合
    start = max(max_dot - window, 0)
    end = min(max_dot + window, len(largest_contour))
    s = np.linspace(0, 1, end - start)
    spline_x = UnivariateSpline(s, largest_contour[start:end, 0], k=4, s=0)
    spline_y = UnivariateSpline(s, largest_contour[start:end, 1], k=4, s=0)
    
    # 在特定点上计算曲率数组
    curvature_array = curvature(s, spline_x(s), spline_y(s))

    # 获取窗口中间点的曲率
    mid_point_index = len(curvature_array) // 2
    curvature_at_point = curvature_array[mid_point_index]
    curvatures.append(curvature_at_point)

def squashed_curv(curv):
    return 2/(1+ np.exp(-0.125*curv)) -1

# 输出计算的曲率
for angle, curvature_value in zip(angles, curvatures):
    print(f"Angle {angle}°: Curvature = {squashed_curv(curvature_value)}")

# 可视化
plt.imshow(image, cmap='gray')
plt.scatter([cx], [cy], color='red')

