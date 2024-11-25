import json
import numpy as np
import open3d as o3d
import cv2

# 加载 JSON 文件
with open('data_sample/calib.json', 'r') as f:
    calib_data = json.load(f)

# 遍历每个相机的外参矩阵和内参矩阵
extrinsic_matrices = {}
intrinsic_matrices = {}
for cam_name, cam_data in calib_data.items():
    # 提取外参矩阵
    lidar2image = cam_data.get("lidar2image", None)
    if lidar2image:
        extrinsic_matrix = np.array(lidar2image)
        extrinsic_matrices[cam_name] = extrinsic_matrix

    # 提取内参矩阵，去掉最后一列以获得 3x3 矩阵
    intrinsic = cam_data.get("intrinsic", None)
    if intrinsic:
        intrinsic_matrix = np.array(intrinsic)[:, :3]
        intrinsic_matrices[cam_name] = intrinsic_matrix

# 读取 LiDAR 点云数据
pcd = o3d.io.read_point_cloud("filtered_cars_only.pcd")
points = np.asarray(pcd.points)

# 遍历每个相机，进行投影
for cam_name in extrinsic_matrices:
    extrinsic_matrix = extrinsic_matrices[cam_name]
    intrinsic_matrix = intrinsic_matrices[cam_name]

    # 将点云的3D坐标转换为齐次坐标
    if points.size == 0:
        print(f"No points found for {cam_name}")
        continue

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 使用外参矩阵将点云转换到相机坐标系
    points_camera = (extrinsic_matrix @ points_homogeneous.T).T

    # 只保留 z > 0 的点（确保点在相机前方）
    points_camera = points_camera[points_camera[:, 2] > 0]

    # 使用内参矩阵将相机坐标系的点投影到图像平面
    points_2d_homogeneous = (intrinsic_matrix @ points_camera[:, :3].T).T

    # 将齐次坐标转换为图像坐标
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]

    # 读取原始图像和mask
    image_path = f"data_sample/1729643342700000000{cam_name[-1]}.png"
    mask_path = f"outputs/{cam_name[-1]}/mask.jpg"
    raw_img = cv2.imread(image_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度mask

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)  # 转换为三通道
    mask_colored = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)  # 映射为伪彩色

    # 确保尺寸一致
    if raw_img.shape[:2] != mask_colored.shape[:2]:
        mask_colored = cv2.resize(mask_colored, (raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_AREA)

    # 确保通道数一致
    if len(mask_colored.shape) != 3 or mask_colored.shape[2] != 3:
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)

    # 叠加原始图像和mask
    overlaid_img = cv2.addWeighted(raw_img, 0.7, mask_colored, 0.5, 0)  # 融合图像

    # 在融合图像上绘制投影点
    for i in range(points_2d.shape[0]):
        x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
        if 0 <= x < overlaid_img.shape[1] and 0 <= y < overlaid_img.shape[0]:  # 确保点在图像范围内
            cv2.circle(overlaid_img, (x, y), 2, (0, 255, 0), -1)  # 绿色点表示投影点

    # 显示结果
    cv2.namedWindow(f'Overlay - {cam_name}', cv2.WINDOW_NORMAL)
    cv2.imshow(f'Overlay - {cam_name}', overlaid_img)
    cv2.resizeWindow(f'Overlay - {cam_name}', overlaid_img.shape[1], overlaid_img.shape[0])

    cv2.waitKey(0)

cv2.destroyAllWindows()
