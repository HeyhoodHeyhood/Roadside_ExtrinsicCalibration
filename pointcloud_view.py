import json
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree

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

# 打印每个相机的外参矩阵
for cam_name, extrinsic_matrix in extrinsic_matrices.items():
    print(f"Camera: {cam_name} - Extrinsic Matrix")
    print(extrinsic_matrix)
    print()

# 打印每个相机的内参矩阵
for cam_name, intrinsic_matrix in intrinsic_matrices.items():
    print(f"Camera: {cam_name} - Intrinsic Matrix")
    print(intrinsic_matrix)
    print()

# 读取 LiDAR 点云数据
pcd = o3d.io.read_point_cloud("data_sample/1729643342700000000.pcd")
points = np.asarray(pcd.points)

# 使用 RANSAC 拟合地面平面
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(points[:, :3])
plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=0.2,
                                                  ransac_n=3,
                                                  num_iterations=1000)
# 提取地面和平面以上的点
inlier_cloud = pcd_filtered.select_by_index(inliers)
outlier_cloud = pcd_filtered.select_by_index(inliers, invert=True)

# 只保留非地面的点（即车辆和其他物体），并筛除高度小于 0.2 的点和大于 3 米的点
points_objects = np.asarray(outlier_cloud.points)
points_objects = points_objects[(points_objects[:, 2] >= 0.2) & (points_objects[:, 2] <= 3.0)]

car_height_min = 1.4  # 车辆最低高度
car_height_max = 4.0  # 车辆最高高度
distance_threshold = 0.01  # 欧氏距离阈值，单位米
min_cluster_size = 10  # 最小点簇大小


def euclidean_clustering(points, distance_threshold, min_cluster_size):
    """
    基于欧氏距离的点簇提取
    """
    tree = cKDTree(points)  # 构建 KD 树
    visited = set()  # 已访问点集合
    clusters = []  # 存储点簇列表

    for idx in range(len(points)):
        if idx in visited:
            continue
        # 查找与当前点在距离阈值内的所有点
        neighbors = tree.query_ball_point(points[idx], distance_threshold)
        if len(neighbors) < min_cluster_size:
            # 若邻域点数不足最小簇大小，则跳过
            continue
        # 创建新簇
        cluster = []
        queue = neighbors
        while queue:
            point_idx = queue.pop()
            if point_idx in visited:
                continue
            visited.add(point_idx)
            cluster.append(point_idx)
            # 添加邻域点到队列
            point_neighbors = tree.query_ball_point(points[point_idx], distance_threshold)
            if len(point_neighbors) >= min_cluster_size:
                queue.extend(point_neighbors)
        clusters.append(cluster)
    return clusters


# 对非地面点进行聚类
clusters = euclidean_clustering(points_objects, distance_threshold, min_cluster_size)

# 按高度过滤点簇
filtered_points = []
for cluster_indices in clusters:
    cluster_points = points_objects[cluster_indices]
    cluster_heights = cluster_points[:, 2]  # 提取簇中所有点的高度
    if car_height_min <= cluster_heights.max() <= car_height_max:  # 按高度范围筛选
        filtered_points.append(cluster_points)

# 合并所有符合条件的点簇
if filtered_points:
    filtered_points = np.vstack(filtered_points)
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
else:
    print("No clusters found matching height criteria.")
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(np.empty((0, 3)))

# 更新点云
pcd_filtered = filtered_cloud

# 遍历每个相机，进行投影
for cam_name in extrinsic_matrices:
    extrinsic_matrix = extrinsic_matrices[cam_name]
    intrinsic_matrix = intrinsic_matrices[cam_name]

    # 将点云的3D坐标转换为齐次坐标
    if points_objects.size == 0:
        print(f"No points found for {cam_name}")
        continue

    points_homogeneous = np.hstack((points_objects, np.ones((points_objects.shape[0], 1))))

    # 使用外参矩阵将点云转换到相机坐标系
    points_camera = (extrinsic_matrix @ points_homogeneous.T).T

    # 只保留 z > 0 的点（确保点在相机前方）
    points_camera = points_camera[points_camera[:, 2] > 0]

    # 使用内参矩阵将相机坐标系的点投影到图像平面
    points_2d_homogeneous = (intrinsic_matrix @ points_camera[:, :3].T).T

    # 将齐次坐标转换为图像坐标
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]

    # 读取对应的图像
    image_path = f"data_sample/1729643342700000000{cam_name[-1]}.png"
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image {image_path} not found.")
        continue

    # 在图像上绘制投影结果
    for i in range(points_2d.shape[0]):
        x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # 确保点在图像范围内
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    # 显示投影结果
    cv2.imshow(f'Projected Points - {cam_name}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
