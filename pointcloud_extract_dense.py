import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

# 读取点云
pcd = o3d.io.read_point_cloud("dynamic_objects_from_ground.pcd")
points = np.asarray(pcd.points)

# Step 1: 聚类点云
# 使用DBSCAN对点云聚类
eps = 1.0  # 聚类邻域半径，可调整
min_samples = 10  # 每个簇的最小点数
db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
labels = db.labels_  # 每个点的标签
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"共找到 {num_clusters} 个聚类")

# Step 2: 筛选符合条件的点云簇
car_points = []

for cluster_id in range(num_clusters):
    # 提取当前簇的点
    cluster_points = points[labels == cluster_id]
    if len(cluster_points) == 0:
        continue

    # 计算簇的边界框（Bounding Box）
    bbox_min = cluster_points.min(axis=0)
    bbox_max = cluster_points.max(axis=0)
    bbox_size = bbox_max - bbox_min  # [长度, 宽度, 高度]
    length, width, height = bbox_size
    volume = np.prod(bbox_size)  # 体积 = 长×宽×高

    # 筛选条件：符合汽车尺寸范围
    is_car = (
        volume > 3.0  # 最小体积限制
    )

    if is_car:
        car_points.append(cluster_points)

# Step 3: 保存筛选后的汽车点云
car_points = np.vstack(car_points)
car_pcd = o3d.geometry.PointCloud()
car_pcd.points = o3d.utility.Vector3dVector(car_points)

# 保存和可视化结果
o3d.io.write_point_cloud("filtered_cars_only.pcd", car_pcd)
o3d.visualization.draw_geometries([car_pcd])
