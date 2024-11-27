import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def extract_cars_from_pcd(dynamic_pcd, eps=1.0, min_samples=10, min_volume=3.0):
    """
    从动态点云中提取符合汽车尺寸的点云簇

    参数：
    - dynamic_pcd: 点云的 open3d 对象
    - eps: DBSCAN 聚类的半径
    - min_samples: DBSCAN 聚类的最小点数
    - min_volume: 筛选汽车的最小体积阈值

    返回：
    - car_pcd: 只包含汽车的点云的 open3d 对象
    """
    # 将点云转换为 numpy 数组
    points = np.asarray(dynamic_pcd.points)

    # 使用 DBSCAN 对点云进行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 筛选符合汽车尺寸的点云簇
    car_points = []
    for cluster_id in range(num_clusters):
        cluster_points = points[labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        bbox_min = cluster_points.min(axis=0)
        bbox_max = cluster_points.max(axis=0)
        bbox_size = bbox_max - bbox_min  # [长度, 宽度, 高度]
        volume = np.prod(bbox_size)  # 体积 = 长×宽×高

        # 筛选条件：符合汽车尺寸范围
        if volume > min_volume:
            car_points.append(cluster_points)

    # 如果找到符合条件的点云簇，将其转换为 open3d 点云对象
    if car_points:
        car_pcd = o3d.geometry.PointCloud()
        car_pcd.points = o3d.utility.Vector3dVector(np.vstack(car_points))
    else:
        car_pcd = o3d.geometry.PointCloud()

    return car_pcd

if __name__ == '__main__':
    # 配置参数
    input_file = "dynamic_objects.pcd"
    output_file = "filtered_cars_only.pcd"
    eps = 1.0  # DBSCAN 聚类半径
    min_samples = 10  # DBSCAN 最小点数
    min_volume = 3.0  # 最小体积限制

    # 读取点云
    dynamic_pcd = o3d.io.read_point_cloud(input_file)

    # 提取汽车点云
    car_pcd = extract_cars_from_pcd(dynamic_pcd, eps=eps, min_samples=min_samples, min_volume=min_volume)

    # 保存并可视化结果
    o3d.io.write_point_cloud(output_file, car_pcd)
    print(f"点云已保存至 {output_file}")
    o3d.visualization.draw_geometries([car_pcd])
