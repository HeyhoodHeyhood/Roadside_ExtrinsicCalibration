import open3d as o3d
import numpy as np
import os
from collections import defaultdict


# 载入所有点云数据
def load_point_clouds(pcd_folder, voxel_size=0.5):
    pcd_files = sorted([os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder) if f.endswith('.pcd')])
    point_clouds = []

    for pcd_file in pcd_files:
        pcd = o3d.io.read_point_cloud(pcd_file)
        # 使用体素滤波简化点云，以减少计算复杂度
        pcd = pcd.voxel_down_sample(voxel_size)
        point_clouds.append(pcd)

    return point_clouds


# 配准并聚合点云并计算出现频率
def register_and_aggregate_point_clouds_with_frequency(point_clouds, distance_threshold=0.5, dynamic_threshold=0.1):
    frequency_dict = defaultdict(int)
    trans_init = np.identity(4)  # 初始化转换矩阵

    size = len(point_clouds)

    # 将所有点云配准到同一坐标系
    for i in range(1, size):
        print(f"finish work ({i}/{size})")
        source = point_clouds[i]
        target = point_clouds[0] if i == 1 else combined_cloud

        # 将源点云和目标点云进行ICP配准
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        source.transform(reg_p2p.transformation)

        # 聚合配准后的点云
        combined_cloud = target + source

    # 结合配准后的点云并计算出现频率
    points = np.asarray(combined_cloud.points)
    for point in points:
        point_key = tuple(np.round(point / distance_threshold))
        frequency_dict[point_key] += 1

    # 处理轻微变化的场景（如树叶飘动）
    for key, count in list(frequency_dict.items()):
        if count < len(point_clouds) * 0.6 and count > len(point_clouds) * dynamic_threshold:
            frequency_dict[key] += 1

    # 过滤出现次超过一定门限的点作为静态场景点
    min_occurrence = len(point_clouds) * 0.9  # 出现次数超过90%的应该认为静态
    static_points = np.array(
        [np.array(key) * distance_threshold for key, count in frequency_dict.items() if count >= min_occurrence])
    return static_points


# 主函数
def main():
    # 设定点云数据的文件夹路径
    pcd_folder = r"D:\data\20241101-data\data\G32050700002M00_20241023082902\lidar\pcd"

    # 载入和聚合点云数据
    point_clouds = load_point_clouds(pcd_folder)

    # 聚合点云并计算出现频率，提取静态场景
    static_points = register_and_aggregate_point_clouds_with_frequency(point_clouds)
    static_pcd = o3d.geometry.PointCloud()
    static_pcd.points = o3d.utility.Vector3dVector(static_points)

    # 保存静态场景点云文件
    o3d.io.write_point_cloud("static_scene.pcd", static_pcd)
    print("静态场景已保存为 static_scene.pcd")

    # 可视化静态场景
    o3d.visualization.draw_geometries([static_pcd])


if __name__ == "__main__":
    main()
