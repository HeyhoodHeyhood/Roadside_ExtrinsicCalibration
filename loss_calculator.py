import os

import cv2
import numpy as np
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from grounded_sam_demo import process_image
from pointcloud_projection import project_points_to_image, load_point_cloud, modifyMask


def single_loss(img, points):
    """
    计算总损失，包括 Overlap Point Loss 和 Projection Distance Loss。

    参数:
    - img: 输入图像，形状为 (H, W, C)
    - points: 图像中的点列表，每个点为 (x, y)

    返回:
    - total_loss: 总损失
    """



    # 确保 points 在图像范围内
    points = np.array([point for point in points if
                       0 <= int(point[0]) < img.shape[1] and 0 <= int(point[1]) < img.shape[0]])

    # Overlap Point Loss (ROI)
    n_total_points = len(points)
    n_projected_points = 0

    for point in points:
        x, y = int(point[0]), int(point[1])

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # 确保点在图像范围内
            pixel_color = tuple(img[y, x])  # 获取像素颜色
            if pixel_color != (0, 0, 0):  # 比较颜色
                n_projected_points += 1

    # 显示图像的功能
    img_with_points = img.copy()
    for point in points:
        x, y = int(point[0]), int(point[1])
        # 使用 cv2.circle 在图像上绘制点，半径为 5，颜色为红色
        pixel_color = tuple(img[y, x])  # 获取像素颜色
        if pixel_color != (0, 0, 0):  # 比较颜色
            cv2.circle(img_with_points, (x, y), 5, (0, 255, 0), -1)
        else:
            cv2.circle(img_with_points, (x, y), 5, (0, 0, 255), -1)

    # 显示图像
    cv2.imshow('Projected Points', img_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # overlap_loss = 1 - (n_projected_points / n_total_points) if n_total_points > 0 else 0
    overlap_loss = n_total_points - n_projected_points

    # Projection Distance Loss
    total_distance = 0

    # # 构造掩码，筛选 ROI 中的非背景点
    # mask = ~np.all(img == (0, 0, 0), axis=-1)
    # roi_points = np.argwhere(mask)  # 获取 ROI 中所有点的坐标
    #
    # valid_pixel_count = roi_points.shape[0]
    # if valid_pixel_count == 0:
    #     return overlap_loss  # 如果没有有效点，直接返回 Overlap Loss
    #
    # distances_list = []
    #
    # for idx, (y, x) in enumerate(roi_points):
    #     current_point = np.array([x, y])
    #
    #     # 计算到点云所有点的距离
    #     distances = np.linalg.norm(points - current_point, axis=1)
    #
    #     # 找到最近点的距离平方
    #     min_distance_squared = np.min(distances ** 2)
    #
    #     # 累加最小距离平方
    #     distances_list.append(min_distance_squared)
    #
    # total_distance = np.sum(distances_list)
    #
    # # 归一化 Projection Distance Loss，使用最大距离进行归一化
    # max_distance = np.max(distances_list) if len(distances_list) > 0 else 1
    # projection_distance_loss = total_distance / (valid_pixel_count * max_distance)
    #
    # print("Projection Distance Loss (Normalized):", projection_distance_loss)

    # Total Loss
    total_loss = overlap_loss
    # print("Total Loss (Normalized):", projection_distance_loss)

    return total_loss


# 对某个外参下的所有的loss计算
def loss_calculator(extrinsic, intrinsic, img_folder_path, pcd_folder_path):
    loss = 0.0

    for name in os.listdir(img_folder_path):
        # 构造点云文件和图像文件的完整路径
        pcd_file = os.path.join(pcd_folder_path, name[:-4] + ".pcd")
        points = load_point_cloud(pcd_file)  # 得到array

        img_file = os.path.join(img_folder_path, name)
        img = cv2.imread(img_file)

        points_2d = project_points_to_image(points, extrinsic, intrinsic)  # 得到投影后的点

        this_loss = single_loss(img, points_2d)

        loss += this_loss

    return loss


def findBestExtrinsic(primary_extrinsic,
                      intrinsic,
                      img_folder_path,
                      pcd_folder_path="dynamic_pcd_outputs_extrinsic"):
    """
    给定图片和点云文件，以及初始外参，执行两阶段搜索优化，首先进行广泛搜索，然后进行细致优化。
    """

    # Step 1: 初始分数计算
    best_loss = loss_calculator(primary_extrinsic, intrinsic, img_folder_path, pcd_folder_path)
    best_extrinsic = primary_extrinsic

    # Step 2: 广泛搜索 (Wide Area Search)
    euler_angles_range = np.arange(-20, 21, 2)  # -20°到+20°，步长为2°
    translation_range = np.arange(-50, 51, 5)  # -50cm到+50cm，步长为5cm

    for euler_x in euler_angles_range:
        for euler_y in euler_angles_range:
            for euler_z in euler_angles_range:
                for tx in translation_range:
                    for ty in translation_range:
                        for tz in translation_range:
                            # 生成新的外参矩阵
                            extrinsic = np.copy(primary_extrinsic)
                            # 更新旋转矩阵部分，假设我们有旋转函数 `update_rotation_matrix`
                            extrinsic[:3, :3] = update_rotation_matrix(extrinsic[:3, :3], euler_x, euler_y, euler_z)
                            # 更新平移向量部分
                            extrinsic[:3, 3] = [tx, ty, tz]

                            # 计算新的损失
                            loss = loss_calculator(extrinsic, intrinsic, img_folder_path, pcd_folder_path)

                            # 如果新损失更小，更新最优外参和损失
                            if loss < best_loss:
                                best_loss = loss
                                best_extrinsic = extrinsic

    # Step 3: 细致优化搜索 (Refinement Search)
    # 在[-2°, +2°] 和 [-10cm, +10cm] 的范围内进行随机搜索，搜索次数为1000次
    for _ in range(1000):
        # 随机生成欧拉角和位移
        euler_x = np.random.uniform(-2, 2)
        euler_y = np.random.uniform(-2, 2)
        euler_z = np.random.uniform(-2, 2)
        tx = np.random.uniform(-10, 10)
        ty = np.random.uniform(-10, 10)
        tz = np.random.uniform(-10, 10)

        # 生成新的外参矩阵
        extrinsic = np.copy(primary_extrinsic)
        # 更新旋转矩阵部分
        extrinsic[:3, :3] = update_rotation_matrix(euler_x, euler_y, euler_z)
        # 更新平移向量部分
        extrinsic[:3, 3] = [tx, ty, tz]

        # 计算新的损失
        loss = loss_calculator(extrinsic, intrinsic, img_folder_path, pcd_folder_path)

        # 如果新损失更小，更新最优外参和损失
        if loss < best_loss:
            best_loss = loss
            best_extrinsic = extrinsic

    return best_extrinsic

def update_rotation_matrix(roll, pitch, yaw):
    """
    根据欧拉角 (roll, pitch, yaw) 生成旋转矩阵。
    roll: 绕 x 轴的旋转 (单位：度)
    pitch: 绕 y 轴的旋转 (单位：度)
    yaw: 绕 z 轴的旋转 (单位：度)
    """
    # 将角度转换为弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # 分别计算绕 x, y, z 轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 旋转矩阵按 ZYX 顺序组合 (先绕 z，再绕 y，再绕 x)
    R = Rz @ Ry @ Rx
    return R

def read_calib(path):
    extrinsic_matrices = {}
    intrinsic_matrices = {}

    with open(path, 'r') as f:
        calib_data = json.load(f)

    # 遍历每个相机的外参矩阵和内参矩阵
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

    return extrinsic_matrices, intrinsic_matrices


if __name__ == '__main__':
    calib_path = r"D:\Tangyuan\extrinsic calibration\project\data_sample\calib.json"
    pointcloud_path = r"D:\Tangyuan\extrinsic calibration\project\dynamic_objects.pcd"
    image_paths = {
        "R2_Aw_CamS": r"D:\Tangyuan\extrinsic calibration\project\outputs\S\mask.jpg",
        "R2_Bn_CamW": r"D:\Tangyuan\extrinsic calibration\project\outputs\W\mask.jpg",
        "R2_Ce_CamN": r"D:\Tangyuan\extrinsic calibration\project\outputs\N\mask.jpg",
        "R2_Ds_CamE": r"D:\Tangyuan\extrinsic calibration\project\outputs\E\mask.jpg"
    }

    # 读取相机的初始外参和内参
    extrinsic, intrinsic = read_calib(calib_path)

    # 读取3D点云
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    ground_truth_points = np.asarray(pcd.points)

    pcd_file = "filtered_cars_only.pcd"
    points = load_point_cloud(pcd_file)

    # 计算每个相机的损失
    for idx, cam_name in enumerate(extrinsic.keys()):
        # 提取相机的外参矩阵
        extrinsic_matrix = extrinsic[cam_name]

        # 提取相机的内参矩阵
        camera_matrix = intrinsic[cam_name]

        # 将点云投影到图像平面
        points_2d = project_points_to_image(points, extrinsic_matrix, camera_matrix)

        if points_2d.size == 0:
            print(f"No points found for {cam_name}")
            continue

        # Get the masked img
        original_image_path = f"data_sample/1729643342700000000{cam_name[-1]}.png"
        mask_path = image_paths[cam_name]
        mask_img = modifyMask(mask_path, original_image_path)

        # Calculate loss
        print("loss start")
        # loss = loss_calculator(mask_img, points_2d)
        # print(loss)

        print(extrinsic_matrix)

        print("find start")
        extrinsic_matrix = findBestExtrinsic(extrinsic_matrix, camera_matrix, points, mask_img)
        print(extrinsic_matrix)
