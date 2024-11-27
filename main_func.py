import copy
import os
import json

import cv2
import open3d as o3d
import time

from grounded_sam_demo import process_image
from loss_calculator import loss_calculator, findBestExtrinsic
from pointcloud_dynamic_extraction import extract_dynamic_objects
from pointcloud_extract_dense import extract_cars_from_pcd
from pointcloud_projection import project_points_to_image, load_point_cloud, parse_calibration_data
from pointcloud_static_background import register_and_aggregate_point_clouds_with_frequency, load_point_clouds

import os
import json
import glob


def parse_folder(path):
    parent_dir = os.path.dirname(path)

    # 读取 calib.json 文件
    calib_file_path = os.path.join(parent_dir, 'calib.json')
    with open(calib_file_path, 'r') as f:
        calib_data = json.load(f)

    # 定义结果存储字典
    parsed_paths = {
        'calib_pos': calib_file_path,
        'lidar_folder_pos': os.path.join(path, 'lidar', 'pcd'),
        'camera_paths': {},
        'camera_params': {},  # 用于存储相机的内参和外参
        'image_files': {}  # 用于存储相机图像文件的位置
    }

    # 遍历 calib.json 中的相机数据，并更新 camera_paths 和相机参数
    for camera_name, params in calib_data.items():
        # 设置相机的路径，根据相机名动态生成路径
        camera_path = os.path.join(path, camera_name, 'image_dc')
        parsed_paths['camera_paths'][camera_name] = camera_path

        # 提取相机的内参和外参
        intrinsic = params.get('intrinsic', None)
        lidar2image = params.get('lidar2image', None)

        # 将内外参存储到 camera_params 中
        parsed_paths['camera_params'][camera_name] = {
            'intrinsic': intrinsic,
            'lidar2image': lidar2image
        }

        # 获取相机图片的完整路径
        image_files = glob.glob(os.path.join(camera_path, '*'))
        parsed_paths['image_files'][camera_name] = image_files

    return parsed_paths


def getExtrinsic(path, saveCalib=False, saveStatic=False, saveDynamic=False, device='cpu'):
    parsed_path = parse_folder(path)

    # 第一步：点云-建立一个静态场景
    # 设定点云数据的文件夹路径
    print("Start building static background...")
    pcd_folder = parsed_path["lidar_folder_pos"]

    # 载入和聚合点云数据
    point_clouds = load_point_clouds(pcd_folder)

    # 聚合点云并计算出现频率，提取静态场景
    # static_points = register_and_aggregate_point_clouds_with_frequency(point_clouds)
    # static_pcd = o3d.geometry.PointCloud()
    # static_pcd.points = o3d.utility.Vector3dVector(static_points)
    #
    # print("Finish building static background!")
    #
    # if saveStatic:
    #     o3d.io.write_point_cloud("static_scene.pcd", static_pcd)
    #     print("Static background has been saved: static_scene.pcd")

    # Checkpoint: 如果已经有了静态场景pcd 可以使用这个
    # static_pcd = o3d.io.read_point_cloud("static_scene.pcd")

    # 第二步：更新每个点云，提取动态目标
    # print("Start extracting dynamic cars...")
    # for pcd_name in os.listdir(pcd_folder):
    #     # 依据静态场景提取动态pcd
    #     pcd_path = os.path.join(pcd_folder, pcd_name)
    #     general_pcd = o3d.io.read_point_cloud(pcd_path)
    #
    #     static_pcd_copy = copy.deepcopy(static_pcd)
    #     dynamic_pcd = extract_dynamic_objects(static_pcd_copy,general_pcd)
    #
    #     # 针对性提取出汽车
    #     dynamic_pcd = extract_cars_from_pcd(dynamic_pcd)
    #
    #     # 保存到本地
    #     if not os.path.exists("dynamic_pcd_outputs_extrinsic"):
    #         os.makedirs("dynamic_pcd_outputs_extrinsic")
    #
    #     output_path = os.path.join("dynamic_pcd_outputs_extrinsic", pcd_name)
    #     o3d.io.write_point_cloud(output_path, dynamic_pcd)
    #
    # print("Finish extracting dynamic cars!")

    # 第三步：依次从SNWE四个角度进行外参更新
    for camera_name in parsed_path['camera_paths']:
        print(f"Start updating extrinsic parameters for Camera {camera_name}")
        img_folder_path = parsed_path['camera_paths'][camera_name]

        # 第四步：读取外参
        intrinsic_params = parsed_path['camera_params'][camera_name]['intrinsic']
        extrinsic_params = parsed_path['camera_params'][camera_name]['lidar2image']

        intrinsic_params = parse_calibration_data(intrinsic_params)

        # 第五步：图片分割
        for name in os.listdir(img_folder_path):
            # 图像的完整路径
            img_file = os.path.join(img_folder_path, name)

            mask_image = process_image(
                image_path=img_file,
                device=device
            )  # 得到mask过后的图像

            # 保存至本地
            if not os.path.exists("masked_imgs_outputs"):
                os.makedirs("masked_imgs_outputs")

            save_path = os.path.join("masked_imgs_outputs", name)

            cv2.imwrite(save_path, mask_image)

        # 第六步：将 动态点云、分割好的图片、内外参 输入
        #        找到最佳外参
        best_extrinsic = findBestExtrinsic(extrinsic_params,
                                           intrinsic_params,
                                           img_folder_path="masked_imgs_outputs",
                                           pcd_folder_path="dynamic_pcd_outputs_extrinsic",
                                           device="cpu")

    return best_extrinsic


if __name__ == '__main__':

    # 使用demo
    # 输入整包数据集路径
    file_folder_path = r"D:\Tangyuan\data\20241101-data\data\G32050700002M00_20241023082902"

    t1 = time.time()
    print(getExtrinsic(file_folder_path, device="cpu"))
    t2 = time.time()

    print(f"Total time cost {t2 - t1}")
