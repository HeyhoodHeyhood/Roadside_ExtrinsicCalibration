import json
import numpy as np
import open3d as o3d
import cv2


def load_calibration_data(calibration_file):
    """读取校准数据文件并返回内参和外参数据"""
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)

    extrinsic_matrices = {}
    intrinsic_matrices = {}

    for cam_name, cam_data in calib_data.items():
        # 提取外参矩阵
        lidar2image = cam_data.get("lidar2image", None)
        if lidar2image:
            extrinsic_matrix = np.array(lidar2image)
            extrinsic_matrices[cam_name] = extrinsic_matrix

        # 提取内参矩阵
        intrinsic = cam_data.get("intrinsic", None)
        if intrinsic:
            intrinsic_matrix = np.array(intrinsic)
            intrinsic_matrices[cam_name] = parse_calibration_data(intrinsic_matrix)

    return extrinsic_matrices, intrinsic_matrices

def parse_calibration_data(intrinsic_matrices):
    intrinsic_matrix = np.array(intrinsic_matrices)[:, :3]

    return intrinsic_matrix


def load_point_cloud(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    return points


def project_points_to_image(points, extrinsic_matrix, intrinsic_matrix):
    if points.size == 0:
        return np.array([])

    # 将点云的3D坐标转换为齐次坐标
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 使用外参矩阵将点云转换到相机坐标系
    points_camera = (extrinsic_matrix @ points_homogeneous.T).T

    # 只保留 z > 0 的点（确保点在相机前方）
    points_camera = points_camera[points_camera[:, 2] > 0]

    # 使用内参矩阵将相机坐标系的点投影到图像平面
    points_2d_homogeneous = (intrinsic_matrix @ points_camera[:, :3].T).T

    # 将齐次坐标转换为图像坐标
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]

    return points_2d


def overlay_points_on_image(points_2d, mask_path, original_image_path):
    # 读取mask
    mask_img = modifyMask(mask_path, original_image_path)

    # 在mask图像上绘制投影点
    for i in range(points_2d.shape[0]):
        x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
        if 0 <= x < mask_img.shape[1] and 0 <= y < mask_img.shape[0]:  # 确保点在图像范围内
            cv2.circle(mask_img, (x, y), 2, (0, 255, 0), -1)  # 绿色点表示投影点

    return mask_img


def modifyMask(mask_path, original_image_path):
    # 读取mask
    mask_img = cv2.imread(mask_path, cv2.COLOR_BGR2RGB)
    original_img = cv2.imread(original_image_path)

    # 确保mask图像的尺寸与原始图像一致
    if mask_img.shape[:2] != original_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_AREA)

    return mask_img


if __name__ == "__main__":
    # 初始化校准数据和点云数据
    calib_file = 'data_sample/calib.json'
    extrinsic_matrices, intrinsic_matrices = load_calibration_data(calib_file)

    pcd_file = "filtered_cars_only.pcd"
    points = load_point_cloud(pcd_file)

    # 遍历每个相机，进行投影并显示结果
    for cam_name in extrinsic_matrices:
        extrinsic_matrix = extrinsic_matrices[cam_name]
        intrinsic_matrix = intrinsic_matrices[cam_name]

        # 将点云投影到图像平面
        points_2d = project_points_to_image(points, extrinsic_matrix, intrinsic_matrix)

        if points_2d.size == 0:
            print(f"No points found for {cam_name}")
            continue

        # 读取并叠加投影点
        mask_path = f"outputs/{cam_name[-1]}/mask.jpg"
        original_image_path = f"data_sample/1729643342700000000{cam_name[-1]}.png"
        result_img = overlay_points_on_image(points_2d, mask_path, original_image_path)

        # 显示结果
        cv2.namedWindow(f'Overlay - {cam_name}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'Overlay - {cam_name}', result_img)
        cv2.resizeWindow(f'Overlay - {cam_name}', result_img.shape[1], result_img.shape[0])

        cv2.waitKey(0)

    cv2.destroyAllWindows()
