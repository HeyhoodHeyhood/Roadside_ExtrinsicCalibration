import torch
import cv2
from segment_anything import SamPredictor, sam_model_registry

# Step 1: 安装和加载 SAM 模型
# 这里假设你已经下载了 SAM 模型的预训练权重
sam_checkpoint = "sam_vit_h_4b8939.pth"  # 预训练模型的路径
model_type = "vit_h"  # 模型类型（根据你下载的模型权重类型）
device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查 GPU 是否可用

# 加载 SAM 模型
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Step 2: 定义要处理的图像列表
image_paths = [
    r"\data_sample\1729643342700000000E.png",  # 替换为你图片的实际路径
    r"\data_sample\1729643342700000000S.png",
    r"\data_sample\1729643342700000000W.png",
    r"\data_sample\1729643342700000000N.png"
]

# Step 3: 分割每张图片
def segment_image(image_path, predictor):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    # 设置输入图像
    predictor.set_image(image)

    # 提供一些初始的提示点，用于指示你想分割的区域
    # 这里可以选择使用手动标注的点或一些默认点
    input_point = [[image.shape[1] // 2, image.shape[0] // 2]]  # 图像中心点
    input_label = [1]  # 1 表示前景点

    # 进行预测，得到分割掩膜
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # 显示和保存分割结果
    mask = masks[0]
    result = (mask * 255).astype("uint8")
    result_image_path = image_path.replace(".jpg", "_segmented.jpg")
    cv2.imwrite(result_image_path, result)
    print(f"分割结果已保存: {result_image_path}")

    # 可视化分割结果
    cv2.imshow("Original Image", image)
    cv2.imshow("Segmented Mask", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Step 4: 处理所有图像
for image_path in image_paths:
    segment_image(image_path, predictor)

print("所有图像分割处理完成！")
