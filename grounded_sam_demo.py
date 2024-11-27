import os
import sys
import torch
from PIL import Image
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 调整您的项目结构路径
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# 导入 Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# 导入 Segment Anything
from segment_anything_this.segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

def load_image(image_path):
    # 加载图像
    image_pil = Image.open(image_path).convert("RGB")

    # 图像转换
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)  # 3 x H x W
    return image_pil, image_tensor

def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("模型加载结果:", load_res)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    image = image.to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, vocab_size)
    boxes = outputs["pred_boxes"][0]  # (num_queries, 4)

    # 筛选输出
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # 获取短语
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if with_logits:
            pred_phrases.append(f"{pred_phrase}({logit.max().item():.2f})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def generate_mask_image(mask_list):
    # 生成掩码图像，背景为黑色，ROI 为非黑色
    # mask_list: 掩码列表 (N, 1, H, W)
    # 返回值: mask_image (H, W, 3) 的 NumPy 数组

    # 获取掩码的高度和宽度
    N, _, H, W = mask_list.shape

    # 初始化掩码图像，背景为黑色
    mask_image = np.zeros((H, W, 3), dtype=np.uint8)

    # 为每个掩码生成非黑色的随机颜色
    colors = []
    for _ in range(N):
        while True:
            color = np.random.randint(0, 256, size=3)
            if not np.array_equal(color, [0, 0, 0]):
                break
        colors.append(color)

    # 应用颜色到掩码区域
    for idx, mask in enumerate(mask_list):
        mask_np = mask.cpu().numpy().squeeze()  # 形状: (H, W)
        color = colors[idx]
        mask_indices = mask_np > 0
        mask_image[mask_indices] = color

    return mask_image

def process_image(
        image_path,
        device="cpu",
        config_file="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounded_checkpoint="groundingdino_swint_ogc.pth",
        sam_checkpoint="sam_vit_h_4b8939.pth",
        text_prompt="cars",
        box_threshold=0.3,
        text_threshold=0.25,
        sam_version="vit_h",
        sam_hq_checkpoint=None,
        use_sam_hq=False,
        bert_base_uncased_path=None
):
    # 加载图像
    image_pil, image_tensor = load_image(image_path)
    image_size = image_pil.size  # (宽度, 高度)

    # 加载 Grounding DINO 模型
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # 获取 Grounding DINO 的输出
    boxes_filt, pred_phrases = get_grounding_output(
        model, image_tensor, text_prompt, box_threshold, text_threshold, device=device
    )

    # 初始化 SAM 预测器
    if use_sam_hq:
        sam_model = sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device)
    else:
        sam_model = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam_model)

    # 为 SAM 准备图像
    image_cv = cv2.imread(image_path)
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv_rgb)

    # 调整边界框到图像尺寸
    W, H = image_size
    scale_factors = torch.tensor([W, H, W, H], dtype=torch.float).to(device)
    boxes_filt = boxes_filt * scale_factors
    boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2  # 中心点转换为左上角坐标
    boxes_filt[:, 2:] += boxes_filt[:, :2]  # 宽高转换为右下角坐标

    # 为 SAM 转换边界框
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv_rgb.shape[:2]).to(device)

    # 从 SAM 获取掩码
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # 生成掩码图像
    mask_image = generate_mask_image(masks)

    # 返回掩码图像和其他数据（如果需要）
    return mask_image
