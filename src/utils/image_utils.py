from pathlib import Path
from typing import List, Union
from PIL import Image, ImageDraw, ImageFont


def normalize_bbox(
    bbox: List[Union[int, float]], img_width: int, img_height: int
) -> List[int]:
    """
    将不同格式的 bounding box 坐标统一归一化为像素坐标 [x1, y1, x2, y2]。

    支持的格式:
    1. Qwen-VL 0-1000 标准归一化格式: [ymin, xmin, ymax, xmax] (0 ~ 1000)
    2. 0.0 ~ 1.0 归一化比例格式: [ymin, xmin, ymax, xmax] 或 [x1, y1, x2, y2]
    3. 绝对像素坐标格式: [x1, y1, x2, y2]
    """
    if len(bbox) != 4:
        raise ValueError(f"Bounding box 必须包含 4 个坐标值，收到: {bbox}")

    # 检测是否为 Qwen-VL 0-1000 归一化格式 [ymin, xmin, ymax, xmax]
    if all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in bbox) and any(
        v > 1.0 for v in bbox
    ):
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]
        x1 = int((xmin / 1000.0) * img_width)
        y1 = int((ymin / 1000.0) * img_height)
        x2 = int((xmax / 1000.0) * img_width)
        y2 = int((ymax / 1000.0) * img_height)

    # 检测是否为 0.0 ~ 1.0 归一化比例格式
    elif all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in bbox):
        val1, val2, val3, val4 = bbox
        x1 = int(val2 * img_width)
        y1 = int(val1 * img_height)
        x2 = int(val4 * img_width)
        y2 = int(val3 * img_height)

    # 绝对像素坐标
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox]

    # 确保坐标在图片有效边界内
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))

    return [x1, y1, x2, y2]


def crop_image_region(
    image_path: Union[str, Path], bbox: List[int], output_path: Union[str, Path]
) -> Path:
    """根据像素坐标框 [x1, y1, x2, y2] 裁剪页面图片并保存"""
    img_path = Path(image_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as img:
        cropped = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        cropped.save(out_path)

    return out_path.resolve()


def draw_bounding_box_overlay(
    image_path: Union[str, Path],
    bbox: List[int],
    output_path: Union[str, Path],
    label: str = "Evidence Region",
    box_color: str = "red",
    line_width: int = 4,
) -> Path:
    """在原始页面图片上绘制红框 overlay 和标签 banner，保存为标注图"""
    img_path = Path(image_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)

        # 绘制红框
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
            outline=box_color,
            width=line_width,
        )

        # 绘制文本标签 background banner
        if label:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            text_width = text_bbox[2] - text_bbox[0] + 10
            text_height = text_bbox[3] - text_bbox[1] + 6

            banner_y1 = max(0, bbox[1] - text_height)
            banner_y2 = bbox[1]

            draw.rectangle(
                [(bbox[0], banner_y1), (bbox[0] + text_width, banner_y2)],
                fill=box_color,
            )
            draw.text(
                (bbox[0] + 5, banner_y1 + 2),
                label,
                fill="white",
                font=font,
            )

        img.save(out_path)

    return out_path.resolve()


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    计算两个 2D 矩形框 [x1, y1, x2, y2] 的交并比 (Intersection over Union, IoU)。

    :param bbox1: 矩形框 1 像素坐标 [x1, y1, x2, y2]
    :param bbox2: 矩形框 2 像素坐标 [x1, y1, x2, y2]
    :return: IoU 得分 (0.0 ~ 1.0)
    """
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0

    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, b1_x2 - b1_x1) * max(0, b1_y2 - b1_y1)
    area2 = max(0, b2_x2 - b2_x1) * max(0, b2_y2 - b2_y1)

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return round(inter_area / union_area, 4)

