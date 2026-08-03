"""逐张预测小幅 RGB TIFF 验证图像，并输出漏检和误检清单。

在仓库根目录、dfine 环境执行：

    python myscript/切片推理预测.py

本脚本用于分析验证集的困难样本，不用于替代 valid.py 的正式模型指标。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import MODEL_IMAGE_SIZE
from ultralytics import YOLO

try:
    import rasterio
except ImportError:
    rasterio = None


# =============================================================================
# 用户配置区
# =============================================================================

# 用于分析的权重。请与 valid.py 中验证的权重保持一致。
WEIGHTS_PATH = Path(r"E:\YOLO\yolo26\runs\train\yolo26s_200epochs_640_训练集708_map50\weights\best_map50.pt")

# 将所有待检查的小 TIFF 放在此目录；支持子文件夹。
INPUT_DIR = Path(r"F:\3能源金三角基础设施识别\火力发电厂\优化不同数据集版本\减少不像的图_第1版\训练集优化掉的图片与标签\金三角内优化掉的\图片")

# 验证集 YOLO 标签目录。保留该设置可自动识别漏检和误检；不需要标签时设为 None。
LABEL_DIR: Path | None = Path(r"F:\3能源金三角基础设施识别\火力发电厂\优化不同数据集版本\减少不像的图_第1版\训练集优化掉的图片与标签\金三角内优化掉的\标签")
#LABEL_DIR = None

# 每次实验使用独立输出目录，名称应包含所用置信度。
OUTPUT_DIR = Path(r"F:\3能源金三角基础设施识别\火力发电厂\优化不同数据集版本\减少不像的图_第1版\训练集优化掉的图片与标签\test可删\yolo26_map50_置信度30")

# 先使用 valid.py 报告的最优 F1 置信度约 0.30；可按实验改为 0.20、0.25、0.35 等。
CONFIDENCE = 0.30
NMS_IOU = 0.70
MATCH_IOU = 0.50  # 仅用于与 YOLO 标签匹配，从而统计 TP、FP、FN。
DEVICE: int | str = 0
USE_HALF = True
MAX_DETECTIONS = 300

# =============================================================================

TIFF_SUFFIXES = {".tif", ".tiff"}


def validate_config() -> None:
    """Validate required paths and numerical settings before inference."""
    if rasterio is None:
        raise RuntimeError("当前环境缺少 rasterio，无法读取 TIFF；请安装 rasterio 后重试。")
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"模型权重不存在：{WEIGHTS_PATH}")
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"待预测目录不存在：{INPUT_DIR}")
    if LABEL_DIR is not None and not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"标签目录不存在：{LABEL_DIR}")
    if MODEL_IMAGE_SIZE <= 0 or MODEL_IMAGE_SIZE % 32:
        raise ValueError("MODEL_IMAGE_SIZE 必须为能被 32 整除的正整数。")
    if not all(0 <= value <= 1 for value in (CONFIDENCE, NMS_IOU, MATCH_IOU)):
        raise ValueError("CONFIDENCE、NMS_IOU 和 MATCH_IOU 必须位于 [0, 1]。")
    if MAX_DETECTIONS <= 0:
        raise ValueError("MAX_DETECTIONS 必须大于 0。")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 设置为 GPU，但当前 PyTorch 没有检测到可用 CUDA。")


def find_tiffs(directory: Path) -> list[Path]:
    """Return every TIFF in a directory tree in deterministic order."""
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES)


def read_rgb_tif(path: Path) -> np.ndarray:
    """Read an 8-bit, at-least-three-band TIFF into an RGB HWC array."""
    with rasterio.open(path) as source:
        if source.count < 3:
            raise ValueError(f"TIFF 少于 3 个波段，无法作为 RGB 输入：{path}")
        if source.dtypes[0] != "uint8":
            raise ValueError(f"TIFF 不是 uint8，不能保证与训练数据一致：{path} ({source.dtypes[0]})")
        return np.moveaxis(source.read([1, 2, 3]), 0, -1)


def read_labels(path: Path | None, width: int, height: int) -> np.ndarray:
    """Read YOLO normalized labels as pixel-space [class, x1, y1, x2, y2] rows."""
    if path is None or not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return np.empty((0, 5), dtype=np.float32)

    targets = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"标签格式错误：{path} 第 {line_number} 行")
        class_id, center_x, center_y, box_width, box_height = map(float, values)
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        x2 = (center_x + box_width / 2) * width
        y2 = (center_y + box_height / 2) * height
        targets.append((int(class_id), x1, y1, x2, y2))
    return np.asarray(targets, dtype=np.float32)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculate IoU between one xyxy box and multiple xyxy boxes."""
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, intersection_x2 - intersection_x1) * np.maximum(0, intersection_y2 - intersection_y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / (box_area + boxes_area - intersection + 1e-9)


def match_predictions(predictions: np.ndarray, targets: np.ndarray) -> tuple[int, int, int]:
    """Greedily match confidence-sorted predictions to same-class targets at MATCH_IOU."""
    if not len(predictions):
        return 0, 0, len(targets)
    if not len(targets):
        return 0, len(predictions), 0

    matched_targets: set[int] = set()
    true_positives = 0
    for prediction in predictions[np.argsort(-predictions[:, 5])]:
        candidates = np.where(targets[:, 0] == prediction[0])[0]
        candidates = np.asarray([index for index in candidates if index not in matched_targets])
        if not len(candidates):
            continue
        best_local_index = box_iou(prediction[1:5], targets[candidates, 1:5]).argmax()
        best_target = candidates[best_local_index]
        if box_iou(prediction[1:5], targets[[best_target], 1:5])[0] >= MATCH_IOU:
            matched_targets.add(int(best_target))
            true_positives += 1
    false_positives = len(predictions) - true_positives
    false_negatives = len(targets) - true_positives
    return true_positives, false_positives, false_negatives


def save_plot(path: Path, result) -> None:
    """Save Ultralytics' BGR annotation image as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.plot()[:, :, ::-1]).save(path)


def main() -> None:
    """Predict each TIFF and create per-image outputs plus error-analysis reports."""
    validate_config()
    image_paths = find_tiffs(INPUT_DIR)
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到 TIFF：{INPUT_DIR}")

    output_images = OUTPUT_DIR / "预测预览图"
    output_labels = OUTPUT_DIR / "预测标签"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"发现 {len(image_paths)} 张 TIFF，正在加载模型：{WEIGHTS_PATH.name}")
    model = YOLO(str(WEIGHTS_PATH))

    per_image_rows = []
    detection_rows = []
    no_prediction_images = []
    missed_target_images = []
    false_positive_images = []
    total_tp = total_fp = total_fn = 0

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(INPUT_DIR)
        rgb = read_rgb_tif(image_path)
        result = model.predict(
            source=rgb[:, :, ::-1].copy(),  # Ultralytics treats numpy input as BGR.
            imgsz=MODEL_IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            quantize=16 if USE_HALF and DEVICE != "cpu" else None,
            classes=[0],
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or not len(boxes):
            predictions = np.empty((0, 6), dtype=np.float32)
        else:
            predictions = np.column_stack(
                (
                    boxes.cls.detach().cpu().numpy(),
                    boxes.xyxy.detach().cpu().numpy(),
                    boxes.conf.detach().cpu().numpy(),
                )
            )
        label_path = LABEL_DIR / relative_path.with_suffix(".txt") if LABEL_DIR is not None else None
        targets = read_labels(label_path, rgb.shape[1], rgb.shape[0])
        tp, fp, fn = match_predictions(predictions, targets)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        save_plot((output_images / relative_path).with_suffix(".png"), result)
        result.save_txt((output_labels / relative_path).with_suffix(".txt"), save_conf=True)
        if not len(predictions):
            no_prediction_images.append(str(relative_path))
        if fn:
            missed_target_images.append(str(relative_path))
        if fp:
            false_positive_images.append(str(relative_path))
        per_image_rows.append(
            {
                "image": str(relative_path),
                "targets": len(targets),
                "predictions": len(predictions),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        for class_id, x1, y1, x2, y2, confidence in predictions:
            detection_rows.append(
                {
                    "image": str(relative_path),
                    "class_id": int(class_id),
                    "confidence": f"{confidence:.6f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                }
            )
        print(
            f"[{index}/{len(image_paths)}] {relative_path} | "
            f"预测={len(predictions)} TP/FP/FN={tp}/{fp}/{fn}"
        )

    with (OUTPUT_DIR / "逐图统计.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "targets", "predictions", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(per_image_rows)
    with (OUTPUT_DIR / "检测框.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        writer.writerows(detection_rows)
    for name, images in (
        ("无预测图像.txt", no_prediction_images),
        ("存在漏检图像.txt", missed_target_images),
        ("存在误检图像.txt", false_positive_images),
    ):
        (OUTPUT_DIR / name).write_text("\n".join(images) + ("\n" if images else ""), encoding="utf-8")

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "weights": str(WEIGHTS_PATH),
        "input_dir": str(INPUT_DIR),
        "label_dir": str(LABEL_DIR) if LABEL_DIR else None,
        "model_image_size": MODEL_IMAGE_SIZE,
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
        "matching_iou": MATCH_IOU,
        "images": len(image_paths),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "no_prediction_images": len(no_prediction_images),
        "missed_target_images": len(missed_target_images),
        "false_positive_images": len(false_positive_images),
    }
    (OUTPUT_DIR / "汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== 小 TIFF 预测与误差分析完成 ==========")
    print(f"TP/FP/FN: {total_tp}/{total_fp}/{total_fn}")
    print(f"Precision/Recall/F1: {precision:.6f}/{recall:.6f}/{f1:.6f}")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
