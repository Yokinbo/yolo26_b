"""验证训练完成的 YOLO26 目标检测模型并保存精度报告。.

修改下方“用户验证参数配置区”后直接运行：
    python valid.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from experiment_config import MODEL_IMAGE_SIZE

# 当前环境中的 Transformers 与 YOLO26 检测无关，隐藏其 PyTorch 版本提示。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch

from ultralytics import YOLO
from ultralytics.utils.metrics import smooth

# =============================================================================
# 用户验证参数配置区（全部使用绝对路径）
# =============================================================================
# 应用验证优先使用 mAP@0.5 最佳的 best_map50.pt；best.pt 保留为官方 mAP@0.5:0.95 最佳权重。
WEIGHTS_PATH = Path(r"E:\YOLO\yolo26\runs\train\yolo26m_240epochs_640_第四版数据集793_map50\weights\best_map50.pt")
DATASET_CONFIG = Path(r"E:\YOLO\yolo26\ultralytics\cfg\datasets\a_myhdc.yaml")

# 应与训练和正式推理使用的尺寸保持一致。
IMAGE_SIZE = MODEL_IMAGE_SIZE
VAL_BATCH_SIZE = 6
DEVICE = 0  # 单张 NVIDIA 显卡用 0；CPU 用 "cpu"
WORKERS = 2

# 第一遍使用官方低阈值生成完整 PR 曲线和 COCO mAP；第二遍使用固定阈值与 D-FINE 对齐 P/R/F1。
AP_CONFIDENCE = 0.001
FIXED_CONFIDENCE = 0.50
FIXED_MATCH_IOU = 0.50
NMS_IOU = 0.7
MAX_DETECTIONS = 300
PLOTS = True

# 验证输出与训练输出分开保存。
PROJECT = Path(r"E:\YOLO\yolo26\runs\val")
RUN_NAME = "yolo26m_240epochs_640_第四版数据集793_map50"
EXIST_OK = False  # False：同名目录存在时自动创建带数字后缀的新目录
# =============================================================================


def format_elapsed_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS。."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_inputs() -> None:
    """在加载模型前检查路径和常用参数。."""
    required_files = {
        "模型权重": WEIGHTS_PATH,
        "数据集配置": DATASET_CONFIG,
    }
    missing = [f"{name}：{path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下验证文件不存在：\n" + "\n".join(missing))
    if IMAGE_SIZE <= 0 or IMAGE_SIZE % 32:
        raise ValueError("IMAGE_SIZE 必须是能被 32 整除的正整数。")
    if VAL_BATCH_SIZE <= 0 or WORKERS < 0 or MAX_DETECTIONS <= 0:
        raise ValueError("VAL_BATCH_SIZE、MAX_DETECTIONS 必须大于 0，WORKERS 不能小于 0。")
    thresholds = (AP_CONFIDENCE, FIXED_CONFIDENCE, FIXED_MATCH_IOU, NMS_IOU)
    if not all(0 <= value <= 1 for value in thresholds):
        raise ValueError("置信度和 IoU 阈值必须在 [0, 1] 范围内。")
    if FIXED_MATCH_IOU != 0.50:
        raise ValueError("当前固定阈值统计使用验证器的 IoU=0.50 匹配列，FIXED_MATCH_IOU 必须保持 0.50。")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 使用 GPU，但当前 PyTorch 没有检测到可用 CUDA。")


def fixed_threshold_metrics(metrics) -> dict:
    """汇总固定置信度、IoU=0.50 匹配条件下的 TP、FP、FN、P、R 和 F1。."""
    image_metrics = metrics.box.image_metrics.values()
    tp = sum(item["tp"] for item in image_metrics)
    fp = sum(item["fp"] for item in metrics.box.image_metrics.values())
    fn = sum(item["fn"] for item in metrics.box.image_metrics.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confidence_threshold": FIXED_CONFIDENCE,
        "matching_iou_threshold": FIXED_MATCH_IOU,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def optimal_f1_confidence(metrics) -> float:
    """Return the confidence at the smoothed maximum-F1 operating point."""
    if not len(metrics.box.f1_curve) or not len(metrics.box.px):
        return 0.0
    index = smooth(metrics.box.f1_curve.mean(0), 0.1).argmax()
    return float(metrics.box.px[index])


def save_reports(model: YOLO, metrics, threshold_metrics: dict, started_at: datetime, elapsed_seconds: float) -> Path:
    """在 Ultralytics 实际验证目录中保存 JSON 和中文 TXT 报告。."""
    save_dir = Path(metrics.save_dir)
    model_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    speed = {name: float(value) for name, value in metrics.speed.items()}
    coco_metrics = {
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.75": float(metrics.box.map75),
        "mAP@0.5:0.95": float(metrics.box.map),
    }
    optimal_metrics = {
        "Confidence": optimal_f1_confidence(metrics),
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
        "F1-score": float(metrics.box.f1.mean()) if len(metrics.box.f1) else 0.0,
    }

    report = {
        "weights": str(WEIGHTS_PATH),
        "dataset": str(DATASET_CONFIG),
        "input_size": IMAGE_SIZE,
        "targets_per_class": [int(value) for value in metrics.nt_per_class],
        "fixed_threshold_metrics": threshold_metrics,
        "coco_metrics": coco_metrics,
        "ultralytics_optimal_f1_point_metrics": optimal_metrics,
        "speed_ms_per_image": speed,
        "model_parameters": model_parameters,
        "validation_elapsed_seconds": elapsed_seconds,
    }
    json_path = save_dir / "metrics.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    finished_at = datetime.now().astimezone()
    txt_path = save_dir / "精度指标报告.txt"
    lines = [
        "YOLO26 HDC 目标检测验证报告",
        "=" * 52,
        f"模型权重：{WEIGHTS_PATH}",
        f"数据配置：{DATASET_CONFIG}",
        f"输入尺寸：{IMAGE_SIZE} × {IMAGE_SIZE}",
        f"目标数量：{sum(report['targets_per_class'])}",
        "",
        "一、论文主表固定阈值指标（与 D-FINE 口径一致）",
        f"置信度阈值           : {FIXED_CONFIDENCE:.2f}",
        f"匹配 IoU 阈值        : {FIXED_MATCH_IOU:.2f}",
        *[
            f"{name:20s}: {value:.6f}" if isinstance(value, float) else f"{name:20s}: {value}"
            for name, value in threshold_metrics.items()
            if name not in {"confidence_threshold", "matching_iou_threshold"}
        ],
        "",
        "二、COCO 标准 AP 指标",
        *[f"{name:20s}: {value:.6f}" for name, value in coco_metrics.items()],
        "说明：mAP@0.5:0.95 是 IoU=0.50 至 0.95、步长 0.05 的平均 AP。",
        "",
        "三、Ultralytics 最优 F1 工作点指标（仅作补充，不与 D-FINE 固定阈值指标直接比较）",
        *[f"{name:20s}: {value:.6f}" for name, value in optimal_metrics.items()],
        "",
        "四、每张图片平均速度",
        *[f"{name:20s}: {value:.4f} ms" for name, value in speed.items()],
        f"模型参数量           : {model_parameters / 1e6:.3f} M",
        "",
        "五、验证耗时",
        f"开始时间：{started_at.isoformat(timespec='seconds')}",
        f"结束时间：{finished_at.isoformat(timespec='seconds')}",
        f"总耗时：{format_elapsed_time(elapsed_seconds)}",
        f"总秒数：{elapsed_seconds:.2f}",
        f"结果目录：{save_dir}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_dir


def main() -> None:
    """加载最佳权重，在验证集计算并保存检测精度指标。."""
    validate_inputs()
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()

    print("\n========== YOLO26 模型验证配置 ==========")
    print(f"模型权重：{WEIGHTS_PATH}")
    print(f"数据配置：{DATASET_CONFIG}")
    print(f"输入尺寸：{IMAGE_SIZE} | Batch：{VAL_BATCH_SIZE} | 设备：{DEVICE}")
    print(f"输出目录：{PROJECT / RUN_NAME}\n")

    model = YOLO(WEIGHTS_PATH)
    metrics = model.val(
        data=DATASET_CONFIG,
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        conf=AP_CONFIDENCE,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        plots=PLOTS,
        save_json=False,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=EXIST_OK,
    )

    fixed_metrics = model.val(
        data=DATASET_CONFIG,
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        conf=FIXED_CONFIDENCE,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        plots=False,
        save_json=False,
        project=Path(metrics.save_dir).parent,
        name=Path(metrics.save_dir).name,
        exist_ok=True,
    )
    threshold_metrics = fixed_threshold_metrics(fixed_metrics)

    elapsed_seconds = time.perf_counter() - start_time
    save_dir = save_reports(model, metrics, threshold_metrics, started_at, elapsed_seconds)

    print("\n========== 固定阈值精度（论文主表） ==========")
    print(f"置信度阈值     ：{FIXED_CONFIDENCE:.2f}")
    print(f"匹配 IoU 阈值  ：{FIXED_MATCH_IOU:.2f}")
    print(f"Precision      ：{threshold_metrics['Precision']:.6f}")
    print(f"Recall         ：{threshold_metrics['Recall']:.6f}")
    print(f"F1-score       ：{threshold_metrics['F1-score']:.6f}")
    print(f"TP / FP / FN   ：{threshold_metrics['TP']} / {threshold_metrics['FP']} / {threshold_metrics['FN']}")
    print("\n========== COCO 标准精度 ==========")
    print(f"mAP@0.5        ：{metrics.box.map50:.6f}")
    print(f"mAP@0.75       ：{metrics.box.map75:.6f}")
    print(f"mAP@0.5:0.95   ：{metrics.box.map:.6f}")
    print(f"最优 F1 置信度 ：{optimal_f1_confidence(metrics):.6f}")
    print(f"验证总耗时     ：{format_elapsed_time(elapsed_seconds)}")
    print(f"完整结果已保存 ：{save_dir}")


if __name__ == "__main__":
    main()
