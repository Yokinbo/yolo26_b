"""验证 YOLO26，使用统一阈值扫描输出论文指标、推荐置信度和效率指标。"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

from experiment_config import (
    MODEL_IMAGE_SIZE,
    MODEL_TAG,
    ensure_loaded_model_size,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import get_flops


# =============================================================================
# 用户验证参数配置区（全部使用绝对路径）
# =============================================================================
WEIGHTS_PATH = Path(r"G:\b\模型训练结果\yolo26\yolo26s_240epochs_640\weights\best_map50.pt")
DATASET_CONFIG = Path(r"E:\YOLO\yolo26\ultralytics\cfg\datasets\a_myhdc.yaml")

IMAGE_SIZE = MODEL_IMAGE_SIZE
VAL_BATCH_SIZE = 6
DEVICE = 0
WORKERS = 2

# AP 使用低阈值保留完整预测；P/R/F1 使用与 D-FINE 一致的显式阈值扫描。
AP_CONFIDENCE = 0.001
FIXED_CONFIDENCE = 0.50
MATCH_IOU = 0.50
CONFIDENCE_START = 0.00
CONFIDENCE_END = 1.00
CONFIDENCE_STEP = 0.01
NMS_IOU = 0.70
MAX_DETECTIONS = 300
PLOTS = True

# 单张图像效率测试：模型前向 + YOLO 检测后处理，不含磁盘读取和 DataLoader 变换。
ENABLE_FPS_BENCHMARK = True
EFFICIENCY_CONFIDENCE = 0.50
FPS_WARMUP_ITERS = 10
FPS_TEST_ITERS = 100

# 验证结果输出根目录：请直接填写 Windows 绝对路径。
PROJECT = Path(r"E:\YOLO\yolo26\runs\val")
# RUN_NAME：本次验证的输出子目录名，最终结果保存在 PROJECT / RUN_NAME。
# 可自定义且不影响精度计算；不要填写绝对路径。EXIST_OK=False 时重名会自动加 -2、-3。
RUN_NAME = "第四版数据集793_map50"
# PAPER_MODEL_NAME：只用于输出报告/论文表格中的 Model 名称，不会选择模型或权重。
# 可自定义，但必须如实对应已加载权重的规模和 IMAGE_SIZE，防止论文表格误标。
PAPER_MODEL_NAME = "yolo26s_512"
EXIST_OK = False
# =============================================================================


class ConfidenceSweepValidator(DetectionValidator):
    """在官方验证过程中保留逐图预测与真值，供统一阈值扫描使用。"""

    latest_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_records: list[dict[str, torch.Tensor]] = []
        self.benchmark_image: torch.Tensor | None = None
        type(self).latest_instance = self

    def update_metrics(self, preds, batch) -> None:
        if self.benchmark_image is None:
            self.benchmark_image = batch["img"][:1].detach().clone()

        for index, pred in enumerate(preds):
            prepared_batch = self._prepare_batch(index, batch)
            prepared_pred = self._prepare_pred(pred)
            self.confidence_records.append(
                {
                    "pred_boxes": prepared_pred["bboxes"].detach().cpu(),
                    "pred_scores": prepared_pred["conf"].detach().cpu(),
                    "pred_labels": prepared_pred["cls"].detach().cpu(),
                    "gt_boxes": prepared_batch["bboxes"].detach().cpu(),
                    "gt_labels": prepared_batch["cls"].detach().cpu(),
                }
            )
        super().update_metrics(preds, batch)


def format_elapsed_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_inputs() -> None:
    required_files = {"模型权重": WEIGHTS_PATH, "数据集配置": DATASET_CONFIG}
    missing = [f"{name}：{path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下验证文件不存在：\n" + "\n".join(missing))
    if IMAGE_SIZE <= 0 or IMAGE_SIZE % 32:
        raise ValueError("IMAGE_SIZE 必须是能被 32 整除的正整数。")
    if VAL_BATCH_SIZE <= 0 or WORKERS < 0 or MAX_DETECTIONS <= 0:
        raise ValueError("VAL_BATCH_SIZE、MAX_DETECTIONS 必须大于 0，WORKERS 不能小于 0。")
    if FPS_WARMUP_ITERS < 0 or FPS_TEST_ITERS <= 0:
        raise ValueError("FPS_WARMUP_ITERS 必须≥0，FPS_TEST_ITERS 必须>0。")
    thresholds = (
        AP_CONFIDENCE,
        FIXED_CONFIDENCE,
        MATCH_IOU,
        CONFIDENCE_START,
        CONFIDENCE_END,
        EFFICIENCY_CONFIDENCE,
        NMS_IOU,
    )
    if not all(0 <= value <= 1 for value in thresholds):
        raise ValueError("置信度和 IoU 阈值必须在 [0, 1] 范围内。")
    if CONFIDENCE_STEP <= 0 or CONFIDENCE_START >= CONFIDENCE_END:
        raise ValueError("置信度扫描范围或步长无效。")
    if MATCH_IOU != 0.50:
        raise ValueError("为保持论文统一口径，MATCH_IOU 必须保持 0.50。")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 使用 GPU，但当前 PyTorch 没有检测到可用 CUDA。")


def metrics_at_threshold(records: list[dict[str, torch.Tensor]], threshold: float) -> dict:
    """在给定置信度和 IoU 下逐图执行类别一致的一对一匹配。"""
    true_positives = false_positives = false_negatives = 0

    for record in records:
        keep = record["pred_scores"] >= threshold
        pred_boxes = record["pred_boxes"][keep]
        pred_labels = record["pred_labels"][keep]
        gt_boxes = record["gt_boxes"]
        gt_labels = record["gt_labels"]
        num_predictions, num_targets = len(pred_boxes), len(gt_boxes)

        if not num_predictions:
            false_negatives += num_targets
            continue
        if not num_targets:
            false_positives += num_predictions
            continue

        ious = box_iou(gt_boxes, pred_boxes)
        valid = (ious >= MATCH_IOU) & (gt_labels[:, None] == pred_labels[None, :])
        gt_indices, pred_indices = torch.nonzero(valid, as_tuple=True)
        matched_predictions: set[int] = set()
        matched_targets: set[int] = set()

        if gt_indices.numel():
            values = ious[gt_indices, pred_indices]
            order = torch.argsort(values, descending=True)
            for item in order.tolist():
                gt_index = int(gt_indices[item])
                pred_index = int(pred_indices[item])
                if gt_index in matched_targets or pred_index in matched_predictions:
                    continue
                matched_targets.add(gt_index)
                matched_predictions.add(pred_index)

        matches = len(matched_predictions)
        true_positives += matches
        false_positives += num_predictions - matches
        false_negatives += num_targets - matches

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confidence": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TPs": int(true_positives),
        "FPs": int(false_positives),
        "FNs": int(false_negatives),
    }


def smooth_curve(values: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """使用与 Ultralytics 和 D-FINE 当前脚本一致的箱式平滑。"""
    filter_size = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(filter_size // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]))
    return np.convolve(padded, np.ones(filter_size) / filter_size, mode="valid")


def compute_confidence_metrics(records: list[dict[str, torch.Tensor]]) -> dict:
    """按0.01步长扫描置信度，并返回平滑F1最大点和固定阈值指标。"""
    count = int(round((CONFIDENCE_END - CONFIDENCE_START) / CONFIDENCE_STEP)) + 1
    thresholds = np.linspace(CONFIDENCE_START, CONFIDENCE_END, count)
    curve = [metrics_at_threshold(records, float(value)) for value in thresholds]
    smoothed_f1 = smooth_curve(np.asarray([item["f1"] for item in curve], dtype=float))
    best = curve[int(smoothed_f1.argmax())].copy()
    fixed_index = int(np.abs(thresholds - FIXED_CONFIDENCE).argmin())
    return {
        "iou_threshold": MATCH_IOU,
        "selection_rule": "maximum 10%-smoothed F1-confidence curve",
        "scan_start": CONFIDENCE_START,
        "scan_end": CONFIDENCE_END,
        "scan_step": CONFIDENCE_STEP,
        "recommended_confidence": best["confidence"],
        "best_f1_metrics": best,
        "fixed_threshold_metrics": curve[fixed_index].copy(),
        "curve": curve,
    }


def benchmark_single_image_fps(model: YOLO, validator: ConfidenceSweepValidator) -> dict:
    """在统一效率置信度下测量单张图像的模型前向与NMS后处理速度。"""
    if not ENABLE_FPS_BENCHMARK or validator.benchmark_image is None:
        return {}

    network = model.model.eval()
    parameter = next(network.parameters())
    image = validator.benchmark_image.to(device=parameter.device, dtype=parameter.dtype)

    def synchronize() -> None:
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)

    original_confidence = validator.args.conf
    validator.args.conf = EFFICIENCY_CONFIDENCE
    try:
        with torch.inference_mode():
            for _ in range(FPS_WARMUP_ITERS):
                validator.postprocess(network(image))
            synchronize()
            started = time.perf_counter()
            for _ in range(FPS_TEST_ITERS):
                validator.postprocess(network(image))
            synchronize()
    finally:
        validator.args.conf = original_confidence

    seconds_per_image = (time.perf_counter() - started) / FPS_TEST_ITERS
    return {
        "FPS_single_image_forward_post": 1.0 / seconds_per_image,
        "latency_ms_single_image_forward_post": seconds_per_image * 1000.0,
        "warmup_iterations": FPS_WARMUP_ITERS,
        "test_iterations": FPS_TEST_ITERS,
        "efficiency_confidence": EFFICIENCY_CONFIDENCE,
    }


def save_confidence_curve_outputs(result_dir: Path, confidence_metrics: dict) -> None:
    curve = confidence_metrics["curve"]
    csv_path = result_dir / "confidence_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib 未安装，已保存置信度 CSV：{csv_path}")
        return

    confidence = [item["confidence"] for item in curve]
    figure, axis = plt.subplots(figsize=(9, 6), dpi=180, constrained_layout=True)
    for key, label, color in (
        ("precision", "Precision", "#1f77b4"),
        ("recall", "Recall", "#ff7f0e"),
        ("f1", "F1", "#2ca02c"),
    ):
        axis.plot(confidence, [item[key] for item in curve], label=label, color=color)
    recommended = confidence_metrics["recommended_confidence"]
    axis.axvline(recommended, color="#d62728", linestyle="--", label=f"Best F1 @ {recommended:.2f}")
    axis.set(xlabel="Confidence threshold", ylabel="Metric", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=0.28, linestyle="--")
    axis.legend()
    figure.savefig(result_dir / "confidence_precision_recall_f1.png", bbox_inches="tight")
    plt.close(figure)


def save_paper_metrics(
    result_dir: Path,
    fixed: dict,
    coco_metrics: dict,
    model_parameters: int,
    flops_g: float,
    speed_metrics: dict,
) -> Path:
    """生成可以直接复制到论文或表格软件中的两张横向指标表。"""
    latency = speed_metrics.get("latency_ms_single_image_forward_post")
    fps = speed_metrics.get("FPS_single_image_forward_post")
    flops_text = f"{flops_g:.3f}" if flops_g > 0 else "N/A"
    latency_text = f"{latency:.4f}" if latency is not None else "N/A"
    fps_text = f"{fps:.4f}" if fps is not None else "N/A"

    accuracy_header = (
        f"{'Model':<16}{'Input':>8}{'Params(M)':>14}{'AP50':>12}{'AP75':>12}"
        f"{'mAP50:95':>14}{'P@0.5':>12}{'R@0.5':>12}{'F1@0.5':>12}"
    )
    accuracy_row = (
        f"{PAPER_MODEL_NAME:<16}{IMAGE_SIZE:>8}{model_parameters / 1e6:>14.3f}"
        f"{coco_metrics['AP@0.5']:>12.4f}{coco_metrics['AP@0.75']:>12.4f}"
        f"{coco_metrics['mAP@0.5:0.95']:>14.4f}{fixed['precision']:>12.4f}"
        f"{fixed['recall']:>12.4f}{fixed['f1']:>12.4f}"
    )
    efficiency_header = (
        f"{'Model':<16}{'Input':>8}{'Params(M)':>14}{'FLOPs(G)':>14}"
        f"{'Latency(ms/image)':>22}{'FPS':>14}"
    )
    efficiency_row = (
        f"{PAPER_MODEL_NAME:<16}{IMAGE_SIZE:>8}{model_parameters / 1e6:>14.3f}"
        f"{flops_text:>14}{latency_text:>22}{fps_text:>14}"
    )

    lines = [
        "YOLO26 论文对比实验指标",
        "=" * 104,
        "一、精度对比表",
        accuracy_header,
        "-" * len(accuracy_header),
        accuracy_row,
        "",
        "二、效率对比表",
        efficiency_header,
        "-" * len(efficiency_header),
        efficiency_row,
        "",
        "评价口径：",
        f"1. P@0.5、R@0.5、F1@0.5：置信度≥{FIXED_CONFIDENCE:.2f}，匹配IoU≥{MATCH_IOU:.2f}。",
        "2. AP50、AP75、mAP50:95：低置信度完整预测曲线上的检测AP。",
        f"3. FLOPs：未融合原始模型，batch=1，输入{IMAGE_SIZE}×{IMAGE_SIZE}。",
        f"4. Latency与FPS：batch=1，置信度={EFFICIENCY_CONFIDENCE:.2f}，模型前向+NMS后处理。",
        f"5. 测速：预热{FPS_WARMUP_ITERS}次，正式测试{FPS_TEST_ITERS}次，不含磁盘读取和DataLoader变换。",
        "",
        "制表提示：以上列使用固定宽度排版；也可按列复制到 Word 或 Excel。",
    ]
    path = result_dir / "论文指标.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_reports(
    metrics,
    confidence_metrics: dict,
    speed_metrics: dict,
    flops_g: float,
    model_parameters: int,
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    save_dir = Path(metrics.save_dir)
    framework_speed = {name: float(value) for name, value in metrics.speed.items()}
    coco_metrics = {
        "mAP@0.5:0.95": float(metrics.box.map),
        "AP@0.5": float(metrics.box.map50),
        "AP@0.75": float(metrics.box.map75),
    }
    fixed = confidence_metrics["fixed_threshold_metrics"]
    best = confidence_metrics["best_f1_metrics"]
    report = {
        "weights": str(WEIGHTS_PATH),
        "dataset": str(DATASET_CONFIG),
        "input_size": IMAGE_SIZE,
        "target_count": int(sum(metrics.nt_per_class)),
        "paper_core_metrics": {
            "Precision": fixed["precision"],
            "Recall": fixed["recall"],
            "F1-score": fixed["f1"],
            "mAP@0.5": coco_metrics["AP@0.5"],
        },
        "coco_metrics": coco_metrics,
        "threshold_metrics": fixed,
        "confidence_metrics": confidence_metrics,
        "speed_metrics": speed_metrics,
        "efficiency_confidence": EFFICIENCY_CONFIDENCE,
        "framework_speed_ms_per_image": framework_speed,
        "model_parameters": model_parameters,
        "FLOPs_G": flops_g if flops_g > 0 else None,
        "validation_elapsed_seconds": elapsed_seconds,
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    finished_at = datetime.now().astimezone()
    lines = [
        "YOLO26 火电厂目标检测验证精度报告",
        "=" * 52,
        f"模型权重：{WEIGHTS_PATH}",
        f"数据配置：{DATASET_CONFIG}",
        f"输入尺寸：{IMAGE_SIZE} × {IMAGE_SIZE}",
        f"目标数量：{report['target_count']}",
        "",
        "一、论文核心指标",
        f"说明：P、R、F1使用置信度阈值{FIXED_CONFIDENCE:.2f}、IoU阈值{MATCH_IOU:.2f}的匹配结果。",
        f"Precision (P)                 : {fixed['precision']:.4f}",
        f"Recall (R)                    : {fixed['recall']:.4f}",
        f"F1-score                      : {fixed['f1']:.4f}",
        f"mAP@0.5                       : {coco_metrics['AP@0.5']:.4f}",
        f"推荐最佳置信度              : {confidence_metrics['recommended_confidence']:.2f}",
        f"该置信度下 F1               : {best['f1']:.4f}",
        "",
        "二、COCO 检测精度指标",
        *[f"{name:30s}: {value:.4f}" for name, value in coco_metrics.items()],
        "",
        f"三、阈值检测统计（置信度≥{FIXED_CONFIDENCE:.2f}，IoU≥{MATCH_IOU:.2f}）",
        *[f"{name:30s}: {value:.4f}" for name, value in fixed.items()],
        "",
        "四、效率参考指标",
        f"模型参数量 (M)                 : {model_parameters / 1e6:.3f}",
        f"FLOPs (G)                       : {flops_g:.3f}" if flops_g > 0 else "FLOPs (G)                       : N/A（未安装 ultralytics-thop）",
        f"效率测试置信度                  : {EFFICIENCY_CONFIDENCE:.2f}",
        *[f"{name:30s}: {value:.4f}" for name, value in speed_metrics.items()],
        "",
        "五、验证耗时",
        f"开始时间：{started_at.isoformat(timespec='seconds')}",
        f"结束时间：{finished_at.isoformat(timespec='seconds')}",
        f"总耗时：{format_elapsed_time(elapsed_seconds)}",
        f"总秒数：{elapsed_seconds:.2f}",
        f"结果目录：{save_dir}",
        "",
        "速度说明：FPS为单张图像的模型前向+NMS后处理速度，不含磁盘读取和DataLoader变换。",
        "FLOPs说明：按单张、当前输入尺寸统计一次前向传播的浮点运算量。",
    ]
    (save_dir / "精度指标报告.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_confidence_curve_outputs(save_dir, confidence_metrics)
    save_paper_metrics(save_dir, fixed, coco_metrics, model_parameters, flops_g, speed_metrics)
    return save_dir


def main() -> None:
    validate_inputs()
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()

    print("\n========== YOLO26 模型验证配置 ==========")
    print(f"共享模型选择：{MODEL_TAG}")
    print(f"模型权重：{WEIGHTS_PATH}")
    print(f"数据配置：{DATASET_CONFIG}")
    print(f"输入尺寸：{IMAGE_SIZE} | Batch：{VAL_BATCH_SIZE} | 设备：{DEVICE}")
    print(f"阈值扫描：{CONFIDENCE_START:.2f}～{CONFIDENCE_END:.2f}，步长 {CONFIDENCE_STEP:.2f}")
    print(f"输出目录：{PROJECT / RUN_NAME}\n")

    model = YOLO(WEIGHTS_PATH)
    ensure_loaded_model_size(model)
    # 验证器会就地融合 Conv/BN，因此参数量和 FLOPs 必须在评估前固定。
    model_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    flops_g = float(get_flops(model.model, imgsz=IMAGE_SIZE))
    metrics = model.val(
        validator=ConfidenceSweepValidator,
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
    validator = ConfidenceSweepValidator.latest_instance
    if validator is None or not validator.confidence_records:
        raise RuntimeError("未能取得验证预测，无法计算统一置信度曲线。")

    confidence_metrics = compute_confidence_metrics(validator.confidence_records)
    speed_metrics = benchmark_single_image_fps(model, validator)
    elapsed_seconds = time.perf_counter() - start_time
    save_dir = save_reports(
        metrics,
        confidence_metrics,
        speed_metrics,
        flops_g,
        model_parameters,
        started_at,
        elapsed_seconds,
    )

    fixed = confidence_metrics["fixed_threshold_metrics"]
    best = confidence_metrics["best_f1_metrics"]
    print("\n========== 固定阈值精度（论文主表） ==========")
    print(f"置信度 / 匹配IoU：{FIXED_CONFIDENCE:.2f} / {MATCH_IOU:.2f}")
    print(f"Precision：{fixed['precision']:.6f}")
    print(f"Recall：{fixed['recall']:.6f}")
    print(f"F1-score：{fixed['f1']:.6f}")
    print(f"TP / FP / FN：{fixed['TPs']} / {fixed['FPs']} / {fixed['FNs']}")
    print("\n========== COCO 标准精度 ==========")
    print(f"mAP@0.5：{metrics.box.map50:.6f}")
    print(f"mAP@0.75：{metrics.box.map75:.6f}")
    print(f"mAP@0.5:0.95：{metrics.box.map:.6f}")
    print("\n========== 验证集最佳 F1 工作点 ==========")
    print(f"推荐置信度：{confidence_metrics['recommended_confidence']:.2f}")
    print(f"Precision：{best['precision']:.6f}")
    print(f"Recall：{best['recall']:.6f}")
    print(f"F1-score：{best['f1']:.6f}")
    print(f"TP / FP / FN：{best['TPs']} / {best['FPs']} / {best['FNs']}")
    print("\n========== 效率指标 ==========")
    print(f"参数量：{model_parameters / 1e6:.3f} M")
    print(f"FLOPs：{flops_g:.3f} G" if flops_g > 0 else "FLOPs：N/A（请安装 ultralytics-thop）")
    for name, value in speed_metrics.items():
        print(f"{name}：{value:.4f}")
    print(f"验证总耗时：{format_elapsed_time(elapsed_seconds)}")
    print(f"完整结果：{save_dir}")


if __name__ == "__main__":
    main()
