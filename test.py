"""YOLO26 火电厂目标检测独立测试集评估脚本。

用途：模型结构、训练轮数、最佳权重和评价规则在验证集上全部确定后，
仅使用独立 test 集生成论文最终核心对比指标。

本脚本与当前 D-FINE ``test.py`` 保持相同的论文口径：

* AP50、AP75、mAP50:95 使用低置信度候选形成完整 PR 曲线；
* P、R、F1 固定使用 confidence=0.50、matching IoU=0.50；
* 不在 test 集扫描或选择最佳置信度；
* 效率测试为 batch=1 的模型前向+官方检测后处理，不含数据读取。

在仓库根目录运行：

    python test.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from experiment_config import MODEL_IMAGE_SIZE, MODEL_TAG, ensure_loaded_model_size

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import get_flops

# =============================================================================
# 用户测试参数配置区
# =============================================================================

# 必须填写已经根据验证集 AP50 选定的权重，禁止根据 test 结果更换权重。
WEIGHTS_PATH = Path(r"E:\YOLO\yolo26\runs\train\yolo26m_240epochs_640_第四版数据集793_map50\weights\best_map50.pt")
DATASET_CONFIG = Path(r"E:\YOLO\yolo26\ultralytics\cfg\datasets\a_myhdc.yaml")

# 与训练和 valid.py 共用 experiment_config.py 中的输入尺寸。
IMAGE_SIZE = MODEL_IMAGE_SIZE
TEST_BATCH_SIZE = 6
DEVICE = 0
WORKERS = 2
SEED = 2026

# AP 必须保留低置信度候选；0.50 只用于固定工作点 P/R/F1。
AP_CONFIDENCE = 0.001
FIXED_EVALUATION_CONFIDENCE = 0.50
MATCH_IOU = 0.50
NMS_IOU = 0.70
MAX_DETECTIONS = 300
PLOTS = True

# batch=1，前向+官方检测后处理；输入张量已在 GPU，不含预处理和传输。
ENABLE_FPS_BENCHMARK = True
FPS_WARMUP_ITERS = 10
FPS_TEST_ITERS = 100

# 测试结果输出根目录：请直接填写 Windows 绝对路径。
TEST_OUTPUT_ROOT = Path(r"E:\YOLO\yolo26\runs\test")
# RUN_NAME：本次测试的输出子目录名，最终结果保存在 TEST_OUTPUT_ROOT / RUN_NAME。
# 可自定义且不影响精度计算；不要填写绝对路径。EXIST_OK=False 时重名会自动加 -2、-3。
RUN_NAME = "第四版数据集793_map50"
# PAPER_MODEL_NAME：只用于“测试集最终核心指标”表中的 Model 名称，不会选择模型或权重。
# 可自定义，但必须如实对应已加载权重的规模和 IMAGE_SIZE，防止论文表格误标。
PAPER_MODEL_NAME = "yolo26s_512"
EXIST_OK = False

# =============================================================================


class IndependentTestValidator(DetectionValidator):
    """在官方测试过程中保留逐图预测与真值，用于固定阈值统计。"""

    latest_instance = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.detection_records: list[dict[str, torch.Tensor]] = []
        self.benchmark_image: torch.Tensor | None = None
        type(self).latest_instance = self

    def update_metrics(self, preds, batch) -> None:
        if self.benchmark_image is None:
            self.benchmark_image = batch["img"][:1].detach().clone()

        for index, pred in enumerate(preds):
            prepared_batch = self._prepare_batch(index, batch)
            prepared_pred = self._prepare_pred(pred)
            self.detection_records.append(
                {
                    "pred_boxes": prepared_pred["bboxes"].detach().cpu(),
                    "pred_scores": prepared_pred["conf"].detach().cpu(),
                    "pred_labels": prepared_pred["cls"].detach().cpu(),
                    "gt_boxes": prepared_batch["bboxes"].detach().cpu(),
                    "gt_labels": prepared_batch["cls"].detach().cpu(),
                    "image_name": Path(prepared_batch["im_file"]).name,
                }
            )
        super().update_metrics(preds, batch)


def format_elapsed_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_dataset_paths() -> tuple[Path, Path]:
    """解析 YAML 的 test 路径，并返回测试图片和标签目录。"""
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8")) or {}
    test_entry = config.get("test")
    if not test_entry:
        raise ValueError(f"数据集 YAML 未配置 test 路径：{DATASET_CONFIG}")
    if isinstance(test_entry, (list, tuple)):
        if len(test_entry) != 1:
            raise ValueError("论文独立测试要求 YAML 的 test 只指向一个明确目录。")
        test_entry = test_entry[0]

    dataset_root = Path(config.get("path", DATASET_CONFIG.parent))
    if not dataset_root.is_absolute():
        dataset_root = (DATASET_CONFIG.parent / dataset_root).resolve()
    image_dir = Path(test_entry)
    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir
    image_dir = image_dir.resolve()

    if image_dir.name.lower() != "images":
        raise ValueError(f"test 路径必须指向 images 目录：{image_dir}")
    return image_dir, image_dir.parent / "labels"


def validate_inputs() -> tuple[Path, Path]:
    required_files = {"模型权重": WEIGHTS_PATH, "数据集配置": DATASET_CONFIG}
    missing = [f"{name}：{path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下测试文件不存在：\n" + "\n".join(missing))

    image_dir, label_dir = resolve_dataset_paths()
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"测试图片或标签目录不存在：\n{image_dir}\n{label_dir}")

    image_suffixes = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_stems = {
        path.stem.casefold()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    }
    label_stems = {path.stem.casefold() for path in label_dir.rglob("*.txt") if path.is_file()}
    if not image_stems:
        raise FileNotFoundError(f"测试目录中没有支持的图片：{image_dir}")
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    if missing_labels or orphan_labels:
        details = []
        if missing_labels:
            details.append(f"缺少标签：{missing_labels[:10]}")
        if orphan_labels:
            details.append(f"孤立标签：{orphan_labels[:10]}")
        raise ValueError("test 图片与标签不一一对应：\n" + "\n".join(details))

    if IMAGE_SIZE <= 0 or IMAGE_SIZE % 32:
        raise ValueError("IMAGE_SIZE 必须是能被 32 整除的正整数。")
    if TEST_BATCH_SIZE <= 0 or WORKERS < 0 or MAX_DETECTIONS <= 0:
        raise ValueError("TEST_BATCH_SIZE、MAX_DETECTIONS 必须大于0，WORKERS不能小于0。")
    if FPS_WARMUP_ITERS < 0 or FPS_TEST_ITERS <= 0:
        raise ValueError("FPS_WARMUP_ITERS必须≥0，FPS_TEST_ITERS必须>0。")
    if not all(
        0 <= value <= 1
        for value in (AP_CONFIDENCE, FIXED_EVALUATION_CONFIDENCE, MATCH_IOU, NMS_IOU)
    ):
        raise ValueError("置信度和IoU阈值必须位于[0, 1]。")
    if FIXED_EVALUATION_CONFIDENCE != 0.50 or MATCH_IOU != 0.50:
        raise ValueError("论文统一测试口径要求confidence=0.50、matching IoU=0.50。")
    if AP_CONFIDENCE >= FIXED_EVALUATION_CONFIDENCE:
        raise ValueError("AP_CONFIDENCE必须低于固定P/R/F1置信度。")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE使用GPU，但当前PyTorch没有检测到可用CUDA。")
    return image_dir, label_dir


def metrics_at_fixed_threshold(records: list[dict[str, torch.Tensor]]) -> dict[str, float | int]:
    """按D-FINE当前规则执行类别一致、IoU优先的一对一匹配。"""
    true_positives = false_positives = false_negatives = 0

    for record in records:
        keep = record["pred_scores"] >= FIXED_EVALUATION_CONFIDENCE
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
            for item in torch.argsort(values, descending=True).tolist():
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
        "confidence": FIXED_EVALUATION_CONFIDENCE,
        "matching_iou": MATCH_IOU,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TPs": int(true_positives),
        "FPs": int(false_positives),
        "FNs": int(false_negatives),
    }


def benchmark_single_image_fps(model: YOLO, validator: IndependentTestValidator) -> dict:
    """测量batch=1的模型前向和YOLO官方检测后处理速度。"""
    if not ENABLE_FPS_BENCHMARK or validator.benchmark_image is None:
        return {}

    network = model.model.eval()
    parameter = next(network.parameters())
    image = validator.benchmark_image.to(device=parameter.device, dtype=parameter.dtype)

    def synchronize() -> None:
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)

    original_confidence = validator.args.conf
    validator.args.conf = FIXED_EVALUATION_CONFIDENCE
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
        "latency_ms_single_image_forward_post": seconds_per_image * 1000.0,
        "FPS_single_image_forward_post": 1.0 / seconds_per_image,
        "warmup_iterations": FPS_WARMUP_ITERS,
        "test_iterations": FPS_TEST_ITERS,
        "efficiency_confidence": FIXED_EVALUATION_CONFIDENCE,
    }


def save_final_comparison_table(
    result_dir: Path,
    coco_metrics: dict,
    fixed_metrics: dict,
    model_parameters: int,
    flops_g: float,
    speed_metrics: dict,
) -> Path:
    latency = speed_metrics.get("latency_ms_single_image_forward_post")
    fps = speed_metrics.get("FPS_single_image_forward_post")
    flops_text = f"{flops_g:.3f}" if flops_g > 0 else "N/A"
    latency_text = f"{latency:.4f}" if latency is not None else "N/A"
    fps_text = f"{fps:.4f}" if fps is not None else "N/A"

    accuracy_header = (
        f"{'Model':<20}{'Input':>8}{'Params(M)':>14}{'AP50':>12}{'AP75':>12}"
        f"{'mAP50:95':>14}{'P@0.5':>12}{'R@0.5':>12}{'F1@0.5':>12}"
    )
    accuracy_row = (
        f"{PAPER_MODEL_NAME:<20}{IMAGE_SIZE:>8}{model_parameters / 1e6:>14.3f}"
        f"{coco_metrics['AP@0.5']:>12.4f}{coco_metrics['AP@0.75']:>12.4f}"
        f"{coco_metrics['mAP@0.5:0.95']:>14.4f}"
        f"{fixed_metrics['precision']:>12.4f}{fixed_metrics['recall']:>12.4f}"
        f"{fixed_metrics['f1']:>12.4f}"
    )
    efficiency_header = (
        f"{'Model':<20}{'Input':>8}{'Params(M)':>14}{'FLOPs(G)':>14}"
        f"{'Latency(ms/image)':>22}{'FPS':>14}"
    )
    efficiency_row = (
        f"{PAPER_MODEL_NAME:<20}{IMAGE_SIZE:>8}{model_parameters / 1e6:>14.3f}"
        f"{flops_text:>14}{latency_text:>22}{fps_text:>14}"
    )

    lines = [
        "YOLO26 独立测试集最终核心对比指标",
        "=" * 112,
        "一、测试集精度对比表",
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
        "1. 本表全部精度来自独立test集；test集未参与训练、选权重或调参。",
        "2. P@0.5、R@0.5、F1@0.5：置信度≥0.50，匹配IoU≥0.50。",
        "3. AP50、AP75、mAP50:95：低阈值候选上的COCO式101点检测AP。",
        f"4. FLOPs：Ultralytics未融合原始模型，batch=1，输入{IMAGE_SIZE}×{IMAGE_SIZE}。",
        "5. Latency与FPS：batch=1，FP32，模型前向+官方检测后处理；",
        "   不含磁盘读取、DataLoader、图像预处理及CPU到GPU传输。",
        f"6. 测速预热{FPS_WARMUP_ITERS}次，正式测试{FPS_TEST_ITERS}次。",
        "",
        "制表提示：将不同模型生成的数值行汇总，即可形成论文最终核心对比表。",
    ]
    path = result_dir / "测试集最终核心指标.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_reports(
    metrics,
    fixed_metrics: dict,
    speed_metrics: dict,
    flops_g: float,
    model_parameters: int,
    test_image_dir: Path,
    started_at: datetime,
    elapsed_seconds: float,
) -> tuple[Path, Path, Path]:
    result_dir = Path(metrics.save_dir)
    coco_metrics = {
        "mAP@0.5:0.95": float(metrics.box.map),
        "AP@0.5": float(metrics.box.map50),
        "AP@0.75": float(metrics.box.map75),
    }
    framework_speed = {name: float(value) for name, value in metrics.speed.items()}
    payload = {
        "evaluation_split": "test",
        "test_set_used_for_tuning": False,
        "weights": str(WEIGHTS_PATH),
        "dataset": str(DATASET_CONFIG),
        "test_images": str(test_image_dir),
        "input_size": IMAGE_SIZE,
        "ap_confidence": AP_CONFIDENCE,
        "nms_iou": NMS_IOU,
        "max_detections": MAX_DETECTIONS,
        "fixed_confidence": FIXED_EVALUATION_CONFIDENCE,
        "matching_iou": MATCH_IOU,
        "target_count": int(sum(metrics.nt_per_class)),
        "coco_metrics": coco_metrics,
        "threshold_metrics": fixed_metrics,
        "speed_metrics": speed_metrics,
        "framework_speed_ms_per_image": framework_speed,
        "model_parameters": model_parameters,
        "FLOPs_G": flops_g if flops_g > 0 else None,
        "test_elapsed_seconds": elapsed_seconds,
    }
    metrics_path = result_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    finished_at = datetime.now().astimezone()
    report_lines = [
        "火电厂目标检测独立测试集精度报告",
        "=" * 52,
        f"模型权重：{WEIGHTS_PATH}",
        f"数据配置：{DATASET_CONFIG}",
        f"测试影像：{test_image_dir}",
        f"输入尺寸：{IMAGE_SIZE} × {IMAGE_SIZE}",
        f"目标数量：{payload['target_count']}",
        "",
        "一、论文核心测试指标",
        "说明：P、R、F1固定使用置信度0.50、匹配IoU 0.50；不在test集搜索最佳阈值。",
        f"Precision (P)                 : {fixed_metrics['precision']:.4f}",
        f"Recall (R)                    : {fixed_metrics['recall']:.4f}",
        f"F1-score                      : {fixed_metrics['f1']:.4f}",
        f"AP@0.5                        : {coco_metrics['AP@0.5']:.4f}",
        f"AP@0.75                       : {coco_metrics['AP@0.75']:.4f}",
        f"mAP@0.5:0.95                  : {coco_metrics['mAP@0.5:0.95']:.4f}",
        "",
        "二、固定阈值检测统计",
        f"TPs                           : {fixed_metrics['TPs']}",
        f"FPs                           : {fixed_metrics['FPs']}",
        f"FNs                           : {fixed_metrics['FNs']}",
        "",
        "三、模型规模与效率",
        f"模型参数量 (M)                 : {model_parameters / 1e6:.3f}",
        f"FLOPs (G)                      : {flops_g:.3f}"
        if flops_g > 0
        else "FLOPs (G)                      : N/A（未安装ultralytics-thop）",
        *[f"{name:34s}: {value:.4f}" for name, value in speed_metrics.items()],
        "",
        "四、测试耗时",
        f"开始时间：{started_at.isoformat(timespec='seconds')}",
        f"结束时间：{finished_at.isoformat(timespec='seconds')}",
        f"总耗时：{format_elapsed_time(elapsed_seconds)}",
        f"总秒数：{elapsed_seconds:.2f}",
    ]
    report_path = result_dir / "测试集精度报告.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    table_path = save_final_comparison_table(
        result_dir,
        coco_metrics,
        fixed_metrics,
        model_parameters,
        flops_g,
        speed_metrics,
    )
    return metrics_path, report_path, table_path


def main() -> None:
    test_image_dir, _ = validate_inputs()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    print("\n========== YOLO26 独立测试集评估 ==========")
    print(f"共享模型选择：{MODEL_TAG}")
    print(f"模型权重：{WEIGHTS_PATH}")
    print(f"数据配置：{DATASET_CONFIG}")
    print(f"测试影像：{test_image_dir}")
    print(f"输入尺寸：{IMAGE_SIZE} | Batch：{TEST_BATCH_SIZE} | 设备：{DEVICE}")
    print(f"输出目录：{TEST_OUTPUT_ROOT / RUN_NAME}\n")

    model = YOLO(WEIGHTS_PATH)
    ensure_loaded_model_size(model)
    # 验证器会就地融合 Conv/BN，因此参数量和 FLOPs 必须在评估前固定。
    model_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    flops_g = float(get_flops(model.model, imgsz=IMAGE_SIZE))
    metrics = model.val(
        validator=IndependentTestValidator,
        data=DATASET_CONFIG,
        split="test",
        imgsz=IMAGE_SIZE,
        batch=TEST_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        conf=AP_CONFIDENCE,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        plots=PLOTS,
        save_json=False,
        project=TEST_OUTPUT_ROOT,
        name=RUN_NAME,
        exist_ok=EXIST_OK,
    )
    validator = IndependentTestValidator.latest_instance
    if validator is None or not validator.detection_records:
        raise RuntimeError("未取得测试集预测，无法计算固定阈值指标。")

    fixed_metrics = metrics_at_fixed_threshold(validator.detection_records)
    speed_metrics = benchmark_single_image_fps(model, validator)
    elapsed_seconds = time.perf_counter() - start_time
    metrics_path, report_path, table_path = save_reports(
        metrics,
        fixed_metrics,
        speed_metrics,
        flops_g,
        model_parameters,
        test_image_dir,
        started_at,
        elapsed_seconds,
    )

    print("\n========== 独立测试集最终结果 ==========")
    print(f"AP@0.5                        : {metrics.box.map50:.4f}")
    print(f"AP@0.75                       : {metrics.box.map75:.4f}")
    print(f"mAP@0.5:0.95                  : {metrics.box.map:.4f}")
    print(f"Precision@0.5                 : {fixed_metrics['precision']:.4f}")
    print(f"Recall@0.5                    : {fixed_metrics['recall']:.4f}")
    print(f"F1@0.5                        : {fixed_metrics['f1']:.4f}")
    print(
        f"TP / FP / FN                  : {fixed_metrics['TPs']} / "
        f"{fixed_metrics['FPs']} / {fixed_metrics['FNs']}"
    )
    print(f"参数量 (M)                    : {model_parameters / 1e6:.3f}")
    print(f"FLOPs (G)                     : {flops_g:.3f}" if flops_g > 0 else "FLOPs (G)：N/A")
    for name, value in speed_metrics.items():
        print(f"{name:34s}: {value:.4f}")
    print(f"\n测试指标JSON：{metrics_path}")
    print(f"测试详细报告：{report_path}")
    print(f"论文核心表格：{table_path}")


if __name__ == "__main__":
    main()
