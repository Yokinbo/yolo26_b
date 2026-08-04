r"""YOLO26-S 火电厂大范围 RGB GeoTIFF 正式推理脚本。.

默认推理流程与 D-FINE 正式大图推理脚本保持一致：

    512/256 重叠滑窗
    + 边界/低置信度候选的 768 扩展视域复检
    + 仅对复检窗口执行翻转 TTA
    + BR-DCF 跨窗口一致性加权融合

这里有两个容易混淆的尺寸：

1. ``BASE_TILE_SIZE=512`` 是从原始 GeoTIFF 中裁取的像素范围，建议与原始训练
   图片尺寸一致。
2. ``MODEL_IMAGE_SIZE`` 是 YOLO26 实际接收的网络输入尺寸，应与本权重训练时的
   ``imgsz`` 一致。Ultralytics 会自动将基础裁块缩放/填充到该尺寸。

使用方法：

1. 修改下方“用户配置区”的 INPUT_TIF、WEIGHTS_PATH 和 OUTPUT_DIR。
2. 在当前 define 环境、YOLO26 仓库根目录执行：

       python myscript/大范围遥感影像推理.py

也可用命令行临时覆盖三个路径：

       python myscript/大范围遥感影像推理.py ^
         --input "F:\\待检测影像.tif" ^
         --weights "E:\\YOLO\\yolo26\\runs\\train\\xxx\\weights\\best.pt" ^
         --output "F:\\YOLO26推理结果"

GeoJSON 和 CSV 始终输出；GPKG/SHP 需要 geopandas、shapely 和相应 GIS 驱动。
本脚本用于无真值的大图应用推理，不计算验证集精度。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# 允许直接执行 ``python myscript/大范围遥感影像推理.py`` 时使用当前仓库源码。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import MODEL_IMAGE_SIZE
from ultralytics import YOLO

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
except ImportError:
    rasterio = None
    Resampling = None
    Window = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# =============================================================================
# 用户配置区：通常只需要修改路径、置信度和显存相关参数
# =============================================================================

# 待检测的镇级/县级 RGB GeoTIFF 绝对路径，至少包含 3 个波段。
INPUT_TIF = Path(
    r"F:\3能源金三角基础设施识别\金三角1.88米tif\榆林市\神木市\镇级tif影像\锦界镇\锦界镇1.86米tif\Level16\锦界镇1.86.tif"
)

# 推荐使用 valid.py 验证过的 best_map50.pt；最终置信度应采用大图滑窗验证后确定的值。
WEIGHTS_PATH = Path(r"E:\YOLO\yolo26\runs\train\yolo26m_240epochs_640_第四版数据集793_map50\weights\best_map50.pt")

# 每次正式推理使用独立目录，避免覆盖其他模型或参数的结果。
OUTPUT_DIR = Path(
    r"F:\3能源金三角基础设施识别\金三角1.88米tif\榆林市\神木市\镇级tif影像\锦界镇\yolo26m_第四版数据集推测结果\置信度0.2"
)

# Shapefile 输出名称，只填写文件名并保留 .shp 后缀；保存位置由 OUTPUT_DIR 决定。
SHP_OUTPUT_NAME = "锦界镇_yolo26m_640_793_map50_置信度0.2.shp"

DEVICE: int | str = 0  # 单张 NVIDIA 显卡用 0；CPU 用 "cpu"
USE_HALF = True  # CUDA 上启用 FP16；CPU 时会自动关闭
BATCH_SIZE = 6  # 显存不足时改为 4、2 或 1

# 模型输入尺寸由仓库根目录的 experiment_config.py 统一设置。

### FINAL_CONFIDENCE 实验不同置信度阈值 最终制图阈值。先填写 valid.py 报告的最优 F1 置信度，再根据真实大图滑窗结果调整并固定。
FINAL_CONFIDENCE = 0.2
NMS_IOU = 0.70  # YOLO 对单个窗口内部候选执行 NMS 的 IoU
MAX_DETECTIONS_PER_VIEW = 50

# 同一目标至少需要多少个不同窗口/TTA 视图支持。1 最稳妥；2 或 4 更严格但可能漏检。
MIN_SUPPORT_COUNT = 1

# 基础滑窗参数：从原始 GeoTIFF 裁 512×512，步长 256，即 50% 重叠。
BASE_TILE_SIZE = 512
OVERLAP_STRIDE = 256

# 用于复检和融合的低阈值候选。
CANDIDATE_CONFIDENCE = 0.05

# 边界候选或不稳定候选使用更大上下文再次推理。
EDGE_MARGIN = 64
EDGE_RISK_THRESHOLD = 0.50
REFINE_MIN_CONFIDENCE = 0.25
REFINE_STABLE_CONFIDENCE = 0.50
REFINE_UNCERTAIN_CANDIDATES = True
REFINE_CONTEXT_SIZE = 768
REFINE_TRIGGER_NMS_IOU = 0.30
MAX_REFINE_WINDOWS = 1000

# 仅复检窗口使用选择性 TTA；original 会始终执行。
SELECTIVE_TTA_MODES = ("hflip", "vflip", "hvflip")

# BR-DCF 跨窗口融合参数。
FUSION_IOU = 0.35
FUSION_GAMMA = 1.0
FUSION_CENTER_FLOOR = 0.20
FUSION_CONSISTENCY_FLOOR = 0.20

# 黑边/无效区域过滤。
BLACK_THRESHOLD = 3
MIN_VALID_RATIO = 0.20

# 训练图片按 8-bit RGB 使用。非 uint8 影像默认停止，防止错误拉伸导致精度失真。
ALLOW_NON_UINT8 = False

# 输出选项。GeoJSON、CSV 和摘要不依赖 geopandas。
WRITE_GPKG = True
WRITE_SHP = True

WRITE_PREVIEW = True
PREVIEW_MAX_SIZE = 2400

# 类别名称；本权重为单类别火电厂检测。
CLASS_NAME = "hdc"

# =============================================================================


@dataclass
class WindowSpec:
    """A source-raster window."""

    x0: int
    y0: int
    size: int
    window_id: str
    is_refine: bool = False


@dataclass
class Detection:
    """A detection expressed in full-raster pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0
    window_id: str = ""
    view: str = "original"
    is_refine: bool = False
    boundary_risk: float = 0.0
    center_weight: float = 1.0
    support_count: int = 1

    @property
    def box(self) -> np.ndarray:
        return np.asarray([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    @property
    def center(self) -> tuple[float, float]:
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)


def require_runtime_dependencies() -> None:
    """Check only dependencies required for reading the raster."""
    if rasterio is None:
        raise RuntimeError(
            "当前环境缺少 rasterio，无法读取带地理坐标的 GeoTIFF。\n"
            "建议执行：conda install -c conda-forge rasterio\n"
            "GPKG/SHP 还需要 geopandas 和 shapely；缺少它们不影响 CSV/GeoJSON 输出。"
        )


def validate_config(input_tif: Path, weights: Path) -> None:
    """Validate paths and parameters before allocating the model."""
    require_runtime_dependencies()
    missing = [
        f"{name}: {path}" for name, path in (("输入 GeoTIFF", input_tif), ("模型权重", weights)) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("以下路径未正确填写：\n" + "\n".join(missing))
    if MODEL_IMAGE_SIZE <= 0 or MODEL_IMAGE_SIZE % 32:
        raise ValueError("MODEL_IMAGE_SIZE 必须是能被 32 整除的正整数。")
    if BASE_TILE_SIZE <= 0 or not 0 < OVERLAP_STRIDE <= BASE_TILE_SIZE:
        raise ValueError("必须满足 BASE_TILE_SIZE > 0 且 0 < OVERLAP_STRIDE <= BASE_TILE_SIZE。")
    if REFINE_CONTEXT_SIZE < BASE_TILE_SIZE:
        raise ValueError("REFINE_CONTEXT_SIZE 不能小于 BASE_TILE_SIZE。")
    if BATCH_SIZE <= 0 or MAX_DETECTIONS_PER_VIEW <= 0:
        raise ValueError("BATCH_SIZE 和 MAX_DETECTIONS_PER_VIEW 必须大于 0。")
    shp_name = Path(SHP_OUTPUT_NAME)
    if shp_name.name != SHP_OUTPUT_NAME or shp_name.suffix.lower() != ".shp":
        raise ValueError("SHP_OUTPUT_NAME 只能填写以 .shp 结尾的文件名，不能包含文件夹路径。")
    probabilities = (
        FINAL_CONFIDENCE,
        CANDIDATE_CONFIDENCE,
        NMS_IOU,
        EDGE_RISK_THRESHOLD,
        REFINE_MIN_CONFIDENCE,
        REFINE_STABLE_CONFIDENCE,
        REFINE_TRIGGER_NMS_IOU,
        FUSION_IOU,
    )
    if not all(0 <= value <= 1 for value in probabilities):
        raise ValueError("置信度和 IoU 参数必须位于 [0, 1]。")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 设置为 GPU，但当前 PyTorch 没有检测到可用 CUDA。")


def start_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Generate full-coverage starts and explicitly include the final edge."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def clip_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    """Clip a box to raster pixel bounds."""
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
    y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
    return np.asarray([x1, y1, x2, y2], dtype=np.float64)


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float64)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def global_nms(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Perform class-agnostic NMS in full-raster pixel coordinates."""
    if not detections:
        return []
    boxes = np.stack([det.box for det in detections])
    scores = np.asarray([det.score for det in detections])
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[overlaps < iou_threshold]
    return [detections[index] for index in keep]


def local_boundary_properties(local_box: Sequence[float], size: int) -> tuple[float, float]:
    """Return boundary risk and Hann-like center reliability."""
    x1, y1, x2, y2 = [float(value) for value in local_box]
    clearance = max(0.0, min(x1, y1, size - x2, size - y2))
    boundary_risk = 1.0 - min(1.0, clearance / max(float(EDGE_MARGIN), 1.0))
    cx = min(max(0.5 * (x1 + x2), 0.0), float(size))
    cy = min(max(0.5 * (y1 + y2), 0.0), float(size))
    hann_center = math.sin(math.pi * cx / size) ** 2 * math.sin(math.pi * cy / size) ** 2
    center_weight = FUSION_CENTER_FLOOR + (1.0 - FUSION_CENTER_FLOOR) * hann_center
    return boundary_risk, center_weight


def convert_patch_to_uint8(patch: np.ndarray) -> np.ndarray:
    """Preserve uint8 data or explicitly rescale an allowed non-uint8 raster."""
    if patch.dtype == np.uint8:
        return patch
    if not ALLOW_NON_UINT8:
        raise TypeError(
            f"输入影像类型为 {patch.dtype}，而模型训练数据按 uint8 RGB 使用。"
            "请先确认影像辐射范围；确认需要自动线性拉伸后再设置 ALLOW_NON_UINT8=True。"
        )
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return np.zeros(patch.shape, dtype=np.uint8)
    if np.issubdtype(patch.dtype, np.integer):
        maximum = float(np.iinfo(patch.dtype).max)
        minimum = 0.0
    else:
        minimum, maximum = np.percentile(finite, [0.5, 99.5]).tolist()
    scale = max(maximum - minimum, 1e-12)
    return np.clip((patch.astype(np.float32) - minimum) / scale * 255.0, 0, 255).astype(np.uint8)


def read_rgb_patch(src: Any, spec: WindowSpec) -> tuple[np.ndarray, float]:
    """Read a boundless RGB window as CHW uint8 and calculate its valid ratio."""
    window = Window(spec.x0, spec.y0, spec.size, spec.size)
    patch = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    patch = convert_patch_to_uint8(patch)
    valid = ~np.all(patch <= BLACK_THRESHOLD, axis=0)
    try:
        valid &= src.dataset_mask(window=window, boundless=True) > 0
    except Exception:
        pass
    return patch, float(valid.mean())


def transform_patch(patch: np.ndarray, mode: str) -> np.ndarray:
    """Apply TTA to a CHW patch."""
    if mode == "original":
        return patch
    if mode == "hflip":
        return patch[:, :, ::-1].copy()
    if mode == "vflip":
        return patch[:, ::-1, :].copy()
    if mode == "hvflip":
        return patch[:, ::-1, ::-1].copy()
    raise ValueError(f"不支持的 TTA 模式: {mode}")


def undo_tta_boxes(boxes: np.ndarray, size: int, mode: str) -> np.ndarray:
    """Map augmented-view boxes back to the original patch coordinates."""
    boxes = boxes.copy()
    if boxes.size == 0 or mode == "original":
        return boxes
    old = boxes.copy()
    if mode in {"hflip", "hvflip"}:
        boxes[:, 0] = size - old[:, 2]
        boxes[:, 2] = size - old[:, 0]
        old = boxes.copy()
    if mode in {"vflip", "hvflip"}:
        boxes[:, 1] = size - old[:, 3]
        boxes[:, 3] = size - old[:, 1]
    return boxes


def infer_batch(
    model: YOLO,
    patches: Sequence[np.ndarray],
    specs: Sequence[WindowSpec],
    views: Sequence[str],
    raster_width: int,
    raster_height: int,
) -> list[Detection]:
    """Run YOLO on one in-memory batch and return global-pixel boxes."""
    detections: list[Detection] = []
    for view in views:
        # Ultralytics treats numpy input as BGR. Rasterio returns RGB, so reverse channels here.
        sources = [np.transpose(transform_patch(patch, view), (1, 2, 0))[:, :, ::-1].copy() for patch in patches]
        results = model.predict(
            source=sources,
            imgsz=MODEL_IMAGE_SIZE,
            conf=CANDIDATE_CONFIDENCE,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS_PER_VIEW,
            device=DEVICE,
            quantize=16 if USE_HALF and DEVICE != "cpu" else None,
            classes=[0],
            verbose=False,
            stream=False,
        )
        for result, spec in zip(results, specs):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            local_boxes = result.boxes.xyxy.detach().cpu().numpy()
            local_boxes = undo_tta_boxes(local_boxes, spec.size, view)
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for local_box, score, class_id in zip(local_boxes, scores, classes):
                boundary_risk, center_weight = local_boundary_properties(local_box, spec.size)
                global_box = clip_box(
                    (
                        local_box[0] + spec.x0,
                        local_box[1] + spec.y0,
                        local_box[2] + spec.x0,
                        local_box[3] + spec.y0,
                    ),
                    raster_width,
                    raster_height,
                )
                if global_box[2] - global_box[0] < 1 or global_box[3] - global_box[1] < 1:
                    continue
                detections.append(
                    Detection(
                        *global_box.tolist(),
                        score=float(score),
                        class_id=int(class_id),
                        window_id=spec.window_id,
                        view=view,
                        is_refine=spec.is_refine,
                        boundary_risk=float(boundary_risk),
                        center_weight=float(center_weight),
                    )
                )
    return detections


def flush_window_batch(
    model: YOLO,
    patches: list[np.ndarray],
    specs: list[WindowSpec],
    views: Sequence[str],
    raster_width: int,
    raster_height: int,
) -> list[Detection]:
    """Infer and clear an accumulated batch."""
    if not patches:
        return []
    detections = infer_batch(model, patches, specs, views, raster_width, raster_height)
    patches.clear()
    specs.clear()
    return detections


def format_duration(seconds: float) -> str:
    """Format elapsed time for terminal output."""
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(label: str, completed: int, total: int, start_time: float, detail: str = "") -> None:
    """Print an in-place progress bar with ETA."""
    completed = min(max(completed, 0), total)
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    ratio = completed / total if total else 1.0
    eta = elapsed * (total - completed) / max(completed, 1)
    filled = round(24 * ratio)
    bar = "#" * filled + "-" * (24 - filled)
    suffix = f" | {detail}" if detail else ""
    print(
        f"\r{label} [{bar}] {ratio:6.2%} | {completed}/{total} | "
        f"耗时 {format_duration(elapsed)} | 剩余 {format_duration(eta)}{suffix}",
        end="",
        flush=True,
    )


def run_base_windows(src: Any, model: YOLO) -> tuple[list[Detection], int, int]:
    """Run the 512/256 overlapping base grid."""
    xs = start_positions(src.width, BASE_TILE_SIZE, OVERLAP_STRIDE)
    ys = start_positions(src.height, BASE_TILE_SIZE, OVERLAP_STRIDE)
    total = len(xs) * len(ys)
    used = skipped = 0
    detections: list[Detection] = []
    patches: list[np.ndarray] = []
    specs: list[WindowSpec] = []
    start_time = time.perf_counter()
    interval = max(1, total // 100)
    print(f"[基础滑窗] 总窗口 {total} | tile={BASE_TILE_SIZE} | stride={OVERLAP_STRIDE}")

    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            spec = WindowSpec(x0, y0, BASE_TILE_SIZE, f"base_r{row}_c{col}")
            patch, valid_ratio = read_rgb_patch(src, spec)
            if valid_ratio < MIN_VALID_RATIO:
                skipped += 1
            else:
                patches.append(patch)
                specs.append(spec)
                used += 1
                if len(patches) >= BATCH_SIZE:
                    detections.extend(flush_window_batch(model, patches, specs, ("original",), src.width, src.height))
            completed = row * len(xs) + col + 1
            if completed % interval == 0 or completed == total:
                print_progress(
                    "[基础滑窗]",
                    completed,
                    total,
                    start_time,
                    f"有效 {used} | 跳过 {skipped} | 候选 {len(detections)}",
                )

    detections.extend(flush_window_batch(model, patches, specs, ("original",), src.width, src.height))
    print_progress(
        "[基础滑窗]",
        total,
        total,
        start_time,
        f"有效 {used} | 跳过 {skipped} | 候选 {len(detections)}",
    )
    print()
    return detections, used, skipped


def select_refine_windows(base_detections: Sequence[Detection], width: int, height: int) -> list[WindowSpec]:
    """Choose spatially de-duplicated edge/uncertain candidates for context refinement."""
    del width, height  # Boundless raster reading handles windows close to outer raster edges.
    candidates = [
        det
        for det in base_detections
        if det.score >= REFINE_MIN_CONFIDENCE
        and (
            det.boundary_risk >= EDGE_RISK_THRESHOLD
            or (REFINE_UNCERTAIN_CANDIDATES and det.score < REFINE_STABLE_CONFIDENCE)
        )
    ]
    candidates = global_nms(candidates, REFINE_TRIGGER_NMS_IOU)
    candidates.sort(key=lambda det: det.score * (0.5 + 0.5 * det.boundary_risk), reverse=True)
    half = REFINE_CONTEXT_SIZE // 2
    minimum_center_distance = max(32, BASE_TILE_SIZE // 4)
    windows: list[WindowSpec] = []
    used_centers: list[tuple[int, int]] = []
    for index, det in enumerate(candidates[:MAX_REFINE_WINDOWS]):
        cx, cy = det.center
        center = round(cx), round(cy)
        if any(math.hypot(center[0] - old[0], center[1] - old[1]) < minimum_center_distance for old in used_centers):
            continue
        used_centers.append(center)
        windows.append(
            WindowSpec(
                center[0] - half,
                center[1] - half,
                REFINE_CONTEXT_SIZE,
                f"refine_{index:05d}",
                True,
            )
        )
    return windows


def run_refine_windows(src: Any, model: YOLO, windows: Sequence[WindowSpec]) -> list[Detection]:
    """Run original and selective-TTA views for refinement windows."""
    if not windows:
        print("[扩展复检] 没有需要复检的候选窗口。")
        return []
    views = ("original", *SELECTIVE_TTA_MODES)
    detections: list[Detection] = []
    patches: list[np.ndarray] = []
    specs: list[WindowSpec] = []
    start_time = time.perf_counter()
    total = len(windows)
    interval = max(1, total // 100)
    print(f"[扩展复检] 窗口 {total} | context={REFINE_CONTEXT_SIZE} | views={views}")

    for index, spec in enumerate(windows, start=1):
        patch, valid_ratio = read_rgb_patch(src, spec)
        if valid_ratio >= MIN_VALID_RATIO:
            patches.append(patch)
            specs.append(spec)
            if len(patches) >= BATCH_SIZE:
                detections.extend(flush_window_batch(model, patches, specs, views, src.width, src.height))
        if index % interval == 0 or index == total:
            print_progress(
                "[扩展复检]",
                index,
                total,
                start_time,
                f"候选 {len(detections)} | 视图 {len(views)}",
            )

    detections.extend(flush_window_batch(model, patches, specs, views, src.width, src.height))
    print_progress(
        "[扩展复检]",
        total,
        total,
        start_time,
        f"候选 {len(detections)} | 视图 {len(views)}",
    )
    print()
    return detections


def cluster_detections(detections: Sequence[Detection], threshold: float) -> list[list[int]]:
    """Cluster overlapping boxes using a spatial hash and union-find."""
    count = len(detections)
    if count == 0:
        return []
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    boxes = np.stack([det.box for det in detections])
    bucket_size = float(BASE_TILE_SIZE)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, box in enumerate(boxes):
        min_col = math.floor(box[0] / bucket_size)
        max_col = math.floor(max(box[0], box[2] - 1e-6) / bucket_size)
        min_row = math.floor(box[1] / bucket_size)
        max_row = math.floor(max(box[1], box[3] - 1e-6) / bucket_size)
        keys = [(row, col) for row in range(min_row, max_row + 1) for col in range(min_col, max_col + 1)]
        possible = sorted({old for key in keys for old in buckets.get(key, [])})
        if possible:
            overlaps = box_iou_one_to_many(box, boxes[np.asarray(possible, dtype=int)])
            for local_index in np.flatnonzero(overlaps >= threshold):
                union(index, possible[int(local_index)])
        for key in keys:
            buckets.setdefault(key, []).append(index)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def brdcf_fusion(detections: Sequence[Detection]) -> list[Detection]:
    """Fuse cross-window boxes using boundary reliability and box consistency."""
    fused: list[Detection] = []
    for cluster in cluster_detections(detections, FUSION_IOU):
        members = [detections[index] for index in cluster]
        boxes = np.stack([det.box for det in members])
        if len(members) == 1:
            consistency = np.ones(1, dtype=np.float64)
        else:
            consistency = np.asarray(
                [
                    float(box_iou_one_to_many(box, np.delete(boxes, index, axis=0)).mean())
                    for index, box in enumerate(boxes)
                ]
            )
        scores = np.asarray([det.score for det in members], dtype=np.float64)
        centers = np.asarray([det.center_weight for det in members], dtype=np.float64)
        consistency_weight = FUSION_CONSISTENCY_FLOOR + (1.0 - FUSION_CONSISTENCY_FLOOR) * consistency
        weights = np.maximum(np.power(scores, FUSION_GAMMA) * centers * consistency_weight, 1e-12)
        fused_box = np.sum(boxes * weights[:, None], axis=0) / weights.sum()
        fused_score = float(np.sum(scores * weights) / weights.sum())
        best = members[int(np.argmax(scores))]
        fused.append(
            Detection(
                *fused_box.tolist(),
                score=fused_score,
                class_id=best.class_id,
                window_id="BRDCF",
                view="fused",
                is_refine=any(det.is_refine for det in members),
                boundary_risk=max(det.boundary_risk for det in members),
                center_weight=float(np.max(centers)),
                support_count=len({(det.window_id, det.view) for det in members}),
            )
        )
    fused.sort(key=lambda det: det.score, reverse=True)
    return fused


def pixel_box_to_world_polygon(box: Sequence[float], transform: Any) -> list[list[float]]:
    """Convert a pixel box to a georeferenced polygon, including rotated affine transforms."""
    x1, y1, x2, y2 = [float(value) for value in box]
    pixels = ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1))
    return [[float(x), float(y)] for x, y in (transform * point for point in pixels)]


def save_predictions_csv(path: Path, detections: Sequence[Detection], transform: Any) -> None:
    """Save pixel boxes and georeferenced centers."""
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "prediction_id",
                "class_name",
                "score",
                "pixel_x1",
                "pixel_y1",
                "pixel_x2",
                "pixel_y2",
                "center_geo_x",
                "center_geo_y",
                "support_count",
                "contains_refine_view",
            ]
        )
        for index, det in enumerate(detections, start=1):
            geo_x, geo_y = transform * det.center
            writer.writerow(
                [
                    f"P{index:04d}",
                    CLASS_NAME,
                    f"{det.score:.6f}",
                    f"{det.x1:.3f}",
                    f"{det.y1:.3f}",
                    f"{det.x2:.3f}",
                    f"{det.y2:.3f}",
                    f"{geo_x:.6f}",
                    f"{geo_y:.6f}",
                    det.support_count,
                    int(det.is_refine),
                ]
            )


def save_predictions_geojson(path: Path, detections: Sequence[Detection], transform: Any, crs: Any) -> None:
    """Save polygons in the source raster coordinate system."""
    features = [
        {
            "type": "Feature",
            "properties": {
                "prediction_id": f"P{index:04d}",
                "class_name": CLASS_NAME,
                "score": det.score,
                "support_count": det.support_count,
                "refined": bool(det.is_refine),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [pixel_box_to_world_polygon(det.box, transform)],
            },
        }
        for index, det in enumerate(detections, start=1)
    ]
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        payload["crs"] = {"type": "name", "properties": {"name": crs.to_string()}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_optional_vectors(result_dir: Path, detections: Sequence[Detection], transform: Any, crs: Any) -> None:
    """Write GPKG/SHP when geopandas is available."""
    if not (WRITE_GPKG or WRITE_SHP):
        return
    if gpd is None:
        print("[提示] 未安装 geopandas，跳过 GPKG/SHP；CSV 和 GeoJSON 已正常输出。")
        return
    try:
        from shapely.geometry import Polygon

        records = [
            {
                "pred_id": f"P{index:04d}",
                "class": CLASS_NAME,
                "score": det.score,
                "support": det.support_count,
                "refined": int(det.is_refine),
                "geometry": Polygon(pixel_box_to_world_polygon(det.box, transform)),
            }
            for index, det in enumerate(detections, start=1)
        ]
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        if WRITE_GPKG:
            frame.to_file(result_dir / "火电厂检测结果.gpkg", layer="power_plants", driver="GPKG")
        if WRITE_SHP:
            frame.to_file(result_dir / SHP_OUTPUT_NAME, encoding="utf-8")
    except Exception as error:
        print(f"[提示] GPKG/SHP 输出失败，但不影响其他结果: {error}")


def save_preview(src: Any, path: Path, detections: Sequence[Detection]) -> None:
    """Save a downsampled overview with red detection boxes."""
    scale = min(1.0, PREVIEW_MAX_SIZE / max(src.width, src.height))
    width = max(1, round(src.width * scale))
    height = max(1, round(src.height * scale))
    rgb = src.read(
        [1, 2, 3],
        out_shape=(3, height, width),
        resampling=Resampling.bilinear,
    )
    rgb = convert_patch_to_uint8(rgb)
    image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, det in enumerate(detections, start=1):
        box = tuple(float(value) * scale for value in det.box)
        draw.rectangle(box, outline=(255, 0, 0), width=2)
        draw.text(
            (box[0], max(0, box[1] - 10)),
            f"P{index}:{det.score:.2f}",
            fill=(255, 0, 0),
            font=font,
        )
    image.save(path)


def synchronize_cuda() -> None:
    """Synchronize CUDA only when it is in use."""
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    """Save a concise human-readable application report."""
    lines = [
        "YOLO26-S 火电厂大范围遥感影像正式应用报告",
        "=" * 60,
        "推理策略: 512/256重叠滑窗 + 768扩展复检 + 选择性TTA + BR-DCF",
        f"输入影像: {summary['input_tif']}",
        f"模型权重: {summary['weights']}",
        f"输出目录: {summary['output_dir']}",
        f"影像尺寸: {summary['raster_width']} × {summary['raster_height']}",
        f"影像坐标系: {summary['crs']}",
        f"模型输入尺寸: {MODEL_IMAGE_SIZE}",
        f"原图裁块/步长: {BASE_TILE_SIZE}/{OVERLAP_STRIDE}",
        "",
        f"有效基础窗口数: {summary['base_windows']}",
        f"跳过无效窗口数: {summary['skipped_windows']}",
        f"二次复检窗口数: {summary['refine_windows']}",
        f"基础低阈值候选框数: {summary['base_candidate_boxes']}",
        f"复检/TTA低阈值候选框数: {summary['refine_candidate_boxes']}",
        f"BR-DCF融合后候选数: {summary['fused_candidate_boxes']}",
        (f"最终输出目标数(conf≥{FINAL_CONFIDENCE:.2f}, support≥{MIN_SUPPORT_COUNT}): {summary['exported_boxes']}"),
        "",
        f"开始时间: {summary['start_time']}",
        f"结束时间: {summary['end_time']}",
        f"大图推理及融合耗时: {format_duration(summary['inference_seconds'])}",
        f"完整运行总耗时: {format_duration(summary['total_seconds'])}",
        "",
        "说明：完整运行总耗时包含模型加载、GeoTIFF读取、推理、融合和结果写出。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_large_raster_inference(input_tif: Path, weights: Path, output_dir: Path) -> None:
    """Run the complete large-raster application workflow."""
    validate_config(input_tif, weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    start_datetime = datetime.now().astimezone()

    print("\n" + "=" * 76)
    print("YOLO26-S 火电厂大范围遥感影像推理")
    print(f"输入影像: {input_tif}")
    print(f"模型权重: {weights}")
    print(f"输出目录: {output_dir}")
    print(
        f"设备: {DEVICE} | FP16: {USE_HALF and DEVICE != 'cpu'} | batch: {BATCH_SIZE} | "
        f"原图裁块: {BASE_TILE_SIZE} | 模型输入: {MODEL_IMAGE_SIZE}"
    )
    print("=" * 76)

    print("[模型] 正在加载 YOLO26 权重...")
    model = YOLO(str(weights))
    print("[模型] 加载完成。")

    with rasterio.open(input_tif) as src:
        if src.count < 3:
            raise ValueError("输入 GeoTIFF 至少需要 3 个 RGB 波段。")
        print(f"[影像] size={src.width}×{src.height} | bands={src.count} | dtype={src.dtypes[0]} | CRS={src.crs}")

        synchronize_cuda()
        inference_start = time.perf_counter()
        base_detections, base_windows, skipped_windows = run_base_windows(src, model)
        refine_specs = select_refine_windows(base_detections, src.width, src.height)
        refine_detections = run_refine_windows(src, model, refine_specs)
        raw_detections = base_detections + refine_detections
        print(f"[BR-DCF] 正在融合 {len(raw_detections)} 个跨窗口候选框...")
        fused_detections = brdcf_fusion(raw_detections)
        synchronize_cuda()
        inference_seconds = time.perf_counter() - inference_start
        print(f"[BR-DCF] 融合完成，得到 {len(fused_detections)} 个低阈值候选。")

        final_detections = [
            det for det in fused_detections if det.score >= FINAL_CONFIDENCE and det.support_count >= MIN_SUPPORT_COUNT
        ]
        final_detections.sort(key=lambda det: det.score, reverse=True)

        save_predictions_csv(output_dir / "火电厂检测结果.csv", final_detections, src.transform)
        save_predictions_geojson(
            output_dir / "火电厂检测结果.geojson",
            final_detections,
            src.transform,
            src.crs,
        )
        save_optional_vectors(output_dir, final_detections, src.transform, src.crs)
        if WRITE_PREVIEW:
            save_preview(src, output_dir / "火电厂检测预览图.png", final_detections)

        summary: dict[str, Any] = {
            "input_tif": str(input_tif),
            "weights": str(weights),
            "output_dir": str(output_dir),
            "device": str(DEVICE),
            "raster_width": src.width,
            "raster_height": src.height,
            "crs": str(src.crs),
            "model_image_size": MODEL_IMAGE_SIZE,
            "base_tile_size": BASE_TILE_SIZE,
            "overlap_stride": OVERLAP_STRIDE,
            "refine_context_size": REFINE_CONTEXT_SIZE,
            "candidate_confidence": CANDIDATE_CONFIDENCE,
            "final_confidence": FINAL_CONFIDENCE,
            "min_support_count": MIN_SUPPORT_COUNT,
            "base_windows": base_windows,
            "skipped_windows": skipped_windows,
            "refine_windows": len(refine_specs),
            "base_candidate_boxes": len(base_detections),
            "refine_candidate_boxes": len(refine_detections),
            "fused_candidate_boxes": len(fused_detections),
            "exported_boxes": len(final_detections),
            "inference_seconds": inference_seconds,
            "start_time": start_datetime.isoformat(timespec="seconds"),
        }

    end_datetime = datetime.now().astimezone()
    summary["end_time"] = end_datetime.isoformat(timespec="seconds")
    summary["total_seconds"] = time.perf_counter() - wall_start
    (output_dir / "正式推理运行参数.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_summary(output_dir / "正式推理结果摘要.txt", summary)

    print("\n========== 大图推理完成 ==========")
    print(f"最终检测数量: {summary['exported_boxes']}")
    print(f"复检窗口数量: {summary['refine_windows']}")
    print(f"推理及融合耗时: {format_duration(summary['inference_seconds'])}")
    print(f"完整运行总耗时: {format_duration(summary['total_seconds'])}")
    print(f"CSV: {output_dir / '火电厂检测结果.csv'}")
    print(f"GeoJSON: {output_dir / '火电厂检测结果.geojson'}")
    if WRITE_PREVIEW:
        print(f"预览图: {output_dir / '火电厂检测预览图.png'}")
    if WRITE_SHP:
        print(f"Shapefile: {output_dir / SHP_OUTPUT_NAME}")
    print(f"运行摘要: {output_dir / '正式推理结果摘要.txt'}")


def parse_args() -> argparse.Namespace:
    """Parse optional path overrides."""
    parser = argparse.ArgumentParser(description="YOLO26-S 火电厂大范围 RGB GeoTIFF 推理")
    parser.add_argument("--input", type=Path, default=INPUT_TIF, help="覆盖顶部 INPUT_TIF")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="覆盖顶部 WEIGHTS_PATH")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="覆盖顶部 OUTPUT_DIR")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_large_raster_inference(args.input, args.weights, args.output)
