"""训练本地 YOLO 格式目标检测数据集。

修改下方“用户训练参数配置区”后，直接运行：
    python train.py
"""

import time
from datetime import datetime
from pathlib import Path

from experiment_config import (
    BASE_MODEL_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    MODEL_IMAGE_SIZE,
    MODEL_TAG,
    PRETRAINED_WEIGHT_PATH,
    ensure_loaded_model_size,
)
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import strip_optimizer

# =============================================================================
# 用户训练参数配置区
# =============================================================================

# 数据集配置文件：内部填写训练集、验证集和类别名称。
DATASET_CONFIG = Path(r"E:\YOLO\yolo26\ultralytics\cfg\datasets\a_myhdc.yaml")

# 模型结构配置文件。
# False：普通微调，结构直接来自 PRETRAINED_WEIGHTS，MODEL_CONFIG 不参与加载。
# True：模型优化实验，先按此 YAML 创建结构，再加载 PRETRAINED_WEIGHTS 中形状匹配的参数。
USE_CUSTOM_MODEL_CONFIG = False
MODEL_CONFIG = MODEL_CONFIG_PATH

# 由 experiment_config.py 的 MODEL_SIZE 在 yolo26s.pt/yolo26m.pt 之间切换。
PRETRAINED_WEIGHTS = PRETRAINED_WEIGHT_PATH

# 基础训练参数
EPOCHS = 240
IMAGE_SIZE = MODEL_IMAGE_SIZE  # 与验证和大图推理共用 experiment_config.py 的设置
BATCH_SIZE = 6  # 显存不足时改为 4 或 2；也可设为 -1 自动估算
DEVICE = 0  # 单张 NVIDIA 显卡用 0；CPU 用 "cpu"；多卡用 [0, 1]
WORKERS = 2  # Windows 下若 DataLoader 报错，可改为 0
PATIENCE = 100
SEED = 2026

# 优化器与学习率；OPTIMIZER="auto" 时，Ultralytics 会自动选择并忽略 LR0。
OPTIMIZER = "auto"  # 可选："auto"、"SGD"、"AdamW"、"MuSGD" 等
LR0 = 0.01
LR_FINAL_FACTOR = 0.01
WEIGHT_DECAY = 0.0005
COSINE_LR = False
AMP = True

# 数据增强与训练控制
CLOSE_MOSAIC = 30  # 训练 240 轮时，最后 30 轮关闭拼图类增强以稳定收敛    200轮对应20
CACHE = False  # True 缓存到内存，"disk" 缓存到磁盘，False 不缓存
MULTI_SCALE = 0.0  # 例如 0.5 表示在 IMAGE_SIZE 的 ±50% 范围内多尺度训练
FRACTION = 1.0  # 使用训练集的比例，快速测试可改为 0.1
DETERMINISTIC = True

# 训练结果输出根目录：请直接填写 Windows 绝对路径。
PROJECT = Path(r"E:\YOLO\yolo26\runs\train")
# RUN_NAME：本次实验的输出子目录名，最终结果保存在 PROJECT / RUN_NAME。
# 可按数据集版本、模型或训练轮数自定义；它不会改变模型结构和精度，不要填写绝对路径。
# train.py 不生成论文最终对比表，因此不需要 PAPER_MODEL_NAME。
RUN_NAME = "第四版数据集793_map50"
SAVE_PERIOD = 10  # 每隔多少个 epoch 额外保存一次权重；-1 表示禁用
PLOTS = True

# 中断后续训：填写上次训练生成的 last.pt 路径；正常新训练保持为空字符串。
# 示例：r"E:\YOLO\yolo26\runs\train\yolo26s_hdc_240epochs\weights\last.pt"
RESUME_CHECKPOINT = ""

MAP50_METRIC = "metrics/mAP50(B)"
BEST_MAP50_FILENAME = "best_map50.pt"


def format_elapsed_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS。"""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_training_time(model: YOLO, started_at: datetime, elapsed_seconds: float) -> Path:
    """将本次训练进程的总耗时写入实际结果目录。"""
    finished_at = datetime.now().astimezone()
    save_dir = Path(model.trainer.save_dir)
    duration_file = save_dir / "training_time.txt"
    duration_file.write_text(
        "YOLO26 训练总时长记录\n"
        f"模式：{'断点续训' if RESUME_CHECKPOINT else '新训练'}\n"
        f"开始时间：{started_at.isoformat(timespec='seconds')}\n"
        f"结束时间：{finished_at.isoformat(timespec='seconds')}\n"
        f"本次总耗时：{format_elapsed_time(elapsed_seconds)}\n"
        f"总秒数：{elapsed_seconds:.2f}\n"
        f"总小时数：{elapsed_seconds / 3600:.4f}\n"
        f"结果目录：{save_dir}\n",
        encoding="utf-8",
    )
    return duration_file


def save_best_map50(trainer) -> None:
    """Save the current checkpoint when detection mAP@0.5 reaches a new maximum."""
    map50 = trainer.metrics.get(MAP50_METRIC)
    if map50 is None:
        return

    history = trainer.read_results_csv().get(MAP50_METRIC, [])
    previous_best = max(history[:-1], default=float("-inf"))
    if map50 > previous_best:
        destination = trainer.wdir / BEST_MAP50_FILENAME
        destination.write_bytes(trainer.last.read_bytes())
        LOGGER.info(f"mAP@0.5 improved to {map50:.5f}; saved {destination}")


def finalize_best_map50(trainer) -> None:
    """Strip optimizer state from the deployment-oriented mAP@0.5 checkpoint."""
    checkpoint = trainer.wdir / BEST_MAP50_FILENAME
    if checkpoint.exists():
        strip_optimizer(checkpoint)


def validate_training_config() -> Path:
    """提前检查常用参数和模型权重，提供清晰的错误信息。"""
    if EPOCHS <= 0 or IMAGE_SIZE <= 0 or BATCH_SIZE == 0 or WORKERS < 0:
        raise ValueError("EPOCHS、IMAGE_SIZE 必须大于 0，BATCH_SIZE 不能为 0，WORKERS 不能小于 0。")
    if not 0 < FRACTION <= 1:
        raise ValueError("FRACTION 必须在 (0, 1] 范围内。")

    required_files = {"数据集配置文件": DATASET_CONFIG}
    if USE_CUSTOM_MODEL_CONFIG:
        # MODEL_CONFIG 是 yolo26s/m.yaml 复合缩放别名，实际文件是 yolo26.yaml。
        required_files["模型基础配置文件"] = BASE_MODEL_CONFIG_PATH
    missing = [f"{name}：{path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下配置文件不存在：\n" + "\n".join(missing))

    checkpoint = Path(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else PRETRAINED_WEIGHTS
    if not checkpoint.is_file():
        kind = "续训权重" if RESUME_CHECKPOINT else "预训练权重"
        raise FileNotFoundError(f"{kind}不存在：{checkpoint}")
    return checkpoint


def main() -> None:
    """开始新训练，或从指定的 last.pt 断点续训。"""
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    checkpoint = validate_training_config()
    resume = bool(RESUME_CHECKPOINT)
    model_source = (
        f"断点权重：{checkpoint}"
        if resume
        else f"自定义结构：{MODEL_CONFIG} + 预训练权重：{PRETRAINED_WEIGHTS}"
        if USE_CUSTOM_MODEL_CONFIG
        else f"预训练模型：{PRETRAINED_WEIGHTS}"
    )

    print("\n========== YOLO26 训练配置 ==========")
    print(f"数据集配置：{DATASET_CONFIG}")
    print(f"共享模型选择：{MODEL_TAG}")
    print(f"模型来源：{model_source}")
    print(f"输入尺寸：{IMAGE_SIZE} | Batch：{BATCH_SIZE} | Epochs：{EPOCHS}")
    print(f"设备：{DEVICE} | AMP：{AMP} | Workers：{WORKERS}")
    print(f"训练输出：{PROJECT / RUN_NAME}")
    print(f"断点续训：{'是' if resume else '否'}\n")

    if resume:
        model = YOLO(checkpoint)
    elif USE_CUSTOM_MODEL_CONFIG:
        model = YOLO(MODEL_CONFIG).load(PRETRAINED_WEIGHTS)
    else:
        model = YOLO(PRETRAINED_WEIGHTS)

    ensure_loaded_model_size(model)

    model.add_callback("on_model_save", save_best_map50)
    model.add_callback("on_train_end", finalize_best_map50)

    model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LR_FINAL_FACTOR,
        weight_decay=WEIGHT_DECAY,
        cos_lr=COSINE_LR,
        amp=AMP,
        close_mosaic=CLOSE_MOSAIC,
        cache=CACHE,
        multi_scale=MULTI_SCALE,
        fraction=FRACTION,
        seed=SEED,
        deterministic=DETERMINISTIC,
        project=PROJECT,
        name=RUN_NAME,
        save_period=SAVE_PERIOD,
        plots=PLOTS,
        resume=resume,
    )

    elapsed_seconds = time.perf_counter() - start_time
    duration_file = save_training_time(model, started_at, elapsed_seconds)
    print("\n========== 训练任务总耗时 ==========")
    print(f"{format_elapsed_time(elapsed_seconds)}（{elapsed_seconds / 3600:.4f} 小时）")
    print(f"耗时记录：{duration_file}")


if __name__ == "__main__":
    main()
