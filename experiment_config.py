"""YOLO26 训练、验证和测试共用的实验配置。

开始 S/M 对比实验前，只需修改 ``MODEL_SIZE``。训练使用的官方预训练
权重、复合缩放模型 YAML，以及报告中的模型名称会同步切换。
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# 支持 "s" 和 "m"。这是 S/M 对比实验的唯一模型规模开关。
MODEL_SIZE = "m"

# 网络预处理后的统一输入尺寸。
MODEL_IMAGE_SIZE = 640

# YOLO26 使用同一份基础 YAML 中的 scales 实现 n/s/m/l/x 复合缩放。
BASE_MODEL_CONFIG_PATH = REPO_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
PRETRAINED_WEIGHT_PATHS = {
    "s": REPO_ROOT / "weights" / "yolo26s.pt",
    "m": REPO_ROOT / "weights" / "yolo26m.pt",
}


def normalized_model_size() -> str:
    """返回并校验共享的 YOLO26 模型规模。"""
    model_size = MODEL_SIZE.lower().strip()
    if model_size not in PRETRAINED_WEIGHT_PATHS:
        raise ValueError(f'MODEL_SIZE must be "s" or "m", got {MODEL_SIZE!r}')
    return model_size


def selected_model_config_path() -> Path:
    """返回带规模后缀的 YOLO26 YAML 路径。

    ``yolo26s.yaml``/``yolo26m.yaml`` 是 Ultralytics 支持的复合缩放别名；
    加载时会读取实际存在的 ``yolo26.yaml``，并根据文件名选择 s/m scale。
    """
    model_size = normalized_model_size()
    return BASE_MODEL_CONFIG_PATH.with_name(f"yolo26{model_size}.yaml")


def ensure_loaded_model_size(model) -> None:
    """确保已加载权重的网络规模与 ``MODEL_SIZE`` 一致。

    验证和测试权重的路径需要人工指定；此检查用于防止将 S 权重
    误标为 M，或将 M 权重误标为 S。
    """
    network = getattr(model, "model", model)
    model_yaml = getattr(network, "yaml", {}) or {}
    loaded_size = str(model_yaml.get("scale", "")).lower().strip()
    expected_size = normalized_model_size()
    if loaded_size and loaded_size != expected_size:
        raise ValueError(
            f'权重内的模型规模为 {loaded_size!r}，但 experiment_config.py '
            f'中 MODEL_SIZE={MODEL_SIZE!r}。请统一后再运行。'
        )


MODEL_TAG = f"YOLO26{normalized_model_size()}"
PRETRAINED_WEIGHT_PATH = PRETRAINED_WEIGHT_PATHS[normalized_model_size()]
MODEL_CONFIG_PATH = selected_model_config_path()
