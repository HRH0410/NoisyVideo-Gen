from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class VideoSample:
    # 原始视频样本的基础信息结构。
    video_id: str
    path: Path
    fps: float | None = None
    num_frames: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class NoiseSpec:
    # 一次扰动任务的配置快照。
    category: str
    name: str
    severity: int | None
    frame_ratio: float
    params: dict[str, Any]


@dataclass
class BenchmarkRecord:
    # manifest 单条记录，对应一个输出视频。
    video_id: str
    clean_path: str
    noisy_path: str
    noise_name: str
    noise_category: str
    severity: int | None
    frame_ratio: float
    sampled_frame_count: int
    sampled_indices: list[int]
    selected_indices: list[int]
    seed: int
    psnr: float | None = None
    ssim: float | None = None


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path 对象。"""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def setup_logger(log_dir: str | Path, log_level: str = "INFO") -> logging.Logger:
    """初始化项目日志器，输出到文件与控制台。"""
    ensure_dir(log_dir)
    logger = logging.getLogger("noisyvideo")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_global_seed(seed: int) -> None:
    """设置全局随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def stable_int_hash(text: str) -> int:
    """将字符串转换为稳定整数哈希，避免内置 hash 的进程随机化。"""
    value = 2166136261
    for ch in text.encode("utf-8"):
        value ^= ch
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def list_video_files(input_dir: str | Path) -> list[Path]:
    """扫描输入目录下支持格式的视频文件（仅第一层）。"""
    root = Path(input_dir)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def uniform_sample_indices(num_frames: int, num_sampled_frames: int) -> list[int]:
    """按时间轴均匀抽样，返回抽样帧索引。"""
    if num_frames <= 0:
        return []
    if num_sampled_frames <= 0:
        return []
    if num_frames <= num_sampled_frames:
        return list(range(num_frames))

    raw = np.linspace(0, num_frames - 1, num=num_sampled_frames)
    return [int(round(v)) for v in raw]


def choose_selected_indices(
    sampled_count: int,
    frame_ratio: float,
    rng: np.random.Generator,
) -> list[int]:
    """按 frame_ratio 随机选择需要施加扰动的 sampled 帧索引。"""
    if sampled_count <= 0:
        return []

    ratio = float(np.clip(frame_ratio, 0.0, 1.0))
    k = max(1, int(round(sampled_count * ratio))) if ratio > 0 else 0
    if k <= 0:
        return []

    selected = rng.choice(sampled_count, size=min(k, sampled_count), replace=False)
    return sorted(int(i) for i in selected)


def write_jsonl_record(manifest_path: str | Path, record: BenchmarkRecord) -> None:
    """将单条 BenchmarkRecord 以 jsonl 追加写入 manifest。"""
    ensure_dir(Path(manifest_path).parent)
    payload = asdict(record)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件并返回字典对象。"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
