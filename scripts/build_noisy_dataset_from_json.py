from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_factory import build_noise, list_available_noises  # noqa: E402
from utils import choose_selected_indices, ensure_dir, load_yaml, stable_int_hash, uniform_sample_indices  # noqa: E402

FAST33_NOISES = [
    "gaussian_noise",
    "poisson_noise",
    "impulse_noise",
    "speckle_noise",
    "frame_drop",
    "frame_replace",
    "frame_repeat",
    "temporal_jitter",
    "gaussian_blur",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "brightness",
    "contrast",
    "color_shift",
    "flicker",
    "overexposure",
    "underexposure",
    "shadow",
    "specular_reflection",
    "frost",
    "snow",
    "rolling_shutter",
    "resolution_degrade",
    "stretch_squish",
    "edge_sawtooth",
    "color_quantization",
    "elastic_transform",
    "random_block",
    "target_block",
    "jpeg_artifact",
    "bit_error",
    "h265_compression",
]
FAST32_NOISES = [
    "poisson_noise",
    "impulse_noise",
    "speckle_noise",
    "frame_drop",
    "frame_replace",
    "frame_repeat",
    "temporal_jitter",
    "gaussian_blur",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "brightness",
    "contrast",
    "color_shift",
    "flicker",
    "overexposure",
    "underexposure",
    "shadow",
    "specular_reflection",
    "frost",
    "snow",
    "rolling_shutter",
    "resolution_degrade",
    "stretch_squish",
    "edge_sawtooth",
    "color_quantization",
    "elastic_transform",
    "random_block",
    "target_block",
    "jpeg_artifact",
    "bit_error",
    "h265_compression",
]
SLOW_NOISES = ["rain", "fog", "glass_blur"]


@dataclass
class OutputRecord:
    video_id: str
    source_rel_path: str
    source_path: str
    output_path: str
    noise_name: str
    severity: int | None
    sampled_indices: list[int]
    selected_indices: list[int]
    sampled_frame_count: int
    fps: float
    status: str
    error: str | None = None


@dataclass
class WorkerContext:
    video_root: Path
    output_root: Path
    noise_catalog: dict[str, Any]
    num_sampled_frames: int
    frame_ratio: float
    output_fps: float
    seed: int
    overwrite: bool


_WORKER_CONTEXT: WorkerContext | None = None


def _init_worker(context: WorkerContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _process_video_task(task: tuple[int, str, str, int | None]) -> tuple[str, OutputRecord]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context has not been initialized")

    idx, rel_path, noise_name, severity = task
    video_root = _WORKER_CONTEXT.video_root
    output_root = _WORKER_CONTEXT.output_root
    num_sampled_frames = _WORKER_CONTEXT.num_sampled_frames
    frame_ratio = _WORKER_CONTEXT.frame_ratio
    output_fps = _WORKER_CONTEXT.output_fps
    seed = _WORKER_CONTEXT.seed
    overwrite = _WORKER_CONTEXT.overwrite
    noise_catalog = _WORKER_CONTEXT.noise_catalog

    src_path, rel_norm = resolve_source_path(video_root, rel_path)
    output_path = output_root / "videos" / noise_name / rel_norm

    if output_path.exists() and not overwrite:
        return (
            "skipped",
            OutputRecord(
                video_id=Path(rel_norm).stem if rel_norm else Path(rel_path).stem,
                source_rel_path=rel_norm,
                source_path=src_path.as_posix(),
                output_path=output_path.as_posix(),
                noise_name=noise_name,
                severity=severity,
                sampled_indices=[],
                selected_indices=[],
                sampled_frame_count=0,
                fps=float(output_fps),
                status="skipped",
                error=None,
            ),
        )

    try:
        if not src_path.exists():
            raise FileNotFoundError(f"源视频不存在: {src_path}")

        frames, src_fps = decode_video(src_path)
        sampled_indices = uniform_sample_indices(len(frames), num_sampled_frames)
        sampled_frames = [frames[i].copy() for i in sampled_indices]
        selected_indices = choose_selected_indices(
            sampled_count=len(sampled_frames),
            frame_ratio=float(frame_ratio),
            rng=np.random.default_rng((int(seed) + stable_int_hash(f"select:{rel_norm}:{idx}")) & 0xFFFFFFFF),
        )

        sample_seed = (int(seed) + stable_int_hash(f"{rel_norm}:{noise_name}:{severity}:{idx}")) & 0xFFFFFFFF
        noise = build_noise(
            name=noise_name,
            params=None,
            noise_catalog=noise_catalog,
            severity=severity,
        )
        noisy_frames = noise.apply(
            sampled_frames,
            selected_indices=selected_indices,
            severity=severity,
            seed=sample_seed,
        )

        write_video(noisy_frames, output_path, output_fps if output_fps > 0 else (src_fps or 8.0))
        return (
            "ok",
            OutputRecord(
                video_id=Path(rel_norm).stem,
                source_rel_path=rel_norm,
                source_path=src_path.as_posix(),
                output_path=output_path.as_posix(),
                noise_name=noise_name,
                severity=severity,
                sampled_indices=sampled_indices,
                selected_indices=selected_indices,
                sampled_frame_count=len(sampled_frames),
                fps=float(output_fps if output_fps > 0 else (src_fps or 8.0)),
                status="ok",
                error=None,
            ),
        )
    except Exception as exc:
        return (
            "failed",
            OutputRecord(
                video_id=Path(rel_norm).stem if rel_norm else Path(rel_path).stem,
                source_rel_path=rel_norm,
                source_path=src_path.as_posix(),
                output_path=output_path.as_posix(),
                noise_name=noise_name,
                severity=severity,
                sampled_indices=[],
                selected_indices=[],
                sampled_frame_count=0,
                fps=float(output_fps),
                status="failed",
                error=str(exc),
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从训练 JSON 中提取视频，均匀抽 8 帧并批量加噪，输出为短视频。"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/data0/lch/llama-factory/formatted_train_data_5000.json",
        help="训练 JSON 路径",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/data0/lch/llama-factory",
        help="JSON 中相对视频路径的根目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data0/lch/noisyvideo_gen_train_outputs",
        help="输出根目录",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="fast33",
        help="噪声选择：all / fast33 / fast32 / slow3 / 单噪声名 / 逗号列表",
    )
    parser.add_argument("--severity", type=str, default="5", help="1-5 或 none")
    parser.add_argument("--num_sampled_frames", type=int, default=8, help="均匀抽样帧数")
    parser.add_argument("--frame_ratio", type=float, default=0.9, help="在抽样帧中施加噪声的比例(0-1)")
    parser.add_argument("--output_fps", type=float, default=8.0, help="输出短视频 FPS")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_videos", type=int, default=0, help="仅处理前 N 个视频，0 表示全量")
    parser.add_argument("--resume", action="store_true", help="遇到已存在输出时跳过")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在输出")
    parser.add_argument("--dry_run", action="store_true", help="只打印计划，不写文件")
    parser.add_argument(
        "--noise_catalog",
        type=str,
        default=str(ROOT / "configs" / "noise_catalog.yaml"),
        help="noise_catalog.yaml 路径",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/data0/lch/noisyvideo_gen_train_outputs/logs",
        help="日志目录",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发工作进程数；1 表示串行",
    )
    return parser


def setup_logger(log_dir: str) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger("noisyvideo_extract")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_severity(value: str) -> int | None:
    if value is None:
        return 5
    if value.lower() == "none":
        return None
    sev = int(value)
    if not 1 <= sev <= 5:
        raise ValueError("severity 必须是 1-5 或 none")
    return sev


def resolve_noise_list(noise_arg: str) -> list[str]:
    available = list_available_noises()
    if noise_arg == "all":
        return available
    if noise_arg == "fast33":
        return FAST33_NOISES
    if noise_arg == "fast32":
        return FAST32_NOISES
    if noise_arg == "slow3":
        return SLOW_NOISES

    selected = [x.strip() for x in noise_arg.split(",") if x.strip()]
    unknown = [x for x in selected if x not in available]
    if unknown:
        raise ValueError(f"未知噪声: {unknown}; 可用: {available}")
    return selected


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("训练 JSON 顶层必须是 list")
    return data


def extract_video_paths(samples: list[dict[str, Any]]) -> list[str]:
    ordered = OrderedDict[str, None]()
    for item in samples:
        if not isinstance(item, dict):
            continue
        videos = item.get("videos")
        if isinstance(videos, list):
            for video_path in videos:
                if isinstance(video_path, str) and video_path.strip():
                    ordered.setdefault(video_path.strip(), None)
        elif isinstance(item.get("video"), str):
            video_path = item["video"].strip()
            if video_path:
                ordered.setdefault(video_path, None)
    return list(ordered.keys())


def resolve_source_path(video_root: Path, rel_path: str) -> tuple[Path, str]:
    rel = Path(rel_path)
    if rel.is_absolute():
        return rel, rel.as_posix().lstrip("/")
    return video_root / rel, rel.as_posix()


def decode_video(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"视频没有有效帧: {video_path}")
    return frames, fps


def write_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("frames 不能为空")

    # Pre-pad all frames to 16x alignment so ffmpeg will not auto-resize and spam warnings.
    h, w = frames[0].shape[:2]
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    if pad_h or pad_w:
        frames = [
            cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)
            for frame in frames
        ]

    ensure_dir(output_path.parent)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    try:
        import imageio.v3 as iio

        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        iio.imwrite(
            temp_path,
            rgb_frames,
            fps=float(max(1.0, fps)),
            codec="libx264",
        )
        temp_path.replace(output_path)
        return
    except Exception:
        pass

    height, width = frames[0].shape[:2]
    if temp_path.exists():
        temp_path.unlink()
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    temp_path.replace(output_path)

def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def append_jsonl(path: Path, record: OutputRecord) -> None:
    ensure_dir(path.parent)
    payload = asdict(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=jsonable) + "\n")


def record_result(
    *,
    noise_name: str,
    manifest_path: Path,
    summary: dict[str, dict[str, int]],
    total_ok: int,
    total_failed: int,
    result: tuple[str, OutputRecord],
    logger: logging.Logger,
) -> tuple[int, int]:
    status, record = result
    if status == "skipped":
        summary[noise_name]["skipped"] += 1
        return total_ok, total_failed

    append_jsonl(manifest_path, record)
    if status == "ok":
        summary[noise_name]["ok"] += 1
        total_ok += 1
        return total_ok, total_failed

    summary[noise_name]["failed"] += 1
    total_failed += 1
    logger.error("failed noise=%s source=%s error=%s", noise_name, record.source_rel_path, record.error)
    return total_ok, total_failed


def main() -> None:
    args = build_parser().parse_args()
    severity = parse_severity(args.severity)
    noise_names = resolve_noise_list(args.noise)

    json_path = Path(args.json_path)
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    noise_catalog = load_yaml(args.noise_catalog)
    samples = load_json(str(json_path))
    video_paths_rel = extract_video_paths(samples)
    logger = setup_logger(args.log_dir)

    if args.max_videos and args.max_videos > 0:
        video_paths_rel = video_paths_rel[: args.max_videos]

    logger.info("json_path=%s", json_path)
    logger.info("video_root=%s", video_root)
    logger.info("output_root=%s", output_root)
    logger.info("videos=%d noises=%d severity=%s", len(video_paths_rel), len(noise_names), severity)
    logger.info("noise_names=%s", ",".join(noise_names))

    if args.dry_run:
        for rel_path in video_paths_rel[:20]:
            src_path, rel_norm = resolve_source_path(video_root, rel_path)
            logger.info("dry_run example source=%s resolved=%s", rel_norm, src_path)
        return

    ensure_dir(output_root)
    ensure_dir(output_root / "manifests")
    ensure_dir(output_root / "videos")

    summary: dict[str, dict[str, int]] = {}
    total_ok = 0
    total_failed = 0
    worker_context = WorkerContext(
        video_root=video_root,
        output_root=output_root,
        noise_catalog=noise_catalog,
        num_sampled_frames=args.num_sampled_frames,
        frame_ratio=float(args.frame_ratio),
        output_fps=float(args.output_fps),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    _init_worker(worker_context)
    tasks = [(idx, rel_path) for idx, rel_path in enumerate(video_paths_rel)]

    for noise_name in noise_names:
        summary[noise_name] = {"ok": 0, "failed": 0, "skipped": 0}
        manifest_path = output_root / "manifests" / f"{noise_name}.jsonl"
        if args.workers > 1:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(worker_context,),
            ) as executor:
                futures = [executor.submit(_process_video_task, (idx, rel_path, noise_name, severity)) for idx, rel_path in tasks]
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    total_ok, total_failed = record_result(
                        noise_name=noise_name,
                        manifest_path=manifest_path,
                        summary=summary,
                        total_ok=total_ok,
                        total_failed=total_failed,
                        result=result,
                        logger=logger,
                    )
                    completed += 1
                    if completed % 50 == 0:
                        logger.info("noise=%s progress=%d/%d", noise_name, completed, len(tasks))
        else:
            for idx, rel_path in tasks:
                result = _process_video_task((idx, rel_path, noise_name, severity))
                total_ok, total_failed = record_result(
                    noise_name=noise_name,
                    manifest_path=manifest_path,
                    summary=summary,
                    total_ok=total_ok,
                    total_failed=total_failed,
                    result=result,
                    logger=logger,
                )
                if (idx + 1) % 50 == 0:
                    logger.info("noise=%s progress=%d/%d", noise_name, idx + 1, len(tasks))

    report = {
        "json_path": json_path.as_posix(),
        "video_root": video_root.as_posix(),
        "output_root": output_root.as_posix(),
        "num_videos": len(video_paths_rel),
        "noise_names": noise_names,
        "severity": severity,
        "num_sampled_frames": args.num_sampled_frames,
        "frame_ratio": args.frame_ratio,
        "output_fps": args.output_fps,
        "total_ok": total_ok,
        "total_failed": total_failed,
        "by_noise": summary,
    }
    with open(output_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("done total_ok=%d total_failed=%d", total_ok, total_failed)


if __name__ == "__main__":
    main()
