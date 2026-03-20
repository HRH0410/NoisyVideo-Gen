from __future__ import annotations

import logging
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from metrics import compute_video_metrics
from noise_factory import build_noise
from utils import (
    BenchmarkRecord,
    choose_selected_indices,
    ensure_dir,
    stable_int_hash,
    uniform_sample_indices,
    write_jsonl_record,
)


class VideoProcessor:
    def __init__(
        self,
        config: dict,
        noise_catalog: dict,
        logger: logging.Logger,
    ) -> None:
        """初始化处理器，注入配置、噪声目录和日志器。"""
        self.config = config
        self.noise_catalog = noise_catalog
        self.logger = logger

    def _decode_video(self, video_path: Path) -> tuple[list[np.ndarray], float]:
        """解码视频并返回全部帧与原始 FPS。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded: {video_path}")
        return frames, fps

    def _write_video(self, frames: list[np.ndarray], output_path: Path, fps: float) -> None:
        """将帧序列编码为输出视频，优先使用 ffmpeg/libx264。"""
        if not frames:
            raise ValueError("Cannot write empty frame list")

        ensure_dir(output_path.parent)
        try:
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            iio.imwrite(
                output_path,
                rgb_frames,
                fps=float(max(1.0, fps)),
                codec="libx264",
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
            return
        except Exception:
            pass

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create writer: {output_path}")

        for frame in frames:
            writer.write(frame)
        writer.release()

    def _save_preview(
        self,
        clean_frames: list[np.ndarray],
        noisy_frames: list[np.ndarray],
        selected_indices: list[int],
        preview_path: Path,
    ) -> None:
        """保存 clean/noisy 对照拼图，便于快速人工检查噪声效果。"""
        if not clean_frames or not noisy_frames:
            return

        selected = set(selected_indices)
        tiles_clean: list[np.ndarray] = []
        tiles_noisy: list[np.ndarray] = []
        target_h = 180

        for i, (clean, noisy) in enumerate(zip(clean_frames, noisy_frames)):
            h, w = clean.shape[:2]
            target_w = max(1, int(round(w * (target_h / max(1, h)))))

            clean_tile = cv2.resize(clean, (target_w, target_h), interpolation=cv2.INTER_AREA)
            noisy_tile = cv2.resize(noisy, (target_w, target_h), interpolation=cv2.INTER_AREA)

            cv2.putText(clean_tile, f"idx={i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tag = "noisy*" if i in selected else "noisy"
            cv2.putText(noisy_tile, f"idx={i} {tag}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            tiles_clean.append(clean_tile)
            tiles_noisy.append(noisy_tile)

        top = np.concatenate(tiles_clean, axis=1)
        bottom = np.concatenate(tiles_noisy, axis=1)
        grid = np.concatenate([top, bottom], axis=0)

        ensure_dir(preview_path.parent)
        cv2.imwrite(str(preview_path), grid)

    def process_video(
        self,
        video_path: Path,
        noise_name: str,
        severity: int | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> BenchmarkRecord:
        """处理单个视频并返回 manifest 记录。"""
        project_seed = int(self.config["project"]["seed"])
        run_seed = int(project_seed if seed is None else seed)

        video_cfg = self.config["video"]
        runtime_cfg = self.config["runtime"]
        paths_cfg = self.config["paths"]

        frames, original_fps = self._decode_video(video_path)
        sampled_indices = uniform_sample_indices(
            num_frames=len(frames),
            num_sampled_frames=int(video_cfg["num_sampled_frames"]),
        )
        sampled_frames = [frames[i].copy() for i in sampled_indices]

        sample_seed = (run_seed + stable_int_hash(f"{video_path.stem}:{noise_name}:{severity}")) & 0xFFFFFFFF
        rng = np.random.default_rng(sample_seed)
        selected_indices = choose_selected_indices(
            sampled_count=len(sampled_frames),
            frame_ratio=float(video_cfg["frame_ratio"]),
            rng=rng,
        )

        noise_obj = build_noise(
            name=noise_name,
            params=noise_params,
            noise_catalog=self.noise_catalog,
            severity=severity,
        )
        noisy_frames = noise_obj.apply(
            sampled_frames,
            selected_indices=selected_indices,
            severity=severity,
            seed=sample_seed,
        )

        metric_psnr = None
        metric_ssim = None
        if bool(runtime_cfg.get("save_metrics", True)):
            metric_result = compute_video_metrics(
                sampled_frames,
                noisy_frames,
                frame_indices=selected_indices if selected_indices else None,
            )
            metric_psnr = metric_result["avg_psnr"]
            metric_ssim = metric_result["avg_ssim"]

        severity_part = f"severity_{severity}" if severity is not None else "severity_none"
        out_dir = Path(paths_cfg["output_dir"]) / noise_name / severity_part
        output_path = out_dir / f"{video_path.stem}.mp4"

        output_fps = float(video_cfg.get("output_fps", 0))
        if output_fps <= 0:
            output_fps = original_fps if original_fps > 0 else 8.0
        self._write_video(noisy_frames, output_path, output_fps)

        if bool(runtime_cfg.get("save_preview", False)):
            preview_dir = Path(paths_cfg["preview_dir"]) / noise_name / severity_part
            preview_path = preview_dir / f"{video_path.stem}.jpg"
            self._save_preview(sampled_frames, noisy_frames, selected_indices, preview_path)

        noise_meta = self.noise_catalog.get(noise_name, {})
        record = BenchmarkRecord(
            video_id=video_path.stem,
            clean_path=video_path.as_posix(),
            noisy_path=output_path.as_posix(),
            noise_name=noise_name,
            noise_category=str(noise_meta.get("category", "unknown")),
            severity=severity,
            frame_ratio=float(video_cfg["frame_ratio"]),
            sampled_frame_count=len(sampled_frames),
            sampled_indices=sampled_indices,
            selected_indices=selected_indices,
            seed=sample_seed,
            psnr=metric_psnr,
            ssim=metric_ssim,
        )

        if bool(runtime_cfg.get("save_manifest", True)):
            manifest_path = Path(paths_cfg["manifest_dir"]) / f"{noise_name}.jsonl"
            write_jsonl_record(manifest_path, record)

        return record

    def process_batch(
        self,
        video_paths: list[Path],
        noise_name: str,
        severity: int | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> tuple[list[BenchmarkRecord], list[tuple[Path, str]]]:
        """按噪声配置批量处理视频，失败样本会被记录但不中断。"""
        records: list[BenchmarkRecord] = []
        failed: list[tuple[Path, str]] = []

        for video_path in tqdm(video_paths, desc=f"noise={noise_name}"):
            try:
                record = self.process_video(
                    video_path=video_path,
                    noise_name=noise_name,
                    severity=severity,
                    noise_params=noise_params,
                    seed=seed,
                )
                records.append(record)
                self.logger.info(
                    "processed video=%s noise=%s psnr=%s ssim=%s",
                    video_path.name,
                    noise_name,
                    record.psnr,
                    record.ssim,
                )
            except Exception as exc:
                msg = str(exc)
                failed.append((video_path, msg))
                self.logger.exception("failed video=%s noise=%s error=%s", video_path.name, noise_name, msg)

        return records, failed
