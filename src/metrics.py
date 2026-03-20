from __future__ import annotations

from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(clean_frame: np.ndarray, noisy_frame: np.ndarray) -> float:
    """计算单帧 PSNR。"""
    clean = clean_frame.astype(np.float32)
    noisy = noisy_frame.astype(np.float32)
    mse = float(np.mean((clean - noisy) ** 2))
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(clean_frame: np.ndarray, noisy_frame: np.ndarray) -> float:
    """计算单帧 SSIM，兼容灰度图与彩色图。"""
    clean = clean_frame.astype(np.uint8)
    noisy = noisy_frame.astype(np.uint8)
    if clean.ndim == 2:
        return float(structural_similarity(clean, noisy, data_range=255))

    return float(
        structural_similarity(
            clean,
            noisy,
            channel_axis=-1,
            data_range=255,
        )
    )


def compute_video_metrics(
    clean_frames: list[np.ndarray],
    noisy_frames: list[np.ndarray],
    frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    """聚合视频级指标，支持只统计指定索引帧。"""
    if len(clean_frames) != len(noisy_frames):
        raise ValueError("clean_frames and noisy_frames length mismatch")
    if not clean_frames:
        return {
            "avg_psnr": None,
            "avg_ssim": None,
            "min_psnr": None,
            "max_psnr": None,
            "std_psnr": None,
            "min_ssim": None,
            "max_ssim": None,
            "std_ssim": None,
        }

    if frame_indices is None:
        pairs = list(zip(clean_frames, noisy_frames))
    else:
        valid_idx = [i for i in frame_indices if 0 <= i < len(clean_frames)]
        pairs = [(clean_frames[i], noisy_frames[i]) for i in valid_idx]

    if not pairs:
        return {
            "avg_psnr": None,
            "avg_ssim": None,
            "min_psnr": None,
            "max_psnr": None,
            "std_psnr": None,
            "min_ssim": None,
            "max_ssim": None,
            "std_ssim": None,
        }

    psnr_vals = [compute_psnr(c, n) for c, n in pairs]
    ssim_vals = [compute_ssim(c, n) for c, n in pairs]

    finite_psnr = [v for v in psnr_vals if np.isfinite(v)]
    if not finite_psnr:
        finite_psnr = psnr_vals

    psnr_arr = np.array(finite_psnr, dtype=np.float32)
    ssim_arr = np.array(ssim_vals, dtype=np.float32)

    return {
        "avg_psnr": float(np.mean(psnr_arr)),
        "avg_ssim": float(np.mean(ssim_arr)),
        "min_psnr": float(np.min(psnr_arr)),
        "max_psnr": float(np.max(psnr_arr)),
        "std_psnr": float(np.std(psnr_arr)),
        "min_ssim": float(np.min(ssim_arr)),
        "max_ssim": float(np.max(ssim_arr)),
        "std_ssim": float(np.std(ssim_arr)),
    }
