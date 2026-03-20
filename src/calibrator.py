from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

from metrics import compute_video_metrics
from noise_factory import build_noise
from utils import uniform_sample_indices
from video_processor import VideoProcessor


DEFAULT_SEVERITY_TARGETS: dict[int, float] = {
    1: 0.95,
    2: 0.90,
    3: 0.85,
    4: 0.80,
    5: 0.75,
}


class Calibrator:
    def __init__(self, config: dict, noise_catalog: dict, logger: logging.Logger) -> None:
        """初始化校准器，复用视频处理器进行解码与采样。"""
        self.config = config
        self.noise_catalog = noise_catalog
        self.logger = logger
        self.processor = VideoProcessor(config=config, noise_catalog=noise_catalog, logger=logger)

    def _score(self, metric: float | None, target: float | None) -> float:
        """计算指标与目标值的距离分数。"""
        if metric is None or target is None:
            return float("inf")
        return abs(metric - target)

    def _iter_param_candidates(self, search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """展开多参数搜索空间为候选参数组合。"""
        keys = list(search_space.keys())
        value_lists = [list(search_space[k]) for k in keys]
        combos = itertools.product(*value_lists)
        return [{k: v for k, v in zip(keys, values)} for values in combos]

    def _psnr_range_penalty(
        self,
        avg_psnr: float | None,
        psnr_range: tuple[float, float] | None,
    ) -> float:
        """若 PSNR 超出合理区间，返回惩罚值。"""
        if avg_psnr is None or psnr_range is None:
            return 0.0

        low, high = psnr_range
        if avg_psnr < low:
            return low - avg_psnr
        if avg_psnr > high:
            return avg_psnr - high
        return 0.0

    def calibrate(
        self,
        noise_name: str,
        sample_videos: list[Path],
        target_ssim: float | None = None,
        target_psnr: float | None = None,
        psnr_range: tuple[float, float] | None = None,
        search_space: dict[str, list[Any]] | None = None,
        severity: int | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """对指定噪声做参数搜索，输出最优参数建议。"""
        if not sample_videos:
            raise ValueError("sample_videos is empty")

        noise_meta = self.noise_catalog.get(noise_name, {})
        if not search_space:
            search_space = dict(noise_meta.get("calibration_space", {}))
        if not search_space:
            raise ValueError(
                f"search_space is empty for noise={noise_name}; "
                "provide search_space or configure calibration_space in noise_catalog"
            )

        candidates = self._iter_param_candidates(search_space)

        best: dict[str, Any] | None = None

        for params in candidates:
            noise = build_noise(noise_name, params=params, noise_catalog=self.noise_catalog)

            psnr_list: list[float] = []
            ssim_list: list[float] = []

            for video_path in sample_videos:
                frames, _ = self.processor._decode_video(video_path)
                sampled_idx = uniform_sample_indices(
                    len(frames), int(self.config["video"]["num_sampled_frames"])
                )
                sampled_frames = [frames[i] for i in sampled_idx]
                selected = list(range(len(sampled_frames)))
                noisy_frames = noise.apply(
                    sampled_frames,
                    selected_indices=selected,
                    severity=severity,
                    seed=seed,
                )
                metrics = compute_video_metrics(
                    sampled_frames,
                    noisy_frames,
                    frame_indices=selected,
                )
                if metrics["avg_psnr"] is not None:
                    psnr_list.append(float(metrics["avg_psnr"]))
                if metrics["avg_ssim"] is not None:
                    ssim_list.append(float(metrics["avg_ssim"]))

            avg_psnr = float(sum(psnr_list) / len(psnr_list)) if psnr_list else None
            avg_ssim = float(sum(ssim_list) / len(ssim_list)) if ssim_list else None

            score_psnr = self._score(avg_psnr, target_psnr) if target_psnr is not None else 0.0
            score_ssim = self._score(avg_ssim, target_ssim) if target_ssim is not None else 0.0
            score_psnr_range = self._psnr_range_penalty(avg_psnr, psnr_range)
            # 以 SSIM 对齐为主，PSNR 只作为合理区间约束。
            score = score_ssim + 0.2 * score_psnr + 0.5 * score_psnr_range

            trial = {
                "params": params,
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
                "score": score,
            }

            if best is None or score < best["score"]:
                best = trial

        assert best is not None

        result = {
            "noise_name": noise_name,
            "severity": severity,
            "target_psnr": target_psnr,
            "psnr_range": list(psnr_range) if psnr_range is not None else None,
            "target_ssim": target_ssim,
            "best": best,
            "search_space": search_space,
            "num_samples": len(sample_videos),
        }

        report_dir = Path(self.config["paths"]["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"calibration_{noise_name}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self.logger.info("calibration saved: %s", report_path)
        return result

    def calibrate_severity_profile(
        self,
        noise_name: str,
        sample_videos: list[Path],
        severity_targets: dict[int, float] | None = None,
        psnr_range: tuple[float, float] | None = None,
        search_space_by_severity: dict[int, dict[str, list[Any]]] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """按多个 severity 逐一校准，产出可写回 noise_catalog 的参数表。"""
        if severity_targets is None:
            severity_targets = dict(DEFAULT_SEVERITY_TARGETS)

        severity_params: dict[str, dict[str, Any]] = {}
        trials: dict[str, Any] = {}

        for severity, target_ssim in sorted(severity_targets.items(), key=lambda x: x[0]):
            search_space = None
            if search_space_by_severity:
                search_space = search_space_by_severity.get(severity)

            result = self.calibrate(
                noise_name=noise_name,
                sample_videos=sample_videos,
                target_ssim=float(target_ssim),
                target_psnr=None,
                psnr_range=psnr_range,
                search_space=search_space,
                severity=int(severity),
                seed=seed,
            )
            best = result["best"]
            severity_params[str(int(severity))] = dict(best["params"])
            trials[str(int(severity))] = {
                "target_ssim": float(target_ssim),
                "avg_ssim": best["avg_ssim"],
                "avg_psnr": best["avg_psnr"],
                "score": best["score"],
            }

        profile = {
            "noise_name": noise_name,
            "severity_params": severity_params,
            "trials": trials,
            "psnr_range": list(psnr_range) if psnr_range is not None else None,
            "num_samples": len(sample_videos),
        }

        report_dir = Path(self.config["paths"]["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"calibration_profile_{noise_name}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        self.logger.info("calibration profile saved: %s", report_path)
        return profile

    def write_profile_to_catalog(
        self,
        noise_name: str,
        profile: dict[str, Any],
        catalog_path: str | Path,
    ) -> None:
        """将校准得到的 severity_params 回写到 noise_catalog.yaml。"""
        import yaml

        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"noise catalog not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            catalog = yaml.safe_load(f) or {}

        if noise_name not in catalog:
            raise KeyError(f"noise '{noise_name}' not found in catalog")

        severity_params = dict(profile.get("severity_params", {}))
        if not severity_params:
            raise ValueError("profile.severity_params is empty")

        catalog[noise_name]["severity_params"] = severity_params

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(catalog, f, sort_keys=False, allow_unicode=True)

        self.logger.info("catalog updated with severity_params: %s", path)
