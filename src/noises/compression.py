from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import BaseNoiseLike


class JPEGArtifact(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """初始化 JPEG 压缩伪影扰动。"""
        super().__init__(name="jpeg_artifact", category="compression", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """JPEG artifact：选中帧按指定 q 进行 JPEG 编解码。"""
        quality = int(np.clip(int(self.params.get("quality", 1)), 1, 100))

        selected = set(selected_indices)
        out: list[np.ndarray] = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            ok, enc = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not ok:
                out.append(frame.copy())
                continue

            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            out.append(frame.copy() if dec is None else dec)
        return out


class BitError(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """Bit error：随机大矩形内竖条复制伪影。"""
        super().__init__(name="bit_error", category="compression", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        sev = 5 if severity is None else int(np.clip(severity, 1, 5))
        if sev <= 2:
            stripe_w = 15
        elif sev == 3:
            stripe_w = 10
        else:
            stripe_w = 5

        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            h, w = frame.shape[:2]
            corrupted = frame.copy()

            # 论文：区域覆盖 1/4 到 1/2 的整帧面积。
            area_ratio = float(rng.uniform(0.25, 0.5))
            target_area = area_ratio * h * w

            # 先采样一个高，再由目标面积推宽。
            region_h = int(rng.integers(max(1, h // 4), h + 1))
            region_w = int(round(target_area / max(region_h, 1)))
            region_w = int(np.clip(region_w, 1, w))

            # 反向按宽修正高，保证面积更接近 target_area。
            region_h = int(round(target_area / max(region_w, 1)))
            region_h = int(np.clip(region_h, 1, h))

            y0 = int(rng.integers(0, h - region_h + 1))
            x0 = int(rng.integers(0, w - region_w + 1))
            y1 = y0 + region_h
            x1 = x0 + region_w

            for sx in range(x0, x1, stripe_w):
                stripe_end = min(sx + stripe_w, x1)
                src_col = int(rng.integers(x0, x1))
                col_pixels = corrupted[y0:y1, src_col:src_col + 1, :]
                corrupted[y0:y1, sx:stripe_end, :] = col_pixels

            out.append(corrupted)
        return out


class H265Compression(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """H.265 压缩失真。"""
        super().__init__(name="h265_compression", category="compression", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """H.265 artifacts：先降到 1/4 分辨率，再恢复原尺寸，并在 full-size 上做两次 H.265 重编码。"""

        def _severity_level(sev: int | None) -> int:
            if sev is None:
                return 2
            sev_i = int(np.clip(sev, 1, 5))
            if sev_i <= 1:
                return 0
            if sev_i == 2:
                return 1
            if sev_i == 3:
                return 2
            return 3

        crf_table = [32, 38, 45, 51]
        bitrate_table = ["400k", "200k", "100k", "50k"]

        level = _severity_level(severity)
        crf = int(self.params.get("crf", crf_table[level]))
        bitrate = str(self.params.get("bitrate", bitrate_table[level]))
        preset = str(self.params.get("preset", "slow"))

        def _h265_roundtrip(frame_bgr: np.ndarray, crf_value: int, bitrate_value: str) -> np.ndarray:
            import imageio.v3 as iio

            with tempfile.TemporaryDirectory() as td:
                video_path = Path(td) / "tmp.mp4"
                target_h, target_w = frame_bgr.shape[:2]

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                iio.imwrite(
                    video_path,
                    [rgb],
                    fps=1,
                    codec="libx265",
                    ffmpeg_params=[
                        "-crf", str(crf_value),
                        "-b:v", bitrate_value,
                        "-preset", preset,
                        "-pix_fmt", "yuv420p",
                        "-tag:v", "hvc1",
                        "-movflags", "+faststart",
                    ],
                )

                decoded = iio.imread(video_path, index=0)
                if decoded.ndim == 4:
                    decoded = decoded[0]

                decoded = decoded.astype(np.uint8)
                decoded_bgr = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
                if decoded_bgr.shape[:2] != (target_h, target_w):
                    decoded_bgr = cv2.resize(decoded_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                return decoded_bgr

        selected = set(selected_indices)
        out: list[np.ndarray] = []
        fallback = JPEGArtifact(params={"quality": 5})

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            try:
                h, w = frame.shape[:2]
                q_w = max(1, w // 4)
                q_h = max(1, h // 4)

                # 1) 先降到 1/4 分辨率
                low_res = cv2.resize(frame, (q_w, q_h), interpolation=cv2.INTER_AREA)

                # 2) 再恢复到原尺寸（full size）
                restored = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)

                # 3) 在 full size 上做 two-pass H.265 编解码
                first_pass = _h265_roundtrip(restored, crf, bitrate)
                second_pass = _h265_roundtrip(first_pass, crf, bitrate)

                out.append(second_pass)

            except Exception:
                out.append(fallback.apply([frame], [0], severity=severity, seed=seed)[0])

        return out
