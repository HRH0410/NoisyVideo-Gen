from __future__ import annotations

import cv2
import numpy as np

from . import BaseNoiseLike


class Brightness(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="brightness", category="lighting", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Brightness transform：按 c_s=0.1*s 提升亮度。"""
        sev = 5 if severity is None else int(np.clip(severity, 1, 5))
        c_s = float(self.params.get("c_s", 0.1 * sev))
        c_s = float(np.clip(c_s, -1.0, 1.0))
        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                norm = frame.astype(np.float32) / 255.0
                lifted = np.clip(norm + c_s, 0.0, 1.0)
                out.append((lifted * 255.0).astype(np.uint8))
            else:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 2] = np.clip(hsv[..., 2] / 255.0 + c_s, 0.0, 1.0) * 255.0
                out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
        return out


class Contrast(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="contrast", category="lighting", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Contrast transform：围绕均值按随机 alpha 对称缩放。"""
        m = float(self.params.get("m", -5.0))
        M = float(self.params.get("M", 5.0))
        if m > M:
            m, M = M, m
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            alpha = float(rng.uniform(m, M))
            if alpha < 0.0:
                alpha = -1.0 / alpha
            mu = float(frame.mean())
            f = frame.astype(np.float32)
            out.append(np.clip(alpha * (f - mu) + mu, 0, 255).astype(np.uint8))
        return out


class ColorShift(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="color_shift", category="lighting", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Color shift：对通道乘以随机增益并裁剪。"""
        s = float(self.params.get("shift", 1.0))
        s = max(0.0, s)
        per_channel = bool(self.params.get("per_channel", True))
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            f = frame.astype(np.float32)
            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                alpha = float(rng.uniform(1.0 - s, 1.0 + s))
                shifted = f * alpha
            else:
                if per_channel:
                    gains = rng.uniform(1.0 - s, 1.0 + s, size=(1, 1, 3)).astype(np.float32)
                    shifted = f * gains
                else:
                    alpha = float(rng.uniform(1.0 - s, 1.0 + s))
                    shifted = f * alpha
            out.append(np.clip(shifted, 0, 255).astype(np.uint8))
        return out


class Flicker(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="flicker", category="lighting", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Flicker：逐帧采样 alpha~N(1,rho^2) 并缩放亮度。"""
        rho = float(self.params.get("rho", 0.2))
        rho = max(0.0, rho)
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue
            alpha = float(rng.normal(1.0, rho))
            out.append(np.clip(frame.astype(np.float32) * alpha, 0, 255).astype(np.uint8))
        return out


class Overexposure(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="overexposure", category="lighting", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Overexposure：正亮度偏移 + 逆gamma。"""
        b_min = float(self.params.get("b_min", 0.1))
        b_max = float(self.params.get("b_max", 0.3))
        g_min = float(self.params.get("gamma_min", 1.1))
        g_max = float(self.params.get("gamma_max", 1.4))
        if b_min > b_max:
            b_min, b_max = b_max, b_min
        if g_min > g_max:
            g_min, g_max = g_max, g_min
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            b = float(rng.uniform(b_min, b_max))
            gamma = float(rng.uniform(g_min, g_max))
            norm = frame.astype(np.float32) / 255.0
            shifted = np.clip(norm + b, 0.0, 1.0)
            corrected = np.power(shifted, 1.0 / gamma)
            out.append(np.clip(corrected * 255.0, 0, 255).astype(np.uint8))
        return out


class Underexposure(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="underexposure", category="lighting", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Underexposure：负亮度偏移 + 逆gamma。"""
        b_min = float(self.params.get("b_min", -0.3))
        b_max = float(self.params.get("b_max", -0.1))
        g_min = float(self.params.get("gamma_min", 0.6))
        g_max = float(self.params.get("gamma_max", 0.9))
        if b_min > b_max:
            b_min, b_max = b_max, b_min
        if g_min > g_max:
            g_min, g_max = g_max, g_min
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            b = float(rng.uniform(b_min, b_max))
            gamma = float(rng.uniform(g_min, g_max))
            norm = frame.astype(np.float32) / 255.0
            shifted = np.clip(norm + b, 0.0, 1.0)
            corrected = np.power(shifted, 1.0 / gamma)
            out.append(np.clip(corrected * 255.0, 0, 255).astype(np.uint8))
        return out

