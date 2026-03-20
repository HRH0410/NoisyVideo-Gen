from __future__ import annotations

import cv2
import numpy as np

from . import BaseNoiseLike


def _severity_index(severity: int | None) -> int:
    """将 severity 映射到 1~5。"""
    if severity is None:
        return 3
    return max(1, min(5, int(severity)))


def _apply_kernel_per_channel(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对每个通道执行 2D 卷积并裁剪到合法范围。"""
    out = cv2.filter2D(frame.astype(np.float32), -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    """生成指定方向的运动模糊核。"""
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    mat = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, mat, (size, size))
    kernel /= max(kernel.sum(), 1e-12)
    return kernel


def _disk_kernel(radius: int, sigma: float) -> np.ndarray:
    """生成带抗锯齿的圆盘核。"""
    k = radius * 2 + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = ((x * x + y * y) <= (radius * radius)).astype(np.float32)
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask /= max(mask.sum(), 1e-12)
    return mask


def _zoom_once(frame: np.ndarray, factor: float) -> np.ndarray:
    """按给定因子做一次中心裁剪+双线性回缩放。"""
    if factor <= 1.0:
        return frame.copy()

    h, w = frame.shape[:2]
    ch = max(1, int(round(h / factor)))
    cw = max(1, int(round(w / factor)))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    crop = frame[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


class GaussianBlur(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """初始化高斯模糊扰动。"""
        super().__init__(name="gaussian_blur", category="blur", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Gaussian blur: kernel=101, sigma=20（可由配置覆盖）。"""
        kernel = int(self.params.get("kernel_size", 101))
        if kernel % 2 == 0:
            kernel += 1
        kernel = max(3, kernel)
        sigma = float(self.params.get("sigma", 20.0))

        selected = set(selected_indices)
        out: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
            else:
                out.append(cv2.GaussianBlur(frame, (kernel, kernel), sigmaX=sigma, sigmaY=sigma))
        return out


class DefocusBlur(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="defocus_blur", category="blur", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Defocus blur: severity表驱动的 radius/sigma 圆盘核。"""
        s = _severity_index(severity)
        radius_table = {1: 3, 2: 4, 3: 6, 4: 8, 5: 10}
        sigma_table = {1: 0.1, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}

        radius = int(self.params.get("radius", radius_table[s]))
        sigma = float(self.params.get("anti_alias_sigma", sigma_table[s]))
        kernel = _disk_kernel(radius=radius, sigma=sigma)

        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            if i in selected:
                out.append(_apply_kernel_per_channel(frame, kernel))
            else:
                out.append(frame.copy())
        return out


class GlassBlur(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="glass_blur", category="blur", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Glass blur: 先高斯->局部洗牌->再高斯。"""
        s = _severity_index(severity)
        sigma_table = {1: 0.7, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.5}
        delta_table = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}
        iter_table = {1: 2, 2: 1, 3: 3, 4: 2, 5: 2}

        sigma = float(self.params.get("sigma", sigma_table[s]))
        max_delta = int(self.params.get("max_delta", delta_table[s]))
        n_iter = int(self.params.get("iterations", iter_table[s]))

        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            img = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
            h, w = img.shape[:2]

            for _ in range(max(1, n_iter)):
                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        dx = int(rng.integers(-max_delta, max_delta + 1))
                        dy = int(rng.integers(-max_delta, max_delta + 1))
                        nx = int(np.clip(x + dx, 0, w - 1))
                        ny = int(np.clip(y + dy, 0, h - 1))
                        img[y, x], img[ny, nx] = img[ny, nx].copy(), img[y, x].copy()

            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return out


class MotionBlur(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="motion_blur", category="blur", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Motion blur: 45° 方向，kernel=101（可由配置覆盖）。"""
        length = int(self.params.get("length", 101))
        length = max(3, length)
        if length % 2 == 0:
            length += 1
        angle = float(self.params.get("angle", 45.0))

        kernel = _motion_kernel(length, angle)

        selected = set(selected_indices)
        out = []
        for i, frame in enumerate(frames):
            out.append(_apply_kernel_per_channel(frame, kernel) if i in selected else frame.copy())
        return out


class ZoomBlur(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="zoom_blur", category="blur", params=params or {})

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        """Zoom blur: 按 severity 构建 zoom factors，做中心裁剪缩放平均。"""
        s = _severity_index(severity)
        delta_table = {1: 0.01, 2: 0.01, 3: 0.02, 4: 0.02, 5: 0.03}
        delta = float(self.params.get("delta", delta_table[s]))
        max_zoom = float(self.params.get("max_zoom", 1.0 + 0.1 * s))

        factors = []
        f = 1.0
        while f <= max_zoom + 1e-8:
            factors.append(float(f))
            f += delta

        selected = set(selected_indices)
        out = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            acc = frame.astype(np.float32)
            for z in factors:
                acc += _zoom_once(frame, z).astype(np.float32)
            out.append(np.clip(acc / (len(factors) + 1), 0, 255).astype(np.uint8))
        return out
