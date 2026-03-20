from __future__ import annotations

import numpy as np

from . import BaseNoiseLike, severity_scale


class GaussianNoise(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """高斯噪声。"""
        super().__init__(name="gaussian_noise", category="quality", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """对选中帧叠加高斯噪声。"""
        # 论文默认：零均值高斯噪声，标准差 100。
        sigma = float(self.params.get("sigma", 100.0)) * severity_scale(severity, 0.6, 1.8)
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)

        output: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            if i not in selected:
                output.append(frame.copy())
                continue
            noise = rng.normal(0.0, sigma, frame.shape).astype(np.float32)
            noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            output.append(noisy)
        return output


class PoissonNoise(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """泊松噪声。"""
        super().__init__(name="poisson_noise", category="quality", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        # 论文默认：先乘 gain (0.01) 作为泊松率，再除 gain 恢复尺度。
        gain = float(self.params.get("gain", 0.01)) * severity_scale(severity, 0.6, 1.8)
        gain = max(1e-6, gain)
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out: list[np.ndarray] = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue
            lam = np.clip(frame.astype(np.float32), 0.0, 255.0) * gain
            noisy = rng.poisson(lam).astype(np.float32) / gain
            out.append(np.clip(noisy, 0, 255).astype(np.uint8))
        return out


class ImpulseNoise(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """椒盐噪声。"""
        super().__init__(name="impulse_noise", category="quality", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        # 论文默认：以概率 p=0.7 执行冲击噪声；<p/2 置黑，p/2~p 置白。
        p = float(self.params.get("probability", 0.7)) * severity_scale(severity, 0.6, 1.6)
        p = float(np.clip(p, 0.0, 1.0))
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out: list[np.ndarray] = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue
            noisy = frame.copy()
            mask = rng.random(noisy.shape[:2])
            black = mask < (p * 0.5)
            white = (mask >= (p * 0.5)) & (mask < p)
            noisy[black] = 0
            noisy[white] = 255
            out.append(noisy)
        return out


class SpeckleNoise(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """乘性斑点噪声。"""
        super().__init__(name="speckle_noise", category="quality", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        # 论文默认：I_noisy = I * (1 + noise)，intensity=0.7。
        intensity = float(self.params.get("intensity", 0.7)) * severity_scale(severity, 0.6, 1.8)
        grain_size = int(self.params.get("grain_size", 1))
        grain_size = max(1, grain_size)
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out: list[np.ndarray] = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue
            h, w = frame.shape[:2]
            if grain_size == 1:
                noise = rng.normal(0.0, intensity, size=(h, w)).astype(np.float32)
            else:
                gh = max(1, int(np.ceil(h / grain_size)))
                gw = max(1, int(np.ceil(w / grain_size)))
                low = rng.normal(0.0, intensity, size=(gh, gw)).astype(np.float32)
                noise = np.repeat(np.repeat(low, grain_size, axis=0), grain_size, axis=1)[:h, :w]

            f = frame.astype(np.float32)
            noisy = f * (1.0 + noise[..., None])
            out.append(np.clip(noisy, 0, 255).astype(np.uint8))
        return out
