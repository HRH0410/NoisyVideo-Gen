from __future__ import annotations

import numpy as np

from . import BaseNoiseLike


def _sample_indices(rng: np.random.Generator, n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    k = min(n, k)
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


def _apply_frame_drop(frames: list[np.ndarray], indices: list[int]) -> list[np.ndarray]:
    drop_set = set(indices)
    out: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if i in drop_set:
            out.append(np.zeros_like(frame, dtype=np.uint8))
        else:
            out.append(frame.copy())
    return out


def _apply_frame_replace_shuffle(
    frames: list[np.ndarray],
    indices: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """将选中帧集合内部随机置换后再写回原位置。"""
    if len(indices) <= 1:
        return [f.copy() for f in frames]

    out = [f.copy() for f in frames]
    perm = list(indices)
    rng.shuffle(perm)
    for dst, src in zip(indices, perm):
        out[dst] = frames[src].copy()
    return out


class FrameDrop(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """初始化丢帧扰动。"""
        super().__init__(name="frame_drop", category="temporal", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Frame drop：选中帧直接置黑（像素全0）。"""
        return _apply_frame_drop(frames, selected_indices)


class FrameReplace(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """初始化替帧扰动。"""
        super().__init__(name="frame_replace", category="temporal", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Frame replace：选中帧子集内部随机置换并回填。"""
        rng = np.random.default_rng(seed)
        return _apply_frame_replace_shuffle(frames, selected_indices, rng)


class FrameRepeat(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """Frame repeat：其余帧替换为最近的参考帧。"""
        super().__init__(name="frame_repeat", category="temporal", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        ref = sorted(set(int(i) for i in selected_indices if 0 <= i < len(frames)))
        if not ref:
            return [f.copy() for f in frames]

        out: list[np.ndarray] = []
        for i in range(len(frames)):
            if i in ref:
                out.append(frames[i].copy())
                continue
            nearest = min(ref, key=lambda r: abs(r - i))
            out.append(frames[nearest].copy())
        return out

class TemporalJitter(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """时间抖动：在局部邻域打乱帧次序。"""
        super().__init__(name="temporal_jitter", category="temporal", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Temporal jitter：按 alpha/beta 比例顺序混合 replace 和 drop。"""
        rng = np.random.default_rng(seed)
        n = len(frames)
        if n == 0:
            return []

        rho = float(len(selected_indices) / n) if n > 0 else 0.0
        alpha = float(rho * rng.random())
        beta = float(1.0 - alpha)

        k_replace = int(round(alpha * n))
        k_drop = int(round(beta * n))

        replace_indices = _sample_indices(rng, n, k_replace)
        drop_indices = _sample_indices(rng, n, k_drop)

        order = str(self.params.get("order", "replace_first"))
        out = [f.copy() for f in frames]

        if order == "drop_first":
            out = _apply_frame_drop(out, drop_indices)
            out = _apply_frame_replace_shuffle(out, replace_indices, rng)
        else:
            out = _apply_frame_replace_shuffle(out, replace_indices, rng)
            out = _apply_frame_drop(out, drop_indices)

        return out
