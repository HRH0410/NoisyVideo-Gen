from __future__ import annotations

import numpy as np

from . import BaseNoiseLike


def _sample_indices(rng: np.random.Generator, n: int, k: int) -> list[int]:
    if n <= 1 or k <= 0:
        return []
    # 用户要求：drop一半，repeat一半。这里的 k 应该被设为 n // 2。
    k = n // 2
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


def _sample_from_pool(rng: np.random.Generator, pool: list[int], k: int) -> list[int]:
    """从给定候选池中无放回采样索引。"""
    if not pool:
        return []
    # 强制采样池的一半
    k = max(1, len(pool) // 2)
    picked = rng.choice(len(pool), size=k, replace=False)
    return sorted(int(pool[int(i)]) for i in picked)


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
    """将选中帧集合内部随机置换后再写回原位置，避免恒等置换。"""
    indices = sorted(set(int(i) for i in indices if 0 <= i < len(frames)))
    if len(indices) <= 1:
        return [f.copy() for f in frames]

    out = [f.copy() for f in frames]
    perm = list(indices)

    # 避免完全不变
    while True:
        rng.shuffle(perm)
        if perm != indices:
            break

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

        base_indices = sorted(set(int(i) for i in selected_indices if 0 <= i < n))
        if not base_indices:
            return [f.copy() for f in frames]

        # 噪声比率由上游 selected_indices 决定，这里在该集合内做 jitter 拆分。
        rho = float(len(base_indices) / n)
        alpha = float(rho * rng.random())
        beta = float(1.0 - alpha)

        # replace 比例按 alpha 对全体帧计，再裁剪到候选池规模。
        k_replace = min(len(base_indices), int(round(alpha * n)))
        replace_indices = _sample_from_pool(rng, base_indices, k_replace)

        replace_set = set(replace_indices)
        drop_pool = [i for i in base_indices if i not in replace_set]

        # drop 比例按 beta 在候选池上采样，避免越界并保持集合拆分语义。
        k_drop = min(len(drop_pool), int(round(beta * len(base_indices))))
        drop_indices = _sample_from_pool(rng, drop_pool, k_drop)

        order = str(self.params.get("order", "replace_first"))
        out = [f.copy() for f in frames]

        if order == "drop_first":
            out = _apply_frame_drop(out, drop_indices)
            out = _apply_frame_replace_shuffle(out, replace_indices, rng)
        else:
            out = _apply_frame_replace_shuffle(out, replace_indices, rng)
            out = _apply_frame_drop(out, drop_indices)

        return out
