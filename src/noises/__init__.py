from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaseNoiseLike:
    name: str
    category: str
    params: dict

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """对输入帧执行扰动并返回等长帧序列。"""
        raise NotImplementedError


def severity_scale(severity: int | None, min_scale: float = 0.5, max_scale: float = 2.0) -> float:
    """将离散 severity 映射为连续强度倍率。"""
    if severity is None:
        return 1.0
    s = max(1, min(5, int(severity)))
    return min_scale + (max_scale - min_scale) * (s - 1) / 4.0
