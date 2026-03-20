from __future__ import annotations

import cv2
import numpy as np

from . import BaseNoiseLike


class RollingShutter(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="rolling_shutter", category="digital", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Rolling shutter：按列/行计算时间偏移并从历史帧取样。"""
		rng = np.random.default_rng(seed)
		delay_factor = float(self.params.get("delay_factor", 1.0))
		buffer_size = int(self.params.get("buffer_size", 5))
		buffer_size = max(1, buffer_size)

		selected = set(selected_indices)
		out = []
		for t, frame in enumerate(frames):
			if t not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			or_horizontal = bool(rng.integers(0, 2))
			reverse = bool(rng.integers(0, 2))
			limit = min(t, buffer_size)
			warped = np.zeros_like(frame)

			if or_horizontal:
				den = max(1, w - 1)
				for x in range(w):
					idx = (w - 1 - x) if reverse else x
					delta = int(np.floor((idx / den) * delay_factor * limit))
					delta = min(delta, buffer_size)
					src_t = max(0, t - delta)
					warped[:, x, :] = frames[src_t][:, x, :]
			else:
				den = max(1, h - 1)
				for y in range(h):
					idx = (h - 1 - y) if reverse else y
					delta = int(np.floor((idx / den) * delay_factor * limit))
					delta = min(delta, buffer_size)
					src_t = max(0, t - delta)
					warped[y, :, :] = frames[src_t][y, :, :]

			out.append(warped)
		return out


class ResolutionDegrade(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="resolution_degrade", category="digital", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Resolution degrade：双三次下采样再上采样。"""
		s = float(self.params.get("scale", 0.1))
		s = float(np.clip(s, 0.01, 1.0))
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue
			h, w = frame.shape[:2]
			small_h = max(1, int(np.floor(s * h)))
			small_w = max(1, int(np.floor(s * w)))
			small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
			restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
			out.append(np.clip(restored, 0, 255).astype(np.uint8))
		return out


class StretchSquish(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="stretch_squish", category="digital", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Stretch/squish：沿随机单轴强压缩后再恢复。"""
		rng = np.random.default_rng(seed)
		s = float(self.params.get("scale", 1.0 / 30.0))
		s = float(np.clip(s, 0.01, 1.0))
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			horizontal = bool(rng.integers(0, 2))
			if horizontal:
				small_w = max(1, int(np.floor(s * w)))
				tmp = cv2.resize(frame, (small_w, h), interpolation=cv2.INTER_CUBIC)
				restored = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_CUBIC)
			else:
				small_h = max(1, int(np.floor(s * h)))
				tmp = cv2.resize(frame, (w, small_h), interpolation=cv2.INTER_CUBIC)
				restored = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_CUBIC)
			out.append(np.clip(restored, 0, 255).astype(np.uint8))
		return out


class EdgeSawtooth(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="edge_sawtooth", category="digital", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Edge sawtooth：边缘邻域随机替换为噪声颜色。"""
		rho = float(self.params.get("rho", 0.3))
		rho = float(np.clip(rho, 0.0, 1.0))
		rng = np.random.default_rng(seed)
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, threshold1=50, threshold2=150)
			edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
			surround = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
			surround_mask = (surround > 0) & (edges == 0)
			rand_mask = rng.random(frame.shape[:2]) < rho
			mask = surround_mask & rand_mask

			corrupted = frame.copy()
			rand_colors = rng.integers(0, 256, size=(mask.sum(), 3), dtype=np.uint8)
			corrupted[mask] = rand_colors
			out.append(corrupted)
		return out


class ColorQuantization(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="color_quantization", category="digital", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Color quantization：将像素均匀量化到 2^b 级。"""
		b = int(self.params.get("bits", 3))
		b = int(np.clip(b, 1, 8))
		levels = 2**b
		delta = 255.0 / max(1, (levels - 1))
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue
			q = np.round(frame.astype(np.float32) / delta) * delta
			out.append(np.clip(q, 0, 255).astype(np.uint8))
		return out


class ElasticTransform(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="elastic_transform", category="digital", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Elastic transform：随机位移场 + 高斯平滑 + 双线性重映射。"""
		alpha_table = {1: 12.5, 2: 16.25, 3: 21.25, 4: 25.0, 5: 30.0}
		sev = 3 if severity is None else int(np.clip(severity, 1, 5))
		alpha = float(self.params.get("alpha", alpha_table[sev]))
		rng = np.random.default_rng(seed)

		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			delta_max = 0.005 * min(h, w)
			sigma = 0.01 * min(h, w)

			dx = rng.uniform(-delta_max, delta_max, size=(h, w)).astype(np.float32)
			dy = rng.uniform(-delta_max, delta_max, size=(h, w)).astype(np.float32)
			dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, sigmaY=sigma)
			dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, sigmaY=sigma)

			grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
			map_x = grid_x + alpha * dx
			map_y = grid_y + alpha * dy

			warped = cv2.remap(
				frame,
				map_x,
				map_y,
				interpolation=cv2.INTER_LINEAR,
				borderMode=cv2.BORDER_REFLECT_101,
			)
			out.append(np.clip(warped, 0, 255).astype(np.uint8))
		return out

