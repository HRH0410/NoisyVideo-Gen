from __future__ import annotations

import cv2
import numpy as np

from . import BaseNoiseLike


def _alpha_blend(frame: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
	return np.clip(frame.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha, 0, 255).astype(
		np.uint8
	)


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
	length = max(1, int(length))
	k = np.zeros((length, length), dtype=np.float32)
	k[length // 2, :] = 1.0
	center = (length / 2 - 0.5, length / 2 - 0.5)
	rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	k = cv2.warpAffine(k, rot, (length, length))
	s = float(k.sum())
	if s <= 0:
		return np.ones((1, 1), dtype=np.float32)
	return k / s


def _next_pow2(v: int) -> int:
	if v <= 1:
		return 1
	return 1 << int(np.ceil(np.log2(v)))


def _diamond_square_map(side: int, decay: float, rng: np.random.Generator) -> np.ndarray:
	"""生成 diamond-square 分形图，输出 shape=(side, side) 且范围约为[0,1]。"""
	decay = max(1.01, float(decay))
	n = int(side)
	grid = np.zeros((n + 1, n + 1), dtype=np.float32)
	grid[0, 0] = rng.random()
	grid[0, n] = rng.random()
	grid[n, 0] = rng.random()
	grid[n, n] = rng.random()

	step = n
	scale = 1.0
	while step > 1:
		half = step // 2
		for y in range(half, n, step):
			for x in range(half, n, step):
				a = grid[y - half, x - half]
				b = grid[y - half, x + half]
				c = grid[y + half, x - half]
				d = grid[y + half, x + half]
				grid[y, x] = (a + b + c + d) * 0.25 + rng.uniform(-scale, scale)

		for y in range(0, n + 1, half):
			x_start = half if ((y // half) % 2 == 0) else 0
			for x in range(x_start, n + 1, step):
				vals = []
				if y - half >= 0:
					vals.append(grid[y - half, x])
				if y + half <= n:
					vals.append(grid[y + half, x])
				if x - half >= 0:
					vals.append(grid[y, x - half])
				if x + half <= n:
					vals.append(grid[y, x + half])
				grid[y, x] = np.mean(vals) + rng.uniform(-scale, scale)

		scale /= decay
		step //= 2

	frag = grid[:n, :n]
	frag -= frag.min()
	den = float(frag.max())
	if den <= 1e-8:
		return np.zeros_like(frag)
	return frag / den


def _simple_perlin_like_noise(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
	base = rng.random((h, w)).astype(np.float32)
	noise = cv2.GaussianBlur(base, (0, 0), sigmaX=max(1.0, w * 0.03), sigmaY=max(1.0, h * 0.03))
	noise -= noise.min()
	den = float(noise.max())
	if den > 1e-8:
		noise /= den
	return noise


class Shadow(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="shadow", category="scene", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Shadow：随机椭圆阴影 + 高斯软化。"""
		rng = np.random.default_rng(seed)
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			x0 = float(rng.uniform(w / 4.0, 3.0 * w / 4.0))
			y0 = float(rng.uniform(h / 4.0, 3.0 * h / 4.0))
			a = int(rng.uniform(w / 4.0, w / 2.0))
			b = int(rng.uniform(h / 4.0, h / 2.0))
			theta = float(rng.uniform(0.0, 180.0))

			mask = np.zeros((h, w), dtype=np.float32)
			cv2.ellipse(mask, (int(x0), int(y0)), (max(1, a), max(1, b)), theta, 0, 360, 1.0, -1)
			mask = cv2.GaussianBlur(mask, (51, 51), sigmaX=0)

			shaded = frame.astype(np.float32) * (1.0 - 0.7 * mask[..., None])
			out.append(np.clip(shaded, 0, 255).astype(np.uint8))
		return out


class SpecularReflection(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="specular_reflection", category="scene", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Specular reflection：高亮掩码增强 + Perlin-like 噪声融合。"""
		rng = np.random.default_rng(seed)
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			gray_mean = frame.mean(axis=2)
			highlight_mask = (gray_mean > 100).astype(np.float32)
			blurred = cv2.GaussianBlur(frame, (15, 15), sigmaX=5, sigmaY=5)

			orig = frame.astype(np.float32)
			bl = blurred.astype(np.float32)
			highlight = np.clip(1.5 * orig + 0.5 * bl, 0, 255)
			mixed = orig * (1.0 - highlight_mask[..., None]) + highlight * highlight_mask[..., None]

			h, w = frame.shape[:2]
			noise = _simple_perlin_like_noise(h, w, rng)
			noise_rgb = np.repeat((noise[..., None] * 255.0), 3, axis=2)
			final = np.clip(0.9 * mixed + 0.1 * noise_rgb, 0, 255)
			out.append(final.astype(np.uint8))
		return out


class Frost(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="frost", category="scene", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Frost：按(λ, μ)融合原图与霜纹理。"""
		rng = np.random.default_rng(seed)
		sev = 3 if severity is None else int(np.clip(severity, 1, 5))
		pairs = {
			1: (1.0, 0.4),
			2: (0.8, 0.6),
			3: (0.7, 0.7),
			4: (0.65, 0.7),
			5: (0.6, 0.75),
		}
		lam, mu = pairs[sev]
		texture_paths = self.params.get("texture_paths", [])
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			tex = None
			if isinstance(texture_paths, list) and texture_paths:
				p = str(texture_paths[int(rng.integers(0, len(texture_paths)))])
				img = cv2.imread(p, cv2.IMREAD_COLOR)
				if img is not None:
					tex = img

			if tex is None:
				n = rng.normal(180, 55, size=(h, w, 3)).astype(np.float32)
				n = cv2.GaussianBlur(np.clip(n, 0, 255).astype(np.uint8), (0, 0), sigmaX=2.0, sigmaY=2.0)
				tex = cv2.Canny(cv2.cvtColor(n, cv2.COLOR_BGR2GRAY), 80, 180)
				tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=1.2)
				tex = cv2.applyColorMap(np.clip(tex, 0, 255).astype(np.uint8), cv2.COLORMAP_BONE)

			th, tw = tex.shape[:2]
			rep_y = int(np.ceil(h / max(1, th)))
			rep_x = int(np.ceil(w / max(1, tw)))
			tiled = np.tile(tex, (rep_y, rep_x, 1))[:h, :w]
			f = frame.astype(np.float32)
			fr = tiled.astype(np.float32)
			out.append(np.clip(lam * f + mu * fr, 0, 255).astype(np.uint8))
		return out


class Snow(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="snow", category="scene", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Snow：按给定参数表生成雪层并与图像融合。"""
		rng = np.random.default_rng(seed)
		sev = 3 if severity is None else int(np.clip(severity, 1, 5))
		table = {
			1: (0.10, 0.30, 3.0, 0.50, 10, 4, 0.80),
			2: (0.20, 0.30, 2.0, 0.50, 12, 4, 0.70),
			3: (0.55, 0.30, 4.0, 0.90, 12, 8, 0.70),
			4: (0.55, 0.30, 4.5, 0.85, 12, 8, 0.65),
			5: (0.55, 0.30, 2.5, 0.85, 12, 12, 0.55),
		}
		mu_s, sigma_s, r_s, t_s, R_s, Sigma_s, beta_s = table[sev]
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			I = frame.astype(np.float32) / 255.0
			S = rng.normal(mu_s, sigma_s, size=(h, w)).astype(np.float32)
			S = np.where(S >= t_s, S, 0.0)

			zh = max(1, int(np.round(h * r_s)))
			zw = max(1, int(np.round(w * r_s)))
			Sz = cv2.resize(S, (zw, zh), interpolation=cv2.INTER_LINEAR)
			if zh > h:
				y0 = (zh - h) // 2
			else:
				y0 = 0
			if zw > w:
				x0 = (zw - w) // 2
			else:
				x0 = 0
			Sz = Sz[y0 : y0 + min(h, zh), x0 : x0 + min(w, zw)]
			if Sz.shape[0] != h or Sz.shape[1] != w:
				Sz = cv2.resize(Sz, (w, h), interpolation=cv2.INTER_LINEAR)

			K = int(2 * np.ceil(sigma_s) + 1)
			kernel = _motion_blur_kernel(K, angle_deg=float(rng.uniform(0.0, 180.0)))
			Sm = cv2.filter2D(Sz, -1, kernel)
			Sm = cv2.GaussianBlur(Sm, (0, 0), sigmaX=max(0.1, Sigma_s), sigmaY=max(0.1, Sigma_s))
			Sm = Sm * (float(R_s) / max(1.0, float(np.max([R_s, 1]))))

			gray = cv2.cvtColor((I * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
			gray3 = np.repeat(gray[..., None], 3, axis=2)
			base = np.maximum(I, 1.5 * gray3 + 0.5)
			snow_layer = np.repeat(Sm[..., None], 3, axis=2)
			snow_rot = np.rot90(snow_layer, 2)

			Is = np.clip(beta_s * I + (1.0 - beta_s) * base + snow_layer + snow_rot, 0.0, 1.0)
			out.append((Is * 255.0).astype(np.uint8))
		return out


class Fog(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="fog", category="scene", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Fog：diamond-square 分形雾图融合。"""
		rng = np.random.default_rng(seed)
		sev = 3 if severity is None else int(np.clip(severity, 1, 5))
		decays = {1: 2.0, 2: 2.0, 3: 1.7, 4: 1.5, 5: 1.4}
		c_values = {1: 1.5, 2: 2.0, 3: 2.5, 4: 2.5, 5: 3.0}
		d_s = decays[sev]
		c_s = c_values[sev]
		selected = set(selected_indices)
		out = []

		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			side = _next_pow2(max(h, w))
			F = _diamond_square_map(side, d_s, rng)[:h, :w]
			F3 = np.repeat(F[..., None], 3, axis=2)
			I = frame.astype(np.float32) / 255.0
			If = np.clip((I + c_s * F3) / (1.0 + c_s), 0.0, 1.0)
			out.append((If * 255.0).astype(np.uint8))
		return out


class Rain(BaseNoiseLike):
	def __init__(self, params: dict | None = None) -> None:
		super().__init__(name="rain", category="scene", params=params or {})

	def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
		"""Rain：按参数表生成雨层并融合雾。"""
		rng = np.random.default_rng(seed)
		sev = 3 if severity is None else int(np.clip(severity, 1, 5))
		table = {
			1: (0.05, 8, 1, 0.20, 1.03, 0.05),
			2: (0.07, 10, 2, 0.25, 1.04, 0.07),
			3: (0.09, 15, 3, 0.30, 1.05, 0.10),
			4: (0.11, 18, 4, 0.35, 1.06, 0.13),
			5: (0.13, 22, 5, 0.40, 1.07, 0.15),
		}
		m_s, L_s, rho_s, b_s, c_s, f_s = table[sev]
		kernel = _motion_blur_kernel(int(L_s), float(rng.uniform(0.0, 180.0)))
		selected = set(selected_indices)
		out = []
		for i, frame in enumerate(frames):
			if i not in selected:
				out.append(frame.copy())
				continue

			h, w = frame.shape[:2]
			density_map = (rng.random((h, w)) < m_s).astype(np.float32)
			rain_layer = cv2.filter2D(density_map, -1, kernel)
			rain_layer = cv2.GaussianBlur(rain_layer, (0, 0), sigmaX=0.5, sigmaY=0.5)
			rain_layer = np.clip(rain_layer * float(rho_s), 0.0, 1.0)

			I = frame.astype(np.float32)
			R3 = np.repeat(rain_layer[..., None], 3, axis=2)
			Irain = np.clip(I * (1.0 - 0.3 * R3) + 200.0 * R3 * b_s, 0.0, 255.0)
			Irain = np.clip(Irain * c_s, 0.0, 255.0)

			fog_add = np.full_like(Irain, 255.0)
			Irain = np.clip((1.0 - f_s) * Irain + f_s * fog_add, 0.0, 255.0)
			out.append(Irain.astype(np.uint8))
		return out

