from __future__ import annotations

from typing import Any

from noises.blur import DefocusBlur, GaussianBlur, GlassBlur, MotionBlur, ZoomBlur
from noises.compression import BitError, H265Compression, JPEGArtifact
from noises.digital import (
    ColorQuantization,
    EdgeSawtooth,
    ElasticTransform,
    ResolutionDegrade,
    RollingShutter,
    StretchSquish,
)
from noises.lighting import Brightness, ColorShift, Contrast, Flicker, Overexposure, Underexposure
from noises.occlusion import RandomBlock, TargetBlock
from noises.quality import GaussianNoise, ImpulseNoise, PoissonNoise, SpeckleNoise
from noises.scene import Fog, Frost, Rain, Shadow, Snow, SpecularReflection
from noises.temporal import FrameDrop, FrameRepeat, FrameReplace, TemporalJitter


NOISE_REGISTRY = {
    # 1) Quality (4)
    "gaussian_noise": GaussianNoise,
    "poisson_noise": PoissonNoise,
    "impulse_noise": ImpulseNoise,
    "speckle_noise": SpeckleNoise,
    # 2) Temporal (4)
    "frame_drop": FrameDrop,
    "frame_replace": FrameReplace,
    "frame_repeat": FrameRepeat,
    "temporal_jitter": TemporalJitter,
    # 3) Blur (5)
    "gaussian_blur": GaussianBlur,
    "defocus_blur": DefocusBlur,
    "glass_blur": GlassBlur,
    "motion_blur": MotionBlur,
    "zoom_blur": ZoomBlur,
    # 4) Lighting / Color (6)
    "brightness": Brightness,
    "contrast": Contrast,
    "color_shift": ColorShift,
    "overexposure": Overexposure,
    "underexposure": Underexposure,
    "flicker": Flicker,
    # 5) Scene Interference (6)
    "shadow": Shadow,
    "specular_reflection": SpecularReflection,
    "frost": Frost,
    "snow": Snow,
    "fog": Fog,
    "rain": Rain,
    # 6) Digital Distortion (6)
    "rolling_shutter": RollingShutter,
    "resolution_degrade": ResolutionDegrade,
    "stretch_squish": StretchSquish,
    "edge_sawtooth": EdgeSawtooth,
    "color_quantization": ColorQuantization,
    "elastic_transform": ElasticTransform,
    # 7) Occlusion (2)
    "random_block": RandomBlock,
    "target_block": TargetBlock,
    # 8) Compression (3)
    "jpeg_artifact": JPEGArtifact,
    "bit_error": BitError,
    "h265_compression": H265Compression,
}


def list_available_noises() -> list[str]:
    """返回当前支持的噪声名称列表。"""
    return sorted(NOISE_REGISTRY.keys())


def _get_catalog_severity_params(
    noise_meta: dict[str, Any],
    severity: int | None,
) -> dict[str, Any]:
    """从 noise catalog 中读取指定 severity 的参数覆盖。"""
    if severity is None:
        return {}

    severity_params = noise_meta.get("severity_params", {})
    if not isinstance(severity_params, dict):
        return {}

    return dict(severity_params.get(str(int(severity)), {}))


def build_noise(
    name: str,
    params: dict[str, Any] | None = None,
    noise_catalog: dict[str, Any] | None = None,
    severity: int | None = None,
):
    """根据噪声名和参数构造噪声实例。"""
    key = name.strip()
    if key not in NOISE_REGISTRY:
        raise KeyError(f"Unsupported noise: {name}. Available: {list_available_noises()}")

    merged_params: dict[str, Any] = {}
    if noise_catalog and key in noise_catalog:
        noise_meta = noise_catalog[key]
        merged_params.update(noise_meta.get("default_params", {}))
        # 若存在按 severity 的校准参数，优先覆盖 default_params。
        merged_params.update(_get_catalog_severity_params(noise_meta, severity))
    if params:
        merged_params.update(params)

    return NOISE_REGISTRY[key](params=merged_params)
