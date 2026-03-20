from __future__ import annotations

import argparse
from pathlib import Path

from calibrator import Calibrator, DEFAULT_SEVERITY_TARGETS
from noise_factory import list_available_noises
from utils import list_video_files, load_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    """解析校准脚本参数。"""
    parser = argparse.ArgumentParser(description="Calibrate severity profiles and update noise catalog")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--noise", type=str, default=None, help="calibrate one noise only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=4)
    parser.add_argument("--psnr_min", type=float, default=20.0)
    parser.add_argument("--psnr_max", type=float, default=40.0)
    parser.add_argument("--dry_run", action="store_true", help="run calibration without writing catalog")
    return parser.parse_args()


def main() -> None:
    """执行校准并可选回写 noise_catalog。"""
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    config = load_yaml(config_path)
    noise_catalog_path = config_path.parent / "noise_catalog.yaml"
    noise_catalog = load_yaml(noise_catalog_path)

    logger = setup_logger(config["paths"]["log_dir"], config["runtime"].get("log_level", "INFO"))

    input_dir = config["paths"]["input_dir"]
    videos = list_video_files(input_dir)
    if not videos:
        raise ValueError(f"no input videos found in {input_dir}")

    sample_videos = videos[: max(1, int(args.sample_limit))]
    logger.info("calibration sample videos: %s", [p.name for p in sample_videos])

    if args.noise:
        noises = [args.noise]
    else:
        noises = list(config["benchmark"].get("enabled_noises", []))

    supported = set(list_available_noises())
    unknown = [n for n in noises if n not in supported]
    if unknown:
        raise ValueError(f"unknown noises: {unknown}")

    calibrator = Calibrator(config=config, noise_catalog=noise_catalog, logger=logger)

    for noise_name in noises:
        logger.info("start calibration noise=%s", noise_name)
        profile = calibrator.calibrate_severity_profile(
            noise_name=noise_name,
            sample_videos=sample_videos,
            severity_targets=dict(DEFAULT_SEVERITY_TARGETS),
            psnr_range=(float(args.psnr_min), float(args.psnr_max)),
            seed=int(args.seed),
        )

        if not args.dry_run:
            calibrator.write_profile_to_catalog(
                noise_name=noise_name,
                profile=profile,
                catalog_path=noise_catalog_path,
            )
            # 内存中的 catalog 也同步，便于后续同进程继续校准。
            noise_catalog[noise_name]["severity_params"] = dict(profile["severity_params"])

    logger.info("calibration finished")


if __name__ == "__main__":
    main()
