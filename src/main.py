from __future__ import annotations

import argparse
import json
from pathlib import Path

from noise_factory import list_available_noises
from utils import ensure_dir, list_video_files, load_yaml, set_global_seed, setup_logger
from video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数，用于覆盖配置并控制单次运行行为。"""
    parser = argparse.ArgumentParser(description="NoisyVideo-Gen benchmark generator")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--noise", type=str, default=None, help="only run one noise name")
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """将 CLI 覆盖项写回配置字典，形成最终运行配置。"""
    if args.input_dir:
        config["paths"]["input_dir"] = args.input_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["project"]["seed"] = int(args.seed)
    return config


def main() -> None:
    """程序主入口：加载配置、批处理视频并写出运行摘要。"""
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml(config_path)
    config = apply_overrides(config, args)

    noise_catalog_path = config_path.parent / "noise_catalog.yaml"
    if not noise_catalog_path.exists():
        raise FileNotFoundError(f"Noise catalog not found: {noise_catalog_path}")
    noise_catalog = load_yaml(noise_catalog_path)

    paths_cfg = config["paths"]
    runtime_cfg = config["runtime"]

    for key in ["output_dir", "manifest_dir", "log_dir", "report_dir", "preview_dir"]:
        ensure_dir(paths_cfg[key])

    logger = setup_logger(paths_cfg["log_dir"], runtime_cfg.get("log_level", "INFO"))

    set_global_seed(int(config["project"]["seed"]))
    logger.info("config=%s", config_path.as_posix())
    logger.info("input_dir=%s", paths_cfg["input_dir"])
    logger.info("output_dir=%s", paths_cfg["output_dir"])

    videos = list_video_files(paths_cfg["input_dir"])
    if not videos:
        logger.warning("no videos found under input_dir=%s", paths_cfg["input_dir"])
        return

    if args.noise:
        # 支持只跑单个噪声，适合调试与对比实验。
        enabled_noises = [args.noise]
    else:
        enabled_noises = list(config["benchmark"].get("enabled_noises", []))

    supported = set(list_available_noises())
    unknown = [n for n in enabled_noises if n not in supported]
    if unknown:
        raise ValueError(f"Unknown noises: {unknown}. Supported: {sorted(supported)}")

    processor = VideoProcessor(config=config, noise_catalog=noise_catalog, logger=logger)

    all_records = 0
    all_failed = 0
    summary: dict[str, dict] = {}

    for noise_name in enabled_noises:
        logger.info("start noise=%s", noise_name)
        records, failed = processor.process_batch(
            video_paths=videos,
            noise_name=noise_name,
            severity=args.severity,
            noise_params=None,
            seed=args.seed,
        )

        all_records += len(records)
        all_failed += len(failed)

        summary[noise_name] = {
            "processed": len(records),
            "failed": len(failed),
            "severity": args.severity,
        }

    report_path = Path(paths_cfg["report_dir"]) / "run_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_processed": all_records,
                "total_failed": all_failed,
                "by_noise": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("done processed=%d failed=%d", all_records, all_failed)


if __name__ == "__main__":
    main()
