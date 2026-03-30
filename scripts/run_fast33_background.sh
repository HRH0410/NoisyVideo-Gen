#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
#   bash scripts/run_fast33_background.sh data/input data/output_fast33 42
# 参数：
#   $1 输入目录（默认 data/input）
#   $2 输出目录（默认 data/output_fast33）
#   $3 随机种子（默认 42）

INPUT_DIR="${1:-data/input}"
OUTPUT_DIR="${2:-data/output_fast33}"
SEED="${3:-42}"

# 36 种中排除 3 种慢噪声：rain、fog、glass_blur
FAST_NOISES=(
  gaussian_noise poisson_noise impulse_noise speckle_noise
  frame_drop frame_replace frame_repeat temporal_jitter
  gaussian_blur defocus_blur motion_blur zoom_blur
  brightness contrast color_shift flicker overexposure underexposure
  shadow specular_reflection frost snow
  rolling_shutter resolution_degrade stretch_squish edge_sawtooth color_quantization elastic_transform
  random_block target_block
  jpeg_artifact bit_error h265_compression
)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] root=$ROOT_DIR"
echo "[INFO] input_dir=$INPUT_DIR"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] seed=$SEED"
echo "[INFO] fast_noises_count=${#FAST_NOISES[@]}"

for noise in "${FAST_NOISES[@]}"; do
  echo "[RUN] noise=$noise"
  docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
    python3 src/main.py \
      --config configs/config.yaml \
      --noise "$noise" \
      --input_dir "$INPUT_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --seed "$SEED"
done

echo "[DONE] Fast 33 noises finished."
