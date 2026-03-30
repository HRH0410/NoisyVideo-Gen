# NoisyVideo-Gen
一个用于构建视频噪声基准集的数据生成工程。项目支持 36 种视频扰动，按配置批量读取输入视频，输出扰动视频、指标、manifest 与运行报告。

> 说明：本项目已移除 calibration（自动校准）相关逻辑，当前仅使用论文参数（`noise_catalog.yaml` 中的 `default_params`）与命令行显式参数运行。

## 1. 项目架构

### 1.1 顶层目录

- `configs/`
  - `config.yaml`：主运行配置（输入输出路径、采样策略、启用噪声列表）
  - `noise_catalog.yaml`：各噪声默认参数与元信息
- `src/`
  - `main.py`：CLI 入口，按配置调度全部噪声
  - `video_processor.py`：视频解码、采样、加噪、编码、指标计算、manifest 写入
  - `noise_factory.py`：噪声注册表与实例构建
  - `metrics.py`：PSNR / SSIM 计算
  - `utils.py`：配置、日志、采样、jsonl 写入等工具
  - `noises/`：8 大类 36 种噪声实现
- `docker/`
  - `Dockerfile`、`docker-compose.yml`：统一运行环境
- `data/`
  - `input/`：输入视频
  - `output/`：输出视频
  - `manifests/`：每种噪声一份 jsonl 记录
- `outputs/`
  - `logs/`：运行日志
  - `reports/`：汇总报告（`run_summary.json`）
  - `previews/`：可视化拼图预览（可选）

### 1.2 处理流程

1. 读取 `config.yaml` 与 `noise_catalog.yaml`
2. 列举输入目录视频
3. 对每个启用噪声：
   - 抽样帧
   - 按 `frame_ratio` 选择被扰动帧
   - 调用 `noise_factory` 构造噪声并执行
   - 输出视频到 `data/output/<noise>/severity_<x|none>/`
   - 写入 manifest 到 `data/manifests/<noise>.jsonl`
4. 生成 `outputs/reports/run_summary.json`

## 2. 支持噪声（36）

### 2.1 Quality (4)

- `gaussian_noise`
- `poisson_noise`
- `impulse_noise`
- `speckle_noise`

### 2.2 Temporal (4)

- `frame_drop`
- `frame_replace`
- `frame_repeat`
- `temporal_jitter`

### 2.3 Blur (5)

- `gaussian_blur`
- `defocus_blur`
- `glass_blur`
- `motion_blur`
- `zoom_blur`

### 2.4 Lighting / Color (6)

- `brightness`
- `contrast`
- `color_shift`
- `flicker`
- `overexposure`
- `underexposure`

### 2.5 Scene Interference (6)

- `rain`
- `fog`
- `snow`
- `frost`
- `specular_reflection`
- `shadow`

### 2.6 Digital Distortion (6)

- `rolling_shutter`
- `resolution_degrade`
- `stretch_squish`
- `edge_sawtooth`
- `color_quantization`
- `elastic_transform`

### 2.7 Occlusion (2)

- `random_block`
- `target_block`

### 2.8 Compression (3)

- `jpeg_artifact`
- `bit_error`
- `h265_compression`

## 3. 如何运行（重点）

下面给出最推荐的 Docker 方式，以及本地 Python 方式。

### 3.1 Docker 方式（推荐）

在项目根目录执行：

```bash
cd /data1/hrh/NoisyVideo-Gen
docker compose -f docker/docker-compose.yml build
```

#### 运行全部启用噪声（默认 36 种）

```bash
cd /data1/hrh/NoisyVideo-Gen
docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
  python3 src/main.py --config configs/config.yaml
```

#### 只运行一种噪声

```bash
cd /data1/hrh/NoisyVideo-Gen
docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
  python3 src/main.py --config configs/config.yaml --noise rain
```

#### 指定 severity（例如 3）

```bash
cd /data1/hrh/NoisyVideo-Gen
docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
  python3 src/main.py --config configs/config.yaml --severity 3
```

#### 单视频测试 36 种噪声（常用）

```bash
cd /data1/hrh/NoisyVideo-Gen
mkdir -p data/input_single
cp data/input/sample_blazes.mp4 data/input_single/

docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
  python3 src/main.py \
    --config configs/config.yaml \
    --input_dir data/input_single \
    --output_dir data/output_single36 \
    --seed 42
```

### 3.2 本地 Python 方式

```bash
cd /data1/hrh/NoisyVideo-Gen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py --config configs/config.yaml
```

### 3.3 参数来源说明（重要）

- 默认参数来源：`configs/noise_catalog.yaml` 的 `default_params`
- 本项目不再支持 calibration profile / 校准回写
- 如需调参，请直接修改 `noise_catalog.yaml` 或在命令行通过业务脚本传入 `params`
- 默认分档策略：未显式传 `--severity` 时，`supports_severity=true` 的噪声自动使用 `severity=5`（最强档）；`supports_severity=false` 的噪声使用 `None`，仅按默认参数运行
- 若显式传入 `--severity`，则以命令行为准

## 4. 结果怎么看

### 4.1 关键输出文件

- `outputs/reports/run_summary.json`
  - 总处理数量、失败数量、每种噪声 processed/failed
- `data/manifests/*.jsonl`
  - 每条样本记录（`psnr`、`ssim`、采样索引、输出路径等）
- `outputs/logs/run.log`
  - 完整运行日志和异常堆栈
- `outputs/previews/`（若 `save_preview=true`）
  - 每种噪声的 clean/noisy 对照拼图

### 4.2 快速检查是否全部成功

```bash
cat outputs/reports/run_summary.json
```

`total_failed` 为 `0` 即表示本次运行没有失败样本。

## 5. 常见问题

### 5.1 运行中断后 manifest 出现重复行怎么办？

重新跑前建议先清理：

```bash
rm -f data/manifests/*.jsonl
rm -f outputs/reports/run_summary.json
```

再重新执行一次完整任务，结果会更干净。

### 5.2 33 种先跑，3 种慢噪声后跑（后台任务）

如果你想先跳过 `rain`、`fog`、`glass_blur`，项目已提供脚本：

- `scripts/run_fast33_background.sh`

这个脚本会顺序运行其余 33 种噪声，对输入目录中的所有视频逐个加扰动。

#### 后台启动 33 种（断开终端不影响）

```bash
cd /data1/hrh/NoisyVideo-Gen
nohup bash scripts/run_fast33_background.sh data/input data/output_fast33 42 \
  > outputs/logs/fast33_nohup.log 2>&1 &
echo $! > outputs/logs/fast33.pid
```

查看进度：

```bash
tail -f outputs/logs/fast33_nohup.log
```

停止任务：

```bash
kill "$(cat outputs/logs/fast33.pid)"
```

> 说明：`nohup` 可保证你退出当前终端/SSH 后任务继续；但如果机器关机或重启，任务仍会中断。

#### 慢噪声单独跑（rain / fog / glass_blur）

```bash
cd /data1/hrh/NoisyVideo-Gen

docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
  python3 src/main.py --config configs/config.yaml --noise rain \
  --input_dir data/input --output_dir data/output_slow3 --seed 42

docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
  python3 src/main.py --config configs/config.yaml --noise fog \
  --input_dir data/input --output_dir data/output_slow3 --seed 42

docker compose -f docker/docker-compose.yml run --rm -T noisyvideo-gen \
  python3 src/main.py --config configs/config.yaml --noise glass_blur \
  --input_dir data/input --output_dir data/output_slow3 --seed 42
```
