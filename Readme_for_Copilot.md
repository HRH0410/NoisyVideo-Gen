# NoisyVideo-Gen

一个用于生成 **视频视觉扰动 benchmark** 的工程项目。  
项目目标是：**对一批原始视频按统一协议施加多种视觉扰动（visual corruptions / perturbations），生成可批量处理、可追踪、可复现的 noisy video benchmark**，用于后续视频多模态模型（Video MLLM / Video-Language Model / Multimodal Video Model）的鲁棒性评测。

---

# 1. 项目目标

本项目负责完成以下工作：

1. 读取一个目录下的原始视频
2. 对每个视频进行抽帧（frame sampling）
3. 在抽样帧中选择部分帧施加视觉扰动
4. 将扰动后的帧重新编码为输出视频
5. 记录 clean/noisy 对应关系及元数据
6. 计算 PSNR / SSIM 等指标
7. 支持按配置批量生成多种噪声版本

本项目**不是训练项目**，也**不是模型推理项目**。  
它是一个 **benchmark generator / benchmark construction pipeline**。

---

# 2. 项目范围

本项目聚焦于 **benchmark 数据生成**，不负责以下内容：

- 不负责训练视频模型
- 不负责微调多模态模型
- 不负责完成下游问答评测
- 不负责实现完整论文中的全部实验表格

本项目的输出是：

- 带噪视频（noisy videos）
- 对应的 manifest / metadata
- clean/noisy 差异指标（PSNR / SSIM）
- 可选的预览图与统计报告

---

# 3. 设计要求

项目实现必须满足以下要求：

## 3.1 清晰
代码结构必须清晰，职责明确，方便阅读与维护。

## 3.2 可扩展
后续能够继续补充更多噪声类型，而不需要推倒整体结构。

## 3.3 可批量处理
必须支持对一个文件夹中的所有视频自动处理。

## 3.4 可复现
所有随机过程必须受随机种子控制，输出结果必须可追踪。

## 3.5 Docker-first
项目默认运行环境是 Docker 容器，而不是宿主机裸环境。

## 3.6 GPU-compatible
项目能够在实验室服务器的 NVIDIA GPU 环境中运行。服务器上有 4 张 4090，可按需使用，不要求必须全部占用。

---

# 4. 项目结构

```text
NoisyVideo-Gen/
├── docker/
│   ├── Dockerfile               # Docker 环境：CUDA + Python + FFmpeg + OpenCV 等依赖
│   └── docker-compose.yml       # 容器启动配置：GPU、挂载目录、工作目录
├── configs/
│   ├── config.yaml              # 全局配置：输入输出路径、抽帧数、frame_ratio、seed 等
│   └── noise_catalog.yaml       # 噪声目录：定义支持的噪声、类别、默认参数
├── src/
│   ├── main.py                  # 主入口：读取配置并启动批量处理
│   ├── video_processor.py       # 视频处理主模块：读视频、抽帧、选帧、写视频
│   ├── noise_factory.py         # 噪声工厂：统一管理所有噪声实现与调度
│   ├── calibrator.py            # 校准模块：根据 PSNR/SSIM 调节噪声强度
│   ├── metrics.py               # 指标模块：计算 PSNR、SSIM
│   ├── utils.py                 # 通用工具：seed、日志、路径处理、dataclass
│   └── noises/                  # 各类噪声具体实现
│       ├── __init__.py
│       ├── quality.py           # 高斯噪声、椒盐噪声、泊松噪声等
│       ├── blur.py              # 高斯模糊、运动模糊、失焦模糊等
│       ├── temporal.py          # 丢帧、乱序、重复帧等
│       ├── compression.py       # JPEG、H.265 等压缩失真
│       ├── occlusion.py         # 随机遮挡、目标遮挡
│       ├── digital.py           # rolling shutter、分辨率退化等
│       ├── lighting.py          # 亮度、对比度、偏色、过曝欠曝
│       └── scene.py             # 雨、雾、雪、阴影、反光
├── data/
│   ├── input/                   # 原始输入视频
│   ├── output/                  # 生成的带噪视频
│   └── manifests/               # JSONL/CSV 元数据记录
├── outputs/
│   ├── logs/                    # 运行日志
│   ├── reports/                 # 校准报告、统计结果
│   └── previews/                # 抽样预览图、对比图
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明
````

---

# 5. 每个目录和文件的职责

## 5.1 `docker/`

### `docker/Dockerfile`

定义项目运行环境，包括：

* CUDA runtime
* Python 版本
* FFmpeg
* OpenCV 相关系统依赖
* Python 依赖安装
* 工作目录
* 环境变量设置

它是整个项目运行环境的标准定义。

---

### `docker/docker-compose.yml`

定义容器启动方式，包括：

* GPU 暴露方式
* 目录挂载
* 工作目录
* 环境变量
* 容器名称
* 交互式启动设置

---

## 5.2 `configs/`

### `configs/config.yaml`

项目主配置文件，用于控制：

* 输入输出目录
* 抽帧数
* frame ratio
* 随机种子
* 是否保存指标
* 是否保存 manifest
* 是否输出 preview
* 运行时启用哪些噪声

---

### `configs/noise_catalog.yaml`

噪声目录配置文件，用于定义：

* 支持的噪声名称
* 噪声所属类别
* 默认参数
* 是否支持 severity
* 是否参与 calibration
* 参数搜索范围（若需要）

---

## 5.3 `src/`

核心代码目录。

---

### `src/main.py`

项目主入口。

职责：

* 读取配置文件
* 初始化日志
* 设置随机种子
* 扫描输入目录
* 构造处理器与噪声对象
* 执行批量处理
* 输出运行总结

`main.py` 只负责总调度，不直接实现具体噪声算法。

---

### `src/video_processor.py`

视频处理核心模块。

职责：

1. 打开视频并读取帧
2. 执行抽帧（frame sampling）
3. 根据 frame ratio 选择待扰动帧
4. 调用噪声对象施加扰动
5. 计算 clean/noisy 对应指标
6. 将结果写出为新视频
7. 生成 manifest record
8. 可选生成预览图

这是整个项目中最核心的流程控制模块。

---

### `src/noise_factory.py`

噪声工厂模块。

职责：

* 维护 noise registry
* 根据 `noise_name` 返回对应噪声实现
* 根据配置参数初始化噪声对象
* 对外提供统一的噪声构造接口

这里负责“找到并构造噪声”，不负责堆放所有噪声算法细节。

---

### `src/calibrator.py`

校准模块。

职责：

* 对某种噪声的关键参数进行强度搜索
* 使用 PSNR / SSIM 衡量 clean/noisy 差异
* 生成参数建议或校准报告

适合实现：

* binary search
* bounded search
* coarse-to-fine search

---

### `src/metrics.py`

指标模块。

职责：

* 计算单帧 PSNR
* 计算单帧 SSIM
* 计算视频级平均 PSNR / SSIM
* 生成视频级指标汇总

---

### `src/utils.py`

通用工具模块。

职责建议包括：

* 设置随机种子
* 构造 logger
* 定义 dataclass
* 写入 JSONL manifest
* 路径检查与创建
* 文件名规范化
* 一些通用 helper 函数

---

## 5.4 `src/noises/`

噪声实现目录。
所有视觉扰动算法按类别组织，便于扩展和维护。

---

### `src/noises/quality.py`

质量类噪声。

适合放：

* Gaussian noise
* Impulse noise
* Poisson noise
* Speckle noise

特点：以像素级随机扰动为主。

---

### `src/noises/blur.py`

模糊类扰动。

适合放：

* Gaussian blur
* Motion blur
* Defocus blur
* Glass blur
* Zoom blur

特点：基于卷积核或图像滤波。

---

### `src/noises/temporal.py`

时序扰动。

适合放：

* Frame drop
* Frame replace
* Frame repeat
* Temporal jitter

特点：直接操作帧序列，而不是单帧像素。

---

### `src/noises/compression.py`

压缩类失真。

适合放：

* JPEG artifact
* H.265 artifact
* Bit error simulation

特点：优先通过真实编码流程产生压缩伪影。

---

### `src/noises/occlusion.py`

遮挡类扰动。

适合放：

* Random block
* Target block

特点：通过遮挡局部区域干扰关键信息。

---

### `src/noises/digital.py`

数字失真类。

适合放：

* Rolling shutter
* Resolution degrade
* Stretch / squish
* Edge sawtooth
* Color quantization
* Elastic transform

---

### `src/noises/lighting.py`

光照与颜色类扰动。

适合放：

* Brightness
* Contrast
* Color shift
* Flicker
* Overexposure
* Underexposure

---

### `src/noises/scene.py`

场景干扰类。

适合放：

* Rain
* Fog
* Snow
* Frost
* Shadow
* Reflection

---

## 5.5 `data/`

### `data/input/`

原始 clean videos 输入目录。

---

### `data/output/`

带噪视频输出目录。

建议按噪声类型组织输出，例如：

```text
data/output/
├── gaussian_blur/
├── frame_drop/
├── jpeg_artifact/
└── ...
```

也可以进一步按 severity 分层。

---

### `data/manifests/`

元数据输出目录。

存放 JSONL / CSV 形式的 clean/noisy 对应记录。

---

## 5.6 `outputs/`

### `outputs/logs/`

运行日志目录。

---

### `outputs/reports/`

统计报告目录。
可存放：

* calibration summary
* metric summary
* failed cases
* dataset statistics

---

### `outputs/previews/`

预览图目录。
可存放：

* sampled frame grid
* clean/noisy comparison image
* debug visualization

---

# 6. 输入输出定义

## 6.1 输入

输入是一个目录下的一批视频文件。

支持的格式可以包括但不限于：

* `.mp4`
* `.avi`
* `.mov`
* `.mkv`

第一阶段建议优先保证 `.mp4`。

---

## 6.2 输出

输出包括三类：

### A. 带噪视频

保存到 `data/output/`

### B. manifest

保存到 `data/manifests/`

### C. 报告/日志/预览

保存到 `outputs/`

---

# 7. 核心数据结构

建议在 `utils.py` 中使用 dataclass 定义以下对象。

---

## 7.1 `VideoSample`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VideoSample:
    video_id: str
    path: Path
    fps: float | None = None
    num_frames: int | None = None
    width: int | None = None
    height: int | None = None
```

用途：表示一个原始视频样本。

---

## 7.2 `NoiseSpec`

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class NoiseSpec:
    category: str
    name: str
    severity: int | None
    frame_ratio: float
    params: dict[str, Any]
```

用途：表示一次扰动配置。

---

## 7.3 `BenchmarkRecord`

```python
from dataclasses import dataclass

@dataclass
class BenchmarkRecord:
    video_id: str
    clean_path: str
    noisy_path: str
    noise_name: str
    noise_category: str
    severity: int | None
    frame_ratio: float
    sampled_frame_count: int
    sampled_indices: list[int]
    selected_indices: list[int]
    seed: int
    psnr: float | None = None
    ssim: float | None = None
```

用途：表示一个输出样本的完整元数据记录。

---

# 8. 视频处理协议

## 8.1 基本思想

视频模型通常不会消费原始视频的每一帧。
实际使用中往往会：

1. 解码视频
2. 抽取少量代表帧
3. 将这些帧输入视觉编码器
4. 送入多模态模型

因此，本项目围绕 **sampled frames** 构建 benchmark。

---

## 8.2 抽帧策略

默认采用：

* `uniform sampling`
* `num_sampled_frames = 8`

含义：在时间轴上均匀抽取 8 帧。

后续可扩展：

* random sampling
* keyframe sampling

但第一阶段只需实现 uniform sampling。

---

## 8.3 frame ratio

默认：

* `frame_ratio = 0.9`

含义：在 sampled frames 中，约 90% 的帧会被施加扰动。

例如：

* 抽 8 帧
* 扰动其中约 7 帧

---

## 8.4 selected indices

被扰动的 sampled frames 由带 seed 的随机选择生成。

要求：

* 可复现
* 可记录到 manifest
* 同一 seed 下输出一致

---

## 8.5 输出视频策略

**直接将 sampled frames 编码为新的输出视频**

优点：

* 实现简单
* 与 benchmark 生成任务直接匹配
* 后续可用于模型评测


---

# 9. 噪声接口设计

每个噪声类建议遵循统一接口：

```python
class BaseNoiseLike:
    name: str
    category: str

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs
    ) -> list[np.ndarray]:
        ...
```

要求如下：

1. 输入是 sampled frames
2. `selected_indices` 指示哪些帧被扰动
3. 输出长度必须与输入一致
4. 未选中的帧保持原状
5. 所有随机操作必须受 `seed` 控制

---

# 10. NoiseFactory 设计要求

`noise_factory.py` 应维护一个注册表，例如：

```python
NOISE_REGISTRY = {
    "gaussian_noise": GaussianNoise,
    "gaussian_blur": GaussianBlur,
    "frame_drop": FrameDrop,
    ...
}
```

推荐提供：

```python
build_noise(name: str, params: dict | None = None)
list_available_noises()
```

要求：

* factory 负责查找和构造
* 具体噪声实现写在 `src/noises/*.py`
* 不要把所有算法实现都堆在 factory 中

---

# 11. metrics 设计要求

建议至少实现以下函数：

```python
def compute_psnr(clean_frame: np.ndarray, noisy_frame: np.ndarray) -> float:
    ...

def compute_ssim(clean_frame: np.ndarray, noisy_frame: np.ndarray) -> float:
    ...

def compute_video_metrics(
    clean_frames: list[np.ndarray],
    noisy_frames: list[np.ndarray]
) -> dict[str, float]:
    ...
```

视频级指标至少返回：

* `avg_psnr`
* `avg_ssim`

可扩展返回：

* `min_psnr`
* `max_psnr`
* `std_psnr`
* `min_ssim`
* `max_ssim`
* `std_ssim`

---

# 12. calibrator 设计要求

校准模块用于让某一类噪声达到指定的强度区间。

推荐接口：

```python
class Calibrator:
    def calibrate(
        self,
        noise_name: str,
        sample_videos: list[Path],
        target_ssim: float | None = None,
        target_psnr: float | None = None,
        search_space: dict | None = None,
    ) -> dict:
        ...
```

职责：

* 对选定视频样本做多次试验
* 比较平均 SSIM / PSNR
* 调节关键参数
* 输出最合适的参数建议

---

# 13. manifest 设计要求

manifest 是强制要求。
每个带噪视频都必须有对应记录。

推荐使用 `jsonl`。

每条记录至少包含：

* `video_id`
* `clean_path`
* `noisy_path`
* `noise_name`
* `noise_category`
* `severity`
* `frame_ratio`
* `sampled_frame_count`
* `sampled_indices`
* `selected_indices`
* `seed`
* `psnr`
* `ssim`

示例：

```json
{
  "video_id": "vid_0001",
  "clean_path": "data/input/vid_0001.mp4",
  "noisy_path": "data/output/gaussian_blur/severity_3/vid_0001.mp4",
  "noise_name": "gaussian_blur",
  "noise_category": "blur",
  "severity": 3,
  "frame_ratio": 0.9,
  "sampled_frame_count": 8,
  "sampled_indices": [0, 12, 24, 36, 48, 60, 72, 84],
  "selected_indices": [0, 1, 2, 3, 4, 5, 7],
  "seed": 42,
  "psnr": 23.51,
  "ssim": 0.71
}
```

---

# 14. 第一阶段推荐实现的噪声集合

第一阶段建议优先实现这 8 种：

* `gaussian_noise`
* `gaussian_blur`
* `frame_drop`
* `frame_replace`
* `jpeg_artifact`
* `h265_artifact`
* `random_block`
* `color_shift`

原因：

* 覆盖主要噪声形态
* 足以验证整体 pipeline
* 工程量可控
* 后续扩展自然

---

# 15. 配置文件建议

## 15.1 `configs/config.yaml`

建议结构：

```yaml
project:
  name: "NoisyVideo-Gen"
  seed: 42

paths:
  input_dir: "data/input"
  output_dir: "data/output"
  manifest_dir: "data/manifests"
  log_dir: "outputs/logs"
  report_dir: "outputs/reports"
  preview_dir: "outputs/previews"

video:
  sample_method: "uniform"
  num_sampled_frames: 8
  frame_ratio: 0.9
  output_fps: 8

runtime:
  save_metrics: true
  save_manifest: true
  save_preview: true
  log_level: "INFO"

benchmark:
  enabled_noises:
    - gaussian_noise
    - gaussian_blur
    - frame_drop
    - frame_replace
    - jpeg_artifact
    - h265_artifact
    - random_block
    - color_shift
```

---

## 15.2 `configs/noise_catalog.yaml`

建议结构：

```yaml
gaussian_noise:
  category: "quality"
  supports_severity: true
  default_params:
    sigma: 25

gaussian_blur:
  category: "blur"
  supports_severity: true
  default_params:
    kernel_size: 11
    sigma: 3.0

frame_drop:
  category: "temporal"
  supports_severity: true
  default_params:
    fill_mode: "black"

jpeg_artifact:
  category: "compression"
  supports_severity: true
  default_params:
    quality: 20

random_block:
  category: "occlusion"
  supports_severity: true
  default_params:
    min_ratio: 0.2
    max_ratio: 0.5
```

---

# 16. Docker 设计要求

## 16.1 基本要求

项目必须运行在 Docker 容器中。

原因：

* 服务器为共享环境
* CUDA / FFmpeg / OpenCV 容易冲突
* 环境需要可复现
* 团队协作需要统一运行方式

---

## 16.2 基础镜像

建议使用 GPU-compatible 基础镜像，例如：

```dockerfile
nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

---

## 16.3 系统依赖

容器中至少安装：

* python3
* pip
* ffmpeg
* libgl1
* libglib2.0-0
* build-essential
* git

---

## 16.4 Python 依赖

至少包括：

* numpy
* opencv-python
* scikit-image
* pyyaml
* tqdm
* imageio
* imageio-ffmpeg
* decord（可选）
* torch（如果后续评测或 GPU 模块需要）

---

## 16.5 工作目录与环境变量

建议：

```bash
WORKDIR /workspace/NoisyVideo-Gen
PYTHONPATH=/workspace/NoisyVideo-Gen/src
```

---

## 16.6 容器运行要求

应支持：

* GPU 映射
* 项目目录挂载
* 数据目录持久化
* 交互式 shell
* 在容器中直接执行 `python src/main.py ...`

---

# 17. CLI 设计要求

至少支持：

```bash
python src/main.py --config configs/config.yaml
```

建议支持更多参数覆盖：

* `--config`
* `--noise`
* `--severity`
* `--input_dir`
* `--output_dir`
* `--seed`

例如：

```bash
python src/main.py \
  --config configs/config.yaml \
  --noise gaussian_blur \
  --severity 3 \
  --seed 42
```

---

# 18. 日志与异常处理要求

## 18.1 日志

必须记录：

* 启动时间
* 配置文件路径
* 输入目录
* 输出目录
* 当前噪声类型
* 当前视频名称
* 当前进度
* 指标信息
* 错误信息
* 最终 summary

---

## 18.2 异常处理

必须保证：

* 单个坏视频不会导致全局崩溃
* 解码失败要被捕获并记录
* 编码失败要被捕获并记录
* 路径不可写要显式报错
* 失败样本应写入日志或 report

---

# 19. 编码规范

1. 使用 Python 3.10+
2. 使用 type hints
3. 使用 dataclass 描述结构化数据
4. 保持函数职责单一
5. 保持文件职责清晰
6. 随机性必须可控
7. 不要散落硬编码
8. 不要复制粘贴同类逻辑
9. 注释说明“为什么这样设计”
10. 对外接口尽量稳定

---

# 20. 推荐实现顺序

Copilot 请按以下顺序实现。

## 第一阶段：环境

1. `docker/Dockerfile`
2. `docker/docker-compose.yml`
3. `requirements.txt`

## 第二阶段：基础工具

4. `src/utils.py`
5. `src/metrics.py`

## 第三阶段：主流程

6. `src/noise_factory.py`
7. `src/video_processor.py`
8. `src/main.py`

## 第四阶段：最小噪声集合

9. `src/noises/quality.py` 中实现 `gaussian_noise`
10. `src/noises/blur.py` 中实现 `gaussian_blur`
11. `src/noises/temporal.py` 中实现 `frame_drop`
12. `src/noises/temporal.py` 中实现 `frame_replace`
13. `src/noises/compression.py` 中实现 `jpeg_artifact`
14. `src/noises/compression.py` 中实现 `h265_artifact`
15. `src/noises/occlusion.py` 中实现 `random_block`
16. `src/noises/lighting.py` 中实现 `color_shift`

## 第五阶段：校准

17. `src/calibrator.py`

---

# 21. 第一阶段成功标准

当项目满足以下条件时，视为第一阶段可用：

1. 能在 Docker 容器中运行
2. 能读取 `data/input/` 下的视频
3. 能均匀抽取 8 帧
4. 能按 frame ratio 选择待扰动帧
5. 能施加至少 2 种基础噪声
6. 能输出带噪视频
7. 能输出 manifest
8. 能计算 PSNR / SSIM
9. 能批量处理整个目录
10. 单个错误样本不会使整批任务中断

---

# 22. 不允许的实现方式

以下做法禁止采用：

1. 把所有逻辑堆进一个文件
2. 把所有噪声都堆进 `noise_factory.py`
3. 不记录 manifest
4. 不控制随机种子
5. 跳过 Docker 直接依赖本机环境
6. 只有代码没有日志
7. 只有视频输出没有元数据
8. 处理失败后直接静默跳过不记录

---

# 23. 给 Copilot 的最终实现要求

请将本项目实现为一个：

* 中文说明清晰
* Docker-first
* GPU-compatible
* 目录结构明确
* 代码职责清晰
* 可批量处理
* 可复现
* 后续易于扩展

的视频视觉扰动 benchmark 生成系统。

优先保证：

1. correctness
2. clarity
3. reproducibility
4. extensibility

如有不确定，请优先选择：

* 明确接口
* 配置驱动
* 统一 manifest
* Docker 兼容
* 便于批处理

---

# 24. 一句话总结

本项目的本质是：

> **在 Docker 环境中，对一批视频按照统一抽帧与扰动协议生成可追踪、可复现、可扩展的 noisy benchmark 数据集。**

