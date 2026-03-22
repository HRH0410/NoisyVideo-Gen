from __future__ import annotations

import cv2
import numpy as np

from . import BaseNoiseLike


class RandomBlock(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """Random block：随机采样大矩形区域并置零。"""
        super().__init__(name="random_block", category="occlusion", params=params or {})

    def apply(
        self,
        frames: list[np.ndarray],
        selected_indices: list[int],
        severity: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        selected = set(selected_indices)
        out: list[np.ndarray] = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            h, w = frame.shape[:2]
            min_h = max(1, int(round(0.3 * h)))
            max_h = max(min_h, int(round(0.7 * h)))
            min_w = max(1, int(round(0.3 * w)))
            max_w = max(min_w, int(round(0.7 * w)))

            bh = int(rng.integers(min_h, max_h + 1))
            bw = int(rng.integers(min_w, max_w + 1))
            y0 = int(rng.integers(0, h - bh + 1))
            x0 = int(rng.integers(0, w - bw + 1))

            occ = frame.copy()
            occ[y0 : y0 + bh, x0 : x0 + bw] = 0
            out.append(occ)
        return out


class TargetBlock(BaseNoiseLike):
    def __init__(self, params: dict | None = None) -> None:
        """Target block：检测主目标框并将该区域置零。"""
        super().__init__(name="target_block", category="occlusion", params=params or {})
        self._detector: cv2.dnn_DetectionModel | None = None
        self._class_names: list[str] = []
        self._detector_initialized = False

    def _init_detector_once(self) -> bool:
        """按配置懒加载检测器，支持 model+config 或 model-only（如 ONNX）。"""
        if self._detector_initialized:
            return self._detector is not None

        self._detector_initialized = True
        model_path = self.params.get("detector_model")
        config_path = self.params.get("detector_config")
        classes_path = self.params.get("detector_classes")

        if not model_path:
            return False

        try:
            if config_path:
                net = cv2.dnn.readNet(str(model_path), str(config_path))
            else:
                try:
                    net = cv2.dnn.readNet(str(model_path))
                except Exception:
                    net = cv2.dnn.readNetFromONNX(str(model_path))

            detector = cv2.dnn_DetectionModel(net)
            in_w = int(self.params.get("detector_input_w", 320))
            in_h = int(self.params.get("detector_input_h", 320))
            detector.setInputSize(max(1, in_w), max(1, in_h))

            input_scale = float(self.params.get("detector_input_scale", 1.0 / 127.5))
            detector.setInputScale(input_scale)

            input_mean = self.params.get("detector_input_mean", [127.5, 127.5, 127.5])
            if isinstance(input_mean, (list, tuple)) and len(input_mean) == 3:
                detector.setInputMean(tuple(float(v) for v in input_mean))

            detector.setInputSwapRB(bool(self.params.get("detector_swap_rb", True)))
            self._detector = detector

            if classes_path:
                try:
                    with open(str(classes_path), "r", encoding="utf-8") as f:
                        self._class_names = [ln.strip() for ln in f if ln.strip()]
                except Exception:
                    self._class_names = []
            return True
        except Exception:
            self._detector = None
            self._class_names = []
            return False

    def _detect_with_dnn(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """使用外部预训练检测器（若提供）预测目标框。"""
        if not self._init_detector_once() or self._detector is None:
            return None

        confidence = float(self.params.get("detector_confidence", 0.3))
        nms = float(self.params.get("detector_nms", 0.4))

        try:
            classes, scores, boxes = self._detector.detect(frame, confThreshold=confidence, nmsThreshold=nms)
            if classes is None or len(classes) == 0:
                return None

            primary_class = str(self.params.get("primary_class", "")).strip().lower()
            class_id_offset = int(self.params.get("class_id_offset", -1))

            best_idx = None
            best_score = -1.0
            for idx, (cls_id, score) in enumerate(zip(classes.flatten(), scores.flatten())):
                keep = True
                if primary_class and self._class_names:
                    cls_i = int(cls_id) + class_id_offset
                    name = self._class_names[cls_i].lower() if 0 <= cls_i < len(self._class_names) else ""
                    keep = name == primary_class
                if keep and float(score) > best_score:
                    best_idx = idx
                    best_score = float(score)

            if best_idx is None:
                return None
            x, y, w, h = boxes[best_idx]
            return int(x), int(y), int(x + w), int(y + h)
        except Exception:
            return None

    def _fallback_center_bbox(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        """未提供检测模型时，回退为中心区域遮挡。"""
        h, w = frame.shape[:2]
        ratio = float(self.params.get("fallback_ratio", 0.4))
        ratio = float(np.clip(ratio, 0.2, 0.8))
        bw = max(1, int(round(w * ratio)))
        bh = max(1, int(round(h * ratio)))
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        return x1, y1, x1 + bw, y1 + bh

    def apply(self, frames, selected_indices, severity=None, seed=None, **kwargs):
        selected = set(selected_indices)
        out = []

        for i, frame in enumerate(frames):
            if i not in selected:
                out.append(frame.copy())
                continue

            bbox = self._detect_with_dnn(frame)
            if bbox is None:
                bbox = self._fallback_center_bbox(frame)

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, x1 + 1, w))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, y1 + 1, h))

            occ = frame.copy()
            occ[y1:y2, x1:x2] = 0
            out.append(occ)
        return out

