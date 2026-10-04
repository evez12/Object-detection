"""YOLO26 model loading, device selection and inference."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or downloaded."""


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected object in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)


def select_device(requested: str | None = None) -> str:
    """Resolve the inference device.

    ``None`` or ``"auto"`` picks CUDA when PyTorch can actually run on the GPU,
    otherwise CPU. Any other value (``"cpu"``, ``"0"``, ``"cuda:0"``) is used as-is.
    """
    if requested and requested.lower() != "auto":
        return requested

    if not torch.cuda.is_available():
        return "cpu"

    # A PyTorch build without kernels for this GPU's architecture reports CUDA as
    # available but fails at the first kernel launch. This matters for RTX 50xx
    # (Blackwell, sm_120), which needs a CUDA 12.8+ build of PyTorch.
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if supported and arch not in supported:
        logger.warning(
            "GPU %s (%s) is not supported by this PyTorch build (supports %s). "
            "Falling back to CPU. Install a matching CUDA build of PyTorch to use the GPU.",
            torch.cuda.get_device_name(0),
            arch,
            ", ".join(supported),
        )
        return "cpu"

    return "cuda:0"


class Detector:
    """Thin wrapper around an Ultralytics YOLO model for per-frame detection."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        device: str | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        classes: Sequence[int | str] | None = None,
    ) -> None:
        self.device: str = select_device(device)
        self.half: bool = self.device != "cpu" and not self.device.startswith("mps")
        # Ultralytics' `quantize` replaces the deprecated `half` flag: 16 = FP16, None = FP32.
        # Passing `half=True` still works but logs a warning and rebuilds the predictor every call.
        self.quantize: int | None = 16 if self.half else None
        self.conf: float = conf
        self.imgsz: int = imgsz

        try:
            # Ultralytics downloads official weights (e.g. yolo26n.pt) automatically
            # if they are not present locally.
            self.model: YOLO = YOLO(model_path)
        except Exception as exc:  # network errors, missing file, corrupt weights...
            raise ModelLoadError(
                f"Could not load model '{model_path}'. If this is an official model name, "
                f"check your internet connection so the weights can be downloaded; "
                f"otherwise check that the file path is correct. Cause: {exc}"
            ) from exc

        self.names: dict[int, str] = dict(self.model.names)
        self.classes: list[int] | None = self._resolve_classes(classes)

        if self.device.startswith("cuda") or self.device.isdigit():
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
            logger.info("Using device %s (%s), half precision: %s", self.device, gpu_name, self.half)
        else:
            logger.info("Using device %s, half precision: %s", self.device, self.half)

    def _resolve_classes(self, classes: Sequence[int | str] | None) -> list[int] | None:
        """Convert a mix of class ids and class names into a list of class ids."""
        if not classes:
            return None

        name_to_id: dict[str, int] = {name.lower(): idx for idx, name in self.names.items()}
        resolved: list[int] = []
        for item in classes:
            text = str(item).strip()
            if text.isdigit() and int(text) in self.names:
                resolved.append(int(text))
            elif text.lower() in name_to_id:
                resolved.append(name_to_id[text.lower()])
            else:
                raise ValueError(
                    f"Unknown class '{item}'. Valid names: {', '.join(self.names.values())}"
                )
        return sorted(set(resolved))

    def warmup(self, frame_shape: tuple[int, ...] | None = None, iterations: int = 3) -> None:
        """Run a few dummy inferences so the first real frames are not slow.

        Pass the camera's frame shape: the letterboxed input size depends on the
        frame's aspect ratio, and every new input size triggers cuDNN autotuning.
        """
        shape = frame_shape if frame_shape is not None else (self.imgsz, self.imgsz, 3)
        dummy = np.zeros(shape, dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR frame and return the detections."""
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            quantize=self.quantize,
            classes=self.classes,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        return [
            Detection(
                class_id=int(cls_id),
                class_name=self.names.get(int(cls_id), str(cls_id)),
                confidence=float(score),
                box=(int(x1), int(y1), int(x2), int(y2)),
            )
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confidences, class_ids)
        ]
