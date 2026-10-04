"""Drawing helpers and a smoothed FPS counter."""

from __future__ import annotations

import time
from collections.abc import Iterable

import cv2
import numpy as np
from ultralytics.utils.plotting import colors

from detector import Detection

FONT: int = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Draw bounding boxes with 'class confidence' labels onto the frame in place."""
    height, width = frame.shape[:2]
    thickness = max(1, round((height + width) / 600))
    font_scale = max(0.4, thickness / 3)

    for det in detections:
        x1, y1, x2, y2 = det.box
        color: tuple[int, int, int] = colors(det.class_id, bgr=True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, thickness)
        # Put the label above the box, or inside it when the box touches the top edge.
        label_top = y1 - text_h - baseline - 4
        if label_top < 0:
            label_top = y1
        label_bottom = label_top + text_h + baseline + 4

        cv2.rectangle(frame, (x1, label_top), (x1 + text_w + 4, label_bottom), color, cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_bottom - baseline - 2),
            FONT,
            font_scale,
            _text_color(color),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw the FPS counter in the top-left corner with a dark background for contrast."""
    label = f"FPS: {fps:.1f}"
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (16 + text_w, 16 + text_h + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (12, 12 + text_h), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def _text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text depending on the background brightness (BGR)."""
    blue, green, red = background
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


class FPSCounter:
    """Frames-per-second meter smoothed with an exponential moving average."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing: float = smoothing
        self.fps: float = 0.0
        self._last_time: float | None = None

    def update(self) -> float:
        """Register a new frame and return the current smoothed FPS."""
        now = time.perf_counter()
        if self._last_time is not None:
            elapsed = now - self._last_time
            if elapsed > 0:
                instant = 1.0 / elapsed
                self.fps = (
                    instant if self.fps == 0.0
                    else self.smoothing * self.fps + (1.0 - self.smoothing) * instant
                )
        self._last_time = now
        return self.fps
