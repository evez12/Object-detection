"""Real-time webcam object detection with YOLO26 and OpenCV.

Example:
    python main.py --model yolo26n.pt --conf 0.4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter

import cv2

from detector import Detector, ModelLoadError
from drawing import FPSCounter, draw_detections, draw_fps

WINDOW_NAME: str = "YOLO26 Real-Time Detection"
MAX_CONSECUTIVE_READ_FAILURES: int = 30

logger = logging.getLogger("yolo26")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam object detection with YOLO26.")
    parser.add_argument("--model", type=str, default="yolo26l.pt",
                        help="Model weights: yolo26{n,s,m,l,x}.pt or a path to custom weights.")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height.")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (0-1). Lower finds more objects but adds false positives.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size in pixels.")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' (CUDA if available, else CPU), 'cpu', '0', 'cuda:0', ...")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Only detect these classes, by name or id (e.g. --classes person cup 67).")
    args = parser.parse_args(argv)

    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        parser.error("--imgsz must be a positive multiple of 32")
    return args


def open_camera(index: int, width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows (much faster startup than MSMF).

    The requested resolution is a hint: the driver picks the closest mode it supports.
    """
    backends: list[int] = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            if width and height:
                # MJPG is needed by most USB webcams to deliver HD at 30 FPS.
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        cap.release()
    raise RuntimeError(
        f"Could not open camera with index {index}. Check that it is connected, "
        f"not used by another application, and try a different --source value."
    )


class StableObjectSet:
    """Debounces the per-frame set of class names so one-frame flickers are ignored.

    A class is reported once it has been seen in `appear_frames` consecutive frames,
    and dropped once it has been missing for `disappear_frames` consecutive frames.
    """

    def __init__(self, appear_frames: int = 5, disappear_frames: int = 10) -> None:
        self.appear_frames: int = appear_frames
        self.disappear_frames: int = disappear_frames
        self.present: set[str] = set()
        self._seen: Counter[str] = Counter()
        self._missing: Counter[str] = Counter()

    def update(self, class_names: set[str]) -> bool:
        """Feed one frame's class names; return True if the reported set changed."""
        changed = False
        for name in class_names:
            self._missing.pop(name, None)
            self._seen[name] += 1
            if name not in self.present and self._seen[name] >= self.appear_frames:
                self.present.add(name)
                changed = True
        for name in set(self._seen) - class_names:
            self._seen.pop(name)
        for name in self.present - class_names:
            self._missing[name] += 1
            if self._missing[name] >= self.disappear_frames:
                self.present.discard(name)
                self._missing.pop(name)
                changed = True
        return changed


def format_objects(names: set[str]) -> str:
    return ", ".join(sorted(names)) if names else "(nothing)"


def run(args: argparse.Namespace) -> int:
    try:
        detector = Detector(
            model_path=args.model,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            classes=args.classes,
        )
    except (ModelLoadError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        cap = open_camera(args.source, args.width, args.height)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    fps_counter = FPSCounter()
    stable_objects = StableObjectSet()
    failed_reads = 0

    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera %d opened (%dx%d). Warming up the model...",
                    args.source, frame_width, frame_height)
        detector.warmup((frame_height, frame_width, 3) if frame_width and frame_height else None)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        logger.info("Running. Press 'q' in the video window to quit.")
        print(f"Detected: {format_objects(stable_objects.present)}", flush=True)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= MAX_CONSECUTIVE_READ_FAILURES:
                    logger.error("Failed to read %d consecutive frames; stopping.", failed_reads)
                    return 1
                time.sleep(0.01)
                continue
            failed_reads = 0

            detections = detector.detect(frame)

            if stable_objects.update({det.class_name for det in detections}):
                print(f"Detected: {format_objects(stable_objects.present)}", flush=True)

            draw_detections(frame, detections)
            draw_fps(frame, fps_counter.update())
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Also quit when the window is closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released, windows closed.")

    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
