# YOLO26 Real-Time Object Detection

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-111f68.svg)](https://docs.ultralytics.com/models/yolo26/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5c3ee8.svg)](https://opencv.org/)

Real-time webcam object detection with [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), PyTorch and OpenCV.
The app detects the 80 COCO object classes, draws labelled bounding boxes with an FPS overlay,
and prints a debounced list of the objects currently in view.

**Project page:** <https://evez12.github.io/Object-detection/>

## Features

- **Five YOLO26 model sizes** (`n`, `s`, `m`, `l`, `x`), selectable from the command line.
- **Automatic device selection.** Uses CUDA when PyTorch can actually run on the GPU and falls back to CPU otherwise.
  This includes a guard for RTX 50xx (Blackwell, `sm_120`) GPUs paired with an incompatible PyTorch build.
- **FP16 inference** on GPU, enabled automatically.
- **Class filtering** by name or COCO id, applied inside the model's `predict` call.
- **Stable console output.** `StableObjectSet` debounces per-frame detections so one-frame flickers are not reported.
- **Robust camera handling.** DirectShow is preferred on Windows for fast startup, MJPG is requested for HD at 30 FPS,
  and the loop stops cleanly after repeated read failures.
- **Typed, modular code** split into detection, drawing and CLI layers.

## Project structure

```
Object-detection/
├── main.py           # Entry point: CLI, camera loop, console output
├── detector.py       # Detector class: device selection, model loading, inference
├── drawing.py        # Bounding boxes, labels, FPS overlay, FPS counter
├── requirements.txt  # Pinned PyTorch (CUDA 12.8) and other dependencies
├── docs/             # GitHub Pages site
└── README.md
```

## Requirements

- Python 3.10 or newer
- A webcam
- Optional: an NVIDIA GPU. The pinned PyTorch build targets CUDA 12.8, which is required for RTX 50xx GPUs.

## Installation

```powershell
git clone https://github.com/evez12/Object-detection.git
cd Object-detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.
The pinned `+cu128` wheels are Windows/Linux CUDA builds; on other platforms, edit `requirements.txt` to use a matching
`torch` / `torchvision` build from <https://pytorch.org/get-started/locally/>.

Check that PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False`, you have a CPU-only build. Reinstall the CUDA build:

```powershell
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Model weights are **not** stored in this repository. They are downloaded automatically on first run.

## Usage

```powershell
python main.py                                    # yolo26l.pt, camera 0, 1280x720
python main.py --source 1                         # use camera index 1
python main.py --model yolo26n.pt --conf 0.4      # fastest model, fewer weak detections
python main.py --width 640 --height 480           # lower camera resolution
python main.py --model yolo26s.pt --imgsz 960     # bigger model, higher resolution
python main.py --device cpu                       # force CPU
python main.py --classes person "cell phone"      # keep only some classes
```

Press **q** in the video window (or close it) to quit.

**Wrong camera?** Virtual cameras (such as "Sharing Camera" or OBS) also get an index, and on Windows they can sit at
index 0 before the built-in webcam. If you see a placeholder image instead of yourself, try `--source 1`, `--source 2`, and so on.

### Command-line arguments

| Argument    | Default      | Description                                         |
|-------------|--------------|-----------------------------------------------------|
| `--model`   | `yolo26l.pt` | Official model name or path to custom `.pt` weights |
| `--source`  | `0`          | Camera index                                        |
| `--width`   | `1280`       | Requested camera width (driver picks nearest mode)  |
| `--height`  | `720`        | Requested camera height                             |
| `--conf`    | `0.4`        | Confidence threshold (0-1)                          |
| `--imgsz`   | `640`        | Inference size (multiple of 32)                     |
| `--device`  | `auto`       | `auto`, `cpu`, `0`, `cuda:0`                        |
| `--classes` | all          | Class names or ids to keep                          |

### Using the detector from Python

```python
import cv2
from detector import Detector

detector = Detector(model_path="yolo26n.pt", conf=0.4, classes=["person", "dog"])
frame = cv2.imread("image.jpg")
for det in detector.detect(frame):
    print(det.class_name, f"{det.confidence:.2f}", det.box)
```

## Choosing a model

Order by size and accuracy: `yolo26n.pt` < `yolo26s.pt` < `yolo26m.pt` < `yolo26l.pt` < `yolo26x.pt`.
Larger models are more accurate but slower. The default is `yolo26l.pt`.

Measured on an RTX 5070 Laptop GPU with a single frame showing a computer mouse held in a hand
(an informal one-frame test, not a benchmark):

| Model     | ms / frame | Result for the mouse            |
|-----------|-----------:|---------------------------------|
| yolo26n   | 21         | "cell phone" 0.74 (wrong)       |
| yolo26s   | 19         | "cell phone" 0.61, "mouse" 0.26 |
| yolo26m   | 26         | not detected                    |
| yolo26l   | 38         | "mouse" 0.62                    |
| yolo26x   | 39         | "mouse" 0.53                    |

With `yolo26l` and a 1280x720 webcam, the full loop runs at about 26 FPS.
Raising `--imgsz` above 640 did not help in this test.

## Filtering classes

YOLO26 models are trained on the 80 COCO classes. To keep only some of them:

- **CLI:** `python main.py --classes person car 16`. The list can mix names and ids. Quote names that contain a space, such as `"cell phone"`.
- **Code:** `Detector(model_path="yolo26n.pt", classes=["person", "dog"])`

Filtered classes are never drawn or printed. To list every class name:

```powershell
python -c "from ultralytics import YOLO; print(YOLO('yolo26n.pt').names)"
```

## How it works

1. `main.py` parses arguments, opens the camera and starts the capture loop.
2. `detector.Detector` loads the model, resolves the device and returns a list of typed `Detection` objects per frame.
3. `drawing.py` renders boxes, labels and a smoothed FPS counter onto the frame.
4. `StableObjectSet` reports a class after it has been seen in 5 consecutive frames and drops it after 10 missing frames.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not open camera with index N` | Close other apps using the camera and try another `--source` value. |
| Placeholder image instead of your face | A virtual camera is at index 0. Try `--source 1` or `2`. |
| Warning that the GPU is not supported by this PyTorch build | Install a CUDA 12.8+ build of PyTorch (see Installation). |
| Model download fails | Check your internet connection, or place the `.pt` file in the project folder. |

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26
- [OpenCV](https://opencv.org/) and [PyTorch](https://pytorch.org/)
