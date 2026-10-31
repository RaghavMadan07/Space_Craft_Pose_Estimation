# OrbitLens — Spacecraft Pose Estimation

Detect a spacecraft in an image or video, locate **11 keypoints**, and estimate its **6-DoF pose** — with a space-themed web UI.

**Pipeline** (trained on the [SPEED+](https://zenodo.org/records/5588480) synthetic set)

```
image/frame ─▶ YOLO26s detector ─▶ crop (+7% margin) ─▶ MobileNetV3-Small keypoint regressor ─▶ EPnP pose
              (800 px, 1 class)                          (224×224 → 11 keypoints)               (experimental)
```

| Item | Value |
|---|---|
| Training data | 47,966 synthetic SPEED+ images (11,994 held out for validation) |
| Detector | YOLO26s, 800 px input, ~17 ms/image on a Tesla T4 |
| Keypoint model | MobileNetV3-Small + MLP head, Wing loss, 11 keypoints from 224×224 crops |
| Pose | `cv2.solvePnP` (EPnP) on the predicted keypoints + 3D model |


## Quickstart

```bash
git clone https://github.com/RaghavMadan07/Space_Craft_Pose_Estimation.git
cd Space_Craft_Pose_Estimation
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put your files in `models/` (they are git-ignored):

| File | Purpose | Required |
|---|---|---|
| `yolo_best.pt` | trained YOLO26s detector | yes |
| `keypoint_mobilenet.pth` | trained keypoint checkpoint (`state_dict`, optional `cfg`) | yes |
| `tangoPoints.mat` | 3D keypoints (`tango3Dpoints`) | for pose |
| `camera.json` | SPEED+ camera intrinsics | for pose |

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

If weights are missing the UI shows which files to add. Set `MODELS_DIR` or `DEVICE=cpu|cuda` to override.

## Using the app
- **Image:** drop/paste a file → *Analyze* → box, 11 keypoints, telemetry, optional pose.
- **Video:** the server processes every *N*-th frame (stride), then the browser plays the video with the overlay locked to the frames. Limits: 25 MB images, 200 MB / 600 processed frames for video.
- **Export:** *Export JSON* downloads all results.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | device, models loaded, missing files, pose availability |
| POST | `/api/infer/image` | multipart `file`; form `conf=0.10`, `iou=0.45`, `margin=0.07` |
| POST | `/api/infer/video` | same + `stride=2`; returns per-frame boxes/keypoints |

Interactive docs: `http://localhost:8000/docs`.

## Evaluate rotation / translation error

Needs SPEED+ (`synthetic/validation.json`, `sunlamp/sunlamp.json`, `camera.json`):

```bash
python scripts/evaluate_pose.py --base /path/to/speedplusv2 \
  --yolo models/yolo_best.pt --kpt models/keypoint_mobilenet.pth \
  --tango-mat models/tangoPoints.mat --domains synthetic sunlamp --n 500
```

Prints keypoint / rotation / translation error and the SPEED+ score, and writes a per-image CSV and summary JSON. In Colab/Jupyter, call `main([...])` instead (see the script's docstring).

## Limitations
- Trained on **synthetic SPEED+ imagery of the Tango satellite only**; other spacecraft or ordinary photos are out of distribution.
- There is no per-keypoint confidence. If nothing is detected the app falls back to the full frame and **warns that the output is unreliable**.
- Pose is **experimental**: only computed on 1920×1200 images (the camera's size) with a detection, and not validated beyond `scripts/evaluate_pose.py`.

## Project layout
```
backend/app.py          FastAPI app + inference pipeline
frontend/index.html     single-file space-themed UI (no build step)
scripts/evaluate_pose.py  pose-error evaluation on SPEED+
models/                 your weights (git-ignored)
```

## Credits
[SPEED+](https://zenodo.org/records/5588480) (Stanford SLAB / ESA ACT), [Ultralytics](https://github.com/ultralytics/ultralytics), PyTorch / torchvision. Check the SPEED+ license before redistributing any dataset files or weights trained on it.
