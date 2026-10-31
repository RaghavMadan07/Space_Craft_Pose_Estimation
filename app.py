"""OrbitLens backend: YOLO26s detection -> MobileNetV3-Small keypoints (+ experimental EPnP pose)."""
import json, os, tempfile, threading, time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torchvision import models, transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_MB, MAX_VIDEO_MB, MAX_VIDEO_FRAMES = 25, 200, 600
S: dict = {}
LOCK = threading.Lock()


class KeypointNet(nn.Module):
    """Same layout as the training notebook so state_dict keys match."""
    def __init__(self, n=22):
        super().__init__()
        b = models.mobilenet_v3_small(weights=None)
        d = b.classifier[0].in_features
        b.classifier = nn.Identity()
        self.backbone = b
        self.head = nn.Sequential(
            nn.Linear(d, 256), nn.Hardswish(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(128, n), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.backbone(x))


@asynccontextmanager
async def lifespan(app):
    missing = [f for f in ("yolo_best.pt", "keypoint_mobilenet.pth") if not (MODELS / f).exists()]
    S["missing"] = missing
    if not missing:
        S["yolo"] = YOLO(str(MODELS / "yolo_best.pt"))
        ck = torch.load(MODELS / "keypoint_mobilenet.pth", map_location=DEVICE, weights_only=False)
        S["kpt"] = KeypointNet().to(DEVICE).eval()
        S["kpt"].load_state_dict(ck["state_dict"])
        size = int(ck.get("cfg", {}).get("crop_size", 224))
        S["tfm"] = transforms.Compose([transforms.ToPILImage(), transforms.Resize((size, size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    mat, cam = MODELS / "tangoPoints.mat", MODELS / "camera.json"
    if mat.exists() and cam.exists():
        S["kp3d"] = loadmat(str(mat))["tango3Dpoints"].T.astype(np.float64)
        c = json.load(open(cam))
        S["K"] = np.array(c["cameraMatrix"], np.float64)
        S["dist"] = np.array(c["distCoeffs"], np.float64).ravel()
        S["cam_size"] = (int(c.get("Nu", 1920)), int(c.get("Nv", 1200)))
    yield


app = FastAPI(title="OrbitLens", lifespan=lifespan)


def need_models():
    if S.get("missing"):
        raise HTTPException(503, f"Model weights missing in {MODELS}: {S['missing']}")


def pose_from(kpts, w, h):
    if "kp3d" not in S or (w, h) != S["cam_size"]:
        return None   # intrinsics only valid for the camera's image size
    ok, rv, tv = cv2.solvePnP(S["kp3d"], kpts.astype(np.float64), S["K"], S["dist"], flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(S["kp3d"], rv, tv, S["K"], S["dist"])
    R, _ = cv2.Rodrigues(rv)
    x, y, z, wq = Rotation.from_matrix(R).as_quat()
    q = np.array([wq, x, y, z]) * (1 if wq >= 0 else -1)
    return {"experimental": True, "quaternion_wxyz": q.round(5).tolist(),
            "translation_m": tv.ravel().round(4).tolist(), "range_m": round(float(np.linalg.norm(tv)), 4),
            "reproj_px": round(float(np.linalg.norm(proj.reshape(-1, 2) - kpts, axis=1).mean()), 2)}


def infer(img, conf, iou, margin, with_pose=True):
    h, w = img.shape[:2]
    with LOCK, torch.no_grad():
        t0 = time.perf_counter()
        r = S["yolo"](img, conf=conf, iou=iou, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        t1 = time.perf_counter()
        detected = len(boxes) > 0
        if detected:
            cf = r.boxes.conf.cpu().numpy()
            i = int(cf.argmax()); x1, y1, x2, y2 = boxes[i].astype(float); det_conf = float(cf[i])
            mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
            box = (max(0, x1 - mw), max(0, y1 - mh), min(w, x2 + mw), min(h, y2 + mh))
        else:
            box, det_conf = (0.0, 0.0, float(w), float(h)), None   # full-frame fallback (unreliable)
        x1, y1, x2, y2 = box
        crop = cv2.cvtColor(img[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB)
        p = S["kpt"](S["tfm"](crop).unsqueeze(0).to(DEVICE)).float().cpu().squeeze().numpy()
        if DEVICE == "cuda": torch.cuda.synchronize()
        t2 = time.perf_counter()
    kpts = np.stack([p[0::2] * (x2 - x1) + x1, p[1::2] * (y2 - y1) + y1], -1)
    out = {"detected": detected, "confidence": det_conf,
           "box": [round(v, 1) for v in box], "keypoints": kpts.round(1).tolist(),
           "ms": {"yolo": round((t1 - t0) * 1e3, 1), "keypoints": round((t2 - t1) * 1e3, 1)}}
    if with_pose:
        out["pose"] = pose_from(kpts, w, h) if detected else None
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "device": DEVICE, "models_loaded": not S.get("missing"),
            "missing_files": S.get("missing", []), "pose_available": "kp3d" in S}


@app.post("/api/infer/image")
def infer_image(file: UploadFile = File(...), conf: float = Form(0.10), iou: float = Form(0.45),
                margin: float = Form(0.07)):
    need_models()
    data = file.file.read()
    if len(data) > MAX_IMAGE_MB << 20:
        raise HTTPException(413, f"Image larger than {MAX_IMAGE_MB} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    r = infer(img, conf, iou, margin)
    r["image"] = {"width": img.shape[1], "height": img.shape[0]}
    return r


@app.post("/api/infer/video")
def infer_video(file: UploadFile = File(...), conf: float = Form(0.10), iou: float = Form(0.45),
                margin: float = Form(0.07), stride: int = Form(2)):
    need_models()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "v.mp4").suffix, delete=False) as tmp:
        n = 0
        while chunk := file.file.read(1 << 20):
            n += len(chunk)
            if n > MAX_VIDEO_MB << 20:
                tmp.close(); os.unlink(tmp.name); raise HTTPException(413, f"Video larger than {MAX_VIDEO_MB} MB")
            tmp.write(chunk)
    cap = cv2.VideoCapture(tmp.name)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total <= 0 or w == 0:
            raise HTTPException(400, "Could not read video")
        stride = max(1, stride, -(-total // MAX_VIDEO_FRAMES))   # cap processed frames
        frames, i, t0 = [], 0, time.perf_counter()
        while True:
            ok, img = cap.read()
            if not ok: break
            if i % stride == 0:
                r = infer(img, conf, iou, margin, with_pose=False)
                r.update(index=i, t=round(i / fps, 3)); frames.append(r)
            i += 1
    finally:
        cap.release(); os.unlink(tmp.name)
    det = [f for f in frames if f["detected"]]
    return {"video": {"width": w, "height": h, "fps": fps, "frame_count": i, "stride": stride},
            "summary": {"processed": len(frames), "detection_rate": len(det) / max(1, len(frames)),
                        "mean_confidence": float(np.mean([f["confidence"] for f in det])) if det else None,
                        "effective_fps": round(len(frames) / (time.perf_counter() - t0), 1)},
            "frames": frames}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="web")
