#!/usr/bin/env python3
"""
evaluate_pose.py — run YOLO26s -> MobileNetV3 keypoints -> EPnP on SPEED+ images
and report rotation / translation error against the ground-truth labels.

Works on Kaggle (default paths) or locally (pass paths via flags).

Examples
--------
# Kaggle, defaults, 1000 random images from synthetic validation + sunlamp
python evaluate_pose.py

# All images, custom domains, save per-image CSV + summary JSON
python evaluate_pose.py --domains synthetic sunlamp lightbox --n 0 --out results

# Colab / Jupyter: call main() with a list of args instead of the command line
#   from evaluate_pose import main
#   main(["--base", "/content/drive/MyDrive/speedplusv2", "--yolo", "/content/yolo_best.pt", ...])

# Local machine
python evaluate_pose.py --base /data/speedplusv2 \
    --yolo weights/yolo_best.pt --kpt weights/keypoint_mobilenet.pth \
    --tango-mat weights/tangoPoints.mat
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

# ----------------------------------------------------------------------------
# Defaults (your Kaggle paths)
# ----------------------------------------------------------------------------
DEF_BASE = "/kaggle/input/datasets/isidorotamassia/speed/speedplusv2"
DEF_YOLO = "/kaggle/input/models/raghavmadan07/yolo/pytorch/default/1/best.pt"
DEF_KPT = "/kaggle/input/datasets/raghavmadan07/regress/keypoint_mobilenet.pth"
DEF_MAT = "/kaggle/input/datasets/raghavmadan07/tangopoint/tangoPoints.mat"

CROP_SIZE = 224
NUM_KPTS = 11
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Official SPEED+ score thresholds (below these, the error counts as 0)
THRESH_ROT_DEG = 0.169
THRESH_TRANS_M = 0.002173


# ----------------------------------------------------------------------------
# Geometry helpers (same conventions as the training notebook)
# ----------------------------------------------------------------------------
def quat2dcm(q: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion -> DCM (identical to the notebook's quat2dcm)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    d = np.zeros((3, 3))
    d[0, 0] = 2 * q0**2 - 1 + 2 * q1**2
    d[1, 1] = 2 * q0**2 - 1 + 2 * q2**2
    d[2, 2] = 2 * q0**2 - 1 + 2 * q3**2
    d[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    d[0, 2] = 2 * q1 * q3 - 2 * q0 * q2
    d[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    d[1, 2] = 2 * q2 * q3 + 2 * q0 * q1
    d[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    d[2, 1] = 2 * q2 * q3 - 2 * q0 * q1
    return d


def gt_rotation(q_vbs2tango: np.ndarray) -> np.ndarray:
    """
    Camera-from-object rotation used by the notebook's project_keypoints():
        pose_mat = [ quat2dcm(q).T | t ]
    This is the same frame cv2.solvePnP returns, so errors are directly comparable.
    """
    return quat2dcm(q_vbs2tango).T


def rotation_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices (== 2*acos|<q_gt,q_est>|)."""
    c = (np.trace(R_gt.T @ R_est) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def speed_score(rot_deg: float, trans_m: float, t_gt: np.ndarray) -> float:
    """Per-image SPEED+ score = E_q(rad) + E_t / ||t_gt||, with the official dead-zones."""
    e_q = 0.0 if rot_deg < THRESH_ROT_DEG else np.radians(rot_deg)
    e_t = 0.0 if trans_m < THRESH_TRANS_M else trans_m / np.linalg.norm(t_gt)
    return float(e_q + e_t)


def solve_pose(kpts_2d, kp3d, K, dist, refine=False):
    """EPnP on the 11 predicted keypoints. Returns (R, t, reproj_px) or None."""
    obj = np.ascontiguousarray(kp3d, dtype=np.float64)
    img = np.ascontiguousarray(kpts_2d, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), reproj


def project_gt_keypoints(q, t, kp3d, K, dist):
    """Ground-truth 2D keypoints (px), used only for the keypoint-error column."""
    R = gt_rotation(q)
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(
        np.ascontiguousarray(kp3d, np.float64), rvec,
        np.asarray(t, np.float64).reshape(3, 1), K, dist)
    return pts.reshape(-1, 2)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_camera(path: Path):
    cam = json.load(open(path))
    K = np.array(cam["cameraMatrix"], dtype=np.float64)
    dist = np.array(cam["distCoeffs"], dtype=np.float64).ravel()
    size = (int(cam.get("Nu", 1920)), int(cam.get("Nv", 1200)))
    return K, dist, size


def load_kp3d(path: Path) -> np.ndarray:
    kp = loadmat(str(path))["tango3Dpoints"].T.astype(np.float64)
    if kp.shape != (NUM_KPTS, 3):
        raise ValueError(f"Expected ({NUM_KPTS},3) 3D keypoints, got {kp.shape}")
    return kp


def label_file_for(base: Path, domain: str, synthetic_labels: str) -> Path:
    name = synthetic_labels if domain == "synthetic" else f"{domain}.json"
    p = base / domain / name
    if not p.exists():
        raise FileNotFoundError(
            f"Label file not found: {p}\n"
            f"Check the folder, or the domain has no ground truth.")
    return p


def get_field(ann: dict, *names):
    for n in names:
        if n in ann:
            return ann[n]
    raise KeyError(f"None of {names} found in annotation keys {list(ann)[:8]}...")


# ----------------------------------------------------------------------------
# Models (heavy imports kept inside so the math above can be unit-tested alone)
# ----------------------------------------------------------------------------
def build_models(yolo_path: str, kpt_path: str, device: str | None):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from ultralytics import YOLO

    class KeypointMobileNet(nn.Module):
        def __init__(self, num_coords=NUM_KPTS * 2):
            super().__init__()
            base = models.mobilenet_v3_small(weights=None)
            feat_dim = base.classifier[0].in_features
            base.classifier = nn.Identity()
            self.backbone = base
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.Hardswish(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.Hardswish(), nn.Dropout(0.2),
                nn.Linear(128, num_coords), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(yolo_path)
    ckpt = torch.load(kpt_path, map_location=device, weights_only=False)
    kpt = KeypointMobileNet().to(device).eval()
    kpt.load_state_dict(ckpt["state_dict"])
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    info = {"device": device, "epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss")}
    return yolo, kpt, tfm, device, torch, info


def add_margin(x1, y1, x2, y2, margin, w, h):
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    return max(0, x1 - mw), max(0, y1 - mh), min(w, x2 + mw), min(h, y2 + mh)


def predict(img_bgr, yolo, kpt, tfm, device, torch, conf, iou, margin):
    """Exactly the notebook's predict(): YOLO box (+margin) -> crop -> keypoints."""
    h, w = img_bgr.shape[:2]
    res = yolo(img_bgr, conf=conf, iou=iou, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    detected = len(boxes) > 0
    if detected:
        confs = res.boxes.conf.cpu().numpy()
        best = boxes[confs.argmax()].astype(float)
        box = add_margin(*best, margin, w, h)
        det_conf = float(confs.max())
    else:
        box, det_conf = (0, 0, w, h), float("nan")   # fallback: full frame
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
    inp = tfm(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = kpt(inp).float().cpu().squeeze().numpy()
    kpts = np.stack([pred[0::2] * bw + x1, pred[1::2] * bh + y1], axis=-1)
    return kpts, detected, det_conf


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_domain(domain, args, ctx):
    base = Path(args.base)
    labels_path = label_file_for(base, domain, args.synthetic_labels)
    anns = json.load(open(labels_path))
    img_dir = base / domain / "images"

    random.Random(args.seed).shuffle(anns)
    if args.n > 0:
        anns = anns[: args.n]

    rows, t0 = [], time.time()
    try:
        from tqdm import tqdm
        it = tqdm(anns, desc=domain)
    except ImportError:
        it = anns

    for a in it:
        fn = get_field(a, "filename")
        img = cv2.imread(str(img_dir / fn))
        if img is None:
            continue
        h, w = img.shape[:2]
        q = np.array(get_field(a, "q_vbs2tango_true", "q_vbs2tango"), np.float64)
        t_gt = np.array(get_field(a, "r_Vo2To_vbs_true", "r_Vo2To_vbs"), np.float64)

        kpts, detected, det_conf = predict(
            img, ctx["yolo"], ctx["kpt"], ctx["tfm"], ctx["device"], ctx["torch"],
            args.conf, args.iou, args.margin)

        gt_kpts = project_gt_keypoints(q, t_gt, ctx["kp3d"], ctx["K"], ctx["dist"])
        kp_err = float(np.linalg.norm(kpts - gt_kpts, axis=1).mean())

        row = dict(domain=domain, filename=fn, img_w=w, img_h=h,
                   detected=int(detected), det_conf=det_conf,
                   kpt_err_px=kp_err, range_gt_m=float(np.linalg.norm(t_gt)),
                   rot_err_deg=np.nan, trans_err_m=np.nan, trans_err_rel_pct=np.nan,
                   speed_score=np.nan, reproj_px=np.nan,
                   tx_gt=t_gt[0], ty_gt=t_gt[1], tz_gt=t_gt[2],
                   tx_est=np.nan, ty_est=np.nan, tz_est=np.nan)

        sol = solve_pose(kpts, ctx["kp3d"], ctx["K"], ctx["dist"], refine=args.refine)
        if sol is not None:
            R_est, t_est, reproj = sol
            rot = rotation_error_deg(gt_rotation(q), R_est)
            tr = float(np.linalg.norm(t_est - t_gt))
            row.update(rot_err_deg=rot, trans_err_m=tr,
                       trans_err_rel_pct=100 * tr / np.linalg.norm(t_gt),
                       speed_score=speed_score(rot, tr, t_gt), reproj_px=reproj,
                       tx_est=t_est[0], ty_est=t_est[1], tz_est=t_est[2])
        rows.append(row)

    print(f"[{domain}] {len(rows)} images in {time.time() - t0:.1f}s")
    return rows


def stats(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return dict(mean=None, median=None, p90=None, max=None)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), max=float(x.max()))


def summarize(rows):
    def block(rs):
        n = len(rs)
        rot = [r["rot_err_deg"] for r in rs]
        tr = [r["trans_err_m"] for r in rs]
        rot_a = np.array([v for v in rot if np.isfinite(v)])
        tr_a = np.array([v for v in tr if np.isfinite(v)])
        return dict(
            n=n,
            detection_rate=float(np.mean([r["detected"] for r in rs])) if n else None,
            keypoint_err_px=stats([r["kpt_err_px"] for r in rs]),
            rotation_err_deg=stats(rot),
            translation_err_m=stats(tr),
            translation_err_rel_pct=stats([r["trans_err_rel_pct"] for r in rs]),
            speed_score=stats([r["speed_score"] for r in rs]),
            reproj_px=stats([r["reproj_px"] for r in rs]),
            frac_rot_below_5deg=float((rot_a < 5).mean()) if len(rot_a) else None,
            frac_rot_below_10deg=float((rot_a < 10).mean()) if len(rot_a) else None,
            frac_trans_below_0p1m=float((tr_a < 0.1).mean()) if len(tr_a) else None,
        )
    return dict(all_images=block(rows),
                detected_only=block([r for r in rows if r["detected"]]))


def fmt(v, spec=".2f"):
    return "n/a" if v is None else format(v, spec)


def print_summary(domain, s):
    for key, title in (("all_images", "ALL IMAGES (full-frame fallback if no box)"),
                       ("detected_only", "DETECTED ONLY")):
        b = s[key]
        if not b["n"]:
            continue
        print(f"\n=== {domain.upper()} — {title} — n={b['n']} ===")
        print(f"  detection rate       : {fmt(b['detection_rate'], '.1%')}")
        k, r, t = b["keypoint_err_px"], b["rotation_err_deg"], b["translation_err_m"]
        print(f"  keypoint error (px)  : mean {fmt(k['mean'])} | median {fmt(k['median'])} | p90 {fmt(k['p90'])}")
        print(f"  ROTATION error (deg) : mean {fmt(r['mean'])} | median {fmt(r['median'])} | p90 {fmt(r['p90'])} | max {fmt(r['max'])}")
        print(f"  TRANSLATION err (m)  : mean {fmt(t['mean'], '.3f')} | median {fmt(t['median'], '.3f')} | p90 {fmt(t['p90'], '.3f')} | max {fmt(t['max'], '.3f')}")
        tp = b["translation_err_rel_pct"]
        print(f"  translation (% range): mean {fmt(tp['mean'])} | median {fmt(tp['median'])}")
        sc = b["speed_score"]
        print(f"  SPEED+ score (lower=better): mean {fmt(sc['mean'], '.4f')} | median {fmt(sc['median'], '.4f')}")
        print(f"  reprojection (px)    : mean {fmt(b['reproj_px']['mean'])}")
        print(f"  rot <5° / <10°       : {fmt(b['frac_rot_below_5deg'], '.1%')} / {fmt(b['frac_rot_below_10deg'], '.1%')}"
              f"   |   trans <0.1 m: {fmt(b['frac_trans_below_0p1m'], '.1%')}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEF_BASE, help="speedplusv2 folder")
    ap.add_argument("--domains", nargs="+", default=["synthetic", "sunlamp"],
                    help="any of: synthetic sunlamp lightbox")
    ap.add_argument("--synthetic-labels", default="validation.json",
                    help="synthetic label file (use validation.json — train.json was seen in training)")
    ap.add_argument("--yolo", default=DEF_YOLO)
    ap.add_argument("--kpt", default=DEF_KPT)
    ap.add_argument("--tango-mat", default=DEF_MAT)
    ap.add_argument("--camera", default=None, help="camera.json (default: <base>/camera.json)")
    ap.add_argument("--n", type=int, default=1000, help="random images per domain; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--margin", type=float, default=0.07)
    ap.add_argument("--refine", action="store_true", help="LM-refine EPnP (notebook does not)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--out", default="pose_eval", help="output prefix for CSV/JSON")
    return ap


def main(argv=None):
    # parse_known_args ignores extra flags that Jupyter/Colab inject (e.g. "-f kernel.json")
    args, _ = build_parser().parse_known_args(argv)

    base = Path(args.base)
    K, dist, (nu, nv) = load_camera(Path(args.camera) if args.camera else base / "camera.json")
    kp3d = load_kp3d(Path(args.tango_mat))
    yolo, kpt, tfm, device, torch, info = build_models(args.yolo, args.kpt, args.device)
    print(f"Device: {info['device']} | checkpoint epoch={info['epoch']} val_loss={info['val_loss']}")
    print(f"Camera intrinsics are for {nu}x{nv}; images of other sizes will give wrong pose.")

    ctx = dict(yolo=yolo, kpt=kpt, tfm=tfm, device=device, torch=torch, kp3d=kp3d, K=K, dist=dist)
    all_rows, summaries = [], {}
    for d in args.domains:
        rows = evaluate_domain(d, args, ctx)
        if not rows:
            print(f"[{d}] no images evaluated")
            continue
        if rows[0]["img_w"] != nu or rows[0]["img_h"] != nv:
            print(f"WARNING [{d}]: image size {rows[0]['img_w']}x{rows[0]['img_h']} != camera {nu}x{nv}")
        all_rows += rows
        summaries[d] = summarize(rows)
        print_summary(d, summaries[d])

    if all_rows:
        csv_path, json_path = Path(f"{args.out}_per_image.csv"), Path(f"{args.out}_summary.json")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        json.dump({"args": vars(args), "checkpoint": info, "summary": summaries},
                  open(json_path, "w"), indent=2)
        print(f"\nSaved: {csv_path}  |  {json_path}")


if __name__ == "__main__":
    main()
