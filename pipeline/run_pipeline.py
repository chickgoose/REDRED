"""
Main pipeline: video → detection → fusion → events → CSV

Usage:
    python run_pipeline.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
        --weights ~/yolov7/runs/train/exp/weights/best.pt \
        --names   ~/yolov7/data/names.txt \
        --prices  ../data/prices.csv \
        --out     ../output/submission.csv \
        --conf    0.4 \
        --device  0

RTF is logged automatically.
"""

import argparse
import time
import sys
import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from event_detector import EventDetector
from multi_view_fusion import fuse
from csv_generator import load_prices, events_to_csv


def load_names(names_path: str):
    with open(names_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_model(weights: str, device: str):
    """Load YOLOv7 directly via torch.load (bypasses attempt_download)."""
    yolov7_root = str(Path.home() / "yolov7")
    sys.path.insert(0, yolov7_root)
    from utils.general import non_max_suppression

    import torch
    import torch.nn as nn
    ckpt = torch.load(weights, map_location=device)
    model = (ckpt.get("ema") or ckpt["model"]).float().fuse().eval()
    # PyTorch 1.12+ compatibility fix
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    return model, non_max_suppression


def _preprocess_single(frame, img_size=640):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).float() / 255.0


def infer_batch(model, nms_fn, frames, conf_thres=0.4, iou_thres=0.45,
                img_size=640, device="cpu"):
    """Run one GPU forward pass for all non-None frames; returns per-cam det lists."""
    valid_idx = [i for i, f in enumerate(frames) if f is not None]
    if not valid_idx:
        return [None] * len(frames)

    tensors = [_preprocess_single(frames[i], img_size) for i in valid_idx]
    batch = torch.stack(tensors).to(device)  # [N, 3, H, W]

    with torch.no_grad():
        preds = model(batch)[0]  # [N, anchors, 5+nc]
    preds = nms_fn(preds, conf_thres, iou_thres)  # list of N tensors

    per_cam = [None] * len(frames)
    for out_i, cam_i in enumerate(valid_idx):
        pred = preds[out_i]
        dets = []
        if pred is not None and len(pred):
            for *xyxy, conf, cls in pred.cpu().numpy():
                dets.append({
                    "class_id":   int(cls),
                    "confidence": float(conf),
                    "bbox":       [float(v) for v in xyxy],
                })
        per_cam[cam_i] = dets
    return per_cam


def open_videos(video_paths):
    caps = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"Warning: cannot open {p}")
            caps.append(None)
        else:
            caps.append(cap)
    return caps


def _read_one(cap):
    if cap is None:
        return None
    ret, frame = cap.read()
    return frame if ret else None


def read_frames(caps, executor):
    return list(executor.map(_read_one, caps))


def video_duration(video_paths):
    total = 0.0
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = max(total, n / fps)
        cap.release()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",   nargs="+", required=True)
    parser.add_argument("--weights",  required=True)
    parser.add_argument("--names",    required=True)
    parser.add_argument("--prices",   required=True)
    parser.add_argument("--out",      default="output/submission.csv")
    parser.add_argument("--conf",     type=float, default=0.4)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--img_size", type=int,   default=640)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--skip",     type=int,   default=2,
                        help="Process every Nth frame (speed vs accuracy)")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    print("Loading model...")
    model, nms_fn = load_model(args.weights, device)

    class_names = load_names(args.names)
    prices      = load_prices(args.prices)

    detector = EventDetector(class_names)
    caps     = open_videos(args.videos)
    vid_len  = video_duration(args.videos)

    print(f"Processing {len(caps)} cameras, video length ≈ {vid_len:.1f}s ...")
    t_start = time.time()
    frame_idx = 0

    with ThreadPoolExecutor(max_workers=len(caps)) as executor:
        while True:
            frames = read_frames(caps, executor)
            if all(f is None for f in frames):
                break

            if frame_idx % args.skip != 0:
                frame_idx += 1
                continue

            per_cam_dets = infer_batch(model, nms_fn, frames,
                                       args.conf, args.iou, args.img_size, device)

            fused_counts = fuse(per_cam_dets)

            flat_dets = [
                {"class_id": cls_id, "confidence": 1.0, "bbox": []}
                for cls_id, cnt in fused_counts.items()
                for _ in range(cnt)
            ]

            new_events = detector.update(flat_dets)
            if new_events:
                for ev in new_events:
                    print(f"  [Frame {frame_idx}] {ev.class_name}: {ev.action} "
                          f"({ev.before}→{ev.after})")

            frame_idx += 1

    for cap in caps:  # executor already closed by 'with' block
        if cap:
            cap.release()

    t_end = time.time()
    proc_time = t_end - t_start
    rtf = proc_time / vid_len if vid_len > 0 else float("inf")
    print(f"\nProcessing time: {proc_time:.1f}s  |  Video length: {vid_len:.1f}s  |  RTF: {rtf:.3f}")

    events_to_csv(
        events=detector.all_events,
        prices=prices,
        out_path=args.out,
        initial_inventory={},
    )


if __name__ == "__main__":
    main()
