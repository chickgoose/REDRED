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

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from event_detector import EventDetector
from multi_view_fusion import fuse
from csv_generator import load_prices, events_to_csv


def load_names(names_path: str):
    with open(names_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_model(weights: str, device: str):
    """Load YOLOv7 via torch.hub or direct import."""
    sys.path.insert(0, str(Path(weights).parent.parent.parent))  # yolov7 root
    from models.experimental import attempt_load
    from utils.general import non_max_suppression

    model = attempt_load(weights, map_location=device)
    model.eval()
    return model, non_max_suppression


def preprocess(frame, img_size=640, device="cpu"):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device) / 255.0
    return img.unsqueeze(0)


def infer_frame(model, nms_fn, frame, conf_thres=0.4, iou_thres=0.45,
                img_size=640, device="cpu"):
    tensor = preprocess(frame, img_size, device)
    with torch.no_grad():
        pred = model(tensor)[0]
    pred = nms_fn(pred, conf_thres, iou_thres)[0]

    detections = []
    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred.cpu().numpy():
            detections.append({
                "class_id":   int(cls),
                "confidence": float(conf),
                "bbox":       [float(v) for v in xyxy],
            })
    return detections


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


def read_frames(caps):
    frames = []
    for cap in caps:
        if cap is None:
            frames.append(None)
            continue
        ret, frame = cap.read()
        frames.append(frame if ret else None)
    return frames


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

    while True:
        frames = read_frames(caps)
        if all(f is None for f in frames):
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        per_cam_dets = []
        for frame in frames:
            if frame is None:
                per_cam_dets.append(None)
            else:
                dets = infer_frame(model, nms_fn, frame,
                                   args.conf, args.iou, args.img_size, device)
                per_cam_dets.append(dets)

        fused_counts = fuse(per_cam_dets)

        # Convert fused counts back to flat detection list for EventDetector
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

    for cap in caps:
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
