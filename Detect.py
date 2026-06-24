"""
D3F-Det Inference Script
Usage:
    python detect.py --weights runs/best_model.pth --source test_image.jpg
    python detect.py --weights runs/best_model.pth --source test_images/ --output results/
    python detect.py --weights runs/best_model.pth --source test.mp4 --conf 0.25
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="D3F-Det Inference")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights (.pth or .pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to input image, directory of images, or video file")
    parser.add_argument("--output", type=str, default="inference_results",
                        help="Directory to save inference results (default: inference_results)")
    parser.add_argument("--config", type=str, default="DBDown+DFMS+3S-RN.yaml",
                        help="Model configuration file (default: DBDown+DFMS+3S-RN.yaml)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--device", type=str, default="1",
                        help="CUDA device (default: 1; use 'cpu' for CPU inference)")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save detection results as .txt files (YOLO format)")
    parser.add_argument("--nosave", action="store_true",
                        help="Do not save images/videos with detections")
    parser.add_argument("--novis", action="store_true",
                        help="Do not visualize results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check source exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[ERROR] Source not found: {args.source}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    if torch.cuda.is_available() and args.device != "cpu":
        device_id = int(args.device)
        device = f"cuda:{args.device}"
    else:
        device_id = None
        device = "cpu"
    print(f"[INFO] Loading model from {args.weights} on {device}...")

    # Load model directly from weights file (Ultralytics standard way)
    model = YOLO(args.weights)

    # Prepare source (support image / directory / video)
    if source_path.is_file():
        sources = [str(source_path)]
    elif source_path.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        sources = sorted([
            str(p) for p in source_path.iterdir()
            if p.suffix.lower() in image_exts
        ])
        if not sources:
            print(f"[ERROR] No image files found in {args.source}")
            sys.exit(1)
        print(f"[INFO] Found {len(sources)} image(s) in {args.source}")
    else:
        sources = [str(args.source)]

    # Run inference
    print(f"[INFO] Running inference with conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}...")

    for src in sources:
        print(f"[INFO] Processing: {src}")
        results = model.predict(
            source=src,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device_id,
            verbose=False,
        )

        for i, result in enumerate(results):
            # Build output path
            src_stem = Path(src).stem
            out_img = os.path.join(args.output, f"{src_stem}_detected.jpg")

            if not args.nosave and result.orig_img is not None:
                annotated = result.plot(conf=args.conf)
                cv2.imwrite(out_img, annotated)
                print(f"  -> Saved: {out_img}")

            # Save YOLO-format txt
            if args.save_txt and result.boxes is not None:
                out_txt = os.path.join(args.output, f"{src_stem}.txt")
                with open(out_txt, "w") as f:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        conf_val = box.conf[0].item()
                        xywh = box.xywh[0].tolist()
                        # Normalize to [0,1]
                        h, w = result.orig_shape[:2]
                        x, y, bw, bh = xywh
                        nx, ny = x / w, y / h
                        nbw, nbh = bw / w, bh / h
                        f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nbw:.6f} {nbh:.6f} {conf_val:.6f}\n")
                print(f"  -> Labels saved: {out_txt}")

            # Print detections summary
            if result.boxes is not None:
                num_dets = len(result.boxes)
                print(f"  -> Detections: {num_dets} object(s)")
                if num_dets > 0 and hasattr(result, "names"):
                    cls_ids = result.boxes.cls.int().tolist()
                    for cid in set(cls_ids):
                        name = result.names.get(cid, f"class_{cid}")
                        count = cls_ids.count(cid)
                        print(f"     {name}: {count}")
            else:
                print("  -> No objects detected.")

    print(f"[INFO] Inference complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
