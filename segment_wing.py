#!/usr/bin/env python3
# --------------------------------------------------------------
# segment_wing.py — YOLOv8 + SAM3 wing-cell segmentation
#
# Automatic mode (default):
#   upright-landscape alignment -> YOLOv8 boxes (pa2r/dm) ->
#   native SAM3 bounding-box prompts -> OpenCV contours -> CSV.
# Manual fallback:
#   the original 1 inclusive + 2 exclusive point prompt per cell.
#
# Usage:
#   python3 segment_wing.py <image> [--prompt-mode auto|manual] \
#          [--points points.json] [--model facebook/sam3] \
#          [--out contours_coordinates_X.csv] [--device auto|cpu|cuda] \
#          [--masks-dir DIR]
#
# Automatic detection uses /app/models/yolo_wings_best.pt at imgsz=1024,
# conf=0.5. Class 0 is pa2r (legacy masks5), class 1 is dm (legacy masks9).
# If either class is missing, a partial _detections_<stem>.json is written,
# a filename-specific WARNING is logged, and the process exits with status 2
# so the web UI can offer manual prompting instead of crashing.
#
# Manual points.json format (original-image pixel coordinates):
#   {
#     "pa2r": {"points": [[x, y], ...], "labels": [1, 0, 0]},
#     "dm":   {"points": [[x, y], ...], "labels": [1, 0, 0]}
#   }
#
# The Hugging Face cache (/root/.cache/huggingface) should be mounted as a
# named Docker volume so the SAM3 weights survive image removal.
# --------------------------------------------------------------

import argparse
import json
import os
import resource
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from wing_detection import (
    YOLO_WEIGHTS,
    IncompleteDetectionError,
    WingCellDetector,
    load_aligned_image,
    require_complete_boxes,
    write_detection_json,
)


def log(msg):
    print(msg, flush=True)


def bench(tag, seconds):
    if tag == "peak_rss_mb":
        log(f"SEGMENT_BENCH\t{tag}\t{seconds:.0f} MB")
    else:
        log(f"SEGMENT_BENCH\t{tag}\t{seconds:.1f}s")


def process_wing_cell(model, processor, device, image_obj,
                      points=None, labels=None, box=None):
    """SAM3 box or point prompt -> largest OpenCV contour and binary mask."""
    if box is not None:
        # [image level, box level, xyxy] — native SAM3 bounding-box prompt.
        prompt_inputs = {"input_boxes": [[box]]}
    else:
        # [image level, object level, point level, xy] — manual fallback.
        prompt_inputs = {
            "input_points": [[points]],
            "input_labels": [[labels]],
        }

    inputs = processor(
        images=image_obj,
        return_tensors="pt",
        **prompt_inputs,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    original_sizes = inputs["original_sizes"].cpu()
    if "reshaped_input_sizes" in inputs:
        reshaped_sizes = inputs["reshaped_input_sizes"].cpu()
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes, reshaped_sizes
        )
    else:
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes
        )

    scores = outputs.iou_scores
    best_mask_idx = torch.argmax(scores[0, 0])
    binary_mask = masks[0][0, best_mask_idx].numpy().astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Segmentation failed: no contour found for this cell.")
    return max(contours, key=cv2.contourArea).squeeze(), binary_mask


def main():
    ap = argparse.ArgumentParser(
        description="Automatic YOLOv8-box + SAM3 wing-cell segmentation")
    ap.add_argument("image", help="path to the wing photo (PNG/JPEG)")
    ap.add_argument("--prompt-mode", default="auto", choices=["auto", "manual"],
                    help="auto: YOLOv8 boxes (default); manual: point prompts")
    ap.add_argument("--points", default=None,
                    help="points.json (required only with --prompt-mode manual)")
    ap.add_argument("--yolo-model", default=YOLO_WEIGHTS,
                    help=f"YOLOv8 weights (default: {YOLO_WEIGHTS})")
    ap.add_argument("--detections-out", default=None,
                    help="detection JSON (default: _detections_<stem>.json)")
    ap.add_argument("--aligned-out", default=None,
                    help="aligned auto-mode image (default: _aligned_<stem>.png)")
    ap.add_argument("--model", default="facebook/sam3",
                    help="HF model id (e.g. facebook/sam3 or facebook/sam3-fast)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: contours_coordinates_<stem>.csv)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--masks-dir", default=None,
                    help="optional directory to save the binary masks (PNG) for QC")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log(f"Using compute device: {device}")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    prompts = None
    boxes = None
    aligned_out = None

    # ---- orientation + YOLO detection (automatic mode) ---------------------
    if args.prompt_mode == "auto":
        image_obj, alignment = load_aligned_image(args.image)
        log(f"Image aligned for YOLO/SAM3: {alignment['original_size']} -> "
            f"{alignment['aligned_size']} px, EXIF orientation "
            f"{alignment['exif_orientation']}, portrait rotation "
            f"{alignment['portrait_rotation_ccw']} deg CCW")

        aligned_out = args.aligned_out or f"_aligned_{stem}.png"
        image_obj.save(aligned_out)
        detections_out = args.detections_out or f"_detections_{stem}.json"

        t0 = time.time()
        detector = WingCellDetector(args.yolo_model)
        yolo_load_s = time.time() - t0
        log(f"YOLOv8 model loaded once: {args.yolo_model} ({yolo_load_s:.1f}s)")
        bench("yolo_model_load", yolo_load_s)

        t0 = time.time()
        boxes = detector.detect(image_obj)
        detect_s = time.time() - t0
        bench("yolo_detect", detect_s)
        log(f"YOLOv8 prediction complete ({detect_s:.1f}s, imgsz=1024, conf=0.5)")

        # Save partial results BEFORE the completeness guard for diagnosis.
        write_detection_json(detections_out, args.image, aligned_out,
                             alignment, boxes, args.yolo_model)
        log(f"YOLOv8 detections saved to {detections_out}")
        for cell in ("pa2r", "dm"):
            if cell in boxes:
                detection = boxes[cell]
                log(f"YOLOv8 {cell} ({detection['mask_target']}): "
                    f"box={detection['xyxy']}, "
                    f"confidence={detection['confidence']:.4f}")

        # Emits the required WARNING with filename; exits cleanly via status 2.
        require_complete_boxes(args.image, boxes)

    else:
        if not args.points:
            ap.error("--points is required with --prompt-mode manual")
        # Manual coordinates remain in the original photo's coordinate frame.
        image_obj = Image.open(args.image).convert("RGB")
        with open(args.points) as fh:
            prompts = json.load(fh)
        log(f"Manual prompt fallback: {args.points}")

    # ---- SAM3 (downloads on first run into the HF cache) ---------------------
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        from huggingface_hub import login
        login(token=token)
    else:
        log("Warning: HF_TOKEN is not set — this only works if the weights are "
            "already cached (gated model: https://huggingface.co/facebook/sam3).")

    t0 = time.time()
    processor = Sam3TrackerProcessor.from_pretrained(args.model)
    model = Sam3TrackerModel.from_pretrained(args.model).to(device)
    model.eval()
    load_s = time.time() - t0
    log(f"SAM3 model loaded: {args.model} ({load_s:.1f}s)")
    bench("model_load", load_s)

    # ---- segment each cell (pa2r first, dm second — same order as Colab) ----
    contours_img = {}
    for cell in ("pa2r", "dm"):
        t0 = time.time()
        if args.prompt_mode == "auto":
            detection = boxes[cell]
            contour, mask = process_wing_cell(
                model, processor, device, image_obj,
                box=detection["xyxy"])
            prompt_description = (
                f"YOLO box: {detection['xyxy']}, "
                f"confidence: {detection['confidence']:.4f}")
        else:
            if cell not in prompts:
                raise RuntimeError(f"points.json is missing the '{cell}' prompts.")
            pts = prompts[cell]["points"]
            labels = prompts[cell]["labels"]
            contour, mask = process_wing_cell(
                model, processor, device, image_obj, pts, labels)
            prompt_description = f"manual prompt points: {pts}"

        infer_s = time.time() - t0
        bench(f"inference_{cell}", infer_s)
        log(f"{cell}: {len(contour)} boundary coordinates "
            f"({infer_s:.1f}s, {prompt_description})")
        contours_img[cell] = contour

        if args.masks_dir:
            os.makedirs(args.masks_dir, exist_ok=True)
            from PIL import Image as PILImage
            PILImage.fromarray(mask * 255).save(
                os.path.join(args.masks_dir, f"mask_{cell}.png"))

    # ---- notebook post-processing: y-axis flip, NaN padding, CSV -------------
    flipped = {}
    for cell, cont in contours_img.items():
        c = cont.copy().astype(np.float32)
        c[:, 1] = -c[:, 1]
        flipped[cell] = c

    max_len = max(len(c) for c in flipped.values())
    padded = {}
    for cell, c in flipped.items():
        pad_n = max_len - len(c)
        padded[f"{cell}_x"] = np.pad(c[:, 0], (0, pad_n), "constant",
                                    constant_values=np.nan)
        padded[f"{cell}_y"] = np.pad(c[:, 1], (0, pad_n), "constant",
                                     constant_values=np.nan)

    df = pd.DataFrame({
        "dm_x":   padded["dm_x"],
        "dm_y":   padded["dm_y"],
        "pa2r_x": padded["pa2r_x"],
        "pa2r_y": padded["pa2r_y"],
    })

    out = args.out or f"contours_coordinates_{stem}.csv"
    df.to_csv(out, index=False)
    log(f"Contour coordinates saved to {out} ({max_len} rows)")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    bench("peak_rss_mb", peak_mb)
    log(f"Peak RSS: {peak_mb:.0f} MB")


if __name__ == "__main__":
    try:
        main()
    except IncompleteDetectionError:
        # The detector already logged the actionable filename-specific warning.
        sys.exit(2)
