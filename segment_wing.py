#!/usr/bin/env python3
# --------------------------------------------------------------
# segment_wing.py — SAM3 wing-cell segmentation (container entrypoint)
#
# Desktop/Docker port of the Colab notebook (flieswings_proj.ipynb):
#   - facebook/sam3 via Hugging Face transformers (gated model —
#     requires the HF_TOKEN environment variable and prior access
#     approval at https://huggingface.co/facebook/sam3)
#   - point prompts: per wing cell, 1 inclusive + 2 exclusive points
#     (the desktop equivalent of the Colab click collector)
#   - OpenCV contour extraction, y-axis flip (y * -1), NaN padding,
#     CSV in the exact dm_x, dm_y, pa2r_x, pa2r_y format expected by
#     the classification workflow
#
# Usage:
#   python3 segment_wing.py <image> --points <points.json> \
#          [--model facebook/sam3] [--out contours_coordinates_X.csv] \
#          [--device auto|cpu|cuda] [--masks-dir DIR]
#
# points.json format (pixel coordinates, same as the Colab clicks):
#   {
#     "pa2r": {"points": [[x, y], ...], "labels": [1, 0, 0]},
#     "dm":   {"points": [[x, y], ...], "labels": [1, 0, 0]}
#   }
#   labels: 1 = inclusive, 0 = exclusive
#
# The Hugging Face cache (/root/.cache/huggingface) should be mounted
# as a named Docker volume so the 3.44 GB weights survive image removal.
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


def log(msg):
    print(msg, flush=True)


def bench(tag, seconds):
    if tag == "peak_rss_mb":
        log(f"SEGMENT_BENCH\t{tag}\t{seconds:.0f} MB")
    else:
        log(f"SEGMENT_BENCH\t{tag}\t{seconds:.1f}s")


def process_wing_cell(model, processor, device, image_obj, points, labels):
    """Notebook's process_wing_image(): multi-point prompt -> binary mask ->
    largest OpenCV contour as an (N, 2) [x, y] array."""
    inputs = processor(
        images=image_obj,
        input_points=[[points]],
        input_labels=[[labels]],
        return_tensors="pt",
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
    ap = argparse.ArgumentParser(description="SAM3 wing-cell segmentation")
    ap.add_argument("image", help="path to the wing photo (PNG/JPEG)")
    ap.add_argument("--points", required=True, help="points.json with the prompts")
    ap.add_argument("--model", default="facebook/sam3",
                    help="HF model id (e.g. facebook/sam3 or facebook/sam3-fast)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: contours_coordinates_<stem>.csv)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--masks-dir", default=None,
                    help="optional directory to save the binary masks (PNG) for QC")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        from huggingface_hub import login
        login(token=token)
    else:
        log("Warning: HF_TOKEN is not set — this only works if the weights are "
            "already cached (gated model: https://huggingface.co/facebook/sam3).")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log(f"Using compute device: {device}")

    image_obj = Image.open(args.image).convert("RGB")

    with open(args.points) as fh:
        prompts = json.load(fh)

    # ---- load model (downloads on first run into the HF cache) --------------
    t0 = time.time()
    processor = Sam3TrackerProcessor.from_pretrained(args.model)
    model = Sam3TrackerModel.from_pretrained(args.model).to(device)
    model.eval()
    load_s = time.time() - t0
    log(f"Model loaded: {args.model} ({load_s:.1f}s)")
    bench("model_load", load_s)

    # ---- segment each cell (pa2r first, dm second — same order as Colab) ----
    contours_img = {}
    for cell in ("pa2r", "dm"):
        if cell not in prompts:
            raise RuntimeError(f"points.json is missing the '{cell}' prompts.")
        pts = prompts[cell]["points"]
        labels = prompts[cell]["labels"]
        t0 = time.time()
        contour, mask = process_wing_cell(model, processor, device,
                                          image_obj, pts, labels)
        infer_s = time.time() - t0
        bench(f"inference_{cell}", infer_s)
        log(f"{cell}: {len(contour)} boundary coordinates "
            f"({infer_s:.1f}s, prompt points: {pts})")
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

    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out = f"contours_coordinates_{stem}.csv"
    df.to_csv(out, index=False)
    log(f"Contour coordinates saved to {out} ({max_len} rows)")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    bench("peak_rss_mb", peak_mb)
    log(f"Peak RSS: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
