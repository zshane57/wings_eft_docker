#!/usr/bin/env python3
# --------------------------------------------------------------
# segment_wing_github.py — SAM3 wing-cell segmentation using the
# OFFICIAL GITHUB IMPLEMENTATION (facebookresearch/sam3) instead of
# the Hugging Face transformers port.
#
# The point-prompt API on images comes from the "instance
# interactivity" (SAM1/SAM2-style) predictor:
#   build_sam3_image_model(..., enable_inst_interactivity=True)
#   -> model.predict_inst(state, point_coords=..., point_labels=...)
# Negative points (labels 0) are supported — same 1-inclusive +
# 2-exclusive scheme as the Colab notebook.
#
# CLI contract is identical to segment_wing.py (HF transformers
# version) so both images are drop-in interchangeable:
#   python3 segment_wing_github.py <image> --points points.json \
#       [--out out.csv] [--device auto|cpu|cuda] [--masks-dir DIR]
#
# The checkpoint (sam3.pt) is downloaded from the SAME gated Hugging
# Face repo (facebook/sam3) via hf_hub_download — the HF_TOKEN
# environment variable and prior access approval are still required.
# --------------------------------------------------------------
# NOTE on --model: kept for CLI compatibility; the GitHub build
# downloads "sam3"/"sam3.1" checkpoints by version flag only.
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

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def log(msg):
    print(msg, flush=True)


def bench(tag, seconds):
    log(f"SEGMENT_BENCH\t{tag}\t{seconds:.1f}s")


def process_wing_cell(model, processor, state, points, labels):
    """Point-prompted SAM1-style prediction -> largest OpenCV contour (N, 2)."""
    masks, scores, _ = model.predict_inst(
        state,
        point_coords=np.array(points, dtype=np.float32),
        point_labels=np.array(labels, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    binary_mask = np.ascontiguousarray(masks[best]).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Segmentation failed: no contour found for this cell.")
    return max(contours, key=cv2.contourArea).squeeze(), binary_mask


def main():
    ap = argparse.ArgumentParser(
        description="SAM3 wing-cell segmentation (github facebookresearch/sam3)")
    ap.add_argument("image", help="path to the wing photo (PNG/JPEG)")
    ap.add_argument("--points", required=True, help="points.json with the prompts")
    ap.add_argument("--model", default="facebook/sam3",
                    help="kept for CLI parity (github build: sam3 | sam3.1)")
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
        log("Warning: HF_TOKEN is not set — this only works if the checkpoint is "
            "already cached (gated model: https://huggingface.co/facebook/sam3).")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log(f"Using compute device: {device}")

    image_obj = Image.open(args.image).convert("RGB")

    with open(args.points) as fh:
        prompts = json.load(fh)

    # ---- build model (downloads sam3.pt on first run into the HF cache) ------
    t0 = time.time()
    model = build_sam3_image_model(
        device=device,
        enable_inst_interactivity=True,
    )
    model.eval()
    processor = Sam3Processor(model, device=device)
    load_s = time.time() - t0
    log(f"Model loaded: sam3.pt via build_sam3_image_model ({load_s:.1f}s)")
    bench("model_load", load_s)

    # ---- image -> backbone features (done once, shared by both cells) ---------
    t0 = time.time()
    state = processor.set_image(image_obj)
    encode_s = time.time() - t0
    bench("image_encode", encode_s)
    log(f"Image encoded ({encode_s:.1f}s)")

    # ---- segment each cell (pa2r first, dm second — same order as Colab) ----
    contours_img = {}
    for cell in ("pa2r", "dm"):
        if cell not in prompts:
            raise RuntimeError(f"points.json is missing the '{cell}' prompts.")
        pts = prompts[cell]["points"]
        labels = prompts[cell]["labels"]
        t0 = time.time()
        contour, mask = process_wing_cell(model, processor, state, pts, labels)
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
