#!/usr/bin/env python3
# --------------------------------------------------------------
# qc_masks.py — objective + visual quality control for SAM3
# segmentation runs. Runs INSIDE the segmentation container:
#
#   docker run --rm -v "$PWD:/work" -w /work \
#       --entrypoint python3 flywing-segment qc_masks.py <photo> [options]
#
# Per cell (pa2r, dm):
#   1. mask structure - number of components, solidity
#   2. prompt points  - inclusive point inside the mask, the two
#                       exclusive points outside it (needs --prompts)
#   3. IoU            - overlap with a reference contour CSV, if one
#                       exists for the sample (--known; only the
#                       bundled benchmark photo has a gold standard)
# Writes qc_overlay_<cell>.png: the photo with
#   yellow  = SAM3-produced boundary
#   magenta = reference boundary (only with --known)
#   green/red dots = prompt points
# --------------------------------------------------------------
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser(description="SAM3 mask quality control")
ap.add_argument("photo", help="the segmented wing photo")
ap.add_argument("--masks-dir", default=None,
                help="directory with mask_pa2r.png / mask_dm.png "
                     "(default: masks_<photo stem>)")
ap.add_argument("--prompts", default=None,
                help="points JSON (default: _prompts_<stem>.json if present)")
ap.add_argument("--known", default=None,
                help="reference contour CSV for the IoU check (optional)")
args = ap.parse_args()

photo_path = args.photo
stem = os.path.splitext(os.path.basename(photo_path))[0]
masks_dir = args.masks_dir or f"masks_{stem}"
prompts_path = args.prompts or f"_prompts_{stem}.json"
known_path = args.known

photo = cv2.imread(photo_path)
if photo is None:
    raise SystemExit(f"Could not read the photo: {photo_path}")
H, W = photo.shape[:2]

prompts = None
if os.path.isfile(prompts_path):
    prompts = json.load(open(prompts_path))
else:
    print(f"(no prompts file '{prompts_path}' - skipping point checks)")

known = None
if known_path and os.path.isfile(known_path):
    import pandas as pd
    known = pd.read_csv(known_path)
else:
    print("(no reference CSV - skipping the IoU check)")

print(f"photo: {W}x{H}; masks: {masks_dir}/")
for cell in ("pa2r", "dm"):
    mask_path = os.path.join(masks_dir, f"mask_{cell}.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"Could not read the mask: {mask_path}")
    mask = (mask > 127).astype(np.uint8)

    # ---- 1. structure ------------------------------------------------
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = n_comp - 1
    biggest = int(stats[1:, cv2.CC_STAT_AREA].max()) if comps else 0
    fill = biggest / mask.sum() if mask.sum() else 0
    x, y, w, h = stats[1, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1e-9)

    verdict = []
    if fill > 0.99 and biggest > 1000:
        verdict.append("single blob OK")
    else:
        verdict.append(f"WARNING: largest blob only {100*fill:.1f}% of mask")
    if solidity < 0.5:
        verdict.append(f"WARNING: low solidity {solidity:.3f} (ragged mask?)")

    # ---- 2. prompt-point containment ---------------------------------
    inside = []
    if prompts and cell in prompts:
        pts = prompts[cell]["points"]
        inside = [bool(mask[int(py), int(px)]) for px, py in pts]
        ok = inside and inside[0] and not any(inside[1:])
        verdict.append("prompt points OK" if ok
                       else "WARNING: prompt points misplaced")

    # ---- 3. IoU vs reference --------------------------------------------
    iou = None
    poly_img = None
    if known is not None:
        kx = known[f"{cell}_x"].to_numpy()
        ky = -known[f"{cell}_y"].to_numpy()          # undo the notebook y-flip
        poly = np.stack([kx[~np.isnan(kx)], ky[~np.isnan(kx)]],
                        axis=1).astype(np.int32).reshape(-1, 1, 2)
        ref_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(ref_mask, [poly], 1)
        inter = np.logical_and(mask, ref_mask).sum()
        union = np.logical_or(mask, ref_mask).sum()
        iou = inter / union if union else 0.0
        if iou >= 0.90:
            verdict.append(f"IoU {iou:.3f} - excellent")
        elif iou >= 0.75:
            verdict.append(f"IoU {iou:.3f} - acceptable")
        else:
            verdict.append(f"IoU {iou:.3f} - INVESTIGATE")

    # ---- overlay ----------------------------------------------------------
    ov = photo.copy()
    cv2.drawContours(ov, c, -1, (0, 255, 255), 3)              # produced: yellow
    if known is not None:
        cv2.polylines(ov, [poly], True, (255, 0, 255), 3)      # reference: magenta
    if prompts and cell in prompts:
        for (px, py), lab in zip(prompts[cell]["points"],
                                 prompts[cell]["labels"]):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(ov, (int(px), int(py)), 9, color, 3)
            cv2.circle(ov, (int(px), int(py)), 2, color, -1)
    out = f"qc_overlay_{cell}.png"
    cv2.imwrite(out, ov)

    print(f"\n--- {cell} ---")
    print(f"  components   : {comps} (largest {biggest} px, {100*fill:.1f}% of mask)")
    print(f"  bounding box : x={x} y={y} w={w} h={h}")
    print(f"  solidity     : {solidity:.3f}")
    if inside:
        print(f"  points inside: {inside}")
    if iou is not None:
        print(f"  IoU          : {iou:.3f}")
    print(f"  {'; '.join(verdict)}")
    print(f"  overlay      : {out}")

print("\nOpen the qc_overlay_*.png files and compare with the photo:"
      " the yellow line should trace the cell cleanly.")
