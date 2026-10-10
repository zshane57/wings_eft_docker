#!/usr/bin/env python3
# --------------------------------------------------------------
# gen_example.py — one-off generator for the click-example image
# shown in the web frontend (example_clicks.png).
#
# Crops the B.karnyi.9 test photo to the two wing cells, draws the
# SAM3 boundaries, the cell names and the prompt-point example
# (green inside + two red outside). Run inside the segmentation
# container:
#   docker run --rm -v "$PWD:/work" -w /work \
#       --entrypoint python3 flywing-segment gen_example.py
# --------------------------------------------------------------
import json

import cv2

PHOTO = "B.karnyi.9_cleanup.png"
MASKS = "masks_bkarnyi9"                  # from the benchmark run
PROMPTS = "prompts_bkarnyi9.json"
OUT = "example_clicks.png"

# crop window (both cells + margin): union of the two bounding boxes
X0, Y0, X1, Y1 = 603, 162, 1920, 604

photo = cv2.imread(PHOTO)
crop = photo[Y0:Y1, X0:X1].copy()

prompts = json.load(open(PROMPTS))
for cell in ("pa2r", "dm"):
    m = (cv2.imread(f"{MASKS}/mask_{cell}.png",
                    cv2.IMREAD_GRAYSCALE) > 127).astype("uint8")
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea) - [X0, Y0]      # shift into crop
    cv2.drawContours(crop, [c], -1, (0, 255, 255), 4)          # yellow

    # cell label
    cx = int(c[:, :, 0].mean()) ; cy = int(c[:, :, 1].mean())
    cv2.putText(crop, cell, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(crop, cell, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # prompt points: inclusive green, exclusive red
    for (px, py), lab in zip(prompts[cell]["points"], prompts[cell]["labels"]):
        px, py = px - X0, py - Y0
        color = (0, 200, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(crop, (px, py), 11, color, -1)
        cv2.circle(crop, (px, py), 11, (255, 255, 255), 2)

cv2.imwrite(OUT, crop)
print(f"wrote {OUT}: {crop.shape[1]}x{crop.shape[0]}")
