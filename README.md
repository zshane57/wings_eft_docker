# Fly Wing Species Identification — Docker Workflow (full pipeline)

Self-contained deployment of the complete pipeline — segmentation **and**
classification — in two chained Docker containers, driven by a browser
page. For a terminal-only workflow, use the sibling `docker_workflow/`
package (same images, command-line launchers) instead:

```
wing photo  →  orientation alignment  →  YOLOv8 boxes  →  SAM3 segmentation
            →  contour CSV
            →  EFD (10 harmonics, Momocs)  →  scale/rotate/phase normalisation
            →  traversal-direction canonicalisation  →  fold-1 LDA projection (24 features)
            →  full GUIDE data file  →  GUIDE random forest  →  species + probability
```

Have contours already (from the Colab notebook `flieswings_proj.ipynb`)?
Skip the segmentation stage: classify the CSV directly (GUI mode 2).

## Before you start

1. **Install Docker** — https://www.docker.com/products/docker-desktop/
   (or Docker Engine on Linux).
2. **For the segmentation stage only:** a **Hugging Face token**.
   SAM3's weights are hosted on a gated repository:
   - request access at **https://huggingface.co/facebook/sam3**
     (a short form; approval is usually quick),
   - create a read token at **https://huggingface.co/settings/tokens**.

   The token is passed to the container via an environment variable only —
   it is never written to disk by this workflow. The first segmentation run
   downloads **~3.4 GB of weights once** into the `flywing-sam3-cache` Docker
   volume; after that, later runs work even without the token.
3. **For classification only:** export the contour CSV from the Google Colab
   notebook **`flieswings_proj.ipynb`** <!-- ADD NOTEBOOK LINK HERE --> and
   place it in this directory (columns `dm_x, dm_y, pa2r_x, pa2r_y`).

## Web frontend

`start_web.sh` (macOS/Linux) or `start_web.ps1` (Windows) builds a small
Flask container (~325 MB) and opens the workflow in your browser:

    ./start_web.sh        # or:  powershell -ExecutionPolicy Bypass -File .\start_web.ps1
    → http://localhost:8080

The page offers two modes:

- **Segment a wing photo, then identify it** — upload the photo and press
  **Run**. The image is EXIF-corrected and put into the upright-landscape
  training orientation; YOLOv8 then detects `pa2r` and `dm` at **imgsz=1024,
  conf=0.5**. The highest-confidence box per class becomes a native SAM3
  bounding-box prompt. No clicks are needed. Optionally paste the HF token
  (needed only for the first SAM3 weights download) and pick the species.
  If either cell is missing, the image is skipped cleanly, a warning names
  the photo and missing cell, and the browser opens the original six-point
  **manual prompt fallback**. The result card includes QC overlays (yellow =
  SAM3 boundary, cyan = YOLOv8 box) and detection-JSON download links.
- **Classify an existing contour CSV** — upload a `contours_coordinates*.csv`
  and run.

Both modes add an **EFD coefficients + LDA features CSV** download to the
result card. `efd_lda_features_<sample>.csv` contains one row with the sample
metadata, all 80 final normalised EFD coefficients (40 per cell, including
`a1`, `b1`, and `c1`), and the 24 centred fold-1 LDA features. Coefficient
columns are prefixed `dm_` or `pa2r_`; LD columns retain the `dm-LD*` and
`pa2r-LD*` names used by GUIDE. These are the exact values used for the new
sample's projection, not a reanalysis of the training set.

Automatic runs write `masks_<photo>/` (the segmented cells), the aligned
photo `_aligned_<photo-stem>.png`, `_detections_<photo-stem>.json` (complete
or partial boxes and confidences), and the QC overlays. Manual-fallback runs
save `_prompts_<photo-stem>.json` instead. For deeper checks run `qc_masks.py`
(below); if an automatic overlay looks wrong, inspect the detection box or
retry with manual points.

The server drives the same Docker images as the other frontends, binds to
**localhost only**, passes the token to the container as an environment
variable (never writes it to disk), and removes the workflow containers
and images after each run. It keeps running until you stop it:

    docker rm -f flywing-web

Requires Docker Desktop or any Docker Engine with the standard
`/var/run/docker.sock` (all desktop installs have it).

### Automatic detection files

Every automatic attempt writes `_detections_<photo-stem>.json`, even when a
cell is missing. Boxes are pixel `[x_min, y_min, x_max, y_max]` coordinates in
the aligned image:

```json
{
  "image": "B.karnyi.9_cleanup.png",
  "aligned_image": "_aligned_B.karnyi.9_cleanup.png",
  "predict": {"imgsz": 1024, "conf": 0.5},
  "boxes": {
    "pa2r": {"class_id": 0, "mask_target": "masks5",
             "xyxy": [1200.0, 220.0, 1660.0, 390.0], "confidence": 0.91},
    "dm":   {"class_id": 1, "mask_target": "masks9",
             "xyxy": [900.0, 350.0, 1210.0, 650.0], "confidence": 0.88}
  },
  "complete": true
}
```

Class **0 = pa2r / masks5** and class **1 = dm / masks9**. If a class produces
multiple boxes, only the highest-confidence one is retained. YOLO and SAM3
receive the same EXIF-corrected, RGB, upright-landscape image, so the saved
coordinates and masks share one coordinate frame. Portrait images are rotated
90° counter-clockwise; training already includes left/right flips.

### Manual fallback prompt-points files

Every manual-fallback run saves the clicked points next to the photo as
`_prompts_<photo-stem>.json`:

```json
{ "pa2r": {"points": [[1438, 301], [1031, 416], [1032, 584]], "labels": [1, 0, 0]},
  "dm":   {"points": [[1031, 416], [1438, 301], [1032, 584]], "labels": [1, 0, 0]} }
```

Points are `[x, y]` pixel coordinates in the **original** photo: the first
is inside the cell (label 1), the next two outside it (label 0). The bundled
`prompts_bkarnyi9.json` matches `B.karnyi.9_cleanup.png` and doubles as a
template.

For a terminal-only (browser-free) workflow, use the sibling
**`docker_workflow/`** package instead — same Docker images, command-line
launchers.

## Platform notes

- The **classification image is x86_64 only** (the GUIDE binary). On Apple
  Silicon it runs through Rosetta emulation — enable *Settings → General →
  Use Rosetta for x86_64/amd64 emulation* in Docker Desktop for best speed.
- The **segmentation image runs natively** on both amd64 and arm64.
- The segmentation image contains CPU-only PyTorch, `ultralytics`, SAM3
  support, and the bundled **6.2 MB YOLOv8 weights**. SAM3 weights are
  **3.4 GB** (one-time download, kept in the Docker volume). The measured
  classification image is **1.9 GB**; segmentation-image size depends on the
  resolved Python dependency versions.
- A CPU-only segmentation takes minutes per wing cell (comparable to the
  Colab notebook's CPU timing). A CUDA GPU accelerates this dramatically if
  you have one — the container detects it automatically.
- Rough RAM need: ≈ 5 GB for the segmentation stage, ≈ 1 GB for
  classification. On Macs, raise the Docker Desktop VM memory limit
  accordingly (*Settings → Resources*).

## Files

| File | Purpose |
|---|---|
| `webserver.py` + `templates/` + `Dockerfile.web` | Browser frontend (Flask + Docker SDK CLI in a container; served at localhost:8080) |
| `start_web.sh` / `start_web.ps1` | Launchers for the browser frontend (build the web image, start it, open the page) |
| `gui_logic.py` | Shared logic for the web server (output parsing, coordinate mapping, points-JSON schema) |
| `test_gui_logic.py` | Headless web/pipeline unit tests — `python3 test_gui_logic.py` |
| `segment_wing.py` | Automatic YOLOv8-box + SAM3 segmentation entrypoint; manual-point fallback |
| `wing_detection.py` | Image alignment, one-time YOLO loading, best-box selection, clean incomplete-detection guard |
| `models/yolo_wings_best.pt` | Trained YOLOv8n weights (1024 px; class 0 pa2r/masks5, class 1 dm/masks9) |
| `test_wing_detection.py` | Dependency-free YOLO contract/fallback tests — `python3 test_wing_detection.py` |
| `Dockerfile.segment` | Segmentation image (python:3.11-slim, CPU-only PyTorch + ultralytics) |
| `prompts_bkarnyi9.json` | Example prompt points for `B.karnyi.9_cleanup.png` |
| `example_clicks.png` | Static fallback click-placement illustration for the web frontend (used until a first run produces `qc_overlay_*.png`) |
| `run_segment_benchmark.sh` | One-shot benchmark/validation run (hidden token prompt, writes the CSV + masks + log) |
| `qc_masks.py` | Segmentation quality control — structure, YOLO boxes or prompt-point containment, IoU vs a reference, and QC overlays. Run inside the segmentation container:
  `docker run --rm -v "$PWD:/work" -w /work --entrypoint python3 flywing-segment qc_masks.py <photo> [--masks-dir DIR] [--detections FILE] [--prompts FILE] [--known reference.csv]` |
| `B.karnyi.9_cleanup.png` | Test wing photo (for end-to-end validation) |
| `segment_wing_github.py` / `Dockerfile.segment_github` | Alternative segmentation image using the **official GitHub SAM3** (see below) |
| `Dockerfile.workflow` | Classification image (ubuntu:22.04, R, Momocs 1.5.0, GUIDE) |
| `run_workflow.sh` | Classification entrypoint — orchestrates EFD → LDA → GUIDE |
| `process_features.R`, `normalisation_scale_rotate_phase.R` | Feature extraction & normalisation |
| `lda_loadings_fold_1_{dm,pa2r}.txt`, `mean_ld_train_fd_1_{dm,pa2r}.rds` | Fold-1 LDA model pieces |
| `dm_testing.csv`, `pa2r_testing.csv` | Fold-1 testing set (130 samples) |
| `guide_input/…` | GUIDE base data, command file, variable description |
| `contours_coordinates_bkarnyi.csv` | Bundled classification sample; any additional `contours_coordinates*.csv` files are also baked into the classification image at build time |

## Alternative: official GitHub SAM3 implementation

The default segmentation image uses the **Hugging Face transformers port**
of SAM3 — the exact code path of the Colab notebook. `Dockerfile.segment_github`
builds a manual-point alternative from **https://github.com/facebookresearch/sam3**
(point prompts via its SAM1/SAM2-style interactive predictor). The automatic
YOLOv8 web pipeline uses the default `Dockerfile.segment` image:

- same gated checkpoint (`facebook/sam3`) — **the same HF token works**;
- bigger image (≈ 2.3 GB) and needs a one-line CPU patch applied at build
  time (documented inside the Dockerfile);
- useful as a fallback or for cross-checking masks against the default.

```bash
curl -sL https://github.com/facebookresearch/sam3/archive/refs/heads/main.tar.gz -o sam3-main.tar.gz
docker build -f Dockerfile.segment_github -t flywing-segment-github .
docker run --rm -v "$PWD:/work" -w /work -v flywing-sam3-cache:/root/.cache/huggingface \
  -e HF_TOKEN=… flywing-segment-github photo.png --points prompts.json
```

## Manual build & run

Classification (GUIDE is x86_64-only — use the amd64 platform on Apple Silicon):

```bash
docker build --platform linux/amd64 -f Dockerfile.workflow -t flywing-workflow .
docker run --rm --platform linux/amd64 -v "$PWD:/work" \
  -e FEATURE_CSV=/work/efd_lda_features_bkarnyi.csv \
  flywing-workflow contours_coordinates_bkarnyi.csv B.karnyi
# The combined EFD/LDA export is now ./efd_lda_features_bkarnyi.csv
```

Segmentation (native architecture, weights volume, first run downloads):

```bash
docker build -f Dockerfile.segment -t flywing-segment .
docker run --rm -v "$PWD:/work" -w /work \
  -v flywing-sam3-cache:/root/.cache/huggingface -e HF_TOKEN=… \
  flywing-segment B.karnyi.9_cleanup.png \
  --out contours_coordinates_B.karnyi.9.csv --masks-dir masks_bkarnyi9
```

If YOLOv8 cannot find both cells, inspect `_detections_*.json` and rerun that
photo with manual points:

```bash
docker run --rm -v "$PWD:/work" -w /work \
  -v flywing-sam3-cache:/root/.cache/huggingface -e HF_TOKEN=… \
  flywing-segment B.karnyi.9_cleanup.png --prompt-mode manual \
  --points prompts_bkarnyi9.json --out contours_coordinates_B.karnyi.9.csv \
  --masks-dir masks_bkarnyi9
```

## Clean up

Containers and workflow images are removed automatically after each run. The web
server (`flywing-web`) keeps running until you stop it with
`docker rm -f flywing-web` (its image can then be removed with
`docker rmi flywing-web`). What stays:

- the **build cache** (fast rebuilds) — free with `docker builder prune`;
- the **weights volume** `flywing-sam3-cache` (avoid re-downloading 3.4 GB) —
  free with `docker volume rm flywing-sam3-cache`;
- your samples and the small QC/feature artifacts (`contours_coordinates*.csv`,
  `efd_lda_features_*.csv`, `masks_*/`, `_detections_*.json`,
  `_aligned_*.png`, `_prompts_*.json`).
