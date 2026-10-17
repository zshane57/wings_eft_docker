# Fly Wing Species Identification — Docker Workflow (full pipeline)

Self-contained deployment of the complete pipeline — segmentation **and**
classification — in two chained Docker containers, driven by a browser
page. For a terminal-only workflow, use the sibling `docker_workflow/`
package (same images, command-line launchers) instead:

```
wing photo  →  SAM3 segmentation (point prompts)  →  contour CSV
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

## Web frontend (browser — the only frontend, no Python on the host needed)

`start_web.sh` (macOS/Linux) or `start_web.ps1` (Windows) builds a small
Flask container (~325 MB) and opens the workflow in your browser:

    ./start_web.sh        # or:  powershell -ExecutionPolicy Bypass -File .\start_web.ps1
    → http://localhost:8080

The page offers two modes:

- **Segment a wing photo, then identify it** — upload the photo, click the
  6 prompt points directly on the image (green inside + 2 red outside,
  per cell, with undo/reset), optionally paste the HF token (needed only
  for the first weights download), pick the species, and press **Run**.
  The log streams live; when the run finishes the result card shows the
  prediction together with the **QC overlay images** (yellow = SAM3
  boundary) and download links for the contour CSV.
- **Classify an existing contour CSV** — upload a `contours_coordinates*.csv`
  and run.

Segmentation runs also write `masks_<photo>/` (the segmented cells), save the
clicked points as `_prompts_<photo-stem>.json` (schema below), and produce
the QC overlays. For deeper checks run `qc_masks.py` (below); if an overlay
looks wrong, redo the run with better-placed points.

The server drives the same Docker images as the other frontends, binds to
**localhost only**, passes the token to the container as an environment
variable (never writes it to disk), and removes the workflow containers
and images after each run. It keeps running until you stop it:

    docker rm -f flywing-web

Requires Docker Desktop or any Docker Engine with the standard
`/var/run/docker.sock` (all desktop installs have it).

### Prompt-points files

Every segmentation run saves the clicked points next to the photo as
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
- Measured sizes: classification image **1.9 GB**; segmentation image
  **1.36 GB** (builds in about a minute thanks to layer caching); SAM3
  weights **3.4 GB** (one-time download, kept in the Docker volume).
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
| `test_gui_logic.py` | Headless unit tests — `python3 test_gui_logic.py` (36 checks) |
| `segment_wing.py` | SAM3 segmentation entrypoint (Hugging Face transformers port) |
| `Dockerfile.segment` | Segmentation image (python:3.11-slim, CPU-only PyTorch) |
| `prompts_bkarnyi9.json` | Example prompt points for `B.karnyi.9_cleanup.png` |
| `example_clicks.png` | Fallback click-placement example for the web frontend (used until a first run produces `qc_overlay_*.png`; generated by `gen_example.py`) |
| `run_segment_benchmark.sh` | One-shot benchmark/validation run (hidden token prompt, writes the CSV + masks + log) |
| `qc_masks.py` | Segmentation quality control — structure, prompt-point containment, IoU vs a reference, and QC overlay images. Run inside the segmentation container:
  `docker run --rm -v "$PWD:/work" -w /work --entrypoint python3 flywing-segment qc_masks.py <photo> [--masks-dir DIR] [--prompts FILE] [--known reference.csv]` |
| `B.karnyi.9_cleanup.png` | Test wing photo (for end-to-end validation) |
| `segment_wing_github.py` / `Dockerfile.segment_github` | Alternative segmentation image using the **official GitHub SAM3** (see below) |
| `Dockerfile.workflow` | Classification image (ubuntu:22.04, R, Momocs 1.5.0, GUIDE) |
| `run_workflow.sh` | Classification entrypoint — orchestrates EFD → LDA → GUIDE |
| `process_features.R`, `normalisation_scale_rotate_phase.R` | Feature extraction & normalisation |
| `lda_loadings_fold_1_{dm,pa2r}.txt`, `mean_ld_train_fd_1_{dm,pa2r}.rds` | Fold-1 LDA model pieces |
| `dm_testing.csv`, `pa2r_testing.csv` | Fold-1 testing set (130 samples) |
| `guide_input/…` | GUIDE base data, command file, variable description |
| `contours_coordinates*.csv` | Contour samples — ALL matching files are baked into the classification image at build time |

## Alternative: official GitHub SAM3 implementation

The default segmentation image uses the **Hugging Face transformers port**
of SAM3 — the exact code path of the Colab notebook. `Dockerfile.segment_github`
builds a drop-in alternative from **https://github.com/facebookresearch/sam3**
(point prompts via its SAM1/SAM2-style interactive predictor):

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
docker run --rm --platform linux/amd64 flywing-workflow contours_coordinates_bkarnyi.csv B.karnyi
```

Segmentation (native architecture, weights volume, first run downloads):

```bash
docker build -f Dockerfile.segment -t flywing-segment .
docker run --rm -v "$PWD:/work" -w /work \
  -v flywing-sam3-cache:/root/.cache/huggingface -e HF_TOKEN=… \
  flywing-segment B.karnyi.9_cleanup.png --points prompts_bkarnyi9.json \
  --out contours_coordinates_B.karnyi.9.csv --masks-dir masks_bkarnyi9
```

## Clean up

Containers and workflow images are removed automatically after each run. The web
server (`flywing-web`) keeps running until you stop it with
`docker rm -f flywing-web` (its image can then be removed with
`docker rmi flywing-web`). What stays:

- the **build cache** (fast rebuilds) — free with `docker builder prune`;
- the **weights volume** `flywing-sam3-cache` (avoid re-downloading 3.4 GB) —
  free with `docker volume rm flywing-sam3-cache`;
- your samples and the small QC artifacts (`contours_coordinates*.csv`,
  `masks_*/`, `_prompts_*.json`).
