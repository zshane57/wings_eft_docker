# Fly Wing Species Classification — Docker Workflow

Self-contained deployment of the full pipeline (flywingproj.md Tasks 2–5):

```
contours CSV  →  EFD (10 harmonics, Momocs)  →  scale/rotate/phase normalisation
              →  traversal-direction canonicalisation  →  fold-1 LDA projection (24 features)
              →  full GUIDE data file (train on top, test below, new sample last, weight 0)
              →  GUIDE random forest  →  predicted species + probability
```

## Before you start

1. **Install Docker** on your machine in advance — this workflow runs
   everything inside a Docker container
   (e.g. Docker Desktop: https://www.docker.com/products/docker-desktop/).
2. **Extract the wing contour coordinates** of your sample using the Google
   Colab notebook **`flieswings_proj.ipynb`**: <!-- ADD NOTEBOOK LINK HERE -->
   the notebook exports a `contours_coordinates*.csv` file (columns:
   `dm_x`, `dm_y`, `pa2r_x`, `pa2r_y`). Place the exported CSV into this
   directory before running the workflow.

## Opening the workflow on your platform

**Windows (x86_64)**
1. Open **PowerShell**.
2. Navigate to this folder and start the launcher:

       cd path\to\docker_workflow
       powershell -ExecutionPolicy Bypass -File .\identify_wing.ps1

   (The `-ExecutionPolicy Bypass` switch is only needed if Windows blocks
   script execution; if execution is already allowed, `.\identify_wing.ps1`
   alone works.) No emulation is needed — the container runs natively on
   x86_64 Windows. The bash launcher `identify_wing.sh` remains available
   for WSL / Git Bash users.

**macOS (Apple Silicon and Intel)**
1. Open **Terminal**.
2. Navigate to this folder and start the launcher:

       cd path/to/docker_workflow
       ./identify_wing.sh

   Apple Silicon is detected automatically and the x86_64 image runs through
   emulation (enable *Rosetta* in Docker Desktop for best speed); on Intel
   Macs everything runs natively.

**Linux (x86_64)**
1. Install **Docker Engine** (via your distribution's package manager or
   https://docs.docker.com/engine/install/) and add your user to the docker
   group so docker works without sudo (log out and back in afterwards):

       sudo usermod -aG docker $USER

2. Open a terminal, navigate to this folder and start the launcher — the
   same script as on macOS, no changes needed:

       cd path/to/docker_workflow
       ./identify_wing.sh

   x86_64 Linux is the most native environment of all: the container runs
   directly on the host kernel with no VM and no emulation.

## Files

| File | Purpose |
|---|---|
| `identify_wing.sh` / `identify_wing.ps1` | Interactive launchers (bash for macOS/Linux, PowerShell for Windows) — prompt for input, build, run, then remove the container + image |
| `Dockerfile.workflow` | Image definition (ubuntu:22.04, R, Momocs 1.5.0, GUIDE Ubuntu22 binary) |
| `run_workflow.sh` | Entrypoint — orchestrates Tasks 2–5 and reports the prediction |
| `process_features.R` | EFD → normalisation → LDA → full GUIDE data file |
| `normalisation_scale_rotate_phase.R` | Custom scale/rotate/phase normalisation (sourced by the script) |
| `lda_loadings_fold_1_{dm,pa2r}.txt` | Fold-1 LDA rotation matrices (37 features → 12 LDs per cell) |
| `mean_ld_train_fd_1_{dm,pa2r}.rds` | Fold-1 training-set LDA means (for centering) |
| `dm_testing.csv`, `pa2r_testing.csv` | Fold-1 testing set (130 samples, re-projected at runtime) |
| `guide_input/combined_fd1_dmpa2r_data.txt` | Base GUIDE data file — supplies the 491 training rows (weight = 1) |
| `guide_input/fold_1_guide_data_rf.in` | GUIDE command file (random forest) |
| `guide_input/fold_1_guide_data.dsc` | GUIDE variable description (28 variables, `species` = class, `weight` = w) |
| `contours_coordinates*.csv` | Contour samples (dm_x, dm_y, pa2r_x, pa2r_y) — ALL files matching this pattern in the directory are baked into the image at build time; add new samples freely and rebuild |

## Interactive one-shot run (recommended for non-Docker users)

`identify_wing.sh` wraps the whole cycle: it prompts for the contour CSV and
the species, builds the image, runs the classification, and afterwards
**removes the container and the image** again — nothing stays on the machine
except the (reusable) build cache:

```bash
./identify_wing.sh
```

- Contour file: type the file name (files in this directory are listed) or a
  full path — files from elsewhere are copied in and renamed automatically if
  they do not follow the `contours_coordinates*.csv` convention.
- Species: press ENTER for an unknown sample (top-3 report), or type one of the
  training species to validate a known sample.

## Manual build & run

The GUIDE binary is x86_64 only — build/run with the amd64 platform on Apple Silicon
(enable *Settings → General → Use Rosetta for x86_64/amd64 emulation* for best speed):

```bash
docker build --platform linux/amd64 -f Dockerfile.workflow -t flywing-workflow .

# known sample - validates the prediction against the given true_species
docker run --rm --platform linux/amd64 flywing-workflow contours_coordinates_cvicina.csv C.vicina
docker run --rm --platform linux/amd64 flywing-workflow contours_coordinates_bkarnyi.csv B.karnyi

# unknown sample - omit true_species input; the top-3 most probable species are reported
docker run --rm --platform linux/amd64 flywing-workflow contours_coordinates.csv
```

Notes:
- `run_workflow.sh` is the image ENTRYPOINT — pass only the arguments, not the script path.
- Known sample: 2nd argument = the validation species → MATCH/MISMATCH report.
- Unknown sample: no 2nd argument → the row carries the default placeholder label
  `C.vicina` (GUIDE requires a training-level class label on every row); it is
  excluded from training (weight = 0) and does not affect the prediction —
  the workflow instead reports the top-3 species with the highest probabilities.
- Each container run works on its own copy of the data — no host files are modified.
- To classify a new sample: name its CSV `contours_coordinates*.csv`, put it in this directory, rebuild the image, then run
  `docker run ... flywing-workflow <csv> [<species>]`.

## Clean up

The image is ~1.9 GB. To remove it after running the workflow (so it does not
sit on the machine unused):

```bash
docker rmi flywing-workflow
```

The build cache is kept, so a later rebuild is still fast. To free that space as well:

```bash
docker builder prune
```
