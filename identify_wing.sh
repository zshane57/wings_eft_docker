#!/bin/bash
# --------------------------------------------------------------
# identify_wing.sh — interactive one-shot launcher
#
# Two modes:
#   1) Segment a wing photo with SAM3, then identify it
#        - needs: the wing photo, a Hugging Face token for the
#          gated SAM3 model (first download only - the weights are
#          then kept in the 'flywing-sam3-cache' Docker volume),
#          and a prompt-points JSON file (see below)
#   2) Classify an existing contour CSV (the classic behaviour)
#
# The prompt-points JSON describes where to click: 1 point inside
# each wing cell + 2 points outside it (to exclude neighbours):
#   { "pa2r": {"points": [[x, y], [x, y], [x, y]], "labels": [1, 0, 0]},
#     "dm":   {"points": [[x, y], [x, y], [x, y]], "labels": [1, 0, 0]} }
# The bundled prompts_bkarnyi9.json matches B.karnyi.9_cleanup.png
# and can serve as a template for your own photos. The GUI wizard
# (identify_gui.py) can generate this file by clicking on the photo.
#
# Usage:
#   ./identify_wing.sh        (then answer the prompts)
#
# Nothing else is required: files named contours_coordinates*.csv in
# this directory are baked into the image at build time.
# --------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE="flywing-workflow"
CONTAINER="flywing-session"
SEG_IMAGE="flywing-segment"
SEG_CONTAINER="flywing-segment-session"
CACHE_VOLUME="flywing-sam3-cache"

VALID_SPECIES="C.vicina Ch.albiceps Ch.bezziana Ch.megacephala Ch.nigripes \
Ch.rufifacies L.sericata A.gressitti B.karnyi Le.alba S.princeps Sy.nudiseta Z.aquila"

echo "================================================================"
echo " Forensic fly wing species identification"
echo "================================================================"

# ---- 0. sanity checks --------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker was not found on this machine."; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running. Start Docker Desktop first, then re-run."; exit 1
fi

# The GUIDE binary is x86_64 only - emulate on Apple Silicon
# (the SAM3 segmentation image runs on the host's native architecture)
ARCH="$(docker version --format '{{.Server.Arch}}' 2>/dev/null || true)"
PLATFORM_FLAGS=()
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  PLATFORM_FLAGS=(--platform linux/amd64)
  echo "Note: Apple Silicon detected - the x86_64 classification image"
  echo "      runs through emulation; SAM3 runs natively."
fi

# ---- cleanup: remove containers + images when the script ends -----------------
cleanup() {
  echo
  echo "=== Cleaning up ==="
  docker rm -f "$CONTAINER" "$SEG_CONTAINER" >/dev/null 2>&1 || true
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker rmi "$IMAGE" >/dev/null 2>&1 || true
    echo "Classification container and image '$IMAGE' removed."
  fi
  if docker image inspect "$SEG_IMAGE" >/dev/null 2>&1; then
    docker rmi "$SEG_IMAGE" >/dev/null 2>&1 || true
    echo "Segmentation container and image '$SEG_IMAGE' removed."
  fi
  echo "The build cache is kept, so the next start rebuilds quickly."
  echo "The SAM3 weights volume '$CACHE_VOLUME' is kept"
  echo "  (delete with: docker volume rm $CACHE_VOLUME)."
  echo "To also free the build cache: docker builder prune"
}
trap cleanup EXIT

# ---- 1. choose the mode --------------------------------------------------------
echo
echo " 1) Segment a wing photo with SAM3, then identify it"
echo " 2) Classify an existing contour CSV"
while true; do
  if ! read -r -p "Mode [1/2]: " MODE; then
    echo; echo "No input received - aborting."; exit 1
  fi
  [ "$MODE" = "1" ] || [ "$MODE" = "2" ] && break
  echo "  Please type 1 or 2."
done

BASE=""
if [ "$MODE" = "1" ]; then
  # ---- 1a. the wing photo -------------------------------------------------------
  echo
  while true; do
    if ! read -r -p "Wing photo (file name or full path): " PHOTO; then
      echo; echo "No input received - aborting."; exit 1
    fi
    [ -z "$PHOTO" ] && { echo "  Please enter a file name."; continue; }
    PHOTO="${PHOTO/#\~/$HOME}"
    if [ ! -f "$PHOTO" ]; then echo "  File not found: $PHOTO"; continue; fi
    PHOTO_BASE="$(basename "$PHOTO")"
    if [ "$(cd "$(dirname "$PHOTO")" && pwd)" != "$SCRIPT_DIR" ]; then
      cp "$PHOTO" "$PHOTO_BASE"
      echo "  Copied into this directory: $PHOTO_BASE"
    fi
    break
  done

  # ---- 1b. Hugging Face token (hidden; ENTER = use cached weights) --------------
  echo
  echo "SAM3 weights are hosted on a gated Hugging Face repo:"
  echo "  request access at https://huggingface.co/facebook/sam3, then paste"
  echo "  your token (https://huggingface.co/settings/tokens)."
  read -r -s -p "HF token (ENTER = skip, works once the weights are cached): " HF_TOKEN
  echo
  [ -n "$HF_TOKEN" ] && echo "  (the token is passed to the container only - never stored)"

  # ---- 1c. prompt points JSON -----------------------------------------------------
  echo
  if [ -f prompts_bkarnyi9.json ]; then
    echo "A points file 'prompts_bkarnyi9.json' is bundled for B.karnyi.9_cleanup.png."
  fi
  while true; do
    DEFAULT_POINTS="prompts_bkarnyi9.json"
    if ! read -r -p "Prompt-points JSON${DEFAULT_POINTS:+ (ENTER = $DEFAULT_POINTS)}: " POINTS; then
      echo; echo "No input received - aborting."; exit 1
    fi
    if [ -z "$POINTS" ]; then POINTS="$DEFAULT_POINTS"; fi
    POINTS="${POINTS/#\~/$HOME}"
    if [ ! -f "$POINTS" ]; then
      echo "  File not found: $POINTS (see the schema in the header of identify_wing.sh)"
      continue
    fi
    POINTS_BASE="$(basename "$POINTS")"
    if [ "$(cd "$(dirname "$POINTS")" && pwd)" != "$SCRIPT_DIR" ]; then
      cp "$POINTS" "$POINTS_BASE"
      echo "  Copied into this directory: $POINTS_BASE"
    fi
    break
  done

  # ---- 1d. build + run the segmentation ----------------------------------------------
  STEM="$(basename "$PHOTO_BASE")"; STEM="${STEM%.*}"
  OUT_CSV="contours_coordinates_${STEM// /_}.csv"
  MASKS_DIR="masks_${STEM// /_}"

  echo
  echo "=== Building the SAM3 segmentation image ==="
  docker build -f Dockerfile.segment -t "$SEG_IMAGE" .

  echo
  echo "=== Segmenting the wing cells ==="
  echo "First run downloads ~3.4 GB of gated weights once (kept in the"
  echo "Docker volume '$CACHE_VOLUME' for next time)."
  SEG_FLAGS=(--rm --name "$SEG_CONTAINER" -v "$SCRIPT_DIR:/work" -w /work
             -v "$CACHE_VOLUME:/root/.cache/huggingface")
  if [ -n "$HF_TOKEN" ]; then
    SEG_FLAGS+=(-e "HF_TOKEN=$HF_TOKEN")
  fi
  docker run "${SEG_FLAGS[@]}" "$SEG_IMAGE" "$PHOTO_BASE" \
      --points "$POINTS_BASE" --out "$OUT_CSV" --masks-dir "$MASKS_DIR"

  echo
  echo "Quality-check masks saved in '$MASKS_DIR' (white = segmented cell)."
  BASE="$OUT_CSV"
fi

if [ "$MODE" = "2" ]; then
  # ---- 2a. prompt for the contour file ------------------------------------------
  echo
  echo "Contour files available in this directory:"
  ls contours_coordinates*.csv 2>/dev/null | sed 's/^/  /' || echo "  (none found)"
  echo
  while true; do
    if ! read -r -p "Contour CSV to identify (file name or full path): " CONTOURS; then
      echo; echo "No input received - aborting."; exit 1
    fi
    [ -z "$CONTOURS" ] && { echo "  Please enter a file name."; continue; }
    CONTOURS="${CONTOURS/#\~/$HOME}"
    if [ ! -f "$CONTOURS" ]; then
      echo "  File not found: $CONTOURS"; continue
    fi
    BASE="$(basename "$CONTOURS")"
    # the image build only picks up files named contours_coordinates*.csv
    case "$BASE" in
      contours_coordinates*.csv)
        if [ "$(cd "$(dirname "$CONTOURS")" && pwd)" != "$SCRIPT_DIR" ]; then
          cp "$CONTOURS" "$BASE"
          echo "  Copied into this directory: $BASE"
        fi
        break
        ;;
      *)
        DEST="contours_coordinates_${BASE%.csv}.csv"
        cp "$CONTOURS" "$DEST"
        BASE="$DEST"
        echo "  Renamed to $BASE (image build needs the contours_coordinates*.csv naming)."
        break
        ;;
    esac
  done
fi

# ---- 3. prompt for the species (ENTER = unknown sample) ------------------------
echo
echo "Training species: $VALID_SPECIES"
echo "(Press ENTER without typing anything if the sample is unknown -"
echo " the workflow then reports the top-3 most probable species.)"
while true; do
  if ! read -r -p "Species of the sample (ENTER = unknown): " SPECIES; then
    echo; echo "No input received - aborting."; exit 1
  fi
  if [ -z "$SPECIES" ]; then break; fi
  if echo " $VALID_SPECIES " | grep -q " $SPECIES "; then break; fi
  echo "  '$SPECIES' is not one of the training species - check the spelling (list above)."
done

# ---- 4. build the classification image -------------------------------------------
echo
echo "=== Building the classification image (cached layers keep this quick) ==="
docker build "${PLATFORM_FLAGS[@]}" -f Dockerfile.workflow -t "$IMAGE" .

# ---- 5. run the classification ------------------------------------------------------
echo
if [ -n "$SPECIES" ]; then
  docker run --rm --name "$CONTAINER" "${PLATFORM_FLAGS[@]}" "$IMAGE" "$BASE" "$SPECIES"
else
  docker run --rm --name "$CONTAINER" "${PLATFORM_FLAGS[@]}" "$IMAGE" "$BASE"
fi
