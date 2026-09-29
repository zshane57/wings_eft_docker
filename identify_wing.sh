#!/bin/bash
# --------------------------------------------------------------
# identify_wing.sh — interactive one-shot launcher
#
# Prompts for the contour CSV and (optionally) the sample's species,
# builds the Docker image, runs the classification workflow, and
# afterwards removes the container and the image again.
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
ARCH="$(docker version --format '{{.Server.Arch}}' 2>/dev/null || true)"
PLATFORM_FLAGS=()
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  PLATFORM_FLAGS=(--platform linux/amd64)
  echo "Note: Apple Silicon detected - the x86_64 image runs through emulation."
fi

# ---- cleanup: remove container + image when the script ends -------------------
cleanup() {
  echo
  echo "=== Cleaning up ==="
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker rmi "$IMAGE" >/dev/null 2>&1 || true
    echo "Container and image '$IMAGE' removed."
  else
    echo "Container removed."
  fi
  echo "The build cache is kept, so the next start rebuilds quickly."
  echo "To also free the build cache: docker builder prune"
}
trap cleanup EXIT

# ---- 1. prompt for the contour file -------------------------------------------
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

# ---- 2. prompt for the species (ENTER = unknown sample) -----------------------
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

# ---- 3. build the image --------------------------------------------------------
echo
echo "=== Building the Docker image (cached layers keep this quick) ==="
docker build "${PLATFORM_FLAGS[@]}" -f Dockerfile.workflow -t "$IMAGE" .

# ---- 4. run the classification --------------------------------------------------
echo
if [ -n "$SPECIES" ]; then
  docker run --rm --name "$CONTAINER" "${PLATFORM_FLAGS[@]}" "$IMAGE" "$BASE" "$SPECIES"
else
  docker run --rm --name "$CONTAINER" "${PLATFORM_FLAGS[@]}" "$IMAGE" "$BASE"
fi
