#!/bin/bash
# --------------------------------------------------------------
# run_segment_benchmark.sh — one-shot SAM3 segmentation benchmark
# (Phase 1 of the combined workflow; run this once with your
#  Hugging Face token, then hand the results back for validation)
#
# Builds the segmentation image if needed, downloads the gated
# SAM3 weights into the 'flywing-sam3-cache' Docker volume (first
# run only, ~3.4 GB), segments the bundled B.karnyi.9 test photo,
# and writes into this directory:
#   contours_coordinates_B.karnyi.9.csv   (classification input)
#   masks_bkarnyi9/                      (quality-check masks)
#   _benchmark_last.log                  (full run log)
# The SEGMENT_BENCH lines report timings and peak RAM.
#
# Usage (recommended — the token is typed hidden, never in history):
#   ./run_segment_benchmark.sh
# or pass it directly (ends up in the shell history):
#   ./run_segment_benchmark.sh hf_xxx
# --------------------------------------------------------------
set -e
set -o pipefail
cd "$(dirname "$0")"

IMAGE="flywing-segment"
VOLUME="flywing-sam3-cache"
PHOTO="B.karnyi.9_cleanup.png"
POINTS="prompts_bkarnyi9.json"
OUT="contours_coordinates_B.karnyi.9.csv"
MASKS="masks_bkarnyi9"
LOG="_benchmark_last.log"

if [ "$#" -ge 1 ]; then
  TOKEN="$1"
else
  read -r -s -p "Hugging Face token (hidden; ENTER = use cached weights): " TOKEN
  echo
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "=== Building the SAM3 segmentation image ==="
  docker build -f Dockerfile.segment -t "$IMAGE" .
fi

echo "=== Segmenting $PHOTO ==="
echo "First run downloads ~3.4 GB of gated weights once (kept in the"
echo "Docker volume '$VOLUME' for next time). CPU-only: expect a few minutes."
docker run --rm -v "$PWD:/work" -w /work \
  -v "$VOLUME:/root/.cache/huggingface" \
  -e "HF_TOKEN=$TOKEN" \
  "$IMAGE" "$PHOTO" --points "$POINTS" --out "$OUT" --masks-dir "$MASKS" \
  2>&1 | tee "$LOG"

echo
echo "=== Benchmark summary ==="
grep "SEGMENT_BENCH" "$LOG" || true
echo
echo "Produced: $OUT, $MASKS/, $LOG"
