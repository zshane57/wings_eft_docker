#!/bin/bash
# --------------------------------------------------------------
# run_workflow.sh — full fly-wing classification workflow (Tasks 2-5)
#   Task 2+3: EFD (10 harmonics) -> scale/rotate/phase normalisation
#              -> traversal-direction canonicalisation
#              -> fold-1 LDA projection -> 24 features
#   Task 4  : assemble the full GUIDE data file (training rows on top,
#             testing rows + new sample at the end, weight = 0)
#   Task 5  : run the GUIDE random forest and report the prediction
#
# Usage:
#   ./run_workflow.sh <contours_csv> [true_species] [family]
#
#   Known sample (validation):
#     ./run_workflow.sh contours_coordinates_bkarnyi.csv B.karnyi
#     -> predicted vs expected species, MATCH/MISMATCH verdict
#
#   Unknown sample (no species given):
#     ./run_workflow.sh contours_coordinates_nudiseta.csv
#     -> the row carries the default placeholder label C.vicina
#        (GUIDE requires a valid class label on every row; the label
#        never enters training and does not affect the prediction)
#     -> reports the top-3 species with the highest probabilities
# --------------------------------------------------------------
set -e

CONTOURS_CSV="$1"
TRUE_SPECIES="$2"

if [ -z "$CONTOURS_CSV" ]; then
  echo "Usage: $0 <contours_csv> [true_species] [family]"
  echo
  echo "  true_species : optional."
  echo "    - given    : validate a sample of known identity (MATCH/MISMATCH report)"
  echo "    - omitted  : unknown sample - placeholder label C.vicina is used and"
  echo "                 the top-3 most probable species are reported"
  exit 1
fi

if [ ! -f "$CONTOURS_CSV" ]; then
  echo "Error: contour file not found in the image: $CONTOURS_CSV"
  echo
  echo "Available contour CSVs:"
  ls /app/contours_coordinates*.csv 2>/dev/null | sed 's|^|  |' || echo "  (none)"
  echo
  echo "To identify a new sample, name its CSV 'contours_coordinates*.csv',"
  echo "put it into the docker_workflow directory and rebuild the image:"
  echo "  cd docker_workflow"
  echo "  docker build --platform linux/amd64 -f Dockerfile.workflow -t flywing-workflow ."
  exit 1
fi

if [ -z "$TRUE_SPECIES" ]; then
  # Unknown sample: default placeholder species (must be a training class
  # level for GUIDE to parse the row; weight = 0 keeps it out of training
  # and the label cannot influence the prediction).
  TRUE_SPECIES="C.vicina"
  KNOWN_SPECIES=0
else
  KNOWN_SPECIES=1
fi

DATA_FILE="guide_input/combined_fd1_dmpa2r_data.txt"
GUIDE_IN="fold_1_guide_data_rf.in"
PROB_FILE="guide_input/identification_prob.txt"   # declared inside the .in file

echo "================================================================"
if [ "$KNOWN_SPECIES" = "1" ]; then
  echo " Fly wing species classification workflow (known sample)"
  echo "   contours  : $CONTOURS_CSV"
  echo "   species   : $TRUE_SPECIES (validation label)"
else
  echo " Fly wing species classification workflow (UNKNOWN sample)"
  echo "   contours  : $CONTOURS_CSV"
  echo "   label     : C.vicina (default placeholder - does not affect prediction)"
fi
echo "================================================================"
echo
echo "=== [Tasks 2-4] EFD -> normalisation -> LDA -> GUIDE data file ==="
Rscript process_features.R "$CONTOURS_CSV" "$DATA_FILE" "$TRUE_SPECIES"
echo

echo "=== [Task 5] GUIDE random-forest classification ==="
# GUIDE 46.x reads its command file from stdin (the filename-as-argument
# form is not accepted; it drops to the interactive menu instead)
(cd guide_input && guide < "$GUIDE_IN" > guide_run.log 2>&1) || {
  echo "GUIDE failed. Log tail:"; tail -30 guide_input/guide_run.log; exit 1;
}
tail -12 guide_input/guide_run.log
echo

TRUE_SPECIES="$TRUE_SPECIES" KNOWN_SPECIES="$KNOWN_SPECIES" python3 - <<'PY'
import os, re

known    = os.environ.get("KNOWN_SPECIES", "1") == "1"
expected = os.environ["TRUE_SPECIES"]

lines = [l for l in open("guide_input/identification_prob.txt").read().splitlines() if l.strip()]
hdr = lines[0].split()
row = lines[-1].split()          # last record = the new sample (weight 0)

# class probabilities: header column "P(<species>)" <-> same column index in the row
probs = {}
for i, h in enumerate(hdr):
    m = re.search(r'P\("?([^")]+?)"?\)', h)
    if m:
        probs[m.group(1)] = float(row[i])

predicted = row[-2].strip('"')   # second-to-last column = predicted class
observed  = row[-1].strip('"')   # last column = the label we assigned

top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]

if known:
    print("=== Prediction (known sample) ===")
    print("  top-3 class probabilities:")
    for sp, p in top3:
        print(f"    {sp:<15} {p:.4f}")
    print()
    print(f"  predicted species : {predicted}  (probability {probs.get(predicted, float('nan')):.4f})")
    print(f"  expected species  : {expected}  (label written on the new row: {observed})")
    if predicted == expected:
        print("  RESULT            : *** MATCH - workflow captured the correct species ***")
    else:
        print("  RESULT            : *** MISMATCH - prediction differs from expectation ***")
else:
    print("=== Prediction (unknown sample - top-3 most probable species) ===")
    for rank, (sp, p) in enumerate(top3, start=1):
        marker = "   <-- best match" if rank == 1 else ""
        print(f"  {rank}. {sp:<15} probability {p:.4f}{marker}")
    print()
    print(f"  best match        : {top3[0][0]}  (probability {top3[0][1]:.4f})")
PY
