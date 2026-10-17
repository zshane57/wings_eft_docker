#!/usr/bin/env python3
# --------------------------------------------------------------
# gui_logic.py — pure (tkinter-free) logic for the fly wing GUI.
#
# Everything that does not need a display lives here so it can be
# unit-tested on any machine ("python3 test_gui_logic.py").
#
# Used by webserver.py (browser frontend).
# --------------------------------------------------------------

import fnmatch
import json
import math
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Docker names
CLASSIFY_IMAGE = "flywing-workflow"
CLASSIFY_CONTAINER = "flywing-session"
SEG_IMAGE = "flywing-segment"
SEG_CONTAINER = "flywing-segment-session"
CACHE_VOLUME = "flywing-sam3-cache"          # HF cache / SAM3 weights volume

VALID_SPECIES = [
    "C.vicina", "Ch.albiceps", "Ch.bezziana", "Ch.megacephala",
    "Ch.nigripes", "Ch.rufifacies", "L.sericata", "A.gressitti",
    "B.karnyi", "Le.alba", "S.princeps", "Sy.nudiseta", "Z.aquila",
]
UNKNOWN_LABEL = "(unknown sample - report top-3)"

# Number of prompt clicks per wing cell: 1 inclusive + 2 exclusive
POINTS_PER_CELL = 3
STAGES = ("pa2r", "dm")                       # notebook order


# ----------------------------------------------------------------------------
# Parsing of the classification workflow's prediction output
# ----------------------------------------------------------------------------
def parse_output(text):
    """Extract the prediction result from run_workflow.sh output.

    Returns a dict with keys (any may be None):
        top3       - list of (species, probability) tuples
        predicted  - predicted species (known-sample mode)
        prob        - probability of the predicted species
        expected   - expected species (known-sample mode)
        verdict    - 'MATCH' or 'MISMATCH'
        best        - best-match species (unknown-sample mode)
        best_prob   - probability of the best match
        mode        - 'known' or 'unknown'
    """
    res = {"top3": [], "predicted": None, "prob": None, "expected": None,
           "verdict": None, "best": None, "best_prob": None, "mode": "unknown"}

    m = re.search(r"predicted species : (\S+)\s+\(probability ([0-9.]+)\)", text)
    if m:
        res["mode"] = "known"
        res["predicted"], res["prob"] = m.group(1), float(m.group(2))
    m = re.search(r"expected species\s+:\s+(\S+)", text)
    if m:
        res["expected"] = m.group(1)
    m = re.search(r"RESULT\s+:\s+\*\*\* (MATCH|MISMATCH)", text)
    if m:
        res["verdict"] = m.group(1)
    m = re.search(r"best match\s+:\s+(\S+)\s+\(probability ([0-9.]+)\)", text)
    if m:
        res["best"], res["best_prob"] = m.group(1), float(m.group(2))

    if "(known sample)" in text:
        res["mode"] = "known"
        # lines like: "    B.karnyi        0.4446"
        for sp, p in re.findall(r"^    (\S+)\s+([0-9.]+)\s*$", text, re.M):
            res["top3"].append((sp, float(p)))
    else:
        # lines like: "  1. Ch.albiceps     probability 0.3549"
        for _, sp, p in re.findall(r"^  (\d)\. (\S+)\s+probability ([0-9.]+)",
                                   text, re.M):
            res["top3"].append((sp, float(p)))
    return res


# ----------------------------------------------------------------------------
# Parsing of the segmentation stage output
# ----------------------------------------------------------------------------
_BENCH_NAMES = {
    "model_load": "model load",
    "image_encode": "image encode",
    "inference_pa2r": "pa2r inference",
    "inference_dm": "dm inference",
    "peak_rss_mb": "peak RAM",
}


def parse_segment_bench(text):
    """Parse the 'SEGMENT_BENCH\\t<tag>\\t<value>' lines of segment_wing.py.

    Returns ({tag: float}, csv_path_or_None, n_rows_or_None).
    """
    timings = {}
    for m in re.finditer(r"SEGMENT_BENCH\t(\S+)\t([0-9.]+)", text):
        timings[m.group(1)] = float(m.group(2))

    csv_path, n_rows = None, None
    m = re.search(r"Contour coordinates saved to (\S+) \(([0-9]+) rows\)", text)
    if m:
        csv_path, n_rows = m.group(1), int(m.group(2))
    return timings, csv_path, n_rows


def pretty_bench(timings):
    """Human-readable multi-line summary for the GUI log."""
    order = ["model_load", "image_encode", "inference_pa2r",
             "inference_dm", "peak_rss_mb"]
    out = []
    for tag in order:
        if tag in timings:
            name = _BENCH_NAMES.get(tag, tag)
            val = timings[tag]
            unit = " MB" if tag == "peak_rss_mb" else " s"
            out.append(f"  {name:<16} {val:8.1f}{unit}")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Prompt-point helpers (segmentation stage)
# ----------------------------------------------------------------------------
def scale_to_fit(w, h, max_w, max_h):
    """Largest uniform scale <= 1 that fits (w, h) into (max_w, max_h)."""
    s = min(max_w / w, max_h / h, 1.0)
    return s, int(w * s), int(h * s)


def to_original(cx, cy, scale):
    """Map a click on the scaled canvas back to original-image pixels."""
    return int(round(cx / scale)), int(round(cy / scale))


def points_json(stages):
    """Build the prompts dict/JSON structure segment_wing.py expects.

    stages: {"pa2r": [[x, y], ...], "dm": [[x, y], ...]}  (click order:
    the first point is the inclusive click, the rest are exclusive).
    """
    doc = {}
    for cell in STAGES:
        pts = stages.get(cell) or []
        if len(pts) != POINTS_PER_CELL:
            raise ValueError(
                f"{cell}: expected {POINTS_PER_CELL} points, got {len(pts)}")
        labels = [1] + [0] * (POINTS_PER_CELL - 1)
        doc[cell] = {"points": [[int(x), int(y)] for x, y in pts],
                     "labels": labels}
    return doc


def write_points_json(path, stages):
    with open(path, "w") as fh:
        json.dump(points_json(stages), fh, indent=1)


# ----------------------------------------------------------------------------
# Sample preparation
# ----------------------------------------------------------------------------
def contour_csv_name(path):
    """Validate a user-chosen contour CSV path and return the file name that
    the classification image build must see (contours_coordinates*.csv).
    The caller is responsible for copying/renaming the file."""
    path = os.path.abspath(os.path.expanduser(path.strip().strip('"').strip("'")))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    base = os.path.basename(path)
    if fnmatch.fnmatch(base, "contours_coordinates*.csv"):
        return path, base
    return path, "contours_coordinates_" + os.path.splitext(base)[0] + ".csv"


def image_stem(path):
    """'Wing photo.B.karnyi.9.png' -> 'B.karnyi.9' (spaces are allowed)."""
    base = os.path.basename(path.strip())
    base = os.path.splitext(base)[0]
    return base.replace(" ", "_")
