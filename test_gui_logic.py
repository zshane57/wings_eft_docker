#!/usr/bin/env python3
# --------------------------------------------------------------
# test_gui_logic.py — headless unit tests for gui_logic.py
# (no tkinter, no Docker, no display required).
#
#   python3 test_gui_logic.py
# --------------------------------------------------------------
import json
import os
import sys
import tempfile

import gui_logic as gl

FAILURES = []


def check(name, cond, info=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ----------------------------------------------------------------------------
print("parse_output — known-sample mode")
KNOWN = """
---------------------------------
 predicted species : B.karnyi (probability 0.4446)
 expected species   : B.karnyi
 RESULT  : *** MATCH ***
 (known sample)
    B.karnyi        0.4446
    Ch.albiceps     0.2101
"""
r = gl.parse_output(KNOWN)
check("predicted", r["predicted"] == "B.karnyi")
check("prob", abs(r["prob"] - 0.4446) < 1e-9)
check("expected", r["expected"] == "B.karnyi")
check("verdict", r["verdict"] == "MATCH")
check("mode", r["mode"] == "known")
check("top3 first", r["top3"][0] == ("B.karnyi", 0.4446))

print("parse_output — unknown-sample mode")
UNKNOWN = """
  Best matches for the unknown sample:
  1. Ch.albiceps     probability 0.3549
  2. B.karnyi        probability 0.2931
  3. Ch.rufifacies   probability 0.1502
 best match : Ch.albiceps (probability 0.3549)
"""
r = gl.parse_output(UNKNOWN)
check("best", r["best"] == "Ch.albiceps")
check("best_prob", abs(r["best_prob"] - 0.3549) < 1e-9)
check("mode", r["mode"] == "unknown")
check("top3 len", len(r["top3"]) == 3)
check("top3 order", r["top3"][1] == ("B.karnyi", 0.2931))

# ----------------------------------------------------------------------------
print("parse_segment_bench")
SEG = """Using compute device: cpu
Model loaded: facebook/sam3 (12.0s)
SEGMENT_BENCH\tmodel_load\t12.3s
SEGMENT_BENCH\timage_encode\t145.9s
SEGMENT_BENCH\tinference_pa2r\t163.2s
SEGMENT_BENCH\tinference_dm\t158.8s
pa2r: 2319 boundary coordinates (163.2s, prompt points: [[1438, 301], [1031, 416], [1032, 584]])
Contour coordinates saved to contours_coordinates_B.karnyi.9.csv (2319 rows)
SEGMENT_BENCH\tpeak_rss_mb\t5233 MB
Peak RSS: 5233 MB
"""
timings, csv_path, n_rows = gl.parse_segment_bench(SEG)
check("model_load", abs(timings["model_load"] - 12.3) < 1e-9)
check("image_encode", abs(timings["image_encode"] - 145.9) < 1e-9)
check("pa2r", abs(timings["inference_pa2r"] - 163.2) < 1e-9)
check("dm", abs(timings["inference_dm"] - 158.8) < 1e-9)
check("peak rss", abs(timings["peak_rss_mb"] - 5233) < 1e-9)
check("csv name", csv_path == "contours_coordinates_B.karnyi.9.csv")
check("n rows", n_rows == 2319)
pretty = gl.pretty_bench(timings)
check("pretty has units", " MB" in pretty and " s" in pretty)
check("pretty pa2r line", "pa2r inference" in pretty)

# ----------------------------------------------------------------------------
print("points_json — schema identical to the Colab-derived prompts file")
with open("prompts_bkarnyi9.json") as fh:
    expected = json.load(fh)
stages = {"pa2r": [[1438, 301], [1031, 416], [1032, 584]],
          "dm":   [[1031, 416], [1438, 301], [1032, 584]]}
got = gl.points_json(stages)
check("schema equals prompts_bkarnyi9.json", got == expected)
check("labels", got["pa2r"]["labels"] == [1, 0, 0] and got["dm"]["labels"] == [1, 0, 0])

try:
    gl.points_json({"pa2r": [[1, 2]], "dm": [[1, 2], [3, 4], [5, 6]]})
    check("wrong point count rejected", False)
except ValueError:
    check("wrong point count rejected", True)

with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "pts.json")
    gl.write_points_json(p, stages)
    with open(p) as fh:
        check("write_points_json round trip", json.load(fh) == expected)

# ----------------------------------------------------------------------------
print("scale_to_fit / to_original — canvas mapping round trip")
s, dw, dh = gl.scale_to_fit(2042, 882, 900, 560)
check("fits width", dw <= 900 and dh <= 560, f"{dw}x{dh}")
check("scale < 1", 0 < s <= 1.0)
# clicks near the pa2r/dm interior points used in the prompts file
for ox, oy in [(1438, 301), (1031, 416), (1032, 584), (2041, 881), (0, 0)]:
    cx, cy = ox * s, oy * s
    check(f"round trip ({ox},{oy})", gl.to_original(cx, cy, s) == (ox, oy))
s2, dw2, dh2 = gl.scale_to_fit(500, 400, 900, 560)
check("small image not upscaled", s2 == 1.0 and (dw2, dh2) == (500, 400))

# ----------------------------------------------------------------------------
print("contour_csv_name")
with tempfile.TemporaryDirectory() as tmp:
    a = os.path.join(tmp, "contours_coordinates_x.csv")
    open(a, "w").close()
    path, base = gl.contour_csv_name(a)
    check("keeps valid name", base == "contours_coordinates_x.csv")
    b = os.path.join(tmp, "my_sample.csv")
    open(b, "w").close()
    path, base = gl.contour_csv_name(b)
    check("renames invalid name", base == "contours_coordinates_my_sample.csv")
    try:
        gl.contour_csv_name(os.path.join(tmp, "missing.csv"))
        check("missing file rejected", False)
    except FileNotFoundError:
        check("missing file rejected", True)

# ----------------------------------------------------------------------------
print("image_stem")
check("plain", gl.image_stem("/x/B.karnyi.9_cleanup.png") == "B.karnyi.9_cleanup")
check("spaces", gl.image_stem("/x/wing photo B.karnyi.9.png") == "wing_photo_B.karnyi.9")

# ----------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} TEST(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED")
