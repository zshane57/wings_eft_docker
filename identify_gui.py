#!/usr/bin/env python3
# --------------------------------------------------------------
# identify_gui.py — graphical front-end for the fly wing
# species identification workflow (Tasks 2-6).
#
# Two modes:
#   1. Segment a wing photo, then identify it
#        - click prompt points on the photo (3 per wing cell)
#        - the SAM3 container segments the two cells,
#          producing the contour CSV automatically
#        - classification runs on the result right away
#   2. Classify an existing contour CSV (previous behaviour)
#
# The GUI builds the Docker images, streams the log, shows the
# result, and removes the containers and images afterwards
# (the build cache and the SAM3 weights volume are kept).
#
# Requirements: Python 3.8+ with tkinter, Docker installed & running.
#   Optional: pillow (for JPEG photos and smoother scaling) —
#             pip install pillow
#
#   macOS:    python3 identify_gui.py
#   Windows:  py identify_gui.py
#   Linux:    python3 identify_gui.py   (needs python3-tk installed)
# --------------------------------------------------------------

import math
import os
import queue
import shutil
import subprocess
import sys
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError:
    sys.stderr.write(
        "Error: tkinter is not available in this Python installation.\n"
        "  - macOS:   install Python from python.org, or: brew install python-tk\n"
        "  - Linux:   sudo apt install python3-tk\n"
        "  - Windows: re-run the python.org installer and tick 'tcl/tk and IDLE'\n"
    )
    sys.exit(1)

from gui_logic import (
    CACHE_VOLUME, CLASSIFY_CONTAINER, CLASSIFY_IMAGE, SCRIPT_DIR, SEG_CONTAINER,
    SEG_IMAGE, STAGES, UNKNOWN_LABEL, VALID_SPECIES,
    contour_csv_name, image_stem, parse_output, parse_segment_bench,
    pretty_bench, scale_to_fit, to_original, write_points_json,
)

try:
    from PIL import Image, ImageTk          # optional, see header
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Execution helpers (run inside the worker threads)
# ----------------------------------------------------------------------------
def log_gui(q, msg):
    q.put(("log", msg))


def run_stream(cmd, q, cwd=None):
    """Run a command, stream its output lines to the GUI, return exit code."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_gui(q, line.rstrip("\n"))
    proc.wait()
    return proc.returncode


def prepare_contour_file(path, q):
    """Copy/rename the selected CSV into the workflow directory so the
    image build picks it up. Returns the file name to pass to the container."""
    path, base = contour_csv_name(path)
    if base == os.path.basename(path):
        if os.path.dirname(path) != SCRIPT_DIR:
            shutil.copy2(path, os.path.join(SCRIPT_DIR, base))
            log_gui(q, f"Copied into the workflow directory: {base}")
        return base
    dest = base
    shutil.copy2(path, os.path.join(SCRIPT_DIR, dest))
    log_gui(q, f"Renamed to {dest} (image build needs the contours_coordinates*.csv naming).")
    return dest


def docker_checks(q):
    """Docker present + running; returns the platform flags the
    classification image needs (GUIDE is x86_64-only)."""
    if shutil.which("docker") is None:
        raise RuntimeError("Docker was not found on this machine.")
    rc = run_stream(["docker", "info"], q)
    if rc != 0:
        raise RuntimeError("Docker is not running. Start Docker Desktop first.")
    arch = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Arch}}"],
        capture_output=True, text=True).stdout.strip()
    if arch in ("arm64", "aarch64"):
        log_gui(q, "Note: ARM host detected - the x86_64 image runs through emulation.")
        return ["--platform", "linux/amd64"]
    log_gui(q, "Note: x86_64 host detected - the container runs natively.")
    return []


def docker_cleanup(q, containers, images):
    """Remove containers + images (build cache and volumes are kept)."""
    log_gui(q, "")
    log_gui(q, "=== Cleaning up ===")
    for name in containers:
        subprocess.run(["docker", "rm", "-f", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for image in images:
        inspect = subprocess.run(["docker", "image", "inspect", image],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        if inspect.returncode == 0:
            subprocess.run(["docker", "rmi", image],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_gui(q, f"Container and image '{image}' removed.")
    log_gui(q, "The build cache is kept, so the next start rebuilds quickly.")
    log_gui(q, f"The SAM3 weights volume '{CACHE_VOLUME}' is kept "
               f"(delete with: docker volume rm {CACHE_VOLUME}).")


def build_classify_image(q, platform_flags):
    log_gui(q, "")
    log_gui(q, "=== Building the classification image (cached layers keep this quick) ===")
    rc = run_stream(["docker", "build"] + platform_flags +
                    ["-f", "Dockerfile.workflow", "-t", CLASSIFY_IMAGE, "."],
                    q, cwd=SCRIPT_DIR)
    if rc != 0:
        raise RuntimeError("Docker image build failed.")


def run_classification(q, collected, csv_name, species, platform_flags):
    log_gui(q, "")
    log_gui(q, f"=== Classifying {csv_name} ===")
    cmd = ["docker", "run", "--rm", "--name", CLASSIFY_CONTAINER] + \
        platform_flags + [CLASSIFY_IMAGE, csv_name]
    if species:
        cmd.append(species)
    proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip("\n")
        collected.append(line)
        log_gui(q, line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Classification run failed.")
    result = parse_output("\n".join(collected))
    if result["mode"] == "known" and result["predicted"] is None:
        raise RuntimeError("Could not read the prediction from the workflow output.")
    q.put(("result", result))


# ----------------------------------------------------------------------------
# Worker threads (one per GUI mode)
# ----------------------------------------------------------------------------
def workflow_thread(q, csv_path, species):
    """Mode 2: classify an existing contour CSV."""
    try:
        log_gui(q, "=" * 64)
        log_gui(q, " Forensic fly wing species identification")
        log_gui(q, "=" * 64)

        platform_flags = docker_checks(q)
        base = prepare_contour_file(csv_path, q)
        log_gui(q, "")
        log_gui(q, f"Sample : {base}")
        log_gui(q, f"Species : {species if species else 'unknown (top-3 report)'}")

        build_classify_image(q, platform_flags)
        collected = []
        run_classification(q, collected, base, species, platform_flags)
    except Exception as exc:                                  # noqa: BLE001
        q.put(("error", str(exc)))
    finally:
        docker_cleanup(q, [CLASSIFY_CONTAINER], [CLASSIFY_IMAGE])


def segment_thread(q, image_path, points, token, species):
    """Mode 1: segment the wing photo with SAM3, then classify the result."""
    seg_container, seg_image = SEG_CONTAINER, SEG_IMAGE
    try:
        log_gui(q, "=" * 64)
        log_gui(q, " Forensic fly wing species identification - full mode")
        log_gui(q, "=" * 64)

        platform_flags = docker_checks(q)

        # ---- stage 1: SAM3 segmentation --------------------------------
        stem = image_stem(image_path)
        src = os.path.abspath(os.path.expanduser(image_path.strip()))
        img_base = os.path.basename(src)
        if os.path.dirname(src) != SCRIPT_DIR:
            shutil.copy2(src, os.path.join(SCRIPT_DIR, img_base))
            log_gui(q, f"Copied the wing photo into the workflow directory: {img_base}")

        prompts_name = f"_prompts_{stem}.json"
        write_points_json(os.path.join(SCRIPT_DIR, prompts_name), points)
        log_gui(q, f"Prompt points written to {prompts_name}")

        out_csv = f"contours_coordinates_{stem}.csv"
        masks_dir = f"masks_{stem}"

        log_gui(q, "")
        log_gui(q, "=== Building the SAM3 segmentation image ===")
        rc = run_stream(["docker", "build", "-f", "Dockerfile.segment",
                         "-t", seg_image, "."], q, cwd=SCRIPT_DIR)
        if rc != 0:
            raise RuntimeError("Segmentation image build failed.")

        log_gui(q, "")
        log_gui(q, "=== Segmenting the wing cells ===")
        log_gui(q, "First run downloads ~3.4 GB of gated weights once "
                   "(kept in the Docker volume for next time).")
        cmd = ["docker", "run", "--rm", "--name", seg_container,
               "-v", f"{SCRIPT_DIR}:/work", "-w", "/work",
               "-v", f"{CACHE_VOLUME}:/root/.cache/huggingface"]
        if token:
            cmd += ["-e", f"HF_TOKEN={token}"]
        cmd += [seg_image, img_base,
                "--points", prompts_name,
                "--out", out_csv,
                "--masks-dir", masks_dir]

        collected = []
        proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip("\n")
            collected.append(line)
            log_gui(q, line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Segmentation run failed.")

        timings, csv_from_log, n_rows = parse_segment_bench("\n".join(collected))
        if csv_from_log is None:
            raise RuntimeError("Segmentation did not produce a contour CSV.")
        log_gui(q, "")
        log_gui(q, f"Segmentation finished: {out_csv} ({n_rows} rows)")
        if timings:
            log_gui(q, pretty_bench(timings))
        log_gui(q, f"Quality-check masks saved in '{masks_dir}' "
                   f"(white = segmented cell).")

        # remove the segmentation image right away; keep cache + weights
        subprocess.run(["docker", "rm", "-f", seg_container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        inspect = subprocess.run(["docker", "image", "inspect", seg_image],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        if inspect.returncode == 0:
            subprocess.run(["docker", "rmi", seg_image],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_gui(q, f"Segmentation container and image '{seg_image}' removed.")

        # ---- stage 2: classification -------------------------------------
        log_gui(q, "")
        log_gui(q, f"Sample : {out_csv}")
        log_gui(q, f"Species : {species if species else 'unknown (top-3 report)'}")
        build_classify_image(q, platform_flags)
        run_classification(q, [], out_csv, species, platform_flags)
    except Exception as exc:                                  # noqa: BLE001
        q.put(("error", str(exc)))
    finally:
        docker_cleanup(q, [CLASSIFY_CONTAINER, seg_container],
                       [CLASSIFY_IMAGE, seg_image])


# ----------------------------------------------------------------------------
# Prompt-point picking dialog (segmentation stage)
# ----------------------------------------------------------------------------
class SegmentationDialog(tk.Toplevel):
    """Modal dialog: click 3 prompt points per wing cell on the photo.

    Click order per cell = 1 point INSIDE the cell (green, inclusive)
    followed by 2 points OUTSIDE it (red, exclusive) - the same
    convention as the notebook that produced the training contours.
    """

    MAXW, MAXH = 900, 560
    _INSTRUCTIONS = {
        "pa2r": "Step 1 of 2 - the pa2r cell: click 1 point INSIDE the cell "
                "(green), then 2 points OUTSIDE the cell (red) to exclude "
                "the neighbouring cells.",
        "dm":   "Step 2 of 2 - the dm cell: click 1 point INSIDE the cell "
                "(green), then 2 points OUTSIDE the cell (red).",
    }

    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.title("SAM3 prompt points")
        self.resizable(False, False)
        self.result = None
        self.points = {cell: [] for cell in STAGES}
        self.stage_idx = 0
        self.markers = []
        self.photo = None          # keep a reference or tk will GC it

        try:
            self._load_image(image_path)
        except Exception as exc:                                  # noqa: BLE001
            messagebox.showerror("Could not open the photo", str(exc), parent=self)
            self.destroy()
            return

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.var_info = tk.StringVar(
            value=self._INSTRUCTIONS[STAGES[0]])
        ttk.Label(self, textvariable=self.var_info, wraplength=860,
                  justify="left").pack(fill="x", padx=10, pady=(10, 4))

        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h,
                                bg="#222222", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=4)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw",
                                 tags="photo")
        self.canvas.bind("<Button-1>", self._click)

        row = ttk.Frame(self)
        row.pack(fill="x", padx=10, pady=(2, 8))
        self.btn_undo = ttk.Button(row, text="Undo last point",
                                   command=self._undo)
        self.btn_undo.pack(side="left")
        self.btn_reset = ttk.Button(row, text="Reset this cell",
                                    command=self._reset)
        self.btn_reset.pack(side="left", padx=6)
        self.btn_next = ttk.Button(row, text="Next cell \u25b8",
                                   command=self._next, state="disabled")
        self.btn_next.pack(side="right")
        ttk.Button(row, text="Cancel", command=self._cancel).pack(
            side="right", padx=6)
        self.var_status = tk.StringVar(value="pa2r: 0 of 3 points")
        ttk.Label(self, textvariable=self.var_status,
                  foreground="grey").pack(side="left", padx=10)

        self.transient(parent)
        self.grab_set()
        self.focus_set()

    # ------------------------------------------------------------------ image
    def _load_image(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        if HAS_PIL:
            img = Image.open(path)
            self.orig_w, self.orig_h = img.size
            s, self.disp_w, self.disp_h = scale_to_fit(
                self.orig_w, self.orig_h, self.MAXW, self.MAXH)
            self.scale = s
            self.photo = ImageTk.PhotoImage(
                img.resize((self.disp_w, self.disp_h), Image.LANCZOS))
        else:
            base = tk.PhotoImage(file=path)      # PNG/GIF only
            self.orig_w, self.orig_h = base.width(), base.height()
            s, self.disp_w, self.disp_h = scale_to_fit(
                self.orig_w, self.orig_h, self.MAXW, self.MAXH)
            k = max(1, math.ceil(max(self.orig_w / self.disp_w,
                                     self.orig_h / self.disp_h)))
            self.disp_w = -(-self.orig_w // k)    # ceil division
            self.disp_h = -(-self.orig_h // k)
            self.photo = base.subsample(k, k)
            self.scale = self.orig_w / self.disp_w

    # ----------------------------------------------------------------- events
    def _click(self, event):
        cell = STAGES[self.stage_idx]
        if len(self.points[cell]) >= 3:
            return
        x, y = to_original(event.x, event.y, self.scale)
        self.points[cell].append([x, y])
        color = "limegreen" if len(self.points[cell]) == 1 else "red"
        r = 5
        self.markers.append(self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill=color, outline="white", width=2))
        self._refresh()

    def _undo(self):
        cell = STAGES[self.stage_idx]
        if self.points[cell]:
            self.points[cell].pop()
            if self.markers:
                self.canvas.delete(self.markers.pop())
            self._refresh()

    def _reset(self):
        cell = STAGES[self.stage_idx]
        self.points[cell] = []
        for m in self.markers:
            self.canvas.delete(m)
        self.markers = []
        self._refresh()

    def _next(self):
        cell = STAGES[self.stage_idx]
        if len(self.points[cell]) != 3:
            return
        self.stage_idx += 1
        if self.stage_idx >= len(STAGES):
            from gui_logic import points_json
            self.result = points_json(self.points)
            self.destroy()
        else:
            for m in self.markers:
                self.canvas.delete(m)
            self.markers = []
            self.btn_next.configure(text="Finish \u25b8")
            self._refresh()

    def _cancel(self):
        self.result = None
        self.destroy()

    def _refresh(self):
        cell = STAGES[self.stage_idx]
        n = len(self.points[cell])
        self.var_status.set(f"{cell}: {n} of 3 points")
        self.btn_next.configure(state="normal" if n == 3 else "disabled")
        self.var_info.set(self._INSTRUCTIONS[cell])


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class IdentifyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Forensic Fly Wing Species Identification")
        self.minsize(860, 600)
        self.q = queue.Queue()
        self.running = False
        self.points_result = None

        pad = {"padx": 12, "pady": 4}
        ttk.Label(self, text="Forensic Fly Wing Species Identification",
                  font=("Helvetica", 15, "bold")).grid(row=0, column=0,
                                                       columnspan=3, pady=(12, 8))

        # --- mode selector --------------------------------------------------
        self.var_mode = tk.IntVar(value=1)
        frm_mode = ttk.LabelFrame(self, text="What would you like to do?")
        frm_mode.grid(row=1, column=0, columnspan=3, sticky="we", **pad)
        ttk.Radiobutton(frm_mode, value=1, variable=self.var_mode,
                        text="1. Segment a wing photo with SAM3, then identify it",
                        command=self._mode_switch).pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Radiobutton(frm_mode, value=2, variable=self.var_mode,
                        text="2. Classify an existing contour CSV",
                        command=self._mode_switch).pack(anchor="w", padx=8, pady=(0, 6))

        # --- mode 1 panel -----------------------------------------------------
        self.frm_seg = ttk.LabelFrame(self, text="Segmentation input")
        self.frm_seg.grid(row=2, column=0, columnspan=3, sticky="we", **pad)

        ttk.Label(self.frm_seg, text="Wing photo:").grid(row=0, column=0, sticky="w")
        self.var_img = tk.StringVar()
        ent_img = ttk.Entry(self.frm_seg, textvariable=self.var_img, width=70)
        ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(self.frm_seg, text="Browse\u2026",
                   command=self.browse_image).grid(row=0, column=3, sticky="e")

        ttk.Label(self.frm_seg, text="HF token:").grid(row=1, column=0, sticky="w")
        self.var_token = tk.StringVar()
        ent_tok = ttk.Entry(self.frm_seg, textvariable=self.var_token,
                            width=52, show="*")
        ent_tok.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(self.frm_seg,
                  text="required for the first download of the gated SAM3 model; "
                       "afterwards the weights stay cached in the Docker volume. "
                       "Passed to the container only - never saved.",
                  foreground="grey", wraplength=430,
                  justify="left").grid(row=1, column=2, columnspan=2, sticky="w", padx=6)

        ttk.Label(self.frm_seg, text="Prompt points:").grid(row=2, column=0, sticky="w")
        self.btn_pick = ttk.Button(self.frm_seg,
                                   text="Click on the photo\u2026",
                                   command=self.pick_points)
        self.btn_pick.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        self.var_points = tk.StringVar(value="not picked yet (3 clicks per wing cell)")
        ttk.Label(self.frm_seg, textvariable=self.var_points,
                  foreground="grey").grid(row=2, column=2, columnspan=2,
                                          sticky="w", padx=6)
        self.frm_seg.columnconfigure(1, weight=1)

        # --- mode 2 panel -------------------------------------------------------
        self.frm_csv = ttk.LabelFrame(self, text="Classification input")
        self.frm_csv.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(self.frm_csv, text="Contour CSV:").grid(row=0, column=0, sticky="w")
        self.var_csv = tk.StringVar()
        ttk.Entry(self.frm_csv, textvariable=self.var_csv, width=70).grid(
            row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(self.frm_csv, text="Browse\u2026",
                   command=self.browse_csv).grid(row=0, column=3, sticky="e")
        self.frm_csv.columnconfigure(1, weight=1)

        # --- species row (both modes) ------------------------------------------
        ttk.Label(self, text="Species:").grid(row=3, column=0, sticky="w", **pad)
        self.var_species = tk.StringVar(value=UNKNOWN_LABEL)
        self.cmb_species = ttk.Combobox(
            self, textvariable=self.var_species, state="readonly", width=30,
            values=[UNKNOWN_LABEL] + VALID_SPECIES)
        self.cmb_species.grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(self, text="(leave as unknown if the sample's identity is not known)",
                  foreground="grey").grid(row=3, column=2, sticky="w", **pad)

        # --- run button -------------------------------------------------------------
        self.btn_run = ttk.Button(self, text="Segment && Identify",
                                  command=self.start)
        self.btn_run.grid(row=4, column=0, columnspan=3, pady=10)

        # --- result panel --------------------------------------------------------------
        frm = ttk.LabelFrame(self, text="Result")
        frm.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)
        self.lbl_main = ttk.Label(frm, text="\u2014",
                                  font=("Helvetica", 19, "bold"))
        self.lbl_main.pack(pady=(8, 0))
        self.lbl_prob = ttk.Label(frm, text="", font=("Helvetica", 12))
        self.lbl_prob.pack()
        self.lbl_detail = ttk.Label(frm, text="", foreground="grey",
                                    justify="left")
        self.lbl_detail.pack(pady=(2, 8))

        # --- log ---------------------------------------------------------------------------
        self.txt_log = scrolledtext.ScrolledText(self, height=14,
                                                 state="disabled",
                                                 font=("Menlo", 10))
        self.txt_log.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)

        # --- status bar ---------------------------------------------------------------------
        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.var_status, relief="sunken",
                  anchor="w").grid(row=7, column=0, columnspan=3, sticky="we")

        self.columnconfigure(1, weight=1)
        self.rowconfigure(6, weight=1)
        self._mode_switch()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ helpers
    def _append_log(self, line):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", line + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _mode_switch(self):
        if self.var_mode.get() == 1:
            self.frm_seg.grid()
            self.frm_csv.grid_remove()
            self.btn_run.configure(text="Segment && Identify")
        else:
            self.frm_csv.grid()
            self.frm_seg.grid_remove()
            self.btn_run.configure(text="Identify")

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "log":
                    self._append_log(data)
                elif kind == "result":
                    self._show_result(data)
                elif kind == "error":
                    self.var_status.set("Failed.")
                    messagebox.showerror("Identification failed", data)
                    self.running = False
                    self.btn_run.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _show_result(self, r):
        if r["mode"] == "known":
            self.lbl_main.configure(text=r["predicted"] or "\u2014")
            self.lbl_prob.configure(
                text=f"probability {r['prob']:.4f}"
                if r["prob"] is not None else "")
            self.lbl_detail.configure(
                text=f"expected: {r['expected']}\nRESULT: {r['verdict']}",
                foreground="#1a7a1a" if r["verdict"] == "MATCH" else "#a11a1a")
            self.var_status.set(
                "Done." if r["verdict"] == "MATCH"
                else "Done - prediction differs from expectation.")
        else:
            self.lbl_main.configure(text=r["best"] or "\u2014")
            self.lbl_prob.configure(
                text=f"best match, probability {r['best_prob']:.4f}"
                if r["best_prob"] is not None else "")
            lines = [f"{i+1}. {sp:<15} {p:.4f}"
                     for i, (sp, p) in enumerate(r["top3"])]
            self.lbl_detail.configure(
                text="top-3 most probable species:\n" + "\n".join(lines),
                foreground="#333333")
            self.var_status.set("Done - unknown sample, top-3 reported.")
        self.running = False
        self.btn_run.configure(state="normal")

    # ------------------------------------------------------------------ actions
    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select the wing photo",
            filetypes=[("Wing photos", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                       ("All files", "*.*")])
        if path:
            self.var_img.set(path)
            self.points_result = None
            self.var_points.set("not picked yet (3 clicks per wing cell)")

    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select the contour coordinate CSV",
            filetypes=[("Contour CSV", "contours_coordinates*.csv"),
                       ("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.var_csv.set(path)

    def pick_points(self):
        img = self.var_img.get().strip()
        if not img:
            messagebox.showwarning("Missing input",
                                   "Please choose the wing photo first.",
                                   parent=self)
            return
        dlg = SegmentationDialog(self, img)
        self.wait_window(dlg)
        if dlg.result:
            self.points_result = dlg.result
            n = sum(len(dlg.result[c]["points"]) for c in dlg.result)
            self.var_points.set(f"{n} points collected "
                                f"(pa2r: 3, dm: 3)")
        else:
            self.points_result = None
            self.var_points.set("picking cancelled")

    def start(self):
        if self.running:
            return
        species = self.var_species.get().strip()
        species = "" if species == UNKNOWN_LABEL else species

        if self.var_mode.get() == 1:
            img = self.var_img.get().strip()
            if not img:
                messagebox.showwarning("Missing input",
                                       "Please choose the wing photo first.")
                return
            if self.points_result is None:
                messagebox.showwarning(
                    "Missing prompt points",
                    "Please click the prompt points on the photo first "
                    "(button 'Click on the photo').")
                return
            token = self.var_token.get().strip()
            if not token:
                if not messagebox.askyesno(
                        "No HF token",
                        "Without a Hugging Face token the first SAM3 "
                        "download will fail. Continue anyway (only works if "
                        "the weights are already cached)?"):
                    return
            args = (img, self.points_result, token, species)
            target = segment_thread
        else:
            csv_path = self.var_csv.get().strip()
            if not csv_path:
                messagebox.showwarning("Missing input",
                                       "Please choose the contour CSV file first.")
                return
            args = (csv_path, species)
            target = workflow_thread

        self.running = True
        self.btn_run.configure(state="disabled")
        self.lbl_main.configure(text="\u2014")
        self.lbl_prob.configure(text="")
        self.lbl_detail.configure(text="", foreground="grey")
        self.var_status.set("Running the identification workflow\u2026")
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")

        threading.Thread(target=target, args=(self.q,) + args,
                         daemon=True).start()


def main():
    app = IdentifyGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
