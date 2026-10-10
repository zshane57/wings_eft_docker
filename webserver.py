#!/usr/bin/env python3
# --------------------------------------------------------------
# webserver.py — browser frontend for the fly wing pipeline.
#
# Serves the HTML page and drives the two workflow images through
# the host Docker daemon (docker CLI via the mounted docker.sock).
# The package directory is mounted at /app/work; WORK_HOST_PATH
# tells the server the HOST-side path of that mount (needed for
# the -v flags of the containers it launches).
#
# Security model for a local research tool:
#   - binds 127.0.0.1 only (the launcher also maps 127.0.0.1)
#   - the HF token travels form -> env var -> container, never disk
#   - file routes only serve whitelisted extensions from the work dir
# --------------------------------------------------------------

import json
import os
import re
import subprocess
import threading

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from gui_logic import (
    CLASSIFY_CONTAINER, CLASSIFY_IMAGE, CACHE_VOLUME, SEG_CONTAINER,
    SEG_IMAGE, UNKNOWN_LABEL, VALID_SPECIES, image_stem, parse_output,
    parse_segment_bench, pretty_bench, points_json, write_points_json,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.environ.get("WORK_DIR", os.path.join(APP_DIR, "work"))
HOST_WORK_DIR = os.environ.get("WORK_HOST_PATH", WORK_DIR)
PORT = int(os.environ.get("PORT", "8080"))

PHOTO_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SERVE_EXT = PHOTO_EXT | {".csv", ".log", ".json"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

job = {
    "state": "idle",          # idle | running | done | error
    "phase": "",
    "log": [],
    "result": None,
    "error": None,
    "proc": None,
    "cancel": False,
}
job_lock = threading.Lock()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def run_capture(cmd, extra_log=None):
    """Run a docker command, stream its output into the job log.
    Returns (exit_code, collected_lines)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    job["proc"] = proc
    out = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        out.append(line)
        job["log"].append(line)
    proc.wait()
    job["proc"] = None
    if extra_log:
        out.extend(extra_log)
    return proc.returncode, out


def docker_arch():
    r = subprocess.run(["docker", "version", "--format", "{{.Server.Arch}}"],
                       capture_output=True, text=True)
    return (r.stdout or "").strip()


def cleanup_images():
    for c in (CLASSIFY_CONTAINER, SEG_CONTAINER):
        subprocess.run(["docker", "rm", "-f", c],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in (CLASSIFY_IMAGE, SEG_IMAGE):
        subprocess.run(["docker", "rmi", "-f", i],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    job["log"].append("Containers and images removed (build cache and the "
                      f"weights volume '{CACHE_VOLUME}' are kept).")


def check_cancel():
    if job["cancel"]:
        raise RuntimeError("Cancelled by the user.")


# ----------------------------------------------------------------------------
# the job (one mode-1 or mode-2 run)
# ----------------------------------------------------------------------------
def do_job(payload):
    mode = payload.get("mode", "2")
    species = payload.get("species") or ""
    token = payload.get("token") or ""
    try:
        job["log"].append("=" * 60)
        job["log"].append(" Forensic fly wing species identification")
        job["log"].append("=" * 60)

        r = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError("Docker is not running.")
        arch = docker_arch()
        platform_flags = []
        if arch in ("arm64", "aarch64"):
            platform_flags = ["--platform", "linux/amd64"]
            job["log"].append("Note: ARM host - the x86_64 classification "
                              "image runs through emulation; SAM3 runs natively.")

        # ---- stage 1: segmentation (mode 1 only) --------------------------
        csv_name = None
        if mode == "1":
            job["phase"] = "segmentation"
            photo = payload["photo"]            # already inside WORK_DIR
            stem = image_stem(photo)
            prompts_name = f"_prompts_{stem}.json"
            write_points_json(os.path.join(WORK_DIR, prompts_name),
                              payload["points"])
            out_csv = f"contours_coordinates_{stem}.csv"
            masks_dir = f"masks_{stem}"

            job["log"].append("")
            job["log"].append("=== Building the SAM3 segmentation image ===")
            check_cancel()
            # build: the docker CLI inside THIS container reads the
            # context, so the container-side path (WORK_DIR) is correct
            rc, _ = run_capture(["docker", "build", "-f",
                                  os.path.join(WORK_DIR, "Dockerfile.segment"),
                                  "-t", SEG_IMAGE, WORK_DIR])
            if rc != 0:
                raise RuntimeError("Segmentation image build failed.")

            job["log"].append("=== Segmenting the wing cells ===")
            job["log"].append("First run downloads ~3.4 GB of gated weights "
                              "once (kept in the Docker volume).")
            check_cancel()
            cmd = ["docker", "run", "--rm", "--name", SEG_CONTAINER,
                   "-v", f"{HOST_WORK_DIR}:/work", "-w", "/work",
                   "-v", f"{CACHE_VOLUME}:/root/.cache/huggingface"]
            if token:
                cmd += ["-e", f"HF_TOKEN={token}"]
            cmd += [SEG_IMAGE, photo,
                    "--points", prompts_name,
                    "--out", out_csv,
                    "--masks-dir", masks_dir]
            rc, seg_lines = run_capture(cmd)
            check_cancel()
            if rc != 0:
                raise RuntimeError("Segmentation run failed "
                                   "(token accepted? weights downloaded?)")
            seg_text = "\n".join(seg_lines)
            timings, csv_from_log, n_rows = parse_segment_bench(seg_text)
            if not csv_from_log:
                raise RuntimeError("Segmentation did not produce a CSV.")
            job["log"].append("")
            job["log"].append(f"Segmentation finished: {csv_from_log} "
                              f"({n_rows} rows)")
            if timings:
                job["log"].append(pretty_bench(timings))

            # automatic mask QC (structure + overlays)
            job["log"].append("")
            job["log"].append("=== Mask quality control ===")
            rc, qc_lines = run_capture([
                "docker", "run", "--rm", "-v", f"{HOST_WORK_DIR}:/work",
                "-w", "/work", "--entrypoint", "python3", SEG_IMAGE,
                "qc_masks.py", photo,
                "--masks-dir", masks_dir, "--prompts", prompts_name])
            csv_name = csv_from_log
            job["result_artifacts"] = {
                "csv": csv_from_log,
                "masks_dir": masks_dir,
                "overlay_pa2r": "qc_overlay_pa2r.png",
                "overlay_dm": "qc_overlay_dm.png",
            }

        else:
            job["phase"] = "classification"
            csv_name = payload["csv"]
            job["result_artifacts"] = {"csv": csv_name}

        # ---- stage 2: classification ---------------------------------------
        job["phase"] = "classification"
        job["log"].append("")
        job["log"].append(f"Sample : {csv_name}")
        job["log"].append(f"Species : {species if species else 'unknown (top-3)'}")
        job["log"].append("")
        job["log"].append("=== Building the classification image ===")
        check_cancel()
        rc, _ = run_capture(["docker", "build"] + platform_flags +
                            ["-f",
                             os.path.join(WORK_DIR, "Dockerfile.workflow"),
                             "-t", CLASSIFY_IMAGE, WORK_DIR])
        if rc != 0:
            raise RuntimeError("Classification image build failed.")

        job["log"].append("=== Classifying ===")
        check_cancel()
        cmd = ["docker", "run", "--rm", "--name", CLASSIFY_CONTAINER] + \
            platform_flags + [CLASSIFY_IMAGE, csv_name]
        if species:
            cmd.append(species)
        rc, clf_lines = run_capture(cmd)
        check_cancel()
        if rc != 0:
            raise RuntimeError("Classification run failed.")
        result = parse_output("\n".join(clf_lines))
        if result["mode"] == "known" and result["predicted"] is None:
            raise RuntimeError("Could not read the prediction from the output.")

        job["result"] = result
        job["state"] = "done"
    except Exception as exc:                                  # noqa: BLE001
        job["error"] = str(exc)
        job["state"] = "error"
    finally:
        job["phase"] = ""
        cleanup_images()


# ----------------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           species=[UNKNOWN_LABEL] + VALID_SPECIES)


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    kind = request.form.get("kind", "photo")
    if f is None or not f.filename:
        return jsonify(ok=False, error="No file received."), 400
    name = secure_filename(f.filename)
    if not name:
        return jsonify(ok=False, error="Invalid file name."), 400
    base, ext = os.path.splitext(name)
    ext = ext.lower()
    if kind == "photo" and ext not in PHOTO_EXT:
        return jsonify(ok=False, error="Not a photo file."), 400
    if kind == "csv":
        if ext != ".csv":
            return jsonify(ok=False, error="Not a CSV file."), 400
        # keep the classification naming convention, but do not
        # prefix files that already follow it
        if not name.startswith("contours_coordinates"):
            name = f"contours_coordinates_{base}.csv"
    f.save(os.path.join(WORK_DIR, name))
    return jsonify(ok=True, name=name)


@app.route("/files/<path:name>")
def files(name):
    name = os.path.basename(name)
    if os.path.splitext(name)[1].lower() not in SERVE_EXT:
        return jsonify(ok=False, error="Not allowed."), 400
    try:
        return send_from_directory(WORK_DIR, name, as_attachment=
                                   "download" in request.args)
    except FileNotFoundError:
        return jsonify(ok=False, error="No such file."), 404


@app.route("/api/species")
def species_list():
    return jsonify(species=[UNKNOWN_LABEL] + VALID_SPECIES)


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True)
    with job_lock:
        if job["state"] == "running":
            return jsonify(ok=False, error="A run is already in progress."), 409
        job["state"] = "running"
        job["phase"] = "preparing"
        job["log"] = []
        job["result"] = None
        job["error"] = None
        job["cancel"] = False
        job["result_artifacts"] = {}
    threading.Thread(target=do_job, args=(data,), daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/status")
def api_status():
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0
    with job_lock:
        payload = {
            "state": job["state"],
            "phase": job["phase"],
            "offset": len(job["log"]),
            "lines": job["log"][offset:],
            "result": job["result"] if job["state"] == "done" else None,
            "artifacts": job.get("result_artifacts") or {},
            "error": job["error"],
        }
    return jsonify(payload)


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    with job_lock:
        job["cancel"] = True
        proc = job.get("proc")
    if proc:
        subprocess.run(["docker", "rm", "-f", CLASSIFY_CONTAINER,
                        SEG_CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify(ok=True)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(WORK_DIR, exist_ok=True)
    print(f"Serving  http://localhost:{PORT}  (work dir: {WORK_DIR})")
# NOTE: bind 0.0.0.0 inside the container so the published port reaches
# the server. The security boundary is the HOST-side mapping
# (-p 127.0.0.1:8080:8080 in start_web.sh/.ps1) - the page is never
# exposed to the network.
    app.run(host="0.0.0.0", port=PORT, threaded=True)
