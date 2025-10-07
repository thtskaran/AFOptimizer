import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from frame_optimization_methods.opticalFlow import remove_dead_frames as remove_dead_frames_of
from frame_optimization_methods.frameDifference import remove_dead_frames as remove_dead_frames_fd
from frame_optimization_methods.ssim import process_video as process_video_ssim
from frame_optimization_methods.unsupervised_dedup import deduplicate_frames as deduplicate_frames_unsupervised

ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "uploads"
PROCESSED_DIR = ROOT_DIR / "outputs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

UNSUPERVISED_PRESETS = {
    "gentle": {
        "hash_threshold": 6,
        "ordinal_footrule_threshold": 220.0,
        "feature_similarity": 0.30,
        "flow_static_threshold": 0.08,
        "flow_low_ratio": 0.98,
        "pan_orientation_std": 0.60,
        "safety_keep_seconds": 1.0,
    },
    "balanced": {
        "hash_threshold": 8,
        "ordinal_footrule_threshold": 260.0,
        "feature_similarity": 0.26,
        "flow_static_threshold": 0.09,
        "flow_low_ratio": 0.97,
        "pan_orientation_std": 0.65,
        "safety_keep_seconds": 1.5,
    },
    "aggressive": {
        "hash_threshold": 12,
        "ordinal_footrule_threshold": 320.0,
        "feature_similarity": 0.22,
        "flow_static_threshold": 0.12,
        "flow_low_ratio": 0.94,
        "pan_orientation_std": 0.80,
        "safety_keep_seconds": 2.5,
    },
}

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

JobDict = Dict[str, object]
JOBS: Dict[str, JobDict] = {}
JOBS_LOCK = threading.Lock()


def _init_job(method: str, source_filename: str) -> str:
  job_id = uuid.uuid4().hex
  now = time.time()
  job_data: JobDict = {
      "id": job_id,
      "status": "queued",
      "method": method,
      "source_file": source_filename,
      "message": "Queued",
      "stage": "Queued",
      "progress": 0.0,
      "processed_frames": 0,
      "total_frames": None,
      "start_time": None,
      "end_time": None,
      "elapsed_seconds": 0.0,
      "eta_seconds": None,
      "fps": None,
      "output_file": None,
      "download_url": None,
      "error": None,
      "created_at": now,
      "last_update": now,
  }
  with JOBS_LOCK:
    JOBS[job_id] = job_data
  return job_id


def _get_job(job_id: str) -> Optional[JobDict]:
  with JOBS_LOCK:
    job = JOBS.get(job_id)
    return dict(job) if job else None


def _update_job(job_id: str, **updates) -> Optional[JobDict]:
  with JOBS_LOCK:
    job = JOBS.get(job_id)
    if not job:
      return None
    job.update(updates)
    job["last_update"] = time.time()
    return dict(job)


def _record_progress(job_id: str,
                     processed: int,
                     total: Optional[int],
                     stage: Optional[str] = None) -> None:
  with JOBS_LOCK:
    job = JOBS.get(job_id)
    if not job:
      return

    total_frames = int(total) if total else 0
    if total_frames <= 0 and processed >= 0:
      total_frames = max(processed, 1)

    job["total_frames"] = total_frames
    job["processed_frames"] = max(0, min(int(processed), total_frames))
    if stage:
      job["stage"] = stage

    now = time.time()
    job["last_update"] = now

    start_time = job.get("start_time")
    if start_time:
      elapsed = max(0.0, now - start_time)
      job["elapsed_seconds"] = elapsed
      fps = job["processed_frames"] / elapsed if elapsed > 0 else 0.0
      job["fps"] = fps if fps > 0 else None
      remaining = ((total_frames - job["processed_frames"])
                   / fps) if fps else None
      job["eta_seconds"] = max(0.0, remaining) if remaining is not None else None

    if total_frames > 0:
      job["progress"] = (job["processed_frames"] / total_frames) * 100.0
    else:
      job["progress"] = 0.0


def _mark_job_running(job_id: str, message: str) -> None:
  _update_job(job_id,
              status="running",
              message=message,
              stage="Processing",
              start_time=time.time(),
              elapsed_seconds=0.0,
              eta_seconds=None,
              progress=0.0)


def _finalize_job_success(job_id: str, output_path: Path) -> None:
  job_snapshot = _get_job(job_id) or {}
  total_frames = job_snapshot.get("total_frames") or job_snapshot.get("processed_frames") or 0
  if total_frames:
    _record_progress(job_id, total_frames, total_frames)
  _update_job(job_id,
              status="completed",
              message="Video processed successfully.",
              stage="Completed",
              end_time=time.time(),
              eta_seconds=0.0,
              output_file=output_path.name,
              download_url=f"/download/{output_path.name}")


def _finalize_job_failure(job_id: str, error_message: str) -> None:
  _update_job(job_id,
              status="failed",
              message=error_message,
              stage="Failed",
              end_time=time.time(),
              eta_seconds=None,
              error=error_message)


def _safe_unlink(target: Path) -> None:
  try:
    target.unlink(missing_ok=True)
  except TypeError:
    if target.exists():
      target.unlink()
  except OSError:
    pass


def _run_job(job_id: str, upload_path: Path, method: str,
             params: Dict[str, float]) -> None:
  try:
    _mark_job_running(job_id, "Processing video")
    _record_progress(job_id, 0, None, "Preparing frames")

    def progress_callback(processed: int,
                          total: Optional[int],
                          stage: Optional[str] = None) -> None:
      _record_progress(job_id, processed, total, stage)

    if method == "opticalFlow":
      flow_mag_threshold = params["flow_mag_threshold"]
      output_path = _run_optical_flow(upload_path, flow_mag_threshold,
                                      progress_callback)
    elif method == "frameDifference":
      base_threshold = params["base_threshold"]
      output_path = _run_frame_difference(upload_path, base_threshold,
                                          progress_callback)
    elif method == "ssim":
      ssim_threshold = params["ssim_threshold"]
      output_path = _run_ssim(upload_path, ssim_threshold, progress_callback)
    elif method == "unsupervisedDedup":
      hash_threshold = int(params["hash_threshold"])
      ordinal_footrule = float(params["ordinal_footrule_threshold"])
      feature_similarity = float(params["feature_similarity"])
      flow_static_threshold = float(params["flow_static_threshold"])
      flow_low_ratio = float(params["flow_low_ratio"])
      pan_orientation_std = float(params["pan_orientation_std"])
      safety_keep_seconds = float(params["safety_keep_seconds"])
      output_path = _run_unsupervised_dedup(
          upload_path,
          hash_threshold,
          ordinal_footrule,
          feature_similarity,
          flow_static_threshold,
          flow_low_ratio,
          pan_orientation_std,
          safety_keep_seconds,
          progress_callback)
    else:
      raise ValueError("Unknown optimization method.")

    _finalize_job_success(job_id, output_path)
  except Exception as exc:  # pylint: disable=broad-except
    _finalize_job_failure(job_id, str(exc))
  finally:
    _safe_unlink(upload_path)


def allowed_file(filename: str) -> bool:
  return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _run_optical_flow(
    video_path: Path,
    flow_mag_threshold: float,
    progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None
) -> Path:
  remove_dead_frames_of(str(video_path), flow_mag_threshold,
                        progress_callback=progress_callback)
  output_name = f"{video_path.stem}_opticalFlow.mp4"
  output_path = ROOT_DIR / output_name
  if not output_path.exists():
    raise FileNotFoundError(
        f"Optical flow processing did not produce expected file: {output_name}")
  final_path = PROCESSED_DIR / output_path.name
  output_path.replace(final_path)
  return final_path


def _run_frame_difference(
    video_path: Path,
    base_threshold: float,
    progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None
) -> Path:
  remove_dead_frames_fd(str(video_path), base_threshold,
                        progress_callback=progress_callback)
  output_name = f"{video_path.stem}_frameDifference.mp4"
  output_path = ROOT_DIR / output_name
  if not output_path.exists():
    raise FileNotFoundError(
        f"Frame difference processing did not produce expected file: {output_name}")
  final_path = PROCESSED_DIR / output_path.name
  output_path.replace(final_path)
  return final_path


def _run_ssim(
    video_path: Path,
    ssim_threshold: float,
    progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None
) -> Path:
  output_name = f"{video_path.stem}_ssim.mp4"
  final_path = PROCESSED_DIR / output_name
  process_video_ssim(str(video_path), ssim_threshold, str(final_path),
                     progress_callback=progress_callback)
  if not final_path.exists():
    raise FileNotFoundError(
        f"SSIM processing did not produce expected file: {output_name}")
  return final_path


def _run_unsupervised_dedup(
    video_path: Path,
    hash_threshold: int,
    ordinal_footrule_threshold: float,
    feature_similarity: float,
    flow_static_threshold: float,
    flow_low_ratio: float,
    pan_orientation_std: float,
    safety_keep_seconds: float,
    progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None
) -> Path:
  output_location = deduplicate_frames_unsupervised(
      str(video_path),
      hash_threshold=hash_threshold,
      ordinal_footrule_threshold=ordinal_footrule_threshold,
      feature_similarity=feature_similarity,
      flow_static_threshold=flow_static_threshold,
      flow_low_ratio=flow_low_ratio,
      pan_orientation_std=pan_orientation_std,
      safety_keep_seconds=safety_keep_seconds,
      progress_callback=progress_callback)
  output_path = Path(output_location)
  if not output_path.exists():
    raise FileNotFoundError(
        "Unsupervised deduplication did not produce an output file.")
  final_path = PROCESSED_DIR / output_path.name
  output_path.replace(final_path)
  return final_path


def _save_upload(upload_file) -> Path:
  timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
  secure_name = secure_filename(upload_file.filename)
  stored_name = f"{timestamp}_{secure_name}" if secure_name else timestamp
  target_path = UPLOAD_DIR / stored_name
  upload_file.save(target_path)
  return target_path


@app.route("/")
def index():
  return render_template("index.html", output_dir=str(PROCESSED_DIR))


@app.route("/api/process", methods=["POST"])
def process_video():
  if "video" not in request.files:
    return jsonify({"error": "No video file provided."}), 400

  file = request.files["video"]
  method = request.form.get("method")

  if not file or file.filename == "":
    return jsonify({"error": "Please choose a video file to upload."}), 400

  if not method:
    return jsonify({"error": "Please select an optimization method."}), 400

  if not allowed_file(file.filename):
    allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
    return jsonify({"error": f"Unsupported file type. Allowed: {allowed}"}), 400

  try:
    upload_path = _save_upload(file)
  except OSError as exc:
    return jsonify({"error": f"Failed to save upload: {exc}"}), 500

  try:
    parameters: Dict[str, float] = {}
    if method == "opticalFlow":
      flow_mag_threshold = float(request.form.get("flow_mag_threshold", 0.4))
      parameters["flow_mag_threshold"] = flow_mag_threshold
    elif method == "frameDifference":
      base_threshold = float(request.form.get("base_threshold", 10))
      parameters["base_threshold"] = base_threshold
    elif method == "ssim":
      ssim_threshold = float(request.form.get("ssim_threshold", 0.9587))
      parameters["ssim_threshold"] = ssim_threshold
    elif method == "unsupervisedDedup":
      profile = (request.form.get("dedup_profile", "balanced").strip().lower()
                 or "balanced")
      preset = UNSUPERVISED_PRESETS.get(profile,
                                        UNSUPERVISED_PRESETS["balanced"])
      parameters.update({"profile": profile, **preset})
    else:
      _safe_unlink(upload_path)
      return jsonify({"error": "Unknown optimization method."}), 400
  except ValueError:
    _safe_unlink(upload_path)
    return jsonify({"error": "Invalid numeric parameter provided."}), 400
  except Exception as exc:  # pylint: disable=broad-except
    _safe_unlink(upload_path)
    return jsonify({"error": str(exc)}), 500

  job_id = _init_job(method, file.filename or upload_path.name)
  _update_job(job_id, parameters=parameters)

  worker = threading.Thread(target=_run_job,
                            args=(job_id, upload_path, method, parameters),
                            daemon=True)
  worker.start()

  job_snapshot = _get_job(job_id) or {}
  return jsonify({
      "jobId": job_id,
      "status": job_snapshot.get("status", "queued"),
      "message": "Processing started.",
      "progressUrl": url_for("progress_status", job_id=job_id)
  }), 202


@app.route("/api/progress/<string:job_id>")
def progress_status(job_id: str):
  job = _get_job(job_id)
  if not job:
    return jsonify({"error": "Job not found."}), 404
  return jsonify(job)


@app.route("/download/<path:filename>")
def download_file(filename):
  file_path = PROCESSED_DIR / filename
  if not file_path.exists():
    return jsonify({"error": "File not found."}), 404
  return send_from_directory(PROCESSED_DIR, filename, as_attachment=True)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
