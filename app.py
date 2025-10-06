import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from frame_optimization_methods.opticalFlow import remove_dead_frames as remove_dead_frames_of
from frame_optimization_methods.frameDifference import remove_dead_frames as remove_dead_frames_fd
from frame_optimization_methods.ssim import process_video as process_video_ssim

ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "uploads"
PROCESSED_DIR = ROOT_DIR / "outputs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")


def allowed_file(filename: str) -> bool:
  return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _run_optical_flow(video_path: Path, flow_mag_threshold: float) -> Path:
  remove_dead_frames_of(str(video_path), flow_mag_threshold)
  output_name = f"{video_path.stem}_opticalFlow.mp4"
  output_path = ROOT_DIR / output_name
  if not output_path.exists():
    raise FileNotFoundError(
        f"Optical flow processing did not produce expected file: {output_name}")
  final_path = PROCESSED_DIR / output_path.name
  output_path.replace(final_path)
  return final_path


def _run_frame_difference(video_path: Path, base_threshold: float) -> Path:
  remove_dead_frames_fd(str(video_path), base_threshold)
  output_name = f"{video_path.stem}_frameDifference.mp4"
  output_path = ROOT_DIR / output_name
  if not output_path.exists():
    raise FileNotFoundError(
        f"Frame difference processing did not produce expected file: {output_name}")
  final_path = PROCESSED_DIR / output_path.name
  output_path.replace(final_path)
  return final_path


def _run_ssim(video_path: Path, ssim_threshold: float) -> Path:
  output_name = f"{video_path.stem}_ssim.mp4"
  final_path = PROCESSED_DIR / output_name
  process_video_ssim(str(video_path), ssim_threshold, str(final_path))
  if not final_path.exists():
    raise FileNotFoundError(
        f"SSIM processing did not produce expected file: {output_name}")
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
    if method == "opticalFlow":
      flow_mag_threshold = float(request.form.get("flow_mag_threshold", 0.4))
      output_path = _run_optical_flow(upload_path, flow_mag_threshold)
    elif method == "frameDifference":
      base_threshold = float(request.form.get("base_threshold", 10))
      output_path = _run_frame_difference(upload_path, base_threshold)
    elif method == "ssim":
      ssim_threshold = float(request.form.get("ssim_threshold", 0.9587))
      output_path = _run_ssim(upload_path, ssim_threshold)
    else:
      return jsonify({"error": "Unknown optimization method."}), 400
  except ValueError:
    return jsonify({"error": "Invalid numeric parameter provided."}), 400
  except Exception as exc:  # pylint: disable=broad-except
    return jsonify({"error": str(exc)}), 500
  finally:
    try:
      upload_path.unlink(missing_ok=True)
    except TypeError:
      # Python < 3.8 compatibility
      if upload_path.exists():
        upload_path.unlink()
    except OSError:
      pass

  download_url = url_for("download_file", filename=output_path.name, _external=False)
  return jsonify({
      "message": "Video processed successfully.",
      "downloadUrl": download_url,
      "outputFile": output_path.name
  })


@app.route("/download/<path:filename>")
def download_file(filename):
  file_path = PROCESSED_DIR / filename
  if not file_path.exists():
    return jsonify({"error": "File not found."}), 404
  return send_from_directory(PROCESSED_DIR, filename, as_attachment=True)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
