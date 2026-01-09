"""Video encoding helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

try:
  import imageio_ffmpeg
except ImportError:  
  imageio_ffmpeg = None

from frame_optimization_methods.gpu_acceleration import get_gpu_info, get_hardware_encoder

PathLike = Union[str, Path]


def convert_to_h264(video_path: PathLike,
                    *,
                    crf: int = 18,
                    preset: str = "medium",
                    use_gpu: bool = True) -> Path:
  """Re-encodes an MP4 file to H.264 using ffmpeg and returns the path.
  
  Automatically uses hardware-accelerated encoding when available:
  - NVIDIA: h264_nvenc
  - AMD: h264_amf
  - Intel: h264_qsv
  - Apple Silicon: h264_videotoolbox
  - Fallback: libx264 (CPU)
  
  Args:
    video_path: Path to input video file
    crf: Constant Rate Factor (lower = higher quality, 18-28 recommended)
    preset: Encoding preset (ultrafast to veryslow)
    use_gpu: Whether to attempt GPU-accelerated encoding (default: True)
  
  Returns:
    Path to the encoded video file
  """
  target = Path(video_path)
  if not target.exists():
    raise FileNotFoundError(f"Video not found for transcoding: {target}")

  ffmpeg_path = shutil.which("ffmpeg")
  if ffmpeg_path is None and imageio_ffmpeg is not None:
    try:
      ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  
      ffmpeg_path = None
      last_error = exc
    else:
      last_error = None
  else:
    last_error = None

  if ffmpeg_path is None:
    hint = ("Install ffmpeg system-wide (e.g. `apt install ffmpeg`, `brew install ffmpeg`)"
            " or add the `imageio-ffmpeg` package to your environment.")
    if last_error:
      hint = f"{hint} (imageio-ffmpeg lookup failed: {last_error})"
    raise RuntimeError(f"ffmpeg binary is required to transcode videos to H.264. {hint}")

  with tempfile.NamedTemporaryFile(prefix=f"{target.stem}_h264_",
                                   suffix=target.suffix,
                                   dir=str(target.parent),
                                   delete=False) as tmp_file:
    temp_path = Path(tmp_file.name)

  # Detect GPU and get hardware encoder
  encoder = "libx264"  # Default CPU encoder
  encoder_params = []
  
  if use_gpu:
    gpu_info = get_gpu_info()
    hw_encoder = get_hardware_encoder(gpu_info)
    
    if hw_encoder:
      encoder = hw_encoder
      # Hardware encoder-specific parameters
      if encoder == "h264_nvenc":
        # NVIDIA NVENC
        encoder_params = [
          "-preset", "p4",  # NVENC preset (p1-p7, p4=medium)
          "-rc", "vbr",
          "-cq", str(crf),
          "-b:v", "0",  # Let CRF control bitrate
        ]
        preset = None  # NVENC uses its own preset system
      elif encoder == "h264_amf":
        # AMD AMF
        encoder_params = [
          "-quality", "balanced",  # speed, balanced, quality
          "-rc", "vbr_peak",
          "-qmin", str(max(18, crf - 5)),
          "-qmax", str(min(51, crf + 5)),
        ]
        preset = None
      elif encoder == "h264_qsv":
        # Intel QuickSync
        encoder_params = [
          "-preset", preset if preset else "medium",
          "-global_quality", str(crf),
        ]
      elif encoder == "h264_videotoolbox":
        # Apple VideoToolbox
        encoder_params = [
          "-allow_sw", "1",  # Allow software fallback
          "-realtime", "0",
          "-quality", "1",  # 0=best, 1=realtime, 2=worst
        ]
        # VideoToolbox uses bitrate, estimate from CRF
        # Rough conversion: CRF 18 ≈ 8Mbps, CRF 23 ≈ 2Mbps, CRF 28 ≈ 0.5Mbps
        estimated_bitrate = max(500, int(8000 * (28 - crf) / 10))
        encoder_params.extend(["-b:v", f"{estimated_bitrate}k"])
        preset = None

  # Build ffmpeg command
  cmd = [
      ffmpeg_path,
      "-y",
      "-i",
      str(target),
      "-c:v",
      encoder,
  ]
  
  # Add encoder-specific parameters
  cmd.extend(encoder_params)
  
  # Add preset if using CPU encoder or if not overridden
  if preset and encoder == "libx264":
    cmd.extend(["-preset", preset])
  
  # Add CRF for CPU encoder or if not already set
  if encoder == "libx264":
    cmd.extend(["-crf", str(crf)])
  
  # Common parameters
  cmd.extend([
      "-movflags",
      "+faststart",
      "-pix_fmt",
      "yuv420p",
      str(temp_path),
  ])

  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    message = result.stderr.strip() or result.stdout.strip() or "Unknown ffmpeg error."
    
    # If hardware encoder failed, try CPU fallback
    if encoder != "libx264" and use_gpu:
      print(f"Hardware encoder {encoder} failed, falling back to CPU encoding...")
      return convert_to_h264(video_path, crf=crf, preset=preset, use_gpu=False)
    
    temp_path.unlink(missing_ok=True)
    raise RuntimeError(f"ffmpeg failed to convert {target.name} to H.264: {message}")

  temp_path.replace(target)
  return target
