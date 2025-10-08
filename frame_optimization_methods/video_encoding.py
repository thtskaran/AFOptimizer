"""Video encoding helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

try:
  import imageio_ffmpeg
except ImportError:  
  imageio_ffmpeg = None

PathLike = Union[str, Path]


def convert_to_h264(video_path: PathLike,
                    *,
                    crf: int = 18,
                    preset: str = "medium") -> Path:
  """Re-encodes an MP4 file to H.264 using ffmpeg and returns the path."""
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

  cmd = [
      ffmpeg_path,
      "-y",
      "-i",
      str(target),
      "-c:v",
      "libx264",
      "-preset",
      preset,
      "-crf",
      str(crf),
      "-movflags",
      "+faststart",
      "-pix_fmt",
      "yuv420p",
      str(temp_path),
  ]

  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    message = result.stderr.strip() or result.stdout.strip() or "Unknown ffmpeg error."
    temp_path.unlink(missing_ok=True)
    raise RuntimeError(f"ffmpeg failed to convert {target.name} to H.264: {message}")

  temp_path.replace(target)
  return target
