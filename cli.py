

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

from frame_optimization_methods.frameDifference import (
    remove_dead_frames as remove_dead_frames_frame_difference,
)
from frame_optimization_methods.opticalFlow import (
    remove_dead_frames as remove_dead_frames_optical_flow,
)
from frame_optimization_methods.ssim import process_video as process_video_ssim
from frame_optimization_methods.unsupervised_dedup import deduplicate_frames
from frame_optimization_methods.video_encoding import convert_to_h264
from frame_optimization_methods.presets import UNSUPERVISED_PRESETS


@contextlib.contextmanager
def _working_directory(target: Path) -> Iterator[None]:
  previous = Path.cwd()
  os.chdir(target)
  try:
    yield
  finally:
    os.chdir(previous)


def _default_output_path(input_path: Path, method: str) -> Path:
  suffix = {
      "optical_flow": "_opticalFlow.mp4",
      "frame_difference": "_frameDifference.mp4",
      "ssim": "_ssim.mp4",
      "unsupervised_dedup": "_unsupervisedDedup.mp4",
  }[method]
  return input_path.with_name(f"{input_path.stem}{suffix}")


def _ensure_parent(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_input(path_like: str) -> Path:
  path = Path(path_like).expanduser()
  if not path.exists():
    raise FileNotFoundError(f"Input video not found: {path}")
  if not path.is_file():
    raise FileNotFoundError(f"Input path is not a file: {path}")
  return path.resolve()


def _apply_encoding(path: Path, crf: Optional[int], preset: Optional[str]) -> Path:
  if crf is None and preset is None:
    return path
  applied_crf = 18 if crf is None else crf
  applied_preset = "medium" if preset is None else preset
  print(f"Re-encoding output with H.264 (crf={applied_crf}, preset={applied_preset})...")
  converted = convert_to_h264(path, crf=applied_crf, preset=applied_preset)
  return Path(converted)


def _run_optical_flow(input_path: Path, flow_mag_threshold: float) -> Path:
  with _working_directory(input_path.parent):
    remove_dead_frames_optical_flow(input_path.name, flow_mag_threshold)
  product = input_path.parent / f"{input_path.stem}_opticalFlow.mp4"
  if not product.exists():
    raise FileNotFoundError("Optical flow processing did not produce an output file.")
  return product


def _run_frame_difference(input_path: Path, base_threshold: float) -> Path:
  with _working_directory(input_path.parent):
    remove_dead_frames_frame_difference(input_path.name, base_threshold)
  product = input_path.parent / f"{input_path.stem}_frameDifference.mp4"
  if not product.exists():
    raise FileNotFoundError("Frame difference processing did not produce an output file.")
  return product


def _run_ssim(input_path: Path, output_path: Path, ssim_threshold: float) -> Path:
  _ensure_parent(output_path)
  process_video_ssim(str(input_path), ssim_threshold, str(output_path))
  if not output_path.exists():
    raise FileNotFoundError("SSIM processing did not produce an output file.")
  return output_path


def _run_unsupervised_dedup(input_path: Path, params: dict[str, float]) -> Path:
  output_location = deduplicate_frames(
      str(input_path),
      hash_threshold=int(params["hash_threshold"]),
      ordinal_footrule_threshold=float(params["ordinal_footrule_threshold"]),
      feature_similarity=float(params["feature_similarity"]),
      flow_static_threshold=float(params["flow_static_threshold"]),
      flow_low_ratio=float(params["flow_low_ratio"]),
      pan_orientation_std=float(params["pan_orientation_std"]),
      safety_keep_seconds=float(params["safety_keep_seconds"]),
  )
  product = Path(output_location)
  if not product.is_absolute():
    product = (Path.cwd() / product).resolve()
  if not product.exists():
    raise FileNotFoundError("Unsupervised deduplication did not produce an output file.")
  return product


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run AFOptimizer frame processing methods from the command line.")
  parser.add_argument("input_video", help="Path to the source video file.")
  parser.add_argument("-o", "--output", help="Destination path for the processed video.")
  parser.add_argument(
      "--encoding-crf",
      type=int,
      help="Optional CRF value for final H.264 re-encode (lower is higher quality).",
  )
  parser.add_argument(
      "--encoding-preset", help="Optional ffmpeg preset for the final re-encode (e.g. veryslow)."
  )

  subparsers = parser.add_subparsers(dest="method", required=True)

  optical = subparsers.add_parser("optical-flow", help="Farnebäck dense optical flow filtering")
  optical.add_argument(
      "--flow-mag-threshold",
      type=float,
      default=0.4,
      help="Mean optical flow magnitude threshold for keeping a frame (default: 0.4).",
  )

  frame_diff = subparsers.add_parser(
      "frame-difference", help="Pixel-wise frame differencing filter")
  frame_diff.add_argument(
      "--base-threshold",
      type=float,
      default=10.0,
      help="Base pixel difference threshold used to seed adaptive motion scoring (default: 10).",
  )

  ssim = subparsers.add_parser("ssim", help="Structural Similarity Index pruning")
  ssim.add_argument(
      "--ssim-threshold",
      type=float,
      default=0.9587,
      help="Maximum SSIM similarity allowed before a frame is treated as redundant (default: 0.9587).",
  )

  unsupervised = subparsers.add_parser(
      "unsupervised-dedup", help="Three-stage perceptual deduplication pipeline")
  unsupervised.add_argument(
      "--profile",
      choices=sorted(UNSUPERVISED_PRESETS.keys()),
      default="balanced",
      help="Preset tuned for gentle/balanced/aggressive deduplication (default: balanced).",
  )
  unsupervised.add_argument(
      "--hash-threshold",
      type=int,
      help="Maximum Hamming distance for Walsh-Hadamard hashes before flagging a duplicate.",
  )
  unsupervised.add_argument(
      "--ordinal-footrule-threshold",
      type=float,
      help="Footrule distance threshold for ordinal texture signatures.",
  )
  unsupervised.add_argument(
      "--feature-similarity",
      type=float,
      help="Minimum ORB feature match ratio required to treat frames as equivalent.",
  )
  unsupervised.add_argument(
      "--flow-static-threshold",
      type=float,
      help="Maximum mean optical flow magnitude to treat a window as static.",
  )
  unsupervised.add_argument(
      "--flow-low-ratio",
      type=float,
      help="Required ratio of low-motion pixels to trigger static suppression.",
  )
  unsupervised.add_argument(
      "--pan-orientation-std",
      type=float,
      help="Orientation standard deviation below which pans are considered redundant.",
  )
  unsupervised.add_argument(
      "--safety-keep-seconds",
      type=float,
      help="Minimum seconds between forced keyframes to avoid over-pruning.",
  )

  return parser.parse_args(argv)


def _gather_unsupervised_params(args: argparse.Namespace) -> dict[str, float]:
  params = dict(UNSUPERVISED_PRESETS[args.profile])
  override_fields = (
      "hash_threshold",
      "ordinal_footrule_threshold",
      "feature_similarity",
      "flow_static_threshold",
      "flow_low_ratio",
      "pan_orientation_std",
      "safety_keep_seconds",
  )
  for field in override_fields:
    value = getattr(args, field)
    if value is not None:
      params[field] = value
  return params


def _finalize_output(source_path: Path, desired_path: Optional[str]) -> Path:
  if not desired_path:
    return source_path
  target = Path(desired_path).expanduser()
  _ensure_parent(target)
  if source_path.resolve() == target.resolve():
    return source_path
  source_path.replace(target)
  return target


def main(argv: Optional[Iterable[str]] = None) -> int:
  args = _parse_args(argv)
  try:
    input_path = _resolve_input(args.input_video)
    method_key = args.method.replace("-", "_")

    if method_key == "ssim":
      output_hint = (Path(args.output).expanduser()
                     if args.output else _default_output_path(input_path, method_key))
      product = _run_ssim(input_path, output_hint, args.ssim_threshold)
    elif method_key == "optical_flow":
      product = _run_optical_flow(input_path, args.flow_mag_threshold)
      product = _finalize_output(product, args.output)
    elif method_key == "frame_difference":
      product = _run_frame_difference(input_path, args.base_threshold)
      product = _finalize_output(product, args.output)
    elif method_key == "unsupervised_dedup":
      params = _gather_unsupervised_params(args)
      product = _run_unsupervised_dedup(input_path, params)
      product = _finalize_output(product, args.output)
    else:
      raise RuntimeError(f"Unhandled method: {args.method}")

    product = _apply_encoding(product, args.encoding_crf, args.encoding_preset)
    print(f"✔ Processed video saved to: {product}")
    return 0
  except Exception as exc:  # pylint: disable=broad-except
    print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
