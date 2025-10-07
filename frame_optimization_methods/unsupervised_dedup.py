"""Unsupervised frame deduplication pipeline.

This module implements the three-stage pipeline documented in
"Unsupervised Frame Deduplication: Technical Design and Implementation Guide".
It combines perceptual hashing, local-feature clustering, and
motion-aware grouping to discard redundant frames without requiring
supervised training.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

ProgressCallback = Optional[Callable[[int, Optional[int], Optional[str]], None]]

# Hashing constants
_HASH_SIZE = 64  # 8x8 LF block -> 64 bits
_HASH_RESOLUTION = (64, 64)
_HASH_BLOCK = (8, 8)

# Flow constants
_FLOW_TARGET_WIDTH = 320
_FLOW_LOW_MAG_NORM = 0.2


def _fwht1d(vector: np.ndarray) -> np.ndarray:
  """Fast Walsh–Hadamard Transform for 1D arrays of power-of-two length."""
  output = vector.astype(np.float32, copy=True)
  length = output.shape[0]
  h = 1
  while h < length:
    step = h * 2
    for start in range(0, length, step):
      end = start + h
      tmp0 = output[start:end]
      tmp1 = output[end:end + h]
      output[start:end] = tmp0 + tmp1
      output[end:end + h] = tmp0 - tmp1
    h *= 2
  return output


def _fwht2d(matrix: np.ndarray) -> np.ndarray:
  """Applies FWHT along rows then columns."""
  transformed = matrix.astype(np.float32, copy=True)
  for row_idx in range(transformed.shape[0]):
    transformed[row_idx, :] = _fwht1d(transformed[row_idx, :])
  for col_idx in range(transformed.shape[1]):
    transformed[:, col_idx] = _fwht1d(transformed[:, col_idx])
  return transformed


def _compute_wht_hash(gray_frame: np.ndarray) -> int:
  resized = cv2.resize(gray_frame, _HASH_RESOLUTION, interpolation=cv2.INTER_AREA)
  smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
  transformed = _fwht2d(smoothed)
  block_h, block_w = _HASH_BLOCK
  low_freq_block = transformed[:block_h, :block_w].flatten()
  median = np.median(low_freq_block)
  bits = (low_freq_block > median).astype(np.uint8)
  hash_value = 0
  for bit in bits:
    hash_value = (hash_value << 1) | int(bit)
  return hash_value


def _compute_ordinal_signature(gray_frame: np.ndarray) -> np.ndarray:
  resized = cv2.resize(gray_frame, _HASH_RESOLUTION, interpolation=cv2.INTER_AREA)
  gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
  gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
  magnitude = cv2.magnitude(gx, gy)
  block_h, block_w = _HASH_BLOCK
  features = []
  for y in range(0, magnitude.shape[0], block_h):
    for x in range(0, magnitude.shape[1], block_w):
      block = magnitude[y:y + block_h, x:x + block_w]
      features.append(float(block.mean()))
  features_array = np.asarray(features, dtype=np.float32)
  order = np.argsort(features_array)
  ranks = np.empty_like(order)
  ranks[order] = np.arange(order.size, dtype=np.uint16)
  return ranks.astype(np.uint16)


def _hamming_distance(a: int, b: int) -> int:
  return int((a ^ b).bit_count())


def _ordinal_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
  return float(np.abs(sig_a.astype(np.int32) - sig_b.astype(np.int32)).sum())


def _prepare_flow_frame(gray_frame: np.ndarray) -> np.ndarray:
  height, width = gray_frame.shape[:2]
  if width <= _FLOW_TARGET_WIDTH:
    return gray_frame
  target_height = max(32, int(round(height * (_FLOW_TARGET_WIDTH / width))))
  return cv2.resize(gray_frame, (_FLOW_TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)


def _flow_metrics(prev_gray: np.ndarray,
                  curr_gray: np.ndarray) -> Tuple[float, float, float]:
  flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3,
                                      5, 1.1, 0)
  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
  mean_mag = float(magnitude.mean())
  low_mag_ratio = float(np.mean(magnitude < _FLOW_LOW_MAG_NORM))
  motion_mask = magnitude > 0.05
  if np.any(motion_mask):
    orientation_std = float(np.std(angle[motion_mask]))
  else:
    orientation_std = 0.0
  return mean_mag, low_mag_ratio, orientation_std


def _extract_orb_descriptors(frame: np.ndarray,
                             detector: cv2.ORB) -> Optional[np.ndarray]:
  keypoints, descriptors = detector.detectAndCompute(frame, None)
  if descriptors is None or len(descriptors) == 0:
    return None
  return descriptors


def _orb_similarity(reference: np.ndarray, candidate: np.ndarray,
                    matcher: cv2.BFMatcher) -> float:
  matches = matcher.match(candidate, reference)
  if not matches:
    return 0.0
  good = sum(1 for match in matches if match.distance <= 32)
  normalizer = max(1, min(len(candidate), len(reference)))
  return good / normalizer


def deduplicate_frames(
    video_path: str,
    hash_threshold: int = 8,
    ordinal_footrule_threshold: float = 260.0,
    feature_similarity: float = 0.26,
    flow_static_threshold: float = 0.09,
    flow_low_ratio: float = 0.97,
    pan_orientation_std: float = 0.65,
    safety_keep_seconds: float = 1.5,
    progress_callback: ProgressCallback = None) -> str:
  """Runs the unsupervised dedup pipeline and returns the output path."""

  source_path = Path(video_path)
  if not source_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

  output_path = source_path.with_name(f"{source_path.stem}_unsupervisedDedup.mp4")

  capture = cv2.VideoCapture(str(source_path))
  if not capture.isOpened():
    raise IOError(f"Unable to open video: {video_path}")

  total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
    total_frames = 0
  fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
  if not np.isfinite(fps) or fps <= 1e-3:
    fps = 24.0
  safety_keep_frames = max(1, int(round(fps * safety_keep_seconds)))

  success, frame = capture.read()
  if not success:
    capture.release()
    raise IOError("Failed to read first frame from video.")

  height, width = frame.shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
  if not writer.isOpened():
    capture.release()
    raise IOError(f"Unable to create output video at {output_path}")

  orb_detector = cv2.ORB_create(nfeatures=1000,
                                fastThreshold=12,
                                scoreType=cv2.ORB_FAST_SCORE)
  bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

  hash_history: Deque[Tuple[int, np.ndarray, int]] = deque(maxlen=4096)
  feature_clusters: Deque[Dict[str, object]] = deque(maxlen=96)

  processed_frames = 0
  written_frames = 0
  frames_since_last_kept = 0

  pbar = None
  if progress_callback is None:
    pbar = tqdm(total=total_frames or None,
                desc="Deduplicating frames",
                unit="frame")
  else:
    progress_callback(0, total_frames or None, "Bootstrapping pipeline")

  def notify(stage: str) -> None:
    nonlocal pbar
    if pbar is not None:
      pbar.update(1)
    elif progress_callback:
      progress_callback(processed_frames, total_frames or None, stage)

  # Process first frame (always keep)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  flow_gray = _prepare_flow_frame(gray)
  frame_hash = _compute_wht_hash(gray)
  ordinal_signature = _compute_ordinal_signature(gray)
  descriptors = _extract_orb_descriptors(gray, orb_detector)

  hash_history.append((frame_hash, ordinal_signature.copy(), 0))
  if descriptors is not None:
    feature_clusters.append({
        "index": 0,
        "descriptor": descriptors.copy()
    })

  writer.write(frame)
  written_frames += 1
  processed_frames += 1
  last_kept_flow_gray = flow_gray

  notify("Seeding reference frame")

  frame_index = 1

  max_hash_gap = max(24, int(round(fps * 6)))
  max_cluster_gap = max(12, int(round(fps * 4)))

  while True:
    success, frame = capture.read()
    if not success:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow_gray = _prepare_flow_frame(gray)
    frame_hash = _compute_wht_hash(gray)
    ordinal_signature = _compute_ordinal_signature(gray)

    processed_frames += 1
    stage_label = "Hash filter"
    is_duplicate = False

    # Stage 1: Hash-based near-duplicate detection
    for known_hash, known_ord, known_idx in reversed(hash_history):
      if frame_index - known_idx > max_hash_gap:
        break
      if _hamming_distance(frame_hash, known_hash) <= hash_threshold:
        if _ordinal_distance(ordinal_signature, known_ord) <= ordinal_footrule_threshold:
          is_duplicate = True
          break

    if is_duplicate:
      frames_since_last_kept += 1
      notify(stage_label)
      frame_index += 1
      continue

    # Stage 2: Local-feature clustering (ORB fallback)
    descriptors = _extract_orb_descriptors(gray, orb_detector)
    stage_label = "Feature clustering"
    matched_cluster = False
    if descriptors is not None and feature_clusters:
      for cluster in reversed(feature_clusters):
        if frame_index - int(cluster["index"]) > max_cluster_gap:
          continue
        reference = cluster["descriptor"]
        similarity = _orb_similarity(reference, descriptors, bf_matcher)
        if similarity >= feature_similarity:
          matched_cluster = True
          cluster["descriptor"] = descriptors.copy()
          cluster["index"] = frame_index
          break

    if matched_cluster:
      frames_since_last_kept += 1
      notify(stage_label)
      frame_index += 1
      continue

    # Stage 3: Motion-aware grouping
    stage_label = "Motion grouping"
    mean_flow, low_ratio, orientation_std = _flow_metrics(last_kept_flow_gray,
                                                         flow_gray)
    static_like = (mean_flow < flow_static_threshold and
                   low_ratio >= flow_low_ratio)

    pan_like = (low_ratio > 0.85 and orientation_std < pan_orientation_std)
    if static_like or pan_like:
      if frames_since_last_kept + 1 < safety_keep_frames:
        frames_since_last_kept += 1
        notify(stage_label)
        frame_index += 1
        continue
      stage_label = "Motion grouping (keyframe)"

    # Keep frame
    hash_history.append((frame_hash, ordinal_signature.copy(), frame_index))
    if descriptors is not None:
      feature_clusters.append({
          "index": frame_index,
          "descriptor": descriptors.copy()
      })

    writer.write(frame)
    written_frames += 1
    frames_since_last_kept = 0
    last_kept_flow_gray = flow_gray

    notify(stage_label)
    frame_index += 1

  capture.release()
  writer.release()
  if pbar is not None:
    pbar.close()

  if written_frames == 0:
    raise RuntimeError("No frames written to output video.")

  return str(output_path)
