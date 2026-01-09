from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from frame_optimization_methods.video_processor_base import VideoProcessorBase, ProgressCallback
from frame_optimization_methods.gpu_acceleration import GPUAccelerator


def calculate_initial_threshold(cap, sample_size, base_threshold, gpu_accel: Optional[GPUAccelerator] = None):
  total_movement = 0
  ret, prev_frame = cap.read()
  for _ in range(sample_size - 1):
    ret, current_frame = cap.read()
    if not ret:
      break
    # Use GPU acceleration if available
    if gpu_accel:
      prev_gray = gpu_accel.cvt_color(prev_frame, cv2.COLOR_BGR2GRAY)
      curr_gray = gpu_accel.cvt_color(current_frame, cv2.COLOR_BGR2GRAY)
      frame_diff = gpu_accel.absdiff(prev_gray, curr_gray)
    else:
      prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
      curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
      frame_diff = cv2.absdiff(prev_gray, curr_gray)
    total_movement += np.sum(frame_diff)
    prev_frame = current_frame

  average_movement = total_movement / ((cap.get(cv2.CAP_PROP_FRAME_HEIGHT) *
                                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)) *
                                       (sample_size - 1))
  return base_threshold + average_movement  # Adjust base_threshold based on average movement


def is_significant_movement(prev_frame, current_frame, threshold, gpu_accel: Optional[GPUAccelerator] = None):
  # Use GPU acceleration if available
  if gpu_accel:
    gray_prev = gpu_accel.cvt_color(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = gpu_accel.cvt_color(current_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = gpu_accel.absdiff(gray_prev, gray_current)
  else:
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_prev, gray_current)
  non_zero_count = np.count_nonzero(frame_diff > threshold)
  return non_zero_count > (gray_prev.shape[0] * gray_prev.shape[1] * 0.02)


class FrameDifferenceProcessor(VideoProcessorBase):
    """Frame difference-based video processor."""

    def process(self, video_path: str, base_threshold: float = 10.0,
                progress_callback: ProgressCallback = None) -> Optional[str]:
        """Process video using frame difference method.

        Args:
            video_path: Path to input video file
            base_threshold: Base threshold for motion detection
            progress_callback: Optional callback for progress updates

        Returns:
            Output file path if successful, None otherwise
        """
        output_path = self._generate_output_path(video_path, "_frameDifference")

        # Initialize GPU acceleration
        self._initialize_gpu()

        # Open video and create writer
        if not self._open_video_capture(video_path):
            self._cleanup()
            return None

        # Calculate adaptive threshold
        movement_threshold = calculate_initial_threshold(
            self.cap, 30, base_threshold, self.gpu_accel)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

        if not self._create_video_writer(output_path):
            self._cleanup()
            return None

        print(f"Processing video with initial threshold: {movement_threshold}")

        # Initialize progress tracking
        self._initialize_progress(progress_callback, "Analyzing motion")

        frame_count, written_frame_count = 0, 0
        ret, prev_frame = self.cap.read()

        try:
            while ret:
                ret, current_frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                self._update_progress(frame_count, "Analyzing motion")

                if is_significant_movement(prev_frame, current_frame, movement_threshold, self.gpu_accel):
                    self.out.write(prev_frame)
                    written_frame_count += 1

                prev_frame = current_frame

        finally:
            self._cleanup()

        # Convert to H.264
        self._convert_to_h264(output_path)

        print(f"\nTotal frames processed: {frame_count}")
        print(f"Total frames written: {written_frame_count}")

        return output_path


def remove_dead_frames(video_path: str,
                       base_threshold: float,
                       progress_callback: ProgressCallback = None) -> Optional[str]:
    """Remove dead frames using frame difference method.

    This is a convenience function that maintains backwards compatibility.

    Args:
        video_path: Path to input video file
        base_threshold: Base threshold for motion detection
        progress_callback: Optional callback for progress updates

    Returns:
        Output file path if successful, None otherwise
    """
    processor = FrameDifferenceProcessor()
    return processor.process(video_path, base_threshold, progress_callback)
