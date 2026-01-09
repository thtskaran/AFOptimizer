from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from frame_optimization_methods.video_processor_base import VideoProcessorBase, ProgressCallback
from frame_optimization_methods.gpu_acceleration import GPUAccelerator


def calculate_optical_flow(prev_frame, current_frame, gpu_accel: Optional[GPUAccelerator] = None):
  # Convert frames to grayscale (using GPU if available)
  if gpu_accel:
    prev_gray = gpu_accel.cvt_color(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = gpu_accel.cvt_color(current_frame, cv2.COLOR_BGR2GRAY)
  else:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

  # Calculate dense optical flow using Farneback method
  # Note: Farneback doesn't have direct GPU support, but preprocessing can use GPU
  if gpu_accel:
    flow = gpu_accel.calc_optical_flow_farneback(prev_gray, current_gray)
  else:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3,
                                        15, 3, 5, 1.2, 0)
  return flow


def is_significant_movement_optical_flow(flow, mag_threshold):
  # Compute magnitude and angle of the flow vectors
  magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

  # Check if the average magnitude of the flow is greater than the threshold
  mean_magnitude = np.mean(magnitude)
  return mean_magnitude > mag_threshold


class OpticalFlowProcessor(VideoProcessorBase):
    """Optical flow-based video processor."""

    def process(self, video_path: str, flow_mag_threshold: float = 0.4,
                progress_callback: ProgressCallback = None) -> Optional[str]:
        """Process video using optical flow method.

        Args:
            video_path: Path to input video file
            flow_mag_threshold: Threshold for flow magnitude
            progress_callback: Optional callback for progress updates

        Returns:
            Output file path if successful, None otherwise
        """
        output_path = self._generate_output_path(video_path, "_opticalFlow")

        # Initialize GPU acceleration
        self._initialize_gpu()

        # Open video and create writer
        if not self._open_video_capture(video_path):
            self._cleanup()
            return None

        if not self._create_video_writer(output_path):
            self._cleanup()
            return None

        ret, prev_frame = self.cap.read()
        if not ret:
            print("Error reading the first frame.")
            self._cleanup()
            return None

        # Initialize progress tracking
        self._initialize_progress(progress_callback, "Analyzing motion")

        frame_count, dead_frame_count, written_frame_count = 0, 0, 0

        try:
            while True:
                ret, current_frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                self._update_progress(frame_count, "Analyzing motion")

                flow = calculate_optical_flow(prev_frame, current_frame, self.gpu_accel)
                if is_significant_movement_optical_flow(flow, flow_mag_threshold):
                    self.out.write(prev_frame)
                    written_frame_count += 1
                else:
                    dead_frame_count += 1

                prev_frame = current_frame

        finally:
            self._cleanup()

        # Convert to H.264
        self._convert_to_h264(output_path)

        print(f"\nTotal frames processed: {frame_count}")
        print(f"Dead frames (not written): {dead_frame_count}")
        print(f"Frames written to output: {written_frame_count}")

        return output_path


def remove_dead_frames(video_path: str,
                       flow_mag_threshold: float,
                       progress_callback: ProgressCallback = None) -> Optional[str]:
    """Remove dead frames using optical flow method.

    This is a convenience function that maintains backwards compatibility.

    Args:
        video_path: Path to input video file
        flow_mag_threshold: Threshold for flow magnitude
        progress_callback: Optional callback for progress updates

    Returns:
        Output file path if successful, None otherwise
    """
    processor = OpticalFlowProcessor()
    return processor.process(video_path, flow_mag_threshold, progress_callback)
