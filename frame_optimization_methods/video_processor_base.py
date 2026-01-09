"""Base class for video processing methods.

This module provides a common foundation for all frame optimization methods,
reducing code duplication and ensuring consistent behavior across different algorithms.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
from tqdm import tqdm

from frame_optimization_methods.gpu_acceleration import GPUAccelerator, GPUInfo, get_gpu_info
from frame_optimization_methods.video_encoding import convert_to_h264

ProgressCallback = Optional[Callable[[int, int, Optional[str]], None]]


class VideoProcessorBase(ABC):
    """Abstract base class for video frame optimization methods."""

    def __init__(self):
        """Initialize the video processor."""
        self.gpu_info: Optional[GPUInfo] = None
        self.gpu_accel: Optional[GPUAccelerator] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.out: Optional[cv2.VideoWriter] = None
        self.pbar: Optional[tqdm] = None
        self.progress_callback: ProgressCallback = None

        # Video properties
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0

    def _initialize_gpu(self) -> None:
        """Initialize GPU acceleration if available."""
        self.gpu_info = get_gpu_info()
        if self.gpu_info.available:
            self.gpu_accel = GPUAccelerator(self.gpu_info)
            print(f"Using GPU acceleration: {self.gpu_info.device_name} ({self.gpu_info.backend.value})")
        else:
            self.gpu_accel = None
            print("Using CPU processing")

    def _open_video_capture(self, video_path: str) -> bool:
        """Open video capture and read video properties.

        Args:
            video_path: Path to input video file

        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = max(self.total_frames, 1)

        return True

    def _create_video_writer(self, output_path: str) -> bool:
        """Create video writer for output.

        Args:
            output_path: Path to output video file

        Returns:
            True if successful, False otherwise
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v' if output_path.endswith('.mp4') else 'XVID')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        if not self.out.isOpened():
            print(f"Error creating video writer: {output_path}")
            return False

        return True

    def _initialize_progress(self, progress_callback: ProgressCallback, initial_stage: str = "Processing") -> None:
        """Initialize progress tracking (tqdm or callback).

        Args:
            progress_callback: Optional callback for progress updates
            initial_stage: Stage description for initial progress update
        """
        self.progress_callback = progress_callback

        if progress_callback is None:
            self.pbar = tqdm(total=self.total_frames, desc="Processing Video", unit="frame")
        else:
            progress_callback(0, self.total_frames, initial_stage)

    def _update_progress(self, frame_count: int, stage: Optional[str] = None) -> None:
        """Update progress tracking.

        Args:
            frame_count: Current frame number
            stage: Optional stage description
        """
        if self.pbar:
            self.pbar.update(1)
        elif self.progress_callback:
            self.progress_callback(frame_count, self.total_frames, stage)

    def _finalize_progress(self, stage: str = "Transcoding to H.264") -> None:
        """Finalize progress tracking.

        Args:
            stage: Stage description for final progress update
        """
        if self.pbar:
            self.pbar.close()
            print(stage + "...")
        elif self.progress_callback:
            self.progress_callback(self.total_frames, self.total_frames, stage)

    def _cleanup(self) -> None:
        """Clean up resources (GPU, video capture, video writer)."""
        if self.gpu_accel:
            try:
                self.gpu_accel.cleanup()
            except Exception as e:
                print(f"Warning: GPU cleanup error: {e}")

        if self.cap:
            self.cap.release()

        if self.out:
            self.out.release()

        if self.pbar:
            self.pbar.close()

    def _convert_to_h264(self, output_path: str) -> None:
        """Convert output video to H.264 format.

        Args:
            output_path: Path to video file to convert
        """
        self._finalize_progress("Transcoding to H.264")
        convert_to_h264(output_path)

        if self.progress_callback:
            self.progress_callback(self.total_frames, self.total_frames, "Finalizing output")

    def _generate_output_path(self, video_path: str, suffix: str) -> str:
        """Generate output file path based on input path and method suffix.

        Args:
            video_path: Input video path
            suffix: Suffix to append (e.g., '_frameDifference')

        Returns:
            Output file path
        """
        base_name = os.path.basename(video_path)
        return os.path.splitext(base_name)[0] + suffix + ".mp4"

    @abstractmethod
    def process(self, video_path: str, **kwargs) -> Optional[str]:
        """Process video and return output path.

        This method must be implemented by subclasses to define the specific
        frame optimization algorithm.

        Args:
            video_path: Path to input video file
            **kwargs: Method-specific parameters

        Returns:
            Output file path if successful, None otherwise
        """
        pass

    def process_with_cleanup(self, video_path: str, **kwargs) -> Optional[str]:
        """Process video with automatic cleanup on error.

        This is a convenience wrapper that ensures cleanup even if processing fails.

        Args:
            video_path: Path to input video file
            **kwargs: Method-specific parameters

        Returns:
            Output file path if successful, None otherwise
        """
        try:
            return self.process(video_path, **kwargs)
        except Exception as e:
            print(f"Error during video processing: {e}")
            return None
        finally:
            self._cleanup()
