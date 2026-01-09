from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from typing import Optional

from frame_optimization_methods.video_processor_base import VideoProcessorBase, ProgressCallback

# Try to import CuPy for GPU-accelerated SSIM
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def _compute_ssim_gpu(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
    """Compute SSIM on GPU using CuPy.

    This is a simplified SSIM implementation optimized for GPU processing.
    Falls back to CPU if CuPy is not available.

    Args:
        img1_gray: First grayscale image (numpy array)
        img2_gray: Second grayscale image (numpy array)

    Returns:
        SSIM value between 0 and 1
    """
    if not CUPY_AVAILABLE:
        return float(compare_ssim(img1_gray, img2_gray))

    try:
        # Transfer to GPU
        img1_gpu = cp.asarray(img1_gray, dtype=cp.float32)
        img2_gpu = cp.asarray(img2_gray, dtype=cp.float32)

        # Constants for SSIM calculation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Compute means
        mu1 = cp.mean(img1_gpu)
        mu2 = cp.mean(img2_gpu)

        # Compute variances and covariance
        sigma1_sq = cp.var(img1_gpu)
        sigma2_sq = cp.var(img2_gpu)
        sigma12 = cp.mean((img1_gpu - mu1) * (img2_gpu - mu2))

        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_val = numerator / denominator

        # Transfer back to CPU and return
        return float(cp.asnumpy(ssim_val))

    except Exception as e:
        # Fallback to CPU on any GPU error
        print(f"Warning: GPU SSIM failed ({e}), falling back to CPU")
        return float(compare_ssim(img1_gray, img2_gray))

class SSIMProcessor(VideoProcessorBase):
    """SSIM-based video processor."""

    def process(self, video_path: str, ssim_threshold: float = 0.987,
                output_path: Optional[str] = None,
                progress_callback: ProgressCallback = None) -> Optional[str]:
        """Process video using SSIM method.

        Args:
            video_path: Path to input video file
            ssim_threshold: Threshold for SSIM comparison
            output_path: Optional output path (auto-generated if not provided)
            progress_callback: Optional callback for progress updates

        Returns:
            Output file path if successful, None otherwise
        """
        if output_path is None:
            output_path = self._generate_output_path(video_path, "_ssim")

        # Initialize GPU acceleration
        self._initialize_gpu()

        # Inform about SSIM acceleration
        if CUPY_AVAILABLE and self.gpu_accel:
            print("SSIM calculation will use GPU acceleration (CuPy)")
        else:
            print("SSIM calculation will use CPU (install CuPy for GPU acceleration)")

        # Open video and create writer
        if not self._open_video_capture(video_path):
            self._cleanup()
            raise IOError(f"Error opening video file: {video_path}")

        if not self._create_video_writer(output_path):
            self._cleanup()
            raise IOError(f"Error creating video writer: {output_path}")

        success, prev_frame = self.cap.read()
        if not success:
            self._cleanup()
            raise IOError("Error reading the first frame.")

        # Initialize progress tracking
        self._initialize_progress(progress_callback, "Analyzing structure")

        count, saved_frames = 0, 0

        try:
            while success:
                success, current_frame = self.cap.read()
                if not success:
                    break

                frame_count = count + 1

                if count > 0:
                    # Use GPU acceleration for color conversion if available
                    if self.gpu_accel:
                        gray_prev_frame = self.gpu_accel.cvt_color(prev_frame, cv2.COLOR_BGR2GRAY)
                        gray_current_frame = self.gpu_accel.cvt_color(current_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                    # SSIM calculation (GPU-accelerated if CuPy available, otherwise CPU)
                    ssim = _compute_ssim_gpu(gray_prev_frame, gray_current_frame)
                    if ssim < ssim_threshold:
                        self.out.write(prev_frame)
                        saved_frames += 1

                prev_frame = current_frame
                count += 1
                self._update_progress(frame_count, "Analyzing structure")

        finally:
            # Save the last frame
            if count > 0:
                self.out.write(prev_frame)
                saved_frames += 1

            self._cleanup()

        # Convert to H.264
        self._convert_to_h264(output_path)

        print(f"Processed {count} frames. Saved {saved_frames} frames to {output_path}")

        return output_path


def process_video(video_path: str,
                  ssim_threshold: float,
                  output_path: str,
                  progress_callback: ProgressCallback = None) -> Optional[str]:
    """Process video using SSIM method.

    This is a convenience function that maintains backwards compatibility.

    Args:
        video_path: Path to input video file
        ssim_threshold: Threshold for SSIM comparison
        output_path: Output file path
        progress_callback: Optional callback for progress updates

    Returns:
        Output file path if successful, None otherwise
    """
    processor = SSIMProcessor()
    return processor.process(video_path, ssim_threshold, output_path, progress_callback)
