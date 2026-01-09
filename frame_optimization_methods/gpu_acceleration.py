"""GPU acceleration detection and management for AFOptimizer.

This module provides automatic GPU detection and acceleration support for:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm/OpenCL)
- Intel integrated GPUs (OpenCL/QuickSync)
- Apple Silicon (Metal)

All operations gracefully fall back to CPU if GPU is unavailable.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

# Optional GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None


class GPUBackend(Enum):
    """Available GPU backends."""
    NONE = "none"
    CUDA = "cuda"
    OPENCL = "opencl"
    METAL = "metal"
    QSV = "qsv"  # Intel QuickSync Video


class GPUInfo:
    """Information about available GPU acceleration."""
    
    def __init__(self, backend: GPUBackend, device_name: str = "", available: bool = False):
        self.backend = backend
        self.device_name = device_name
        self.available = available
        self.cv2_backend = None
        self.encoder = None
        
    def __repr__(self) -> str:
        status = "available" if self.available else "unavailable"
        return f"GPUInfo(backend={self.backend.value}, device={self.device_name}, {status})"


def _detect_nvidia_cuda() -> Optional[GPUInfo]:
    """Detect NVIDIA CUDA GPU."""
    device_name = None
    available = False
    
    # First, check for nvidia-smi to get GPU name
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            device_name = result.stdout.strip().split('\n')[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check if CUDA is available in OpenCV
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if not device_name:
                device_name = cv2.cuda.getDevice(0).name() if device_count > 0 else "NVIDIA GPU"
            return GPUInfo(
                backend=GPUBackend.CUDA,
                device_name=device_name or "NVIDIA GPU",
                available=True
            )
    except Exception:
        pass
    
    # Check if CuPy is available (even without OpenCV CUDA)
    if CUPY_AVAILABLE:
        try:
            if cp.cuda.is_available():
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count > 0:
                    if not device_name:
                        # Try to get device name from CuPy
                        try:
                            props = cp.cuda.runtime.getDeviceProperties(0)
                            device_name = props.get('name', 'NVIDIA GPU') if isinstance(props, dict) else 'NVIDIA GPU'
                        except Exception:
                            device_name = device_name or "NVIDIA GPU"
                    return GPUInfo(
                        backend=GPUBackend.CUDA,
                        device_name=device_name or "NVIDIA GPU",
                        available=True
                    )
        except Exception:
            pass
    
    # If we detected GPU via nvidia-smi but no CUDA libraries work
    if device_name:
        return GPUInfo(
            backend=GPUBackend.CUDA,
            device_name=device_name,
            available=False  # GPU detected but CUDA libraries not working
        )
    
    return None


def _detect_amd_rocm() -> Optional[GPUInfo]:
    """Detect AMD GPU (ROCm/OpenCL)."""
    if not OPENCL_AVAILABLE:
        return None
    
    try:
        platforms = cl.get_platforms()
        for platform in platforms:
            platform_name = platform.get_info(cl.platform_info.NAME).lower()
            if 'amd' in platform_name or 'rocm' in platform_name:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    device_name = devices[0].get_info(cl.device_info.NAME)
                    return GPUInfo(
                        backend=GPUBackend.OPENCL,
                        device_name=device_name,
                        available=True
                    )
    except Exception:
        pass
    
    return None


def _detect_intel_gpu() -> Optional[GPUInfo]:
    """Detect Intel integrated GPU."""
    if not OPENCL_AVAILABLE:
        return None
    
    try:
        platforms = cl.get_platforms()
        for platform in platforms:
            platform_name = platform.get_info(cl.platform_info.NAME).lower()
            if 'intel' in platform_name:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    device_name = devices[0].get_info(cl.device_info.NAME)
                    return GPUInfo(
                        backend=GPUBackend.OPENCL,
                        device_name=device_name,
                        available=True
                    )
    except Exception:
        pass
    
    # Check for Intel QuickSync Video support via ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and "h264_qsv" in result.stdout:
            return GPUInfo(
                backend=GPUBackend.QSV,
                device_name="Intel QuickSync Video",
                available=True
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def _detect_apple_silicon() -> Optional[GPUInfo]:
    """Detect Apple Silicon GPU."""
    if platform.system() != "Darwin":
        return None
    
    try:
        # Check for Apple Silicon
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            cpu_info = result.stdout.strip().lower()
            if "apple" in cpu_info or "m1" in cpu_info or "m2" in cpu_info or "m3" in cpu_info:
                # Check for VideoToolbox encoder
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and "h264_videotoolbox" in result.stdout:
                    return GPUInfo(
                        backend=GPUBackend.METAL,
                        device_name="Apple Silicon GPU",
                        available=True
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def _detect_amd_encoder() -> Optional[GPUInfo]:
    """Detect AMD AMF encoder."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and "h264_amf" in result.stdout:
            # Try to get AMD GPU name
            try:
                result = subprocess.run(
                    ["lspci"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'amd' in line.lower() and ('vga' in line.lower() or 'display' in line.lower()):
                            device_name = line.strip()
                            return GPUInfo(
                                backend=GPUBackend.OPENCL,
                                device_name=device_name,
                                available=True
                            )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            return GPUInfo(
                backend=GPUBackend.OPENCL,
                device_name="AMD GPU",
                available=True
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def detect_gpu() -> GPUInfo:
    """Detect available GPU acceleration.
    
    Returns:
        GPUInfo object with backend and availability information.
        Falls back to CPU if no GPU is detected.
    """
    # Detection priority:
    # 1. NVIDIA CUDA (best OpenCV support)
    # 2. Apple Silicon (VideoToolbox)
    # 3. Intel QuickSync
    # 4. AMD (AMF encoder or OpenCL)
    # 5. Generic OpenCL
    
    gpu_info = _detect_nvidia_cuda()
    if gpu_info and gpu_info.available:
        return gpu_info
    
    gpu_info = _detect_apple_silicon()
    if gpu_info and gpu_info.available:
        return gpu_info
    
    gpu_info = _detect_intel_gpu()
    if gpu_info and gpu_info.available:
        return gpu_info
    
    gpu_info = _detect_amd_encoder()
    if gpu_info and gpu_info.available:
        return gpu_info
    
    gpu_info = _detect_amd_rocm()
    if gpu_info and gpu_info.available:
        return gpu_info
    
    # Fallback to CPU
    return GPUInfo(backend=GPUBackend.NONE, device_name="CPU", available=False)


def get_hardware_encoder(gpu_info: GPUInfo) -> Optional[str]:
    """Get the appropriate hardware encoder for ffmpeg.
    
    Args:
        gpu_info: GPUInfo object from detect_gpu()
    
    Returns:
        Encoder name for ffmpeg, or None for CPU encoding.
    """
    if not gpu_info.available:
        return None
    
    encoder_map = {
        GPUBackend.CUDA: "h264_nvenc",
        GPUBackend.METAL: "h264_videotoolbox",
        GPUBackend.QSV: "h264_qsv",
        GPUBackend.OPENCL: "h264_amf",  # AMD AMF, fallback to CPU if not available
    }
    
    encoder = encoder_map.get(gpu_info.backend)
    
    # Verify encoder is available
    if encoder:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and encoder in result.stdout:
                return encoder
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    return None


class GPUAccelerator:
    """Context manager for GPU-accelerated operations."""
    
    def __init__(self, gpu_info: GPUInfo):
        self.gpu_info = gpu_info
        self.use_gpu = gpu_info.available and gpu_info.backend != GPUBackend.NONE
        self.use_cupy = self.use_gpu and CUPY_AVAILABLE and gpu_info.backend == GPUBackend.CUDA
        self.use_cv2_cuda = False
        # Check if OpenCV CUDA is available
        if self.use_gpu and gpu_info.backend == GPUBackend.CUDA:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.use_cv2_cuda = True
            except Exception:
                pass
        self._gpu_mat_prev = None
        self._gpu_mat_curr = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self._gpu_mat_prev is not None:
            try:
                self._gpu_mat_prev.release()
            except Exception:
                pass
            self._gpu_mat_prev = None
        
        if self._gpu_mat_curr is not None:
            try:
                self._gpu_mat_curr.release()
            except Exception:
                pass
            self._gpu_mat_curr = None
    
    def cvt_color(self, frame: np.ndarray, code: int) -> np.ndarray:
        """Convert color space, using GPU if available."""
        if not self.use_gpu:
            return cv2.cvtColor(frame, code)
        
        # Try OpenCV CUDA first
        if self.use_cv2_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_output = cv2.cuda.cvtColor(gpu_frame, code)
                result = gpu_output.download()
                return result
            except Exception:
                pass
        
        # Fallback to CPU (OpenCV CPU is still fast for color conversion)
        return cv2.cvtColor(frame, code)
    
    def resize(self, frame: np.ndarray, size: Tuple[int, int], interpolation: int = cv2.INTER_AREA) -> np.ndarray:
        """Resize image, using GPU if available."""
        if not self.use_gpu:
            return cv2.resize(frame, size, interpolation=interpolation)
        
        # Try OpenCV CUDA first
        if self.use_cv2_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_output = cv2.cuda.resize(gpu_frame, size, interpolation=interpolation)
                result = gpu_output.download()
                return result
            except Exception:
                pass
        
        # Use CuPy for GPU-accelerated resize if available
        if self.use_cupy:
            try:
                gpu_frame = cp.asarray(frame)
                # CuPy doesn't have direct resize, but we can use scipy or fallback to CPU
                # For now, fallback to CPU (OpenCV CPU resize is still reasonably fast)
            except Exception:
                pass
        
        return cv2.resize(frame, size, interpolation=interpolation)
    
    def gaussian_blur(self, frame: np.ndarray, ksize: Tuple[int, int], sigma: float) -> np.ndarray:
        """Apply Gaussian blur, using GPU if available."""
        if not self.use_gpu:
            return cv2.GaussianBlur(frame, ksize, sigma)
        
        # Try OpenCV CUDA first
        if self.use_cv2_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_output = cv2.cuda.GaussianBlur(gpu_frame, ksize, sigma)
                result = gpu_output.download()
                return result
            except Exception:
                pass
        
        # Fallback to CPU (Gaussian blur is reasonably fast on CPU)
        return cv2.GaussianBlur(frame, ksize, sigma)
    
    def absdiff(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute absolute difference, using GPU if available."""
        if not self.use_gpu:
            return cv2.absdiff(frame1, frame2)
        
        # Use CuPy for GPU-accelerated absdiff (works even without OpenCV CUDA)
        if self.use_cupy:
            try:
                gpu_frame1 = cp.asarray(frame1)
                gpu_frame2 = cp.asarray(frame2)
                gpu_output = cp.abs(gpu_frame1 - gpu_frame2)
                return cp.asnumpy(gpu_output)
            except Exception:
                pass
        
        # Try OpenCV CUDA
        if self.use_cv2_cuda:
            try:
                gpu_frame1 = cv2.cuda_GpuMat()
                gpu_frame2 = cv2.cuda_GpuMat()
                gpu_frame1.upload(frame1)
                gpu_frame2.upload(frame2)
                gpu_output = cv2.cuda.absdiff(gpu_frame1, gpu_frame2)
                result = gpu_output.download()
                return result
            except Exception:
                pass
        
        return cv2.absdiff(frame1, frame2)
    
    def calc_optical_flow_farneback(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Calculate optical flow, using GPU if available."""
        # OpenCV's Farneback optical flow doesn't have direct CUDA support
        # But we can use GPU for preprocessing
        if not self.use_gpu or self.gpu_info.backend != GPUBackend.CUDA:
            return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # For now, use CPU (Farneback doesn't have CUDA implementation in OpenCV)
        # Future: could use cv2.cuda.BroxOpticalFlow or other GPU-accelerated methods
        return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    def sobel(self, frame: np.ndarray, dx: int, dy: int, ksize: int = 3) -> np.ndarray:
        """Compute Sobel derivative, using GPU if available."""
        if not self.use_gpu:
            return cv2.Sobel(frame, cv2.CV_32F, dx, dy, ksize=ksize)
        
        # Try OpenCV CUDA first
        if self.use_cv2_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_output = cv2.cuda.Sobel(gpu_frame, cv2.CV_32F, dx, dy, ksize=ksize)
                result = gpu_output.download()
                return result
            except Exception:
                pass
        
        # Fallback to CPU (Sobel is reasonably fast on CPU)
        return cv2.Sobel(frame, cv2.CV_32F, dx, dy, ksize=ksize)
    
    def magnitude(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute magnitude, using GPU if available."""
        if not self.use_gpu:
            return cv2.magnitude(x, y)
        
        # Use CuPy for GPU-accelerated magnitude (works even without OpenCV CUDA)
        if self.use_cupy:
            try:
                x_gpu = cp.asarray(x)
                y_gpu = cp.asarray(y)
                mag_gpu = cp.sqrt(x_gpu**2 + y_gpu**2)
                return cp.asnumpy(mag_gpu)
            except Exception:
                pass
        
        # Fallback to OpenCV CPU
        return cv2.magnitude(x, y)
    
    def cart_to_polar(self, x: np.ndarray, y: np.ndarray, angle_in_degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Convert cartesian to polar coordinates, using GPU if available."""
        if not self.use_gpu:
            return cv2.cartToPolar(x, y, angleInDegrees=angle_in_degrees)
        
        # Use CuPy for GPU-accelerated cart_to_polar (works even without OpenCV CUDA)
        if self.use_cupy:
            try:
                x_gpu = cp.asarray(x)
                y_gpu = cp.asarray(y)
                magnitude = cp.sqrt(x_gpu**2 + y_gpu**2)
                angle = cp.arctan2(y_gpu, x_gpu)
                if angle_in_degrees:
                    angle = cp.degrees(angle)
                return cp.asnumpy(magnitude), cp.asnumpy(angle)
            except Exception:
                pass
        
        # Fallback to OpenCV CPU
        return cv2.cartToPolar(x, y, angleInDegrees=angle_in_degrees)


# Global GPU info cache
_gpu_info_cache: Optional[GPUInfo] = None


def get_gpu_info() -> GPUInfo:
    """Get cached GPU info or detect it."""
    global _gpu_info_cache
    if _gpu_info_cache is None:
        _gpu_info_cache = detect_gpu()
    return _gpu_info_cache


def reset_gpu_cache():
    """Reset GPU detection cache (useful for testing)."""
    global _gpu_info_cache
    _gpu_info_cache = None

