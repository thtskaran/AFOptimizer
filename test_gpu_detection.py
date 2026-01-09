#!/usr/bin/env python3
"""Simple script to test GPU detection and display available acceleration."""

from frame_optimization_methods.gpu_acceleration import (
    detect_gpu,
    get_gpu_info,
    get_hardware_encoder,
    reset_gpu_cache,
)

def main():
    print("=" * 60)
    print("AFOptimizer GPU Detection Test")
    print("=" * 60)
    print()
    
    # Reset cache to force fresh detection
    reset_gpu_cache()
    
    # Detect GPU
    print("Detecting GPU acceleration...")
    gpu_info = detect_gpu()
    
    print(f"\nGPU Backend: {gpu_info.backend.value}")
    print(f"Device Name: {gpu_info.device_name}")
    print(f"Available: {'Yes' if gpu_info.available else 'No'}")
    
    # Check hardware encoder
    encoder = get_hardware_encoder(gpu_info)
    if encoder:
        print(f"Hardware Encoder: {encoder}")
    else:
        print("Hardware Encoder: None (will use CPU encoding)")
    
    print()
    print("=" * 60)
    
    if gpu_info.available:
        print("✓ GPU acceleration is available and will be used!")
    else:
        print("ℹ GPU acceleration not available. System will use CPU processing.")
        print("  This is normal if:")
        print("  - No GPU is installed")
        print("  - GPU drivers are not installed")
        print("  - OpenCV was not built with GPU support")
        print("  - GPU libraries (CuPy, PyOpenCL) are not installed")
    
    print("=" * 60)
    
    # Test cached version
    print("\nTesting cached GPU info...")
    cached_info = get_gpu_info()
    print(f"Cached backend: {cached_info.backend.value}")
    print(f"Cache matches: {cached_info.backend == gpu_info.backend}")

if __name__ == "__main__":
    main()

