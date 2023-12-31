# Anime Frame Optimizer (AFOptimizer)

## Introduction
Anime Frame Optimizer (AFOptimizer) is a Python tool designed to enhance anime viewing experiences by removing static or 'dead' frames from videos. Using OpenCV, AFOptimizer implements three advanced frame analysis methods: Optical Flow, Frame Difference, and Structural Similarity Index (SSIM), each catering to specific video processing needs.

## Features
- **Tri-Method Analysis**: Offers Optical Flow, Frame Difference, and SSIM methods for frame analysis.
- **Efficient Frame Removal**: Detects and removes static frames, enhancing the viewing experience.
- **Customizable Sensitivity**: Adjust movement detection sensitivity for different video types.
- **Command-Line Interface**: User-friendly CLI for easy processing with specified methods and settings.
- **Performance Variation**: Each method varies in processing time and efficiency: Optical Flow > SSIM > Frame Difference.

## Setup and Installation
Ensure Python and OpenCV are installed and configured on your system to use AFOptimizer.

## Usage
Execute `main.py` with the chosen method and video file:
- `python main.py -of --video=path/to/video` for Optical Flow.
- `python main.py -fd --video=path/to/video` for Frame Difference.
- `python main.py -ss --video=path/to/video --ssim_threshold=0.9587` for SSIM. (--ssim_threshold flag is optional , if not used value will be set to 0.9587)

## Methods Explained
### Optical Flow Method
- **Functionality**: Analyzes pixel movement between frames using motion vectors.
- **Technique**: Employs the Farneback algorithm for dense optical flow, quantifying motion for each pixel.
- **Ideal Use**: Suited for videos needing detailed motion analysis, as it precisely captures pixel dynamics.
- **Output**: Creates smoother transitions in videos by omitting frames with minimal pixel motion.
- **Performance**: Generally requires more processing time due to complex vector calculations.

### Frame Difference Method
- **Functionality**: Compares pixel intensity changes between consecutive frames.
- **Technique**: Analyzes grayscale frame differences, adjusting sensitivity based on initial frame sampling.
- **Ideal Use**: Effective for videos with varying lighting or minimal background motion.
- **Output**: Highlights significant movements, removing frames with negligible changes.
- **Performance**: Faster than Optical Flow but less precise in motion detection.

### SSIM Method
- **Functionality**: Evaluates frame similarity using the Structural Similarity Index.
- **Technique**: Measures changes in luminance, contrast, and structure between frames.
- **Ideal Use**: Suitable for videos where perceptual frame similarity is a priority.
- **Output**: Excludes frames that are perceptually similar to their preceding frames.
- **Performance**: Faster than Optical Flow, with a focus on perceptual quality rather than pixel-level changes.

## Development and Contributions
AFOptimizer is continually evolving. We encourage contributions, especially in algorithm refinement and feature expansion.

## Contact and Support
For queries or contributions, reach out to hello@karanprasad.com.
