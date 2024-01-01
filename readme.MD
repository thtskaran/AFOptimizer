# Anime Frame Optimizer (AFOptimizer)

## Introduction
Anime Frame Optimizer (AFOptimizer) is a Python-based tool that enhances anime viewing experiences by automatically removing static or 'dead' frames from videos. Using OpenCV, AFOptimizer employs three sophisticated frame analysis methods: Optical Flow, Frame Difference, and Structural Similarity Index (SSIM), each tailored to specific video processing requirements.

## Features
- **Tri-Method Analysis**: Utilizes Optical Flow, Frame Difference, and SSIM for comprehensive frame analysis.
- **Efficient Frame Removal**: Automatically detects and removes static frames, streamlining the viewing experience.
- **Customizable Sensitivity**: Adjusts movement detection sensitivity for different video types and preferences.
- **Command-Line Interface**: Easy-to-use CLI for processing videos with specified methods and settings.
- **Performance Variation**: Each method varies in processing time and efficiency, offering flexibility based on user needs:
  - **SSIM**: ~2.13 frames/s
  - **Frame Difference**: ~37.12 frames/s
  - **Optical Flow**: ~1.18 frames/s
  - _Note: Performance metrics tested on a 4vCPU, 8GB RAM replit core environment._

## Setup and Installation
To use AFOptimizer, ensure Python and all the dependencies from `requirements.txt` are installed.

## Usage
Run `main.py` with the desired method and video file as arguments:
- `python main.py -of --video=path/to/video` for Optical Flow.
- `python main.py -fd --video=path/to/video` for Frame Difference.
- `python main.py -ss --video=path/to/video --ssim_threshold=0.9587` for SSIM (the `--ssim_threshold` flag is optional; if not used, the value will default to 0.9587).

## Methods Explained
### Optical Flow Method
- **How It Works**: Calculates motion between frames based on pixel changes.
- **Technique**: Uses the Farneback algorithm for dense optical flow, providing a motion vector for each pixel.
- **Ideal Use**: Best for videos where detailed movement detection is crucial.
- **Output**: Generates a video with smoother transitions by excluding frames with minimal pixel motion.
- **Performance**: More processing-intensive due to complex vector calculations.

### Frame Difference Method
- **How It Works**: Assesses the difference in pixel values between consecutive frames.
- **Technique**: Compares grayscale versions of frames and measures the change in pixel intensity.
- **Ideal Use**: Effective for videos with varying lighting conditions or minimal background movement.
- **Output**: Focuses on significant movements by discarding frames with minimal changes.
- **Performance**: Faster than Optical Flow, offering a balance between speed and precision.

### SSIM Method
- **How It Works**: Uses the Structural Similarity Index to evaluate frame similarity.
- **Technique**: Measures changes in luminance, contrast, and structure between frames.
- **Ideal Use**: Suited for videos where perceptual frame similarity is essential.
- **Output**: Removes frames that are perceptually similar to their preceding frames.
- **Performance**: Quicker than Optical Flow, prioritizing perceptual quality over pixel-level changes.

## Development and Contributions
AFOptimizer is in active development. Contributions to enhance functionality, especially in algorithm refinement and feature additions, are welcome.

## Contact and Support
For support, contributions, or inquiries, please contact hello@karanprasad.com.
