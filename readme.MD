# Anime Frame Optimizer (AFOptimizer)

## Introduction
Anime Frame Optimizer (AFOptimizer) is a Python-based tool that enhances the anime viewing experience by automatically removing static or 'dead' frames from videos. Leveraging the powerful OpenCV library, AFOptimizer implements two sophisticated frame analysis methods: Optical Flow and Frame Difference. These methods are designed to optimize videos by identifying and eliminating frames with minimal or no movement.

## Features
- **Dual Detection Methods**: Choose between Optical Flow and Frame Difference methods for frame analysis.
- **Efficient Frame Removal**: Automatically detects and removes static frames, streamlining the viewing experience.
- **Customizable Sensitivity**: Adjust the sensitivity of movement detection to suit different video types and preferences.
- **Command-Line Interface**: Easy-to-use CLI for processing videos with specified methods and settings.

## Setup and Installation
AFOptimizer requires Python and OpenCV. Ensure these are installed and properly configured on your system.

## Usage
Run `main.py` with the desired method and video file as arguments:
- `python main.py -of --video=path/to/video` for Optical Flow method.
- `python main.py -fd --video=path/to/video` for Frame Difference method.

## Methods Explained
### Optical Flow Method
- **How It Works**: This method calculates the motion between two consecutive frames based on pixel changes.
- **Technique**: Utilizes the Farneback algorithm to compute dense optical flow, providing a motion vector for each pixel.
- **Ideal Use**: Best for videos where precision in movement detection is crucial, and for analyzing intricate details in motion.
- **Output**: Generates a video with smoother transitions by removing frames where pixel motion is below the set threshold.

### Frame Difference Method
- **How It Works**: This method assesses the difference in pixel values between consecutive frames.
- **Technique**: Compares grayscale versions of consecutive frames and measures the change in pixel intensity.
- **Dynamic Thresholding**: Initially samples a set of frames to calculate an average movement value, which is used to adjust the sensitivity dynamically.
- **Ideal Use**: Effective in scenarios with varying lighting conditions or when dealing with videos having minimal background movement.
- **Output**: Produces a video emphasizing significant movements by discarding frames with minimal changes.

## Development and Contributions
AFOptimizer is in active development, and contributions to enhance its functionality are welcome. We are particularly interested in refining the movement detection algorithms and adding more features for user customization.

## Contact and Support
For support, contributions, or queries, please reach out through hello@karanprasad.com
