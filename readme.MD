# Anime Frame Optimizer (AFOptimizer)

## Introduction
Anime Frame Optimizer (AFOptimizer) is a Python-based tool that enhances anime videos by automatically removing static, or 'dead', frames. Using OpenCV, this script identifies and eliminates frames with minimal or no movement, streamlining the viewing experience.

## Features
- **Efficient Frame Removal**: Automates the detection and removal of static frames from anime footage.
- **Customizable Threshold**: Users can set the sensitivity of movement detection.
- **Replit Compatibility**: Configured to work on Replit with specific Nix dependencies.

## Setup and Installation
To run AFOptimizer, you need Python and OpenCV. If you're using Replit, ensure the following dependencies are added to your `replit.nix` file for correct functionality:

```nix
{ pkgs }: {
  deps = [
    pkgs.postgresql,
    pkgs.libGL
  ];
}
```

## Usage
- **Specify the input and output video file paths in the script.**
- **Adjust the ```movement_threshold``` variable as per your requirements.**
- **Run the script to process the video.**
## Development and Contributions
**This project is currently under active development. We are working on refining the movement detection threshold to ensure optimal accuracy. Contributions, suggestions, and feedback are highly welcome.**