import argparse
import sys
import os

from frame_optimization_methods.opticalFlow import remove_dead_frames as remove_dead_frames_of
from frame_optimization_methods.frameDifference import remove_dead_frames as remove_dead_frames_fd

def main():
    parser = argparse.ArgumentParser(description='Anime Frame Optimizer (AFOptimizer)',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-of', '--opticalFlow', action='store_true', help='Use the Optical Flow method to remove static frames.')
    parser.add_argument('-fd', '--frameDifference', action='store_true', help='Use the Frame Difference method to remove static frames.')
    parser.add_argument('--video', type=str, help='Path to the input video file.')

    # Display help message if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Check if the video file exists
    if not os.path.exists(args.video):
        print(f"Error: The video file {args.video} does not exist.")
        sys.exit(1)

    base_threshold = 10  # Default threshold, can be made configurable through CLI
    flow_mag_threshold = 0.4  # Example threshold, can be made configurable

    if args.opticalFlow:
        print("Using Optical Flow method...")
        remove_dead_frames_of(args.video, flow_mag_threshold)

    elif args.frameDifference:
        print("Using Frame Difference method...")
        remove_dead_frames_fd(args.video, base_threshold)

    else:
        print("Error: No method selected. Use -of for Optical Flow or -fd for Frame Difference.")
        sys.exit(1)

if __name__ == "__main__":
    main()
