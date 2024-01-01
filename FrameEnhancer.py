import argparse
import sys
import os

from frame_optimization_methods.opticalFlow import remove_dead_frames as remove_dead_frames_of
from frame_optimization_methods.frameDifference import remove_dead_frames as remove_dead_frames_fd
from frame_optimization_methods.ssim import process_video as process_video_ssim


def main():
  parser = argparse.ArgumentParser(
      description='Anime Frame Optimizer (AFOptimizer)',
      formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument(
      '-of',
      '--opticalFlow',
      action='store_true',
      help='Use the Optical Flow method to remove static frames.')
  parser.add_argument(
      '-fd',
      '--frameDifference',
      action='store_true',
      help='Use the Frame Difference method to remove static frames.')
  parser.add_argument('-ss',
                      '--ssim',
                      action='store_true',
                      help='Use the SSIM method to remove similar frames.')
  parser.add_argument('--video',
                      type=str,
                      help='Path to the input video file.')
  parser.add_argument('--ssim_threshold',
                      type=float,
                      default=0.9587,
                      help='SSIM threshold for frame similarity.')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()

  if not args.video:
    print("Error: Please specify the input video file path.")
    sys.exit(1)

  if not os.path.exists(args.video):
    print(f"Error: The video file {args.video} does not exist.")
    sys.exit(1)

  base_threshold = 10  # Default threshold for frame difference
  flow_mag_threshold = 0.4  # Default threshold for optical flow

  if args.opticalFlow:
    print("Using Optical Flow method...")
    remove_dead_frames_of(args.video, flow_mag_threshold)

  elif args.frameDifference:
    print("Using Frame Difference method...")
    remove_dead_frames_fd(args.video, base_threshold)

  elif args.ssim:
    print("Using SSIM method...")
    output_video = os.path.splitext(
        args.video)[0] + "_ssim.mp4"  # Generating output file name
    process_video_ssim(args.video, args.ssim_threshold, output_video)

  else:
    print(
        "Error: No method selected. Use -of for Optical Flow, -fd for Frame Difference, or -ss for SSIM."
    )
    sys.exit(1)


if __name__ == "__main__":
  main()
