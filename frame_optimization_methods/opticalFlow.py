import cv2
import numpy as np
import os
from tqdm import tqdm


def calculate_optical_flow(prev_frame, current_frame):
  # Convert frames to grayscale
  prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
  current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

  # Calculate dense optical flow using Farneback method
  flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3,
                                      15, 3, 5, 1.2, 0)
  return flow


def is_significant_movement_optical_flow(flow, mag_threshold):
  # Compute magnitude and angle of the flow vectors
  magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

  # Check if the average magnitude of the flow is greater than the threshold
  mean_magnitude = np.mean(magnitude)
  return mean_magnitude > mag_threshold


def remove_dead_frames(video_path, flow_mag_threshold):
  base_name = os.path.basename(video_path)
  output_path = os.path.splitext(base_name)[0] + "_opticalFlow.mp4"

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    return

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  ret, prev_frame = cap.read()
  if not ret:
    print("Error reading the first frame.")
    cap.release()
    return

  frame_count, dead_frame_count, written_frame_count = 0, 0, 0

  # Initialize tqdm progress bar
  pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

  while True:
    ret, current_frame = cap.read()
    if not ret:
      break

    frame_count += 1
    pbar.update(1)  # Update the progress bar

    flow = calculate_optical_flow(prev_frame, current_frame)
    if is_significant_movement_optical_flow(flow, flow_mag_threshold):
      out.write(prev_frame)
      written_frame_count += 1
    else:
      dead_frame_count += 1

    prev_frame = current_frame

  cap.release()
  out.release()
  pbar.close()  # Close the progress bar

  print(f"\nTotal frames processed: {frame_count}")
  print(f"Dead frames (not written): {dead_frame_count}")
  print(f"Frames written to output: {written_frame_count}")


# Uncomment below lines for direct usage
# input_video = 'input.mp4'
# output_video = 'output.mp4'
# flow_mag_threshold = 0.4  # Start with this value and adjust based on trials
# remove_dead_frames(input_video, output_video, flow_mag_threshold)
