import cv2
import numpy as np


def is_significant_movement(prev_frame, current_frame, threshold):
  # Convert frames to grayscale
  gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
  gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

  # Compute the absolute difference
  frame_diff = cv2.absdiff(gray_prev, gray_current)

  # Calculate the number of pixels that changed significantly
  non_zero_count = np.count_nonzero(frame_diff > threshold)

  # Return True if the count is above a certain threshold
  return non_zero_count > (gray_prev.shape[0] * gray_prev.shape[1] * 0.02)


def remove_dead_frames(video_path, output_path, movement_threshold):
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print("Error opening video file")
    return

  # Get video properties for the output
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Choose codec based on the format of the output file
  if output_path.endswith('.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  elif output_path.endswith('.avi'):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
  else:
    print("Unsupported file format")
    return

  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  ret, prev_frame = cap.read()

  if not ret:
    print("No frames to read")
    return

  print("Processing video...")
  frame_count = 0
  written_frame_count = 0

  while ret:
    ret, current_frame = cap.read()
    if not ret:
      break

    if is_significant_movement(prev_frame, current_frame, movement_threshold):
      out.write(prev_frame)
      written_frame_count += 1

    prev_frame = current_frame
    frame_count += 1

  print(f"Total frames processed: {frame_count}")
  print(f"Total frames written: {written_frame_count}")

  cap.release()
  out.release()


input_video = 'input.mp4'
output_video = 'output.mp4'
movement_threshold = 60  # Adjust this threshold based on your requirements

remove_dead_frames(input_video, output_video, movement_threshold)
