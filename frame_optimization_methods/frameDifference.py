import cv2
import numpy as np
import os
from tqdm import tqdm


def calculate_initial_threshold(cap, sample_size, base_threshold):
    """Calculates an initial threshold for frame difference based on average movement.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        sample_size (int): Number of frames to sample for calculating average movement.
        base_threshold (int): Base threshold value.

    Returns:
        int: Adjusted threshold based on average movement.
    """
    total_movement = 0
    ret, prev_frame = cap.read()
    for _ in range(sample_size - 1):
        ret, current_frame = cap.read()
        if not ret:
            break
        total_movement += np.sum(
            cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)))
        prev_frame = current_frame

    average_movement = total_movement / ((cap.get(cv2.CAP_PROP_FRAME_HEIGHT) *
                                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)) *
                                       (sample_size - 1))
    return int(base_threshold + average_movement)  # Adjust base_threshold based on average movement


def is_significant_movement(prev_frame, current_frame, threshold, non_zero_pixel_threshold=0.02):
    """Checks if there is significant movement between two frames.

    Args:
        prev_frame (numpy.ndarray): Previous frame in BGR format.
        current_frame (numpy.ndarray): Current frame in BGR format.
        threshold (int): Threshold for pixel difference to be considered significant.
        non_zero_pixel_threshold (float): Percentage of non-zero pixels to consider movement significant.

    Returns:
        bool: True if significant movement detected, False otherwise.
    """
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_prev, gray_current)
    non_zero_count = np.count_nonzero(frame_diff > threshold)
    return non_zero_count > (gray_prev.shape[0] * gray_prev.shape[1] * non_zero_pixel_threshold)


def remove_dead_frames(video_path, base_threshold, non_zero_pixel_threshold=0.02):
    """Removes static frames from a video using frame difference analysis.

    Args:
        video_path (str): Path to the input video file.
        base_threshold (int): Base threshold for pixel difference.
        non_zero_pixel_threshold (float): Percentage of non-zero pixels to consider movement significant.
    """
    base_name = os.path.basename(video_path)
    output_path = os.path.splitext(base_name)[0] + "_frameDifference.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    movement_threshold = calculate_initial_threshold(
        cap, 30, base_threshold)  # Sample 30 frames for initial threshold
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start of video

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(
        *'mp4v' if output_path.endswith('.mp4') else 'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video with initial threshold:", movement_threshold)
    frame_count = 0
    written_frame_count = 0

    # Initialize tqdm progress bar
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    ret, prev_frame = cap.read()
    while ret:
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)  # Update the progress bar

        if is_significant_movement(prev_frame, current_frame, movement_threshold, non_zero_pixel_threshold):
            out.write(prev_frame)
            written_frame_count += 1

        prev_frame = current_frame

    cap.release()
    out.release()
    pbar.close()  # Close the progress bar

    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total frames written: {written_frame_count}")

# Example usage (uncomment to use):
# input_video = 'input.mp4'
# base_threshold = 10  # Base threshold, to be adjusted
# remove_dead_frames(input_video, base_threshold)