import cv2
from skimage.metrics import structural_similarity as compare_ssim
import os


def process_video(video_path, ssim_threshold, output_path):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise IOError(f"Error opening video file: {video_path}")

  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  success, prev_frame = cap.read()
  if not success:
    raise IOError("Error reading the first frame.")

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  count, saved_frames = 0, 0
  while success:
    success, current_frame = cap.read()
    if not success:
      break

    frame_count = count + 1  # Frame count starts from 1
    print(f"Processing frame {frame_count}/{total_frames}...", end='\r')

    if count > 0:
      gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
      gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
      ssim = compare_ssim(gray_prev_frame, gray_current_frame)
      if ssim < ssim_threshold:
        out.write(prev_frame)
        saved_frames += 1

    prev_frame = current_frame
    count += 1

  if count > 0:
    out.write(prev_frame)  # Save the last frame
    saved_frames += 1

  cap.release()
  out.release()
  print(
      f"\nProcessed {count} frames. Saved {saved_frames} frames to {output_path}"
  )


# Uncomment below lines for direct usage
# if __name__ == "__main__":
#     process_video('input.mp4', 0.987, 'output.mp4')
