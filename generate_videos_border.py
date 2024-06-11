import cv2
import numpy as np
import os

# Parameters
video_path = './data/wave/circular_wave.mp4'
num_frames = 90  # Update this if different
frame_height, frame_width = 640, 640  # Update according to your video dimensions
fps = 30

# Prepare output directories
video_basename = video_path[:-4]
output_dirs = [f'{video_basename}_border_top', f'{video_basename}_border_bottom', f'{video_basename}_border_left', f'{video_basename}_border_right']

# Initialize video readers and writers
cap = cv2.VideoCapture(video_path)
writers = {
    direction: cv2.VideoWriter(
        f'{direction}.avi',
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps,
        (frame_width, frame_height), 0
    ) for direction in output_dirs
}

# Define exponential weights
weights_top = np.exp(-np.linspace(0, 5, frame_height))
weights_bottom = weights_top[::-1]
weights_left = np.exp(-np.linspace(0, 5, frame_width))
weights_right = weights_left[::-1]

def find_first_nonzero_masked(arr, weights):
    """Create a masked array based on the first nonzero element per column or row."""
    mask = np.cumsum(arr, axis=0) == 1  # Find the first occurrence of nonzero elements
    return np.dot(weights, arr * mask)

def process_frame(frame):
    # Apply threshold to create a binary frame
    binary_frame = (frame > 127).astype(np.uint8)

    # Calculate weighted sums using masks
    top_to_bottom = find_first_nonzero_masked(binary_frame, weights_top)
    bottom_to_top = find_first_nonzero_masked(np.flipud(binary_frame), weights_bottom)
    left_to_right = find_first_nonzero_masked(binary_frame.T, weights_left)
    right_to_left = find_first_nonzero_masked(np.fliplr(binary_frame.T), weights_right)

    return top_to_bottom, bottom_to_top, left_to_right, right_to_left
    
def convert_to_2d(image_1d, vertical=True):
    """Convert a 1D image to a 2D image by repeating the array."""
    if vertical:
        img_2d = np.tile(image_1d, (frame_height, 1))
    else:
        img_2d = np.tile(image_1d[:, np.newaxis], (1, frame_width))

    # Ensure the image is scaled to 0-255 and of type uint8
    # img_2d = np.clip(img_2d, 0, 255)  # Optional: Clip values to ensure they are within 0-255
    return img_2d.astype(np.uint8)



# Process video
for frame_index in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = (frame > 127).astype(np.uint8)  # Convert to binary

    tb, bt, lr, rl = process_frame(frame)

    # Write the images to their respective videos
    writers[output_dirs[0]].write(convert_to_2d(tb))
    writers[output_dirs[1]].write(convert_to_2d(bt))
    writers[output_dirs[2]].write(convert_to_2d(lr, vertical=False))
    writers[output_dirs[3]].write(convert_to_2d(rl, vertical=False))

# Cleanup
cap.release()
for writer in writers.values():
    writer.release()

print("Processing complete. Videos saved in respective directories.")
