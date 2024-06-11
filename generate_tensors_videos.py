import cv2
import numpy as np
import torch
import os

# Parameters
video_path = './data/wave/blur_7/circular_wave.mp4'
os.makedirs("./output_tensors/wave_blur7")
output_dir = 'output_tensors/wave_blur7'
num_frames = 90  # Update this if different
frame_height, frame_width = 640, 640  # Update according to your video dimensions
fps = 30

# Prepare output directories
video = video_path[:-4]
output_dirs = [f'{video}_top', f'{video}_bottom', f'{video}_left', f'{video}_right']

# Initialize video readers and writers
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Failed to open video"

writers = {
    direction: cv2.VideoWriter(
        f'{direction}.avi',
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps,
        (frame_width, frame_height),
        0
    ) for direction in output_dirs
}

# Exponential weights
weights_top = np.exp(-np.linspace(0, 5, frame_height))
weights_bottom = weights_top[::-1]
weights_left = np.exp(-np.linspace(0, 5, frame_width))
weights_right = weights_left[::-1]

def process_frame(frame, direction):
    global video
    if direction == f'{video}_top':
        weighted_sum = np.dot(frame, weights_top)
    elif direction == f'{video}_bottom':
        weighted_sum = np.dot(frame, weights_bottom)
    elif direction == f'{video}_left':
        weighted_sum = np.dot(frame.T, weights_left)
    elif direction == f'{video}_right':
        weighted_sum = np.dot(frame.T, weights_right)

    # Convert 1D array to 2D using np.outer
    if 'left' in direction or 'right' in direction:
        # print(weighted_sum)
        img_2d = np.outer(np.ones(frame_height), weighted_sum)
    else:
        img_2d = np.outer(weighted_sum, np.ones(frame_width))
    return img_2d.astype(np.uint8)

def process_tensor(frame):
    top_to_bottom = np.dot(frame, weights_top)
    bottom_to_top = np.dot(frame, weights_bottom)
    left_to_right = np.dot(frame.T, weights_left)
    right_to_left = np.dot(frame.T, weights_right)
    tensor = torch.tensor(np.array([top_to_bottom, bottom_to_top, left_to_right, right_to_left]), dtype=torch.float32)
    return tensor

# Process video
for frame_index in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tensor = process_tensor(frame)
    torch.save(tensor, os.path.join(output_dir, f'tensor_{frame_index+1}.pt'))

    for direction in output_dirs:
        img_2d = process_frame(frame, direction)
        writers[direction].write(img_2d)

# Cleanup
cap.release()
for writer in writers.values():
    writer.release()

print("Processing complete. Videos saved in respective directories.")
