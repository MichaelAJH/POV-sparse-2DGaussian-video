import cv2
import numpy as np
import torch
import os

# Parameters
video_path = 'circular_rotating_wave.mp4'
output_dir = 'output_tensors'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
num_frames = 360
frame_height, frame_width = 320, 320
fps = 30

# Initialize video reader
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Failed to open video"

def process_frame(frame):
    # Convert frame to binary if not already
    if frame.max() > 1:
        _, frame = cv2.threshold(frame, 127, 1, cv2.THRESH_BINARY)
    
    # Calculate weighted sums
    top_to_bottom = np.dot(frame, np.arange(1, frame_height + 1))
    bottom_to_top = np.dot(frame, np.arange(frame_height, 0, -1))
    left_to_right = np.dot(frame.T, np.arange(1, frame_width + 1))
    right_to_left = np.dot(frame.T, np.arange(frame_width, 0, -1))
    
    # Stack to a tensor
    tensor = torch.tensor([top_to_bottom, bottom_to_top, left_to_right, right_to_left], dtype=torch.float32)
    return tensor

# Process each frame
for frame_index in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    tensor = process_frame(frame)
    
    # Save the tensor
    torch.save(tensor, os.path.join(output_dir, f'tensor_{frame_index+1}.pt'))

cap.release()
print("Processing complete. Tensors saved.")
