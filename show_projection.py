import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load video
video_path = 'circular_rotating_wave.mp4'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Failed to open video"

# Create figure for plotting
fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # A 3x3 grid

# Remove the unused subplots
for ax in [axs[0][0], axs[0][2], axs[2][0], axs[2][2]]:
    ax.remove()

# Configure axes for the video and bars
ax_video = axs[1][1]
ax_video.axis('off')  # Video in the center

ax_tb = axs[0][1]  # Top
ax_tb.axis('off')

ax_bt = axs[2][1]  # Bottom
ax_bt.axis('off')

ax_lr = axs[1][0]  # Left
ax_lr.axis('off')

ax_rl = axs[1][2]  # Right
ax_rl.axis('off')

# Initialize the video plot
im_video = ax_video.imshow(np.zeros((320, 320)), cmap='gray', vmin=0, vmax=1)

# Initialize the bar plots
bars_tb = ax_tb.bar(range(320), np.zeros(320), color='gray', orientation='horizontal')
bars_bt = ax_bt.bar(range(320), np.zeros(320), color='gray', orientation='horizontal')
bars_lr = ax_lr.barh(range(320), np.zeros(320), color='gray')
bars_rl = ax_rl.barh(range(320), np.zeros(320), color='gray')

def update(frame_index):
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = (frame > 127).astype(np.uint8)  # Ensure binary

    # Calculate weighted sums
    top_to_bottom = np.dot(np.arange(1, 321), frame)
    bottom_to_top = np.dot(np.arange(320, 0, -1), frame)
    left_to_right = np.dot(frame, np.arange(1, 321))
    right_to_left = np.dot(frame, np.arange(320, 0, -1))

    # Update the video frame
    im_video.set_data(frame)

    # Update bar graphs
    for bar, value in zip(bars_tb, top_to_bottom):
        bar.set_height(value)
    for bar, value in zip(bars_bt, bottom_to_top):
        bar.set_height(value)
    for bar, value in zip(bars_lr, left_to_right):
        bar.set_width(value)
    for bar, value in zip(bars_rl, right_to_left):
        bar.set_width(value)

    return [im_video] + list(bars_tb) + list(bars_bt) + list(bars_lr) + list(bars_rl)

# Create animation
ani = FuncAnimation(fig, update, frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), blit=True, repeat=False)
plt.tight_layout()
plt.show()

# Release resources
cap.release()
