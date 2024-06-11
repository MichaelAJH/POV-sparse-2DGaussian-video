import cv2
import numpy as np

# Parameters
width, height = 640, 640
center = (width // 2, height // 2)
radius = 200
num_frames = 90
fps = 30

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./data/wave/circular_wave.mp4', fourcc, fps, (width, height), False)

def polar_to_cartesian(r, theta):
    x = int(r * np.cos(theta) + center[0])
    y = int(r * np.sin(theta) + center[1])
    return (x, y)

def generate_wave_frame(frame_number):
    frame = np.zeros((height, width), dtype=np.uint8)
    for angle in np.linspace(0, 2 * np.pi, 2880, endpoint=False):
        # Sine wave parameters
        sine_value = np.sin(6*angle + 16*np.radians(frame_number)) * 30  # amplitude of the wave
        point_r = radius + sine_value  # radius adjusted by sine value
        point_theta = angle  # angle around the circle
        
        # Calculate Cartesian coordinates
        x, y = polar_to_cartesian(point_r, point_theta)
        
        # Draw the point
        cv2.circle(frame, (x, y), 2, 255, -1)  # draw small circles to form the wave

    return frame

# Generate and write frames to the video
for frame_num in range(num_frames):
    frame = generate_wave_frame(frame_num)
    out.write(frame)

out.release()
