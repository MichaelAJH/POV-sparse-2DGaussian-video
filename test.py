import cv2
import numpy as np

out = cv2.VideoWriter('test_output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (320, 320), 0)

for i in range(60):  # Create 60 frames
    frame = np.random.randint(0, 256, (320, 320), dtype=np.uint8)
    out.write(frame)

out.release()
