import cv2
import sys
import os

def blur_video(input_path, output_path, kernel_size):
    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the video frame width, height, and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to write the video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Gaussian blur to each frame
        blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        out.write(blurred_frame)

    # Release everything when the job is finished
    cap.release()
    out.release()
    print("Video processing complete. Output saved to", output_path)

if __name__ == "__main__":
    kernel_size = 7
    directions = ["top", "bottom", "left", "right"]

    # Kernel size must be positive and odd
    if kernel_size <= 0 or kernel_size % 2 == 0:
        print("Kernel size must be a positive odd number.")
        sys.exit(1)

    # for direction in directions:
    #     input_path = "./data/wave/circular_wave_" + direction + ".avi"
    #     os.makedirs(f"./data/wave/blur_{kernel_size}", exist_ok=True)
    #     output_path = f"./data/wave/blur_{kernel_size}/circular_wave_" + direction + ".avi"
    #     blur_video(input_path, output_path, kernel_size)

    blur_video("./data/wave/circular_wave.mp4", "./data/wave/blur_7/circular_wave.mp4", kernel_size)
