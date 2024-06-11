import cv2
import numpy as np

# Parameters
width, height = 640, 640
num_frames_per_shape = 15
fps = 30

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./data/polygons/shape_transition.mp4', fourcc, fps, (width, height), False)

def generate_polygon_vertices(sides, radius=width//3, center=(width//2, height//2)):
    return [(center[0] + radius * np.cos(2 * np.pi * i / sides), center[1] + radius * np.sin(2 * np.pi * i / sides)) for i in range(sides)]

def interpolate_points(p1, p2, num_frames):
    return [(int(p1[0] + (p2[0] - p1[0]) * i / num_frames), int(p1[1] + (p2[1] - p1[1]) * i / num_frames)) for i in range(num_frames)]

def interpolate_shapes(shape1, shape2, num_frames):
    interpolated_frames = []
    for i in range(num_frames):
        frame = np.zeros((height, width), dtype=np.uint8)
        num_vertices = len(shape1)
        polygon = [interpolate_points(shape1[j], shape2[j], num_frames)[i] for j in range(num_vertices)]
        cv2.polylines(frame, [np.array(polygon, np.int32)], isClosed=True, color=255, thickness=2)
        interpolated_frames.append(frame)
    return interpolated_frames

# Generate frames for each shape transition
shapes = [generate_polygon_vertices(sides) for sides in range(3, 13)]
# shapes.extend(list(reversed(shapes)))

for i in range(len(shapes) - 1):
    frames = interpolate_shapes(shapes[i], shapes[i + 1], num_frames_per_shape)
    for frame in frames:
        out.write(frame)

out.release()
