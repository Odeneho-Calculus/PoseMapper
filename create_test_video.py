import cv2
import numpy as np

# Create a simple test video
width, height = 640, 480
fps = 30
duration = 5  # seconds
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('input.mp4', fourcc, fps, (width, height))

# Generate frames
for i in range(fps * duration):
    # Create a frame with a moving circle
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add a background
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Add a moving circle (simulating a person)
    center_x = int(width // 2 + 100 * np.sin(2 * np.pi * i / (fps * 2)))
    center_y = int(height // 2 + 50 * np.cos(2 * np.pi * i / (fps * 2)))
    cv2.circle(frame, (center_x, center_y), 50, (0, 255, 255), -1)
    
    # Add some text
    cv2.putText(frame, f'Frame {i+1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)

out.release()
print(f"Test video 'input.mp4' created with {fps * duration} frames")