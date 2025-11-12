#!/usr/bin/env python3
"""
Test script for pose transfer functionality
"""

import cv2
import numpy as np
from renderer import PoseTransfer

def test_pose_transfer():
    """Test the pose transfer functionality"""
    # Create a simple test character image (a colored rectangle)
    test_image = np.zeros((200, 100, 3), dtype=np.uint8)
    test_image[:, :] = [0, 255, 0]  # Green rectangle
    cv2.imwrite("test_character.png", test_image)

    # Create pose transfer object
    try:
        pose_transfer = PoseTransfer("test_character.png", "COCO")
        print("PoseTransfer initialized successfully")

        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:, :] = [255, 255, 255]  # White background

        # Create some test keypoints (simple pose)
        test_keypoints = [
            [320, 100],  # nose
            [320, 120],  # neck
            [280, 140],  # right shoulder
            [360, 140],  # left shoulder
            [260, 200],  # right elbow
            [380, 200],  # left elbow
            [240, 260],  # right wrist
            [400, 260],  # left wrist
        ] + [[0, 0]] * 10  # Fill remaining keypoints with zeros

        # Test pose transfer
        result = pose_transfer.transfer_pose(test_keypoints, test_frame)
        print("Pose transfer completed successfully")

        # Save result
        cv2.imwrite("test_result.png", result)
        print("Test result saved as test_result.png")

        return True

    except Exception as e:
        print(f"Error in pose transfer test: {e}")
        return False

if __name__ == "__main__":
    test_pose_transfer()