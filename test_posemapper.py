#!/usr/bin/env python3
"""
Test script for PoseMapper implementation
"""

import cv2
import numpy as np
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIGS, MODEL_TYPE
from openpose_detector import PoseDetector
from renderer import PoseRenderer
from video_processor import VideoProcessor

def create_test_frame():
    """Create a simple test frame with a stick figure"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Draw a simple stick figure
    # Head
    cv2.circle(frame, (320, 100), 30, (200, 200, 200), -1)
    # Body
    cv2.line(frame, (320, 130), (320, 250), (200, 200, 200), 5)
    # Arms
    cv2.line(frame, (320, 150), (250, 200), (200, 200, 200), 5)
    cv2.line(frame, (320, 150), (390, 200), (200, 200, 200), 5)
    # Legs
    cv2.line(frame, (320, 250), (280, 350), (200, 200, 200), 5)
    cv2.line(frame, (320, 250), (360, 350), (200, 200, 200), 5)
    
    return frame

def test_pose_detector():
    """Test the pose detector"""
    print("Testing PoseDetector...")
    
    # Test each model type
    for model_type in MODEL_CONFIGS.keys():
        print(f"\nTesting {model_type} model...")
        model_config = MODEL_CONFIGS[model_type]
        
        try:
            detector = PoseDetector(
                proto_path=model_config["proto"],
                weights_path=model_config["weights"],
                threshold=0.1
            )
            
            # Create test frame
            frame = create_test_frame()
            
            # Detect pose
            points, confidences = detector.detect(frame)
            
            print(f"  Model loaded successfully")
            print(f"  Detected {len(points)} keypoints")
            print(f"  Valid keypoints: {sum(1 for p in points if p is not None)}")
            
        except Exception as e:
            print(f"  Error with {model_type} model: {e}")
    
    print("\nPoseDetector test complete!")

def test_renderer():
    """Test the pose renderer"""
    print("\nTesting PoseRenderer...")
    
    try:
        # Create test frame
        frame = create_test_frame()
        
        # Create dummy pose points (simplified for testing)
        points = [
            (320, 70),   # Nose
            (320, 100),  # Neck
            (250, 150),  # RShoulder
            (200, 200),  # RElbow
            (150, 250),  # RWrist
            (390, 150),  # LShoulder
            (440, 200),  # LElbow
            (490, 250),  # LWrist
            (280, 250),  # RHip
            (260, 320),  # RKnee
            (240, 390),  # RAnkle
            (360, 250),  # LHip
            (380, 320),  # LKnee
            (400, 390),  # LAnkle
            (310, 70),   # REye
            (330, 70),   # LEye
            (300, 70),   # REar
            (340, 70)    # LEar
        ]
        
        # Test different rendering styles
        styles = ["default", "glow", "neon", "minimal"]
        
        for style in styles:
            renderer = PoseRenderer(style=style)
            rendered_frame = renderer.draw_pose(frame.copy(), points)
            
            # Save test image
            output_path = f"test_render_{style}.jpg"
            cv2.imwrite(output_path, rendered_frame)
            print(f"  Saved {style} style test to {output_path}")
        
        print("  PoseRenderer test complete!")
        
    except Exception as e:
        print(f"  Error in PoseRenderer test: {e}")

def test_video_processor():
    """Test the video processor"""
    print("\nTesting VideoProcessor...")
    
    try:
        # Create a simple test video
        create_test_video()
        
        # Test video processor
        with VideoProcessor("input.mp4", "test_output.mp4") as processor:
            frame_count = 0
            while True:
                ret, frame = processor.get_frame()
                if not ret:
                    break
                
                # Simple processing: add text
                cv2.putText(frame, f"Frame {frame_count}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                processor.write_frame(frame)
                frame_count += 1
                
                if frame_count >= 30:  # Only process first 30 frames
                    break
        
        print(f"  Processed {frame_count} frames")
        print("  VideoProcessor test complete!")
        
    except Exception as e:
        print(f"  Error in VideoProcessor test: {e}")

def create_test_video():
    """Create a simple test video if it doesn't exist"""
    if os.path.exists("input.mp4"):
        return
    
    print("  Creating test video...")
    width, height = 640, 480
    fps = 30
    duration = 2  # seconds
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('input.mp4', fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(fps * duration):
        frame = create_test_frame()
        cv2.putText(frame, f'Frame {i+1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print("  Test video created")

def main():
    """Run all tests"""
    print("PoseMapper Test Suite")
    print("=" * 50)
    
    # Test components
    test_pose_detector()
    test_renderer()
    test_video_processor()
    
    print("\n" + "=" * 50)
    print("All tests complete!")
    print("\nTo run PoseMapper with a video:")
    print("python main.py --input input.mp4 --output output.mp4")
    print("\nTo run PoseMapper with webcam:")
    print("python main.py --input 0")

if __name__ == "__main__":
    main()