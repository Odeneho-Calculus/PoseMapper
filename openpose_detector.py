import cv2
import numpy as np
import os
import json
from config import *

# Import model configuration from config
nPoints = nPoints
POSE_PAIRS = POSE_PAIRS
KEYPOINT_NAMES = KEYPOINT_NAMES

class PoseDetector:
    def __init__(self, proto_path=None, weights_path=None, threshold=THR,
                 input_width=IN_WIDTH, input_height=IN_HEIGHT, use_gpu=False):
        """
        Initialize the PoseDetector with model paths and parameters.
        
        Args:
            proto_path: Path to the .prototxt file
            weights_path: Path to the .caffemodel file
            threshold: Confidence threshold for keypoint detection
            input_width: Input width for the network
            input_height: Input height for the network
            use_gpu: Whether to use GPU acceleration
        """
        self.proto_path = proto_path or MODEL_PROTO
        self.weights_path = weights_path or MODEL_WEIGHTS
        self.threshold = threshold
        self.input_width = input_width
        self.input_height = input_height
        self.net = None
        self.use_gpu = use_gpu
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the OpenPose model with error handling."""
        try:
            if not os.path.exists(self.proto_path):
                raise FileNotFoundError(f"Model prototxt file not found: {self.proto_path}")
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Model weights file not found: {self.weights_path}")
                
            self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.weights_path)
            
            if self.use_gpu:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
            print(f"Model loaded successfully. Using {'GPU' if self.use_gpu else 'CPU'}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect(self, frame):
        """
        Detect pose keypoints in the given frame.
        
        Args:
            frame: Input image/frame
            
        Returns:
            List of detected keypoints (x, y) or None for undetected points
        """
        if self.net is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
            
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
            
        try:
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            # Prepare input blob
            inpBlob = cv2.dnn.blobFromImage(
                frame, 1.0 / 255, (self.input_width, self.input_height),
                (0, 0, 0), swapRB=False, crop=False
            )
            
            # Forward pass
            self.net.setInput(inpBlob)
            output = self.net.forward()

            H = output.shape[2]
            W = output.shape[3]

            # Extract keypoints
            points = []
            confidences = []
            
            for i in range(nPoints):
                probMap = output[0, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale point to frame dimensions
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H

                if prob > self.threshold:
                    points.append((int(x), int(y)))
                    confidences.append(float(prob))
                else:
                    points.append(None)
                    confidences.append(0.0)

            return points, confidences
            
        except Exception as e:
            print(f"Error during pose detection: {e}")
            # Return empty results on error
            return [None] * nPoints, [0.0] * nPoints
    
    def detect_multiple(self, frame, max_persons=10):
        """
        Detect poses for multiple people in the frame.
        Note: This is a simplified implementation that focuses on the most prominent person.
        For true multi-person detection, a more sophisticated approach would be needed.
        
        Args:
            frame: Input image/frame
            max_persons: Maximum number of persons to detect
            
        Returns:
            List of pose results, each containing keypoints and confidences
        """
        # For now, return single person detection
        # In a full implementation, this would use a more sophisticated approach
        points, confidences = self.detect(frame)
        return [(points, confidences)]
    
    def get_pose_data(self, frame, frame_number=0):
        """
        Get pose data in a structured format for export.
        
        Args:
            frame: Input image/frame
            frame_number: Frame number for reference
            
        Returns:
            Dictionary containing pose data
        """
        points, confidences = self.detect(frame)
        
        pose_data = {
            "frame": frame_number,
            "keypoints": []
        }
        
        for i, (point, confidence) in enumerate(zip(points, confidences)):
            keypoint_data = {
                "id": i,
                "name": KEYPOINT_NAMES[i],
                "confidence": confidence
            }
            
            if point:
                keypoint_data["position"] = {
                    "x": point[0],
                    "y": point[1]
                }
            
            pose_data["keypoints"].append(keypoint_data)
        
        return pose_data
    
    def export_pose_data(self, frames, output_path="pose_data.json"):
        """
        Export pose data for multiple frames to a JSON file.
        
        Args:
            frames: List of frames or a video capture object
            output_path: Path to save the JSON file
        """
        all_pose_data = []
        
        if isinstance(frames, cv2.VideoCapture):
            frame_count = 0
            while True:
                ret, frame = frames.read()
                if not ret:
                    break
                    
                pose_data = self.get_pose_data(frame, frame_count)
                all_pose_data.append(pose_data)
                frame_count += 1
        else:
            # Assume it's a list of frames
            for i, frame in enumerate(frames):
                pose_data = self.get_pose_data(frame, i)
                all_pose_data.append(pose_data)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(all_pose_data, f, indent=2)
        
        print(f"Pose data exported to {output_path}")
        return all_pose_data