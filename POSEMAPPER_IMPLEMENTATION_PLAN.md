# PoseMapper Implementation Plan

## Overview
PoseMapper is a Python-based software for pose estimation and stickman animation overlay on videos, similar to TikTok effects. It uses OpenCV's DNN module with pre-trained OpenPose models to detect human joints and draw a "stickman" figure over the person in the video.

## Requirements
- Python 3.7+
- OpenCV for video processing and pose estimation
- Pre-trained OpenPose model files

## Libraries and Dependencies
- `opencv-python`
- `numpy`

## Model Download
Download the following files to the `models/` directory:

### Option 1: Use the download script
Run the provided download script to automatically download all required models:
```bash
# For Windows (using curl)
mkdir -p models/pose/{body_25,coco,mpi}

# COCO model (18 keypoints)
curl -L "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt" -o models/pose/coco/pose_deploy_linevec.prototxt
curl -L "http://vcl.snu.ac.kr/OpenPose/models/pose/coco/pose_iter_440000.caffemodel" -o models/pose/coco/pose_iter_440000.caffemodel

# BODY_25 model (25 keypoints)
curl -L "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt" -o models/pose/body_25/pose_deploy.prototxt
curl -L "http://vcl.snu.ac.kr/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel" -o models/pose/body_25/pose_iter_584000.caffemodel

# MPI model (15 keypoints)
curl -L "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec.prototxt" -o models/pose/mpi/pose_deploy_linevec.prototxt
curl -L "http://vcl.snu.ac.kr/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel" -o models/pose/mpi/pose_iter_160000.caffemodel
```

### Option 2: Manual download
Download the following files to the appropriate directories:

**COCO Model (18 keypoints):**
- `models/pose/coco/pose_deploy_linevec.prototxt`: https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
- `models/pose/coco/pose_iter_440000.caffemodel`: http://vcl.snu.ac.kr/OpenPose/models/pose/coco/pose_iter_440000.caffemodel

**BODY_25 Model (25 keypoints):**
- `models/pose/body_25/pose_deploy.prototxt`: https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
- `models/pose/body_25/pose_iter_584000.caffemodel`: http://vcl.snu.ac.kr/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel

**MPI Model (15 keypoints):**
- `models/pose/mpi/pose_deploy_linevec.prototxt`: https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec.prototxt
- `models/pose/mpi/pose_iter_160000.caffemodel`: http://vcl.snu.ac.kr/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel

Note: Weight files (.caffemodel) are downloaded from SNU server as GitHub doesn't host large binary files directly.

## Implementation Steps
1. **Environment Setup**
   - Download OpenPose model files
   - Install Python dependencies
   - Set up project directory structure

2. **Video Input Handling**
   - Load video file or webcam stream using OpenCV
   - Handle different video formats and resolutions

3. **Pose Estimation**
   - Load OpenPose model using OpenCV DNN
   - Process each frame to detect pose keypoints
   - Extract joint coordinates (18 keypoints: nose, neck, shoulders, elbows, wrists, hips, knees, ankles, eyes, ears)
   - Single-person detection (extendable to multi-person)

4. **Stickman Rendering**
   - Draw connections between detected keypoints
   - Customize appearance (colors, thickness, style)
   - Add optional features like joint labels or angles

5. **Output Generation**
   - Display real-time preview
   - Save processed frames to new video file
   - Support different output formats

6. **Advanced Features**
   - Multi-person pose estimation
   - Pose comparison for dance/movement analysis
   - Export pose data (JSON format) for further processing
   - Integration with 3D modeling tools like Blender

## Code Structure
```
PoseMapper/
├── main.py              # Main application script
├── openpose_detector.py # OpenPose integration
├── renderer.py          # Stickman drawing functions
├── video_processor.py   # Video input/output handling
├── config.py            # Configuration settings
└── requirements.txt     # Python dependencies
```

## Key Components
- **OpenPoseDetector**: Wrapper for OpenPose library
- **VideoProcessor**: Handles video capture and writing
- **Renderer**: Draws the stickman overlay on frames
- **Main Loop**: Orchestrates frame processing and display

## Challenges and Considerations
- Model file download and availability
- Computational requirements (GPU recommended for real-time)
- Single-person limitation (current implementation)
- Accuracy in various lighting conditions
- Memory usage for high-resolution videos

## Testing and Validation
- Test with various video types (different resolutions, frame rates)
- Validate pose detection accuracy across different poses
- Performance benchmarking (CPU vs GPU)
- Edge case handling (crowded scenes, partial occlusions)

## Future Enhancements
- 3D pose estimation
- Hand and face detection integration
- Real-time dance comparison features
- Web-based interface
- Machine learning model fine-tuning

## Resources
- OpenPose GitHub: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- OpenCV DNN Pose Estimation Tutorial: https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
- OpenCV Pose Estimation: https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html
- TikTok Effect House: https://effecthouse.tiktok.com/learn/guides/editor-panels/visual-scripting/nodes/body/human-pose-estimation