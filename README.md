# PoseMapper

PoseMapper is a Python-based software for pose estimation and stickman animation overlay on videos, similar to TikTok effects. It uses OpenCV's DNN module with pre-trained OpenPose models to detect human joints and draw a "stickman" figure over the person in the video.

## Features

- Real-time pose estimation using OpenPose
- Multiple rendering styles (default, glow, neon, minimal)
- Webcam and video file support
- Pose data export to JSON
- Customizable appearance (colors, thickness, style)
- Joint angle visualization
- Snapshot functionality
- GPU acceleration support (CUDA)
- Multi-language keypoint labels

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Odeneho-Calculus/PoseMapper.git
cd PoseMapper
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model files:
```bash
# Download all models (recommended)
python download_models.py

# Or download specific models
python download_models.py --model COCO
python download_models.py --model BODY_25
python download_models.py --model MPI

# List available models
python download_models.py --list
```

The models will be downloaded to the `models/` directory with the following structure:
```
models/
├── pose/
│   ├── coco/
│   │   ├── pose_deploy_linevec.prototxt
│   │   └── pose_iter_440000.caffemodel
│   ├── body_25/
│   │   ├── pose_deploy.prototxt
│   │   └── pose_iter_584000.caffemodel
│   └── mpi/
│       ├── pose_deploy_linevec.prototxt
│       └── pose_iter_160000.caffemodel
```

## Usage

### Basic Usage

Process a video file:
```bash
python main.py --input input.mp4 --output output.mp4
```

Use webcam:
```bash
python main.py --input 0
```

### Easy Mode (GUI)
For users who prefer a graphical interface:
```bash
python gui_main.py
```
This provides a user-friendly interface with:
- File picker dialogs for input/output selection
- Dropdown menus for model and style selection
- Checkboxes for display options
- Auto-generation of output filenames
- Command preview before execution

### Advanced Options

#### Rendering Styles
```bash
# Default style
python main.py --style default

# Glow effect
python main.py --style glow

# Neon effect
python main.py --style neon

# Minimal style
python main.py --style minimal
```

#### Custom Appearance
```bash
# Custom colors (B,G,R format)
python main.py --line-color "0,255,0" --circle-color "0,0,255"

# Custom thickness
python main.py --line-thickness 3 --circle-radius 5
```

#### Display Options
```bash
# Show keypoint names and confidence scores
python main.py --show-keypoint-names --show-confidence

# Show joint angles
python main.py --show-angles

# Fullscreen display
python main.py --fullscreen
```

#### Export Options
```bash
# Export pose data to JSON
python main.py --export-json pose_data.json

# Take snapshots every 30 frames
python main.py --snapshot-interval 30
```

#### Performance Options
```bash
# Use GPU acceleration (requires CUDA)
python main.py --use-gpu

# Custom output resolution
python main.py --resolution 1280x720

# Custom FPS
python main.py --fps 30
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `input.mp4` | Input video file or webcam index |
| `--output` | `-o` | `output.mp4` | Output video file |
| `--no-output` | | `False` | Don't save output video |
| `--model-proto` | | `models/pose_deploy.prototxt` | Model .prototxt file |
| `--model-weights` | | `models/pose_iter_440000.caffemodel` | Model .caffemodel file |
| `--threshold` | `-t` | `0.1` | Confidence threshold |
| `--use-gpu` | | `False` | Use GPU acceleration |
| `--style` | | `default` | Rendering style |
| `--line-color` | | `0,255,0` | Line color (B,G,R) |
| `--line-thickness` | | `2` | Line thickness |
| `--circle-color` | | `0,0,255` | Circle color (B,G,R) |
| `--circle-radius` | | `3` | Circle radius |
| `--show-keypoint-names` | | `False` | Show keypoint names |
| `--show-confidence` | | `False` | Show confidence scores |
| `--show-angles` | | `False` | Show joint angles |
| `--fps` | | `None` | Output FPS |
| `--resolution` | | `None` | Output resolution (WIDTHxHEIGHT) |
| `--codec` | | `mp4v` | Video codec |
| `--export-json` | | `None` | Export pose data to JSON |
| `--snapshot-interval` | | `None` | Snapshot every N frames |
| `--no-display` | | `False` | Don't display preview |
| `--fullscreen` | | `False` | Fullscreen display |

### Interactive Controls

When running with display enabled, you can use the following keyboard controls:

- `q`: Quit the application
- `s`: Take a snapshot of the current frame
- `p`: Pause/resume processing

## Project Structure

```
PoseMapper/
├── main.py              # Main application script
├── openpose_detector.py # OpenPose integration
├── renderer.py          # Stickman drawing functions
├── video_processor.py   # Video input/output handling
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── models/             # Model files directory
    ├── pose_deploy.prototxt
    └── pose_iter_440000.caffemodel
```

## Key Components

- **PoseDetector**: Wrapper for OpenPose library with error handling and GPU support
- **PoseRenderer**: Draws the stickman overlay with multiple styles and customization options
- **VideoProcessor**: Handles video capture, writing, and webcam support
- **Main Loop**: Orchestrates frame processing with command-line argument support

## Keypoints

The system detects 18 keypoints following the COCO format:

1. Nose
2. Neck
3. Right Shoulder
4. Right Elbow
5. Right Wrist
6. Left Shoulder
7. Left Elbow
8. Left Wrist
9. Right Hip
10. Right Knee
11. Right Ankle
12. Left Hip
13. Left Knee
14. Left Ankle
15. Right Eye
16. Left Eye
17. Right Ear
18. Left Ear

## Performance Tips

1. **GPU Acceleration**: Use `--use-gpu` if you have a CUDA-compatible GPU
2. **Resolution**: Lower input resolution improves performance
3. **Threshold**: Increase the threshold to reduce false positives
4. **Style**: Minimal style is faster than glow or neon effects

## Troubleshooting

### Model Files Not Found
If you get an error about missing model files, make sure:
1. The `models/` directory exists
2. Both `pose_deploy.prototxt` and `pose_iter_440000.caffemodel` are present
3. The files are not corrupted

### GPU Not Working
If GPU acceleration doesn't work:
1. Ensure you have CUDA installed
2. Install OpenCV with CUDA support: `pip install opencv-python-headless`
3. Check that your GPU is CUDA-compatible

### Performance Issues
If the application runs slowly:
1. Try reducing the input resolution
2. Increase the confidence threshold
3. Use the minimal rendering style
4. Close other applications that might be using the GPU

## Examples

### Basic Video Processing
```bash
python main.py --input dance_video.mp4 --output dance_with_pose.mp4
```

### Webcam with Neon Style
```bash
python main.py --input 0 --style neon --show-keypoint-names
```

### Export Pose Data
```bash
python main.py --input exercise.mp4 --export-json exercise_poses.json --no-display
```

### High Quality Output
```bash
python main.py --input input.mp4 --output high_quality.mp4 --resolution 1920x1080 --fps 60 --codec mp4v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenPose: Real-time multi-person 2D pose estimation
- OpenCV: Computer vision library
- COCO Dataset: Keypoint format and annotations

## Resources

- [OpenPose GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [OpenCV DNN Pose Estimation Tutorial](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/)
- [OpenCV Pose Estimation](https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2017)