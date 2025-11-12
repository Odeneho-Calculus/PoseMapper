# Configuration for PoseMapper

# OpenPose model configurations
# Available models: COCO (18 points), BODY_25 (25 points), MPI (15 points)
MODEL_TYPE = "COCO"  # Options: COCO, BODY_25, MPI

# Model download URLs
# Note: Weight files are downloaded from SNU server as GitHub doesn't host large binary files
MODEL_DOWNLOAD_URLS = {
    "COCO": {
        "proto": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt",
        "weights": "http://vcl.snu.ac.kr/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"
    },
    "BODY_25": {
        "proto": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt",
        "weights": "http://vcl.snu.ac.kr/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel"
    },
    "MPI": {
        "proto": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec.prototxt",
        "weights": "http://vcl.snu.ac.kr/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel"
    }
}

# Model paths for different pose estimation models
MODEL_CONFIGS = {
    "COCO": {
        "proto": "models/pose/coco/pose_deploy_linevec.prototxt",
        "weights": "models/pose/coco/pose_iter_440000.caffemodel",
        "num_points": 18,
        "pose_pairs": [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                      [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                      [1,0], [0,14], [14,16], [0,15], [15,17]],
        "keypoint_names": [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
            "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
            "LEye", "REar", "LEar"
        ]
    },
    "BODY_25": {
        "proto": "models/pose/body_25/pose_deploy.prototxt",
        "weights": "models/pose/body_25/pose_iter_584000.caffemodel",
        "num_points": 25,
        "pose_pairs": [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                      [8,9], [9,10], [10,11], [8,12], [12,13], [13,14],
                      [1,0], [0,15], [15,17], [0,16], [16,18], [14,19],
                      [19,20], [14,21], [21,22], [11,23], [23,24], [11,24]],
        "keypoint_names": [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
            "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle",
            "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar",
            "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
        ]
    },
    "MPI": {
        "proto": "models/pose/mpi/pose_deploy_linevec.prototxt",
        "weights": "models/pose/mpi/pose_iter_160000.caffemodel",
        "num_points": 15,
        "pose_pairs": [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7],
                      [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]],
        "keypoint_names": [
            "Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
            "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip",
            "LKnee", "LAnkle", "Chest"
        ]
    }
}

# Get current model configuration
CURRENT_MODEL_CONFIG = MODEL_CONFIGS[MODEL_TYPE]
MODEL_PROTO = CURRENT_MODEL_CONFIG["proto"]
MODEL_WEIGHTS = CURRENT_MODEL_CONFIG["weights"]
nPoints = CURRENT_MODEL_CONFIG["num_points"]
POSE_PAIRS = CURRENT_MODEL_CONFIG["pose_pairs"]
KEYPOINT_NAMES = CURRENT_MODEL_CONFIG["keypoint_names"]

# Legacy compatibility
MODEL_PROTO_LEGACY = "models/pose_deploy.prototxt"
MODEL_WEIGHTS_LEGACY = "models/pose_iter_440000.caffemodel"

# Video settings
INPUT_VIDEO = "input.mp4"  # Path to input video or 0 for webcam
OUTPUT_VIDEO = "output.mp4"  # Path to output video

# Pose estimation settings
IN_WIDTH = 368
IN_HEIGHT = 368
THR = 0.1  # Threshold for keypoint detection

# Rendering settings
LINE_COLOR = (0, 255, 0)  # Green
LINE_THICKNESS = 2
CIRCLE_RADIUS = 3
CIRCLE_COLOR = (0, 0, 255)  # Red

# Performance settings
ENABLE_GPU = False  # Set to True to use GPU acceleration (requires CUDA)
GPU_DEVICE_ID = 0  # GPU device ID to use
MAX_BATCH_SIZE = 1  # Batch size for processing

# Display settings
WINDOW_NAME = "PoseMapper"
DISPLAY_WIDTH = 1280  # Maximum display width
DISPLAY_HEIGHT = 720  # Maximum display height
FULLSCREEN = False  # Display in fullscreen mode

# Export settings
EXPORT_FORMAT = "json"  # Format for pose data export
EXPORT_PRECISION = 4  # Decimal precision for exported values
SNAPSHOT_FORMAT = "jpg"  # Format for snapshot images
SNAPSHOT_QUALITY = 95  # Quality for snapshot images (0-100)

# Advanced settings
MULTI_PERSON_DETECTION = False  # Enable multi-person pose detection
MAX_PERSONS = 10  # Maximum number of persons to detect
POSE_TRACKING = False  # Enable pose tracking across frames
TRACKING_THRESHOLD = 0.5  # Threshold for pose tracking

# Debug settings
DEBUG_MODE = False  # Enable debug mode
SAVE_DEBUG_FRAMES = False  # Save debug frames
DEBUG_OUTPUT_DIR = "debug"  # Directory for debug output