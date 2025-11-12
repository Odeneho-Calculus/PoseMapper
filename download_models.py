#!/usr/bin/env python3
"""
Download script for OpenPose models required by PoseMapper
"""

import os
import sys
import urllib.request
import argparse
from pathlib import Path

# Model configurations
# Note: Weight files are downloaded from SNU server as GitHub doesn't host large binary files
MODEL_CONFIGS = {
    "COCO": {
        "proto": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt",
        "weights": "http://vcl.snu.ac.kr/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
        "proto_path": "models/pose/coco/pose_deploy_linevec.prototxt",
        "weights_path": "models/pose/coco/pose_iter_440000.caffemodel",
        "description": "COCO model (18 keypoints)"
    },
    "BODY_25": {
        "proto": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt",
        "weights": "http://vcl.snu.ac.kr/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
        "proto_path": "models/pose/body_25/pose_deploy.prototxt",
        "weights_path": "models/pose/body_25/pose_iter_584000.caffemodel",
        "description": "BODY_25 model (25 keypoints)"
    },
    "MPI": {
        "proto": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec.prototxt",
        "weights": "http://vcl.snu.ac.kr/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel",
        "proto_path": "models/pose/mpi/pose_deploy_linevec.prototxt",
        "weights_path": "models/pose/mpi/pose_iter_160000.caffemodel",
        "description": "MPI model (15 keypoints)"
    }
}

def download_file(url, destination, description="file"):
    """Download a file from URL to destination with progress bar."""
    print(f"Downloading {description}...")
    print(f"From: {url}")
    print(f"To: {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    def progress_hook(block_num, block_size, total_size):
        """Progress callback for urllib"""
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            sys.stdout.write(f"\rProgress: {percent}% ({block_num * block_size}/{total_size} bytes)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloaded: {block_num * block_size} bytes")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n✓ Successfully downloaded {description}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {description}: {e}")
        return False

def check_file_exists(filepath):
    """Check if a file exists and is not empty."""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

def download_model(model_name, force=False):
    """Download a specific model."""
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown model '{model_name}'. Available models: {list(MODEL_CONFIGS.keys())}")
        return False
    
    config = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"Downloading {config['description']}")
    print(f"{'='*60}")
    
    # Check if files already exist
    proto_exists = check_file_exists(config["proto_path"])
    weights_exists = check_file_exists(config["weights_path"])
    
    if proto_exists and weights_exists and not force:
        print(f"✓ Model files already exist. Use --force to re-download.")
        return True
    
    # Download proto file
    if not proto_exists or force:
        if not download_file(config["proto"], config["proto_path"], f"{model_name} prototype"):
            return False
    
    # Download weights file
    if not weights_exists or force:
        if not download_file(config["weights"], config["weights_path"], f"{model_name} weights"):
            return False
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download OpenPose models for PoseMapper")
    parser.add_argument("--model", "-m", type=str, choices=["COCO", "BODY_25", "MPI", "all"],
                        default="all", help="Model to download (default: all)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-download even if files exist")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  {name}: {config['description']}")
        return
    
    print("PoseMapper Model Downloader")
    print("=" * 60)
    
    if args.model == "all":
        success = True
        for model_name in MODEL_CONFIGS.keys():
            if not download_model(model_name, args.force):
                success = False
    else:
        success = download_model(args.model, args.force)
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All downloads completed successfully!")
        print("\nYou can now run PoseMapper with:")
        print("  python main.py --input input.mp4 --output output.mp4")
        print("  python main.py --input 0  # For webcam")
    else:
        print("✗ Some downloads failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()