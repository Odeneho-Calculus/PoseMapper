import cv2
import argparse
import os
import sys
import time
import numpy as np
from config import *
from openpose_detector import PoseDetector
from renderer import PoseRenderer, draw_pose
from video_processor import VideoProcessor, VideoUtils

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PoseMapper - Pose estimation and stickman animation overlay")
    
    # Input/Output options
    parser.add_argument("--input", "-i", type=str, default=INPUT_VIDEO,
                        help="Input video file path or webcam index (default: input.mp4)")
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_VIDEO,
                        help="Output video file path (default: output.mp4)")
    parser.add_argument("--no-output", action="store_true",
                        help="Don't save output video")
    
    # Model options
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE,
                        choices=["COCO", "BODY_25", "MPI"],
                        help="Pose estimation model type")
    parser.add_argument("--no-background", action="store_true",
                        help="Use black background (equivalent to --background black)")
    parser.add_argument("--model-proto", type=str, default=None,
                        help="Path to model .prototxt file (overrides model-type)")
    parser.add_argument("--model-weights", type=str, default=None,
                        help="Path to model .caffemodel file (overrides model-type)")
    parser.add_argument("--threshold", "-t", type=float, default=THR,
                        help="Confidence threshold for keypoint detection")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU acceleration (requires CUDA)")
    
    # Rendering options
    parser.add_argument("--style", type=str, default="default",
                        choices=["default", "glow", "neon", "minimal"],
                        help="Rendering style")
    parser.add_argument("--background", type=str, default="original",
                        choices=["black", "white", "transparent", "original"],
                        help="Background type (default: original video)")
    parser.add_argument("--line-color", type=str, default="0,255,0",
                        help="Line color in B,G,R format")
    parser.add_argument("--line-thickness", type=int, default=LINE_THICKNESS,
                        help="Line thickness")
    parser.add_argument("--circle-color", type=str, default="0,0,255",
                        help="Circle color in B,G,R format")
    parser.add_argument("--circle-radius", type=int, default=CIRCLE_RADIUS,
                        help="Circle radius")
    parser.add_argument("--show-keypoint-names", action="store_true",
                        help="Display keypoint names")
    parser.add_argument("--show-confidence", action="store_true",
                        help="Display confidence scores")
    parser.add_argument("--show-angles", action="store_true",
                        help="Display joint angles")
    
    # Processing options
    parser.add_argument("--fps", type=float, default=None,
                        help="Output FPS (default: same as input)")
    parser.add_argument("--resolution", type=str, default=None,
                        help="Output resolution as WIDTHxHEIGHT (default: same as input)")
    parser.add_argument("--codec", type=str, default="mp4v",
                        help="Video codec for output")
    
    # Export options
    parser.add_argument("--export-json", type=str, default=None,
                        help="Export pose data to JSON file")
    parser.add_argument("--snapshot-interval", type=int, default=None,
                        help="Take snapshot every N frames")
    
    # Display options
    parser.add_argument("--no-display", action="store_true",
                        help="Don't display real-time preview")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Display in fullscreen mode")
    
    return parser.parse_args()

def parse_color(color_str):
    """Parse color string in B,G,R format to tuple."""
    try:
        return tuple(map(int, color_str.split(',')))
    except:
        print(f"Invalid color format: {color_str}. Using default.")
        return (0, 255, 0)

def parse_resolution(res_str):
    """Parse resolution string in WIDTHxHEIGHT format to tuple."""
    try:
        width, height = map(int, res_str.split('x'))
        return (width, height)
    except:
        print(f"Invalid resolution format: {res_str}. Using default.")
        return None

def main():
    """Main function to run PoseMapper."""
    print("Starting PoseMapper...")
    args = parse_arguments()
    print(f"Arguments parsed. Input: {args.input}, Output: {args.output}")
    
    # Check if input is webcam
    is_webcam = str(args.input).isdigit()
    print(f"Is webcam: {is_webcam}")
    
    # Initialize pose detector
    try:
        print("Initializing pose detector...")
        # Determine model paths
        if args.model_proto and args.model_weights:
            # Use custom model paths
            proto_path = args.model_proto
            weights_path = args.model_weights
        else:
            # Use model type configuration
            from config import MODEL_CONFIGS
            model_config = MODEL_CONFIGS[args.model_type]
            proto_path = model_config["proto"]
            weights_path = model_config["weights"]
            
            # Update global variables for renderer
            global POSE_PAIRS, KEYPOINT_NAMES, nPoints
            POSE_PAIRS = model_config["pose_pairs"]
            KEYPOINT_NAMES = model_config["keypoint_names"]
            nPoints = model_config["num_points"]
        
        print(f"Loading model from {proto_path} and {weights_path}")
        detector = PoseDetector(
            proto_path=proto_path,
            weights_path=weights_path,
            threshold=args.threshold,
            use_gpu=args.use_gpu
        )
        
        print(f"Using {args.model_type} model with {nPoints} keypoints")
    except Exception as e:
        print(f"Error initializing pose detector: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize renderer
    print("Initializing renderer...")
    line_color = parse_color(args.line_color)
    circle_color = parse_color(args.circle_color)
    
    renderer = PoseRenderer(
        line_color=line_color,
        line_thickness=args.line_thickness,
        circle_color=circle_color,
        circle_radius=args.circle_radius,
        show_keypoint_names=args.show_keypoint_names,
        show_confidence=args.show_confidence,
        style=args.style
    )
    print("Renderer initialized")
    
    # Initialize display_available before try block
    display_available = False
    
    # Initialize video processor
    print("Initializing video processor...")
    output_path = None if args.no_output else args.output
    resolution = parse_resolution(args.resolution)
    
    try:
        print(f"Creating VideoProcessor with input={args.input}, output={output_path}")
        with VideoProcessor(
            input_path=args.input,
            output_path=output_path,
            fps=args.fps,
            resolution=resolution,
            codec=args.codec,
            is_webcam=is_webcam
        ) as processor:
            print("VideoProcessor initialized successfully")
            
            # Setup display window
            if not args.no_display:
                try:
                    window_name = 'PoseMapper'
                    if args.fullscreen:
                        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    display_available = True
                    print("Display window created successfully")
                except Exception as e:
                    print(f"Warning: Could not create display window: {e}")
                    print("Continuing without display (headless mode)")
                    args.no_display = True
            
            # For pose data export
            pose_data_list = []
            
            # Main processing loop
            frame_count = 0
            start_time = time.time()
            print("Starting main processing loop...", flush=True)
            print("Model loaded successfully. Using CPU", flush=True)

            while True:
                ret, frame = processor.get_frame()
                if not ret:
                    if not is_webcam:
                        print("End of video reached")
                    break
                
                if frame_count == 0:
                    print(f"Video file loaded: {args.input}", flush=True)
                    print(f"Total frames: {processor.total_frames}", flush=True)
                    print(f"Video writer initialized: {args.output}", flush=True)
                    print(f"Output resolution: {processor.resolution}, FPS: {processor.fps}", flush=True)
                elif frame_count % 30 == 0:
                    print(f"Processing frame {frame_count}", flush=True)
                
                # Detect pose
                points, confidences = detector.detect(frame)

                # Create background frame if specified
                if args.background == "black":
                    # Use black background
                    render_frame = np.zeros_like(frame)
                elif args.background == "white":
                    render_frame = np.full_like(frame, 255)
                elif args.background == "transparent":
                    # For transparent, we'll use black background for now
                    # (proper alpha channel support would require different codec)
                    render_frame = np.zeros_like(frame)
                else:
                    # Default: use original video background
                    render_frame = frame.copy()

                # Render pose
                render_frame = renderer.draw_pose(render_frame, points, confidences)

                # Show angles if requested
                if args.show_angles:
                    render_frame = renderer.draw_pose_angles(render_frame, points)

                # Write frame to output
                processor.write_frame(render_frame)
                
                # Export pose data if requested
                if args.export_json:
                    pose_data = detector.get_pose_data(render_frame, frame_count)
                    pose_data_list.append(pose_data)

                # Take snapshots if requested
                if args.snapshot_interval and frame_count % args.snapshot_interval == 0:
                    snapshot_path = f"snapshot_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(snapshot_path, render_frame)
                    print(f"Snapshot saved: {snapshot_path}")

                # Display frame
                if not args.no_display and display_available:
                    try:
                        cv2.imshow('PoseMapper', render_frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            print("Quitting...")
                            break
                        elif key == ord('s'):  # Take snapshot on 's' key
                            snapshot_path = f"snapshot_frame_{frame_count:06d}.jpg"
                            cv2.imwrite(snapshot_path, render_frame)
                            print(f"Snapshot saved: {snapshot_path}")
                        elif key == ord('p'):  # Pause on 'p' key
                            print("Paused. Press any key to continue...")
                            cv2.waitKey(0)
                    except Exception as e:
                        print(f"Warning: Display error: {e}")
                        print("Continuing without display")
                        args.no_display = True
                
                # Print progress for video files
                if not is_webcam and frame_count % 10 == 0:  # More frequent updates
                    progress = processor.get_progress()
                    if progress is not None:
                        print(f"Progress: {progress:.1f}%", flush=True)  # Force immediate output
                
                frame_count += 1
            
            # Calculate and print statistics
            end_time = time.time()
            processing_time = end_time - start_time
            
            if not is_webcam:
                print(f"\nProcessing complete!")
                print(f"Total frames processed: {frame_count}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"Average FPS: {frame_count / processing_time:.2f}")
            
            # Export pose data to JSON if requested
            if args.export_json and pose_data_list:
                import json
                with open(args.export_json, 'w') as f:
                    json.dump(pose_data_list, f, indent=2)
                print(f"Pose data exported to {args.export_json}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
    
    finally:
        if not args.no_display and display_available:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Warning: Could not close display windows: {e}")

if __name__ == "__main__":
    main()