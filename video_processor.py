import cv2
import os
from config import *

class VideoProcessor:
    def __init__(self, input_path, output_path=None, fps=None, resolution=None,
                 codec='mp4v', is_webcam=False):
        """
        Initialize the VideoProcessor with input/output settings.
        
        Args:
            input_path: Path to input video file or webcam index (0 for default webcam)
            output_path: Path to output video file (None for no output)
            fps: Output FPS (None to use input FPS)
            resolution: Output resolution as (width, height) tuple (None to use input resolution)
            codec: FourCC codec for output video
            is_webcam: Whether input is from webcam
        """
        self.input_path = input_path
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.codec = codec
        self.is_webcam = is_webcam or str(input_path).isdigit()
        
        # Initialize video capture
        self.cap = None
        self.out = None
        self.frame_count = 0
        self.total_frames = 0
        
        # Setup video capture and writer
        self._setup_capture()
        if output_path:
            self._setup_writer()
    
    def _setup_capture(self):
        """Setup video capture with error handling."""
        try:
            if self.is_webcam:
                # For webcam, convert string index to integer if needed
                cam_index = int(self.input_path) if str(self.input_path).isdigit() else 0
                self.cap = cv2.VideoCapture(cam_index)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open webcam at index {cam_index}")
                
                # Set webcam resolution if specified
                if self.resolution:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                
                print(f"Webcam initialized at index {cam_index}")
            else:
                # For video file
                if not os.path.exists(self.input_path):
                    raise FileNotFoundError(f"Input video file not found: {self.input_path}")
                
                self.cap = cv2.VideoCapture(self.input_path)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open video file: {self.input_path}")
                
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Video file loaded: {self.input_path}")
                print(f"Total frames: {self.total_frames}")
            
            # Get video properties
            self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.input_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.input_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use input properties if not specified
            if self.fps is None:
                self.fps = self.input_fps
            if self.resolution is None:
                self.resolution = (self.input_width, self.input_height)
                
        except Exception as e:
            print(f"Error setting up video capture: {e}")
            raise
    
    def _setup_writer(self):
        """Setup video writer with error handling."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.out = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, self.resolution
            )
            
            if not self.out.isOpened():
                raise RuntimeError(f"Could not open video writer for: {self.output_path}")
            
            print(f"Video writer initialized: {self.output_path}")
            print(f"Output resolution: {self.resolution}, FPS: {self.fps}")
            
        except Exception as e:
            print(f"Error setting up video writer: {e}")
            raise
    
    def get_frame(self):
        """
        Get the next frame from the video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            
            # Resize frame if output resolution is different
            if self.resolution and (frame.shape[1], frame.shape[0]) != self.resolution:
                frame = cv2.resize(frame, self.resolution)
        
        return ret, frame
    
    def write_frame(self, frame, original_frame=None):
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write (with pose overlay)
            original_frame: Original frame from video (for background preservation)
        """
        if self.out is not None and frame is not None:
            # Just use the frame as is (background is already handled in main.py)
            output_frame = frame
            
            # Resize frame if needed
            if self.resolution and (output_frame.shape[1], output_frame.shape[0]) != self.resolution:
                output_frame = cv2.resize(output_frame, self.resolution)
            
            self.out.write(output_frame)
    
    def get_progress(self):
        """
        Get processing progress as a percentage.
        
        Returns:
            Progress percentage (0-100) or None for webcam
        """
        if self.is_webcam or self.total_frames == 0:
            return None
        return (self.frame_count / self.total_frames) * 100
    
    def get_frame_info(self):
        """
        Get information about the current frame.
        
        Returns:
            Dictionary with frame information
        """
        return {
            "frame_number": self.frame_count,
            "fps": self.fps,
            "resolution": self.resolution,
            "progress": self.get_progress()
        }
    
    def set_output_path(self, output_path):
        """
        Set or change the output path.
        
        Args:
            output_path: New output path
        """
        if self.out:
            self.out.release()
        
        self.output_path = output_path
        if output_path:
            self._setup_writer()
    
    def snapshot(self, filename=None):
        """
        Take a snapshot of the current frame.
        
        Args:
            filename: Filename for the snapshot (auto-generated if None)
            
        Returns:
            Path to the saved snapshot or None if failed
        """
        ret, frame = self.get_frame()
        if not ret:
            return None
            
        if filename is None:
            filename = f"snapshot_frame_{self.frame_count}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
        return filename
    
    def release(self):
        """Release video capture and writer resources."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        print("Video resources released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

class VideoUtils:
    @staticmethod
    def get_video_info(video_path):
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    @staticmethod
    def extract_frames(video_path, output_dir, max_frames=None, step=1):
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save frames
            max_frames: Maximum number of frames to extract
            step: Extract every nth frame
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % step == 0:
                filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(filename, frame)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames to {output_dir}")
        return extracted_count