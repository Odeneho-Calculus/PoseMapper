import cv2
import numpy as np
from config import *

class PoseRenderer:
    def __init__(self, line_color=LINE_COLOR, line_thickness=LINE_THICKNESS,
                 circle_color=CIRCLE_COLOR, circle_radius=CIRCLE_RADIUS,
                 show_keypoint_names=False, show_confidence=False,
                 style="default"):
        """
        Initialize the PoseRenderer with customizable parameters.
        
        Args:
            line_color: Color for pose lines (B, G, R)
            line_thickness: Thickness of pose lines
            circle_color: Color for keypoint circles (B, G, R)
            circle_radius: Radius of keypoint circles
            show_keypoint_names: Whether to display keypoint names
            show_confidence: Whether to display confidence scores
            style: Rendering style ("default", "glow", "neon", "minimal")
        """
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.circle_color = circle_color
        self.circle_radius = circle_radius
        self.show_keypoint_names = show_keypoint_names
        self.show_confidence = show_confidence
        self.style = style
        
        # Style-specific configurations
        self._configure_style()
    
    def _configure_style(self):
        """Configure style-specific parameters."""
        if self.style == "glow":
            self.glow_layers = 3
            self.glow_thickness = self.line_thickness + 2
        elif self.style == "neon":
            self.neon_color = (255, 0, 255)  # Magenta
            self.neon_thickness = self.line_thickness + 1
        elif self.style == "minimal":
            self.minimal_thickness = 1
            self.minimal_radius = 2
    
    def draw_pose(self, frame, points, confidences=None):
        """
        Draw pose on the frame with the configured style.
        
        Args:
            frame: Input frame
            points: List of keypoint coordinates
            confidences: List of confidence values for each keypoint
            
        Returns:
            Frame with pose drawn on it
        """
        if self.style == "glow":
            return self._draw_glow_style(frame, points, confidences)
        elif self.style == "neon":
            return self._draw_neon_style(frame, points, confidences)
        elif self.style == "minimal":
            return self._draw_minimal_style(frame, points, confidences)
        else:
            return self._draw_default_style(frame, points, confidences)
    
    def _draw_default_style(self, frame, points, confidences=None):
        """Draw pose with default style."""
        # Draw skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                # Adjust color based on confidence if available
                color = self.line_color
                if confidences:
                    avg_confidence = (confidences[partA] + confidences[partB]) / 2
                    color = tuple(int(c * avg_confidence) for c in self.line_color)
                
                cv2.line(frame, points[partA], points[partB], color, self.line_thickness)

        # Draw keypoints
        for i, point in enumerate(points):
            if point:
                # Adjust color based on confidence if available
                color = self.circle_color
                if confidences:
                    color = tuple(int(c * confidences[i]) for c in self.circle_color)
                
                cv2.circle(frame, point, self.circle_radius, color, -1)
                
                # Add keypoint names if requested
                if self.show_keypoint_names and i < len(KEYPOINT_NAMES):
                    cv2.putText(frame, KEYPOINT_NAMES[i],
                               (point[0] + 10, point[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add confidence values if requested
                if self.show_confidence and confidences and i < len(confidences):
                    cv2.putText(frame, f"{confidences[i]:.2f}",
                               (point[0] + 10, point[1] + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return frame
    
    def _draw_glow_style(self, frame, points, confidences=None):
        """Draw pose with glow effect."""
        # Draw glow layers
        for layer in range(self.glow_layers, 0, -1):
            alpha = 0.1 * (self.glow_layers - layer + 1)
            glow_color = tuple(int(c * alpha) for c in self.line_color)
            
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB],
                            glow_color, self.line_thickness + layer * 2)
        
        # Draw main skeleton
        return self._draw_default_style(frame, points, confidences)
    
    def _draw_neon_style(self, frame, points, confidences=None):
        """Draw pose with neon effect."""
        # Draw neon outline
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB],
                        self.neon_color, self.neon_thickness)
        
        # Draw main skeleton
        return self._draw_default_style(frame, points, confidences)
    
    def _draw_minimal_style(self, frame, points, confidences=None):
        """Draw pose with minimal style."""
        # Draw thin skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB],
                        self.line_color, self.minimal_thickness)

        # Draw small keypoints
        for point in points:
            if point:
                cv2.circle(frame, point, self.minimal_radius, self.circle_color, -1)

        return frame
    
    def draw_pose_angles(self, frame, points, angles_to_show=None):
        """
        Draw angles between specific joints.
        
        Args:
            frame: Input frame
            points: List of keypoint coordinates
            angles_to_show: List of tuples (joint1, joint2, joint3) to calculate angles for
                           Defaults to common body angles
        """
        if angles_to_show is None:
            angles_to_show = [
                (5, 6, 7),   # Left arm
                (2, 3, 4),   # Right arm
                (11, 12, 13), # Left leg
                (8, 9, 10)   # Right leg
            ]
        
        for angle_triplet in angles_to_show:
            joint1, joint2, joint3 = angle_triplet
            
            if (points[joint1] and points[joint2] and points[joint3]):
                # Calculate angle
                angle = self._calculate_angle(
                    points[joint1], points[joint2], points[joint3]
                )
                
                # Draw angle text
                cv2.putText(frame, f"{angle:.1f}°", points[joint2],
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points."""
        # Vector from point2 to point1
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        # Vector from point2 to point3
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Convert to degrees
        return np.degrees(angle)
    
    def draw_pose_comparison(self, frame, points1, points2, color1=(0, 255, 0), color2=(0, 0, 255)):
        """
        Draw two poses on the same frame for comparison.
        
        Args:
            frame: Input frame
            points1: First pose keypoints
            points2: Second pose keypoints
            color1: Color for first pose
            color2: Color for second pose
        """
        # Draw first pose
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points1[partA] and points1[partB]:
                cv2.line(frame, points1[partA], points1[partB], color1, self.line_thickness)
        
        # Draw second pose
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points2[partA] and points2[partB]:
                cv2.line(frame, points2[partA], points2[partB], color2, self.line_thickness)
        
        return frame

# Default function for backward compatibility
def draw_pose(frame, points, confidences=None):
    """
    Default pose drawing function for backward compatibility.
    
    Args:
        frame: Input frame
        points: List of keypoint coordinates
        confidences: List of confidence values for each keypoint
        
    Returns:
        Frame with pose drawn on it
    """
    renderer = PoseRenderer()
    return renderer.draw_pose(frame, points, confidences)