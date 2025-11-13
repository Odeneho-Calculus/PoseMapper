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


class PoseTransfer:
    """
    Class for transferring poses to character images using piece-wise image warping.
    Divides the character into body segments for more natural pose animation.
    """

    def __init__(self, character_image_path, pose_model="COCO"):
        """
        Initialize the PoseTransfer with a character image.

        Args:
            character_image_path: Path to the character image
            pose_model: Pose model type ("COCO", "BODY_25", "MPI")
        """
        self.character_image = cv2.imread(character_image_path)
        if self.character_image is None:
            raise ValueError(f"Could not load character image: {character_image_path}")

        self.pose_model = pose_model
        self.height, self.width = self.character_image.shape[:2]

        # Define reference pose keypoints for the character image
        # These are the "rest pose" positions where the character looks natural
        self._define_reference_pose()

        # Define body segments and their corresponding keypoints
        self._define_body_segments()

        # Create segment masks for piece-wise warping
        self._create_segment_masks()

    def _define_reference_pose(self):
        """Define reference pose keypoints for the character image."""
        # Create a neutral T-pose for the character
        # These coordinates are relative to the character image dimensions
        center_x, center_y = self.width // 2, self.height // 2

        if self.pose_model == "COCO":
            # COCO model has 18 keypoints
            self.reference_keypoints = np.array([
                [center_x, center_y - self.height//4],  # 0: nose
                [center_x, center_y - self.height//6],  # 1: neck
                [center_x - self.width//8, center_y - self.height//8],  # 2: right shoulder
                [center_x + self.width//8, center_y - self.height//8],  # 3: left shoulder
                [center_x - self.width//6, center_y],  # 4: right elbow
                [center_x + self.width//6, center_y],  # 5: left elbow
                [center_x - self.width//4, center_y + self.height//4],  # 6: right wrist
                [center_x + self.width//4, center_y + self.height//4],  # 7: left wrist
                [center_x, center_y + self.height//3],  # 8: waist
                [center_x - self.width//10, center_y + self.height//2],  # 9: right hip
                [center_x + self.width//10, center_y + self.height//2],  # 10: left hip
                [center_x - self.width//8, center_y + self.height//1.5],  # 11: right knee
                [center_x + self.width//8, center_y + self.height//1.5],  # 12: left knee
                [center_x - self.width//6, self.height - 10],  # 13: right ankle
                [center_x + self.width//6, self.height - 10],  # 14: left ankle
                [center_x - self.width//12, center_y - self.height//5],  # 15: right eye
                [center_x + self.width//12, center_y - self.height//5],  # 16: left eye
                [center_x, center_y - self.height//3],  # 17: right ear (using left ear position)
            ], dtype=np.float32)
        else:
            # Simplified version for other models
            self.reference_keypoints = np.array([
                [center_x, center_y - self.height//4],  # nose
                [center_x, center_y - self.height//6],  # neck
                [center_x - self.width//8, center_y - self.height//8],  # right shoulder
                [center_x + self.width//8, center_y - self.height//8],  # left shoulder
                [center_x - self.width//6, center_y],  # right elbow
                [center_x + self.width//6, center_y],  # left elbow
                [center_x - self.width//4, center_y + self.height//4],  # right wrist
                [center_x + self.width//4, center_y + self.height//4],  # left wrist
            ], dtype=np.float32)

    def _define_body_segments(self):
        """Define body segments and their corresponding keypoints."""
        if self.pose_model == "COCO":
            # Define segments: head, torso, left_arm, right_arm, left_leg, right_leg
            self.body_segments = {
                'head': [0, 14, 15, 16, 17],  # nose, eyes, ears
                'torso': [1, 2, 3, 8, 9, 10],  # neck, shoulders, waist, hips
                'left_arm': [3, 5, 7],  # left shoulder, elbow, wrist
                'right_arm': [2, 4, 6],  # right shoulder, elbow, wrist
                'left_leg': [10, 12, 14],  # left hip, knee, ankle
                'right_leg': [9, 11, 13],  # right hip, knee, ankle
            }
        else:
            # Simplified segments for other models
            self.body_segments = {
                'head': [0],  # nose
                'torso': [1, 2, 3],  # neck, shoulders
                'left_arm': [3, 5, 7],  # left shoulder, elbow, wrist
                'right_arm': [2, 4, 6],  # right shoulder, elbow, wrist
                'left_leg': [],  # no leg keypoints in simplified model
                'right_leg': [],  # no leg keypoints in simplified model
            }

    def _create_segment_masks(self):
        """Create binary masks for each body segment."""
        self.segment_masks = {}

        # Convert to grayscale for mask creation
        if len(self.character_image.shape) == 3:
            gray = cv2.cvtColor(self.character_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.character_image

        # Create a base mask (non-transparent areas)
        _, base_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # For now, create simple rectangular masks for each segment
        # In a more advanced implementation, these could be learned or manually defined
        center_x, center_y = self.width // 2, self.height // 2

        # Head mask (top portion)
        head_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        head_mask[:center_y - self.height//6, :] = 255
        self.segment_masks['head'] = cv2.bitwise_and(base_mask, head_mask)

        # Torso mask (middle portion)
        torso_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        torso_mask[center_y - self.height//6:center_y + self.height//4, :] = 255
        self.segment_masks['torso'] = cv2.bitwise_and(base_mask, torso_mask)

        # Left arm mask (left side, middle)
        left_arm_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        left_arm_mask[center_y - self.height//8:center_y + self.height//3, :center_x] = 255
        self.segment_masks['left_arm'] = cv2.bitwise_and(base_mask, left_arm_mask)

        # Right arm mask (right side, middle)
        right_arm_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        right_arm_mask[center_y - self.height//8:center_y + self.height//3, center_x:] = 255
        self.segment_masks['right_arm'] = cv2.bitwise_and(base_mask, right_arm_mask)

        # Left leg mask (bottom left)
        left_leg_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        left_leg_mask[center_y + self.height//4:, :center_x] = 255
        self.segment_masks['left_leg'] = cv2.bitwise_and(base_mask, left_leg_mask)

        # Right leg mask (bottom right)
        right_leg_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        right_leg_mask[center_y + self.height//4:, center_x:] = 255
        self.segment_masks['right_leg'] = cv2.bitwise_and(base_mask, right_leg_mask)

    def transfer_pose(self, target_keypoints, target_frame):
        """
        Transfer the character image to match the target pose keypoints using piece-wise warping.

        Args:
            target_keypoints: Detected pose keypoints from video
            target_frame: Target frame to composite onto

        Returns:
            Frame with character image warped to match the pose
        """
        if target_keypoints is None or len(target_keypoints) == 0:
            return target_frame

        # Start with the target frame as base
        result = target_frame.copy()

        # Process each body segment separately
        for segment_name, keypoint_indices in self.body_segments.items():
            if not keypoint_indices:  # Skip empty segments
                continue

            # Get valid keypoints for this segment
            valid_ref_points = []
            valid_target_points = []

            for idx in keypoint_indices:
                if idx < len(target_keypoints) and target_keypoints[idx] is not None:
                    target_point = target_keypoints[idx]
                    if (len(target_point) >= 2 and
                        target_point[0] >= 0 and target_point[1] >= 0 and
                        target_point[0] < target_frame.shape[1] and target_point[1] < target_frame.shape[0]):
                        valid_ref_points.append(self.reference_keypoints[idx])
                        valid_target_points.append(target_point[:2])

            # Need at least 3 points for affine transformation
            if len(valid_ref_points) >= 3:
                try:
                    valid_ref_points = np.array(valid_ref_points, dtype=np.float32)
                    valid_target_points = np.array(valid_target_points, dtype=np.float32)

                    # Calculate affine transformation for this segment
                    transformation_matrix = cv2.getAffineTransform(
                        valid_ref_points[:3],
                        valid_target_points[:3]
                    )

                    # Warp the segment
                    warped_segment = cv2.warpAffine(
                        self.character_image,
                        transformation_matrix,
                        (target_frame.shape[1], target_frame.shape[0])
                    )

                    # Apply segment mask to warped image
                    segment_mask = cv2.warpAffine(
                        self.segment_masks[segment_name],
                        transformation_matrix,
                        (target_frame.shape[1], target_frame.shape[0])
                    )

                    # Composite this segment onto the result
                    segment_mask_bool = segment_mask > 127
                    result[segment_mask_bool] = warped_segment[segment_mask_bool]

                except Exception as e:
                    # If segment warping fails, continue with other segments
                    print(f"Warning: Failed to warp segment {segment_name}: {e}")
                    continue

        return result