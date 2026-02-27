"""
Feature Extraction Module
=========================
Combines depth estimation and pose estimation to extract comprehensive
features for fall detection (§4.3).

Extracts per-frame features:
- 33 landmarks × 4 (x, y, z, visibility) = 132 dims
- 33 depth values at joint locations = 33 dims
- Bounding box aspect ratio = 1 dim
- Distance-to-floor estimation = 1 dim
- Center of mass height = 1 dim
- Vertical velocity of key joints = 9 dims
Total: 177 dimensions per frame
"""

import numpy as np

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
import config


class FeatureExtractor:
    """
    Combines depth and pose features into a unified feature vector.
    
    Implements the feature extraction pipeline described in §4.3:
    - Vertical motion characteristics
    - Bounding box aspect ratio
    - Distance-to-floor estimation
    """
    
    def __init__(self, depth_estimator, pose_estimator):
        """
        Args:
            depth_estimator: DepthEstimator instance
            pose_estimator: PoseEstimator instance
        """
        self.depth_estimator = depth_estimator
        self.pose_estimator = pose_estimator
        self.prev_key_joints = None  # For velocity computation
        
    def extract_frame_features(self, frame, depth_map=None, landmarks=None):
        """
        Extract complete feature vector from a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
            depth_map: Pre-computed depth map (H, W), or None to compute
            landmarks: Pre-computed landmarks (33, 4), or None to compute
            
        Returns:
            features: numpy array (FEATURE_DIM,) or None if no person detected
        """
        # Compute depth map if not provided
        if depth_map is None:
            depth_map = self.depth_estimator.estimate_depth(frame)
        
        # Compute pose landmarks if not provided
        if landmarks is None:
            landmarks = self.pose_estimator.estimate_pose(frame)
        
        if landmarks is None:
            self.prev_key_joints = None
            return None
        
        # 1. Flatten pose landmarks: (33, 4) → (132,)
        pose_features = landmarks.flatten()  # 132 dims
        
        # 2. Depth values at each joint location: (33,)
        joint_points = [(landmarks[i, 0], landmarks[i, 1]) 
                        for i in range(config.NUM_POSE_LANDMARKS)]
        depth_at_joints = self.depth_estimator.get_depth_at_points(
            depth_map, joint_points
        )  # 33 dims
        
        # 3. Bounding box aspect ratio (§4.3): 1 dim
        bbox_ratio = self.pose_estimator.get_bounding_box_aspect_ratio(landmarks)
        
        # 4. Distance-to-floor estimation (§4.3): 1 dim
        floor_distance = self._estimate_floor_distance(landmarks, depth_at_joints)
        
        # 5. Center of mass height: 1 dim
        com_height = self._compute_center_of_mass_height(landmarks)
        
        # 6. Vertical velocity of key joints (§4.3): len(KEY_JOINTS) dims
        vertical_velocity = self._compute_vertical_velocity(landmarks)
        
        # Concatenate all features
        features = np.concatenate([
            pose_features,                          # 132
            depth_at_joints,                        # 33
            np.array([bbox_ratio], dtype=np.float32),        # 1
            np.array([floor_distance], dtype=np.float32),    # 1
            np.array([com_height], dtype=np.float32),        # 1
            vertical_velocity,                      # 9
        ])
        
        return features
    
    def _estimate_floor_distance(self, landmarks, depth_at_joints):
        """
        Estimate distance from person's lowest point to the floor plane.
        Uses depth values at ankle joints relative to the overall depth (§4.3).
        
        Args:
            landmarks: (33, 4) pose landmarks
            depth_at_joints: (33,) depth values at each joint
            
        Returns:
            float: estimated distance to floor (0 = on floor, 1 = far from floor)
        """
        # Use ankle positions as proxy for ground contact
        left_ankle_idx = config.KEY_JOINTS["left_ankle"]
        right_ankle_idx = config.KEY_JOINTS["right_ankle"]
        
        # Y-coordinate of ankles (higher y = lower in image = closer to floor)
        left_ankle_y = landmarks[left_ankle_idx, 1]
        right_ankle_y = landmarks[right_ankle_idx, 1]
        
        # Depth at ankles
        left_ankle_depth = depth_at_joints[left_ankle_idx]
        right_ankle_depth = depth_at_joints[right_ankle_idx]
        
        # Hip center position
        left_hip_y = landmarks[config.KEY_JOINTS["left_hip"], 1]
        right_hip_y = landmarks[config.KEY_JOINTS["right_hip"], 1]
        hip_center_y = (left_hip_y + right_hip_y) / 2
        
        # Floor distance estimation:
        # When standing: hip is well above ankles → large distance
        # When fallen: hip is at similar height to ankles → small distance
        ankle_y = max(left_ankle_y, right_ankle_y)
        floor_distance = abs(hip_center_y - ankle_y)
        
        # Combine with depth information
        avg_ankle_depth = (left_ankle_depth + right_ankle_depth) / 2
        
        # Weighted combination (depth gives 3D context)
        floor_estimate = floor_distance * 0.7 + avg_ankle_depth * 0.3
        
        return float(floor_estimate)
    
    def _compute_center_of_mass_height(self, landmarks):
        """
        Compute approximate center of mass height.
        Uses weighted average of key joint y-coordinates.
        
        Lower center of mass typically indicates fall/lying state.
        
        Args:
            landmarks: (33, 4) pose landmarks
            
        Returns:
            float: normalized center of mass height (0=top, 1=bottom)
        """
        # Weighted joints (torso and hip carry most body mass)
        weights = {
            "nose": 0.07,
            "left_shoulder": 0.12,
            "right_shoulder": 0.12,
            "left_hip": 0.20,
            "right_hip": 0.20,
            "left_knee": 0.10,
            "right_knee": 0.10,
            "left_ankle": 0.045,
            "right_ankle": 0.045,
        }
        
        com_y = 0.0
        total_weight = 0.0
        
        for joint_name, weight in weights.items():
            idx = config.KEY_JOINTS[joint_name]
            if landmarks[idx, 3] > 0.3:  # Only use visible joints
                com_y += landmarks[idx, 1] * weight
                total_weight += weight
        
        if total_weight > 0:
            com_y /= total_weight
        
        return float(com_y)
    
    def _compute_vertical_velocity(self, landmarks):
        """
        Compute vertical velocity (y-displacement) of key joints
        between current and previous frame (§4.3: vertical motion characteristics).
        
        Sudden downward movement indicates potential fall.
        
        Args:
            landmarks: (33, 4) current frame landmarks
            
        Returns:
            numpy array (len(KEY_JOINTS),): vertical velocity per key joint
        """
        current_joints = {}
        for name, idx in config.KEY_JOINTS.items():
            current_joints[name] = landmarks[idx, 1]  # y-coordinate
        
        if self.prev_key_joints is None:
            velocity = np.zeros(len(config.KEY_JOINTS), dtype=np.float32)
        else:
            velocity = np.array([
                current_joints[name] - self.prev_key_joints.get(name, current_joints[name])
                for name in config.KEY_JOINTS.keys()
            ], dtype=np.float32)
        
        # Update previous joints
        self.prev_key_joints = current_joints
        
        return velocity
    
    def reset(self):
        """Reset temporal state (for new video sequence)."""
        self.prev_key_joints = None
    
    def extract_sequence_features(self, frames, depth_maps=None, landmarks_list=None):
        """
        Extract features from a sequence of frames.
        
        Args:
            frames: list of BGR images
            depth_maps: optional list of pre-computed depth maps
            landmarks_list: optional list of pre-computed landmarks
            
        Returns:
            features: numpy array (num_valid_frames, FEATURE_DIM)
        """
        self.reset()
        features = []
        
        for i, frame in enumerate(frames):
            dm = depth_maps[i] if depth_maps is not None else None
            lm = landmarks_list[i] if landmarks_list is not None else None
            
            feat = self.extract_frame_features(frame, dm, lm)
            if feat is not None:
                features.append(feat)
            else:
                # Pad with zeros if no person detected in this frame
                features.append(np.zeros(config.FEATURE_DIM, dtype=np.float32))
        
        return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    print(f"Feature dimension: {config.FEATURE_DIM}")
    print(f"Key joints: {list(config.KEY_JOINTS.keys())}")
    print("FeatureExtractor module loaded successfully ✓")
