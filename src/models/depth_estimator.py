"""
MiDaS Monocular Depth Estimation Module
========================================
Generates depth maps from single RGB frames using Intel's MiDaS model.
This simulates depth-camera data from a regular single camera (§4.2, §4.3).

Reference: Ranftl et al., "Towards Robust Monocular Depth Estimation," 2020
"""

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
import config


class DepthEstimator:
    """
    Wrapper around MiDaS for monocular depth estimation.
    
    Generates a pseudo-depth map from a single RGB frame, enabling
    depth-assisted analysis without a physical depth camera.
    """
    
    def __init__(self, model_type=None, device=None):
        """
        Initialize MiDaS depth estimator.
        
        Args:
            model_type: MiDaS variant ("MiDaS_small", "DPT_Hybrid", "DPT_Large")
            device: torch device ("cuda" or "cpu")
        """
        self.model_type = model_type or config.MIDAS_MODEL_TYPE
        self.device = device or config.MIDAS_DEVICE
        
        # Check for CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[DepthEstimator] CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        print(f"[DepthEstimator] Loading MiDaS model: {self.model_type} on {self.device}")
        
        # Load model from torch hub
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print("[DepthEstimator] Model loaded successfully")
    
    def estimate_depth(self, frame):
        """
        Estimate depth map from a single RGB frame.
        
        Args:
            frame: BGR image (H, W, 3) from OpenCV
            
        Returns:
            depth_map: Normalized depth map (H, W) with values in [0, 1]
                       Higher values = closer to camera
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply MiDaS transforms
        input_batch = self.transform(rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original frame size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to [0, 1] range (§4.2: depth normalization)
        depth_map = self._normalize_depth(depth_map)
        
        return depth_map
    
    def _normalize_depth(self, depth_map):
        """
        Normalize depth map to [0, 1] range.
        Applies noise reduction via median filtering (§4.2).
        
        Args:
            depth_map: Raw depth prediction
            
        Returns:
            Normalized and filtered depth map
        """
        # Apply median filter to reduce noise (§4.2: noise handling)
        depth_filtered = cv2.medianBlur(
            depth_map.astype(np.float32),
            config.DEPTH_NOISE_KERNEL_SIZE
        )
        
        # Normalize to [0, 1]
        d_min = depth_filtered.min()
        d_max = depth_filtered.max()
        
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_filtered - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_filtered)
        
        return depth_normalized
    
    def get_depth_at_points(self, depth_map, points):
        """
        Sample depth values at specific (x, y) locations.
        Used to get depth at joint positions from pose estimation.
        
        Args:
            depth_map: Normalized depth map (H, W)
            points: List of (x, y) coordinates (normalized 0-1)
            
        Returns:
            List of depth values at each point
        """
        h, w = depth_map.shape
        depth_values = []
        
        for (x, y) in points:
            # Convert normalized coordinates to pixel coordinates
            px = int(np.clip(x * w, 0, w - 1))
            py = int(np.clip(y * h, 0, h - 1))
            depth_values.append(depth_map[py, px])
        
        return np.array(depth_values, dtype=np.float32)


if __name__ == "__main__":
    # Quick test
    print("Testing DepthEstimator...")
    estimator = DepthEstimator()
    
    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = estimator.estimate_depth(dummy_frame)
    
    print(f"Input shape: {dummy_frame.shape}")
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print("DepthEstimator test PASSED ✓")
