"""
Central Configuration for Fall Detection System
================================================
All hyperparameters, paths, and model settings in one place.
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Dataset-specific paths
URFD_RAW_DIR = RAW_DATA_DIR / "urfd"
URFD_PROCESSED_DIR = PROCESSED_DATA_DIR / "urfd"
NTU_RAW_DIR = RAW_DATA_DIR / "ntu_rgbd"
NTU_PROCESSED_DIR = PROCESSED_DATA_DIR / "ntu_rgbd"

# ============================================================
# Dataset Configuration
# ============================================================
URFD_DATASET_URL = "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html"
URFD_NUM_FALL_SEQUENCES = 30
URFD_NUM_ADL_SEQUENCES = 40

# NTU RGB+D fall action class
NTU_FALL_ACTION_ID = 43  # A43: falling down
NTU_NORMAL_ACTION_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Standing, walking, etc.

# ============================================================
# Pre-processing (§4.2)
# ============================================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEPTH_NORMALIZATION_RANGE = (0.0, 1.0)
DEPTH_NOISE_KERNEL_SIZE = 5  # Median filter kernel for depth noise removal

# ============================================================
# MiDaS Depth Estimation
# ============================================================
MIDAS_MODEL_TYPE = "MiDaS_small"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
MIDAS_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# ============================================================
# MediaPipe Pose Estimation (§4.3)
# ============================================================
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0=lite, 1=full, 2=heavy
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
NUM_POSE_LANDMARKS = 33
POSE_FEATURE_DIM = 4  # (x, y, z, visibility) per landmark

# Key joints for analysis (§4.3)
KEY_JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# ============================================================
# Feature Extraction (§4.3)
# ============================================================
# Per-frame feature dimensions:
#   - 33 landmarks × 4 (x, y, z, visibility) = 132
#   - 33 depth values at joints = 33
#   - Bounding box aspect ratio = 1
#   - Distance-to-floor estimation = 1
#   - Center of mass height = 1
#   - Vertical velocity (key joints) = len(KEY_JOINTS)
FEATURE_DIM = NUM_POSE_LANDMARKS * POSE_FEATURE_DIM + NUM_POSE_LANDMARKS + 3 + len(KEY_JOINTS)
# = 132 + 33 + 3 + 9 = 177

# ============================================================
# Temporal Sequence (§4.4)
# ============================================================
SEQUENCE_LENGTH = 30  # Number of consecutive frames per sequence
SEQUENCE_STRIDE = 10  # Stride for sliding window
FPS = 30  # Expected frames per second

# ============================================================
# LSTM Model (§4.4)
# ============================================================
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Classification
NUM_CLASSES = 1  # Binary: fall (1) vs normal (0)
FALL_THRESHOLD = 0.5  # Probability threshold for fall classification

# ============================================================
# Inference
# ============================================================
INFERENCE_BATCH_SIZE = 1
ALERT_COOLDOWN_SECONDS = 5  # Minimum time between consecutive fall alerts

# Fall Confirmation Window
# After initial fall detection, wait this many seconds to confirm
# If person recovers (stands back up), cancel the fall alert
FALL_CONFIRMATION_SECONDS = 3  # Wait 3 seconds before confirming fall
RECOVERY_THRESHOLD = 0.3  # If confidence drops below this, person recovered

# ============================================================
# Visualization
# ============================================================
SKELETON_COLOR = (0, 255, 0)  # Green
FALL_ALERT_COLOR = (0, 0, 255)  # Red
NORMAL_COLOR = (0, 255, 0)  # Green
DEPTH_COLORMAP = "inferno"  # Matplotlib colormap for depth visualization
FONT_SCALE = 0.8
FONT_THICKNESS = 2


def ensure_dirs():
    """Create all necessary directories."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
              RESULTS_DIR, DOCS_DIR, URFD_RAW_DIR, URFD_PROCESSED_DIR,
              NTU_RAW_DIR, NTU_PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Auto-create directories on import
ensure_dirs()
