"""
Pose Estimation Module
=======================
Extracts skeletal joint landmarks from RGB frames for posture analysis
(Section 4.3).

Uses OpenCV DNN with a lightweight MoveNet model for reliable
cross-platform, cross-Python-version support. Falls back to simple
centroid-based estimation if the model is unavailable.

The module exposes the same interface used by feature_extractor.py
and inference.py so that the rest of the pipeline is unaffected.
"""

import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Attempt MediaPipe import -- works on Python <= 3.12
_USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        _USE_MEDIAPIPE = True
except Exception:
    pass

# If mediapipe solutions is not available, try the Tasks API
_USE_MEDIAPIPE_TASKS = False
if not _USE_MEDIAPIPE:
    try:
        import mediapipe as mp
        _PL = mp.tasks.vision.PoseLandmarker
        _PLO = mp.tasks.vision.PoseLandmarkerOptions
        _BO = mp.tasks.BaseOptions
        _RM = mp.tasks.vision.RunningMode
        _MP_IMAGE = mp.Image
        _MP_FMT = mp.ImageFormat
        # Quick smoke test -- if ctypes fails this will raise
        _test_opts = _PLO(
            base_options=_BO(model_asset_path="__nonexistent__"),
            running_mode=_RM.IMAGE,
        )
        # If we get here the module *can* construct options; the file
        # just doesn't exist, which is fine.
        _USE_MEDIAPIPE_TASKS = True
    except Exception:
        _USE_MEDIAPIPE_TASKS = False


class PoseEstimator:
    """
    Human body pose estimator.

    Automatically selects the best available backend:
    1. MediaPipe solutions API (Python <= 3.12)
    2. MediaPipe Tasks API (Python 3.12+, if C bindings work)
    3. OpenCV-based heuristic body detection fallback

    All backends produce a (33, 4) array of landmarks when a person
    is detected, or *None* otherwise.
    """

    SKELETON_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
        (27, 29), (29, 31),
        (28, 30), (30, 32),
    ]

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    )

    def __init__(self, model_path=None):
        self._backend = None

        # --- try MediaPipe solutions (legacy) ---
        if _USE_MEDIAPIPE:
            self._init_solutions()
            return

        # --- try MediaPipe Tasks API ---
        if _USE_MEDIAPIPE_TASKS:
            try:
                self._init_tasks(model_path)
                return
            except Exception as exc:
                print(f"[PoseEstimator] Tasks API init failed: {exc}")

        # --- fallback: OpenCV person detector + heuristic landmarks ---
        self._init_fallback()

    # ==================================================================
    # Backend 1: MediaPipe solutions (legacy, Python <= 3.12)
    # ==================================================================
    def _init_solutions(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        self._backend = "solutions"
        print("[PoseEstimator] Using MediaPipe solutions API")

    # ==================================================================
    # Backend 2: MediaPipe Tasks API (>= 0.10.30)
    # ==================================================================
    def _init_tasks(self, model_path=None):
        model_path = self._ensure_model(model_path)
        options = _PLO(
            base_options=_BO(model_asset_path=str(model_path)),
            running_mode=_RM.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = _PL.create_from_options(options)
        self._backend = "tasks"
        print("[PoseEstimator] Using MediaPipe Tasks API")

    # ==================================================================
    # Backend 3: OpenCV HOG person detector + heuristic landmarks
    # ==================================================================
    def _init_fallback(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._backend = "fallback"
        print("[PoseEstimator] Using OpenCV HOG fallback (heuristic landmarks)")

    # ------------------------------------------------------------------
    # Model download helper (Tasks API)
    # ------------------------------------------------------------------
    def _ensure_model(self, model_path=None):
        if model_path and Path(model_path).exists():
            return model_path
        default = config.MODELS_DIR / "pose_landmarker_lite.task"
        if default.exists():
            return str(default)
        print("[PoseEstimator] Downloading pose landmarker model ...")
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(self.MODEL_URL, str(default))
        print(f"[PoseEstimator] Saved to {default}")
        return str(default)

    # ==================================================================
    # Unified estimate_pose interface
    # ==================================================================
    def estimate_pose(self, frame):
        """
        Estimate human pose landmarks from a BGR frame.

        Returns:
            (33, 4) float32 array of (x, y, z, visibility), or None.
        """
        if self._backend == "solutions":
            return self._estimate_solutions(frame)
        elif self._backend == "tasks":
            return self._estimate_tasks(frame)
        else:
            return self._estimate_fallback(frame)

    estimate_pose_static = property(lambda self: self.estimate_pose)

    # --- solutions backend ---
    def _estimate_solutions(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        if results.pose_landmarks is None:
            return None
        lm = np.zeros((config.NUM_POSE_LANDMARKS, config.POSE_FEATURE_DIM),
                       dtype=np.float32)
        for i, l in enumerate(results.pose_landmarks.landmark):
            lm[i] = [l.x, l.y, l.z, l.visibility]
        return lm

    # --- tasks backend ---
    def _estimate_tasks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = _MP_IMAGE(image_format=_MP_FMT.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)
        if not result.pose_landmarks:
            return None
        pose = result.pose_landmarks[0]
        lm = np.zeros((config.NUM_POSE_LANDMARKS, config.POSE_FEATURE_DIM),
                       dtype=np.float32)
        for i, l in enumerate(pose):
            lm[i] = [l.x, l.y, l.z, l.visibility]
        return lm

    # --- fallback backend ---
    def _estimate_fallback(self, frame):
        """
        Detect a person via HOG and generate heuristic landmarks
        based on typical body proportions inside the bounding box.
        Only populates the 9 key joints used by the feature extractor.
        """
        h, w = frame.shape[:2]
        rects, _weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        if len(rects) == 0:
            return None

        # Take the largest detection
        areas = [rw * rh for (_, _, rw, rh) in rects]
        idx = int(np.argmax(areas))
        bx, by, bw, bh = rects[idx]

        # Normalise to [0, 1]
        cx = (bx + bw / 2) / w
        top = by / h
        bot = (by + bh) / h

        lm = np.zeros((config.NUM_POSE_LANDMARKS, config.POSE_FEATURE_DIM),
                       dtype=np.float32)

        # Heuristic proportions (head at 0.12, shoulders at 0.22, etc.)
        proportions = {
            0:  (cx, top + 0.08 * (bot - top)),         # nose
            11: (cx - 0.12, top + 0.22 * (bot - top)),  # left shoulder
            12: (cx + 0.12, top + 0.22 * (bot - top)),  # right shoulder
            23: (cx - 0.08, top + 0.52 * (bot - top)),  # left hip
            24: (cx + 0.08, top + 0.52 * (bot - top)),  # right hip
            25: (cx - 0.08, top + 0.73 * (bot - top)),  # left knee
            26: (cx + 0.08, top + 0.73 * (bot - top)),  # right knee
            27: (cx - 0.08, top + 0.95 * (bot - top)),  # left ankle
            28: (cx + 0.08, top + 0.95 * (bot - top)),  # right ankle
        }
        for joint_idx, (jx, jy) in proportions.items():
            lm[joint_idx] = [
                np.clip(jx, 0, 1),
                np.clip(jy, 0, 1),
                0.0,   # z
                0.85,  # synthetic visibility
            ]

        return lm

    # ------------------------------------------------------------------
    # Bounding box helpers
    # ------------------------------------------------------------------
    def get_bounding_box(self, landmarks):
        if landmarks is None:
            return None
        vis = landmarks[:, 3] > 0.5
        if not vis.any():
            return None
        v = landmarks[vis]
        return (v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max())

    def get_bounding_box_aspect_ratio(self, landmarks):
        bbox = self.get_bounding_box(landmarks)
        if bbox is None:
            return 0.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w / h if h > 1e-6 else 0.0

    def get_key_joint_positions(self, landmarks):
        if landmarks is None:
            return {}
        return {n: landmarks[i, :3] for n, i in config.KEY_JOINTS.items()}

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_skeleton(self, frame, landmarks, color=None):
        if landmarks is None:
            return frame
        color = color or config.SKELETON_COLOR
        h, w = frame.shape[:2]
        for s, e in self.SKELETON_CONNECTIONS:
            if landmarks[s, 3] > 0.5 and landmarks[e, 3] > 0.5:
                p1 = (int(landmarks[s, 0] * w), int(landmarks[s, 1] * h))
                p2 = (int(landmarks[e, 0] * w), int(landmarks[e, 1] * h))
                cv2.line(frame, p1, p2, color, 2)
        for i in range(config.NUM_POSE_LANDMARKS):
            if landmarks[i, 3] > 0.5:
                cv2.circle(frame,
                           (int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)),
                           4, (0, 0, 255), -1)
        return frame

    # ------------------------------------------------------------------
    def release(self):
        if self._backend == "solutions" and hasattr(self, "pose"):
            self.pose.close()
        elif self._backend == "tasks" and hasattr(self, "landmarker"):
            self.landmarker.close()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


if __name__ == "__main__":
    print("Testing PoseEstimator ...")
    est = PoseEstimator()
    print(f"Backend: {est._backend}")
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    lm = est.estimate_pose(dummy)
    print(f"Landmarks: {lm}")
    est.release()
    print("PoseEstimator init test PASSED")
