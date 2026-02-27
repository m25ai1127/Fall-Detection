"""
Video & Live Camera Inference Script
=====================================
Process video files OR live camera feed through the complete
fall detection pipeline with real-time visualization overlay.

Usage:
  Video file:   python -m src.inference --input video.mp4
  Live camera:  python -m src.inference --live
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.models.depth_estimator import DepthEstimator
from src.models.pose_estimator import PoseEstimator
from src.models.feature_extractor import FeatureExtractor
from src.models.fall_detector import FallDetectorLSTM
from src.utils.visualization import colorize_depth_map, create_info_panel


def load_fall_detector(model_path, device):
    """Load the trained LSTM fall detector."""
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    model_config = checkpoint.get('config', {})
    
    model = FallDetectorLSTM(
        input_dim=model_config.get('feature_dim', config.FEATURE_DIM),
        hidden_size=model_config.get('hidden_size', config.LSTM_HIDDEN_SIZE),
        num_layers=model_config.get('num_layers', config.LSTM_NUM_LAYERS),
        bidirectional=model_config.get('bidirectional', config.LSTM_BIDIRECTIONAL),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def process_video(input_path, output_path, model_path, device,
                  show_depth=True, show_skeleton=True, show_info=True):
    """
    Process a video file through the full fall detection pipeline.
    
    Args:
        input_path: Path to input video
        output_path: Path to save annotated output video
        model_path: Path to trained model checkpoint
        device: torch device
        show_depth: Whether to show depth map overlay
        show_skeleton: Whether to show skeleton overlay
        show_info: Whether to show info panel
    """
    print("=" * 60)
    print("Fall Detection Inference")
    print("=" * 60)
    
    # Initialize models
    print("Loading models...")
    depth_estimator = DepthEstimator(device=device)
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor(depth_estimator, pose_estimator)
    
    # Load fall detector
    if Path(model_path).exists():
        fall_detector = load_fall_detector(model_path, device)
        print("[Inference] Fall detector loaded")
        has_classifier = True
    else:
        print(f"[WARN] Model not found at {model_path}")
        print("[Inference] Running without classifier (visualization only)")
        has_classifier = False
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nInput: {input_path}")
    print(f"Resolution: {width}×{height} @ {fps}fps")
    print(f"Frames: {total_frames}")
    
    # Setup output video
    # Output will be wider to accommodate depth map side panel
    out_width = width * 2 if show_depth else width
    out_height = height
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    
    # Feature sequence buffer for temporal classification
    feature_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    
    # Processing loop
    frame_idx = 0
    fall_detected = False
    fall_confidence = 0.0
    last_fall_time = 0
    
    print(f"\nProcessing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        display_frame = frame.copy()
        
        # 1. Depth estimation
        depth_map = depth_estimator.estimate_depth(frame)
        
        # 2. Pose estimation
        landmarks = pose_estimator.estimate_pose(frame)
        
        # 3. Feature extraction
        features = feature_extractor.extract_frame_features(
            frame, depth_map, landmarks
        )
        
        if features is not None:
            feature_buffer.append(features)
        else:
            feature_buffer.append(np.zeros(config.FEATURE_DIM, dtype=np.float32))
        
        # 4. Classification (when buffer is full)
        if has_classifier and len(feature_buffer) == config.SEQUENCE_LENGTH:
            sequence = np.array(list(feature_buffer), dtype=np.float32)
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prob = fall_detector(sequence_tensor).item()
            
            fall_confidence = prob
            current_time = frame_idx / fps
            
            if prob >= config.FALL_THRESHOLD:
                if current_time - last_fall_time > config.ALERT_COOLDOWN_SECONDS:
                    fall_detected = True
                    last_fall_time = current_time
            else:
                fall_detected = False
        
        # 5. Visualization
        # Draw skeleton
        if show_skeleton and landmarks is not None:
            pose_estimator.draw_skeleton(display_frame, landmarks)
        
        # Draw status info
        if show_info:
            status_text = "FALL DETECTED!" if fall_detected else "Normal"
            status_color = config.FALL_ALERT_COLOR if fall_detected else config.NORMAL_COLOR
            
            # Status bar at top
            cv2.rectangle(display_frame, (0, 0), (config.FRAME_WIDTH, 40), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Status: {status_text}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Confidence bar
            if has_classifier:
                cv2.putText(display_frame, f"Conf: {fall_confidence:.2f}",
                            (350, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Confidence bar
                bar_x = 480
                bar_w = 150
                bar_h = 15
                bar_y = 15
                cv2.rectangle(display_frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
                fill_w = int(bar_w * fall_confidence)
                bar_color = (0, 0, 255) if fall_confidence >= config.FALL_THRESHOLD else (0, 255, 0)
                cv2.rectangle(display_frame, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            
            # Frame counter
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}",
                        (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Fall alert overlay
        if fall_detected:
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                          (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)
            
            cv2.putText(display_frame, "! FALL DETECTED !",
                        (int(config.FRAME_WIDTH * 0.15), int(config.FRAME_HEIGHT * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Create output frame
        if show_depth:
            depth_colored = colorize_depth_map(depth_map)
            depth_colored = cv2.resize(depth_colored, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            
            # Add label to depth panel
            cv2.putText(depth_colored, "Depth Map (MiDaS)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            output_frame = np.hstack([display_frame, depth_colored])
        else:
            output_frame = display_frame
        
        out.write(output_frame)
        
        # Progress
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  Frame {frame_idx}/{total_frames} "
                  f"({100 * frame_idx / max(total_frames, 1):.0f}%) "
                  f"conf={fall_confidence:.2f}")
    
    # Cleanup
    cap.release()
    out.release()
    pose_estimator.release()
    
    print(f"\nOutput saved: {output_path}")
    print(f"Total frames processed: {frame_idx}")
    print("Done")


def process_live_camera(model_path, device, camera_id=0,
                        show_depth=True, show_skeleton=True):
    """
    Run fall detection on live camera feed.
    
    Args:
        model_path: Path to trained model checkpoint
        device: torch device
        camera_id: Camera index (0 = default webcam)
        show_depth: Whether to show depth map panel
        show_skeleton: Whether to show skeleton overlay
    """
    print("=" * 60)
    print("Fall Detection — LIVE CAMERA")
    print("=" * 60)
    print("Press 'q' to quit")
    print()
    
    # Initialize models
    print("Loading models...")
    depth_estimator = DepthEstimator(device=device)
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor(depth_estimator, pose_estimator)
    
    # Load fall detector
    has_classifier = False
    if Path(model_path).exists():
        fall_detector = load_fall_detector(model_path, device)
        print("[Live] Fall detector loaded")
        has_classifier = True
    else:
        print(f"[WARN] Model not found at {model_path}")
        print("[Live] Running without classifier (visualization only)")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}")
        print("Make sure a webcam is connected.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print("\nStarting live detection...\n")
    
    # Feature sequence buffer
    feature_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    
    frame_idx = 0
    fall_confirmed = False       # Final confirmed fall
    fall_pending = False         # Waiting for confirmation
    fall_pending_start = 0       # When pending started
    fall_confidence = 0.0
    last_fall_time = 0
    fps_counter = deque(maxlen=30)
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera")
            break
        
        frame_idx += 1
        current_time = frame_idx / 30.0
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        display_frame = frame.copy()
        
        # 1. Depth estimation
        depth_map = depth_estimator.estimate_depth(frame)
        
        # 2. Pose estimation
        landmarks = pose_estimator.estimate_pose(frame)
        
        # 3. Feature extraction
        features = feature_extractor.extract_frame_features(
            frame, depth_map, landmarks
        )
        
        if features is not None:
            feature_buffer.append(features)
        else:
            feature_buffer.append(np.zeros(config.FEATURE_DIM, dtype=np.float32))
        
        # 4. Classification with Confirmation Window
        if has_classifier and len(feature_buffer) == config.SEQUENCE_LENGTH:
            sequence = np.array(list(feature_buffer), dtype=np.float32)
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prob = fall_detector(sequence_tensor).item()
            
            fall_confidence = prob
            
            if prob >= config.FALL_THRESHOLD:
                if not fall_pending and not fall_confirmed:
                    # Initial fall detected — start confirmation window
                    fall_pending = True
                    fall_pending_start = current_time
                    print(f"  [t={current_time:.1f}s] Fall suspected, waiting {config.FALL_CONFIRMATION_SECONDS}s to confirm...")
                
                elif fall_pending:
                    # Still in confirmation window — check if time elapsed
                    elapsed_since_pending = current_time - fall_pending_start
                    if elapsed_since_pending >= config.FALL_CONFIRMATION_SECONDS:
                        # Person stayed down for 3 seconds — CONFIRMED FALL
                        fall_confirmed = True
                        fall_pending = False
                        last_fall_time = current_time
                        print(f"  [t={current_time:.1f}s] FALL CONFIRMED! Person did not recover.")
            
            else:
                # Confidence dropped below threshold
                if fall_pending and prob < config.RECOVERY_THRESHOLD:
                    # Person recovered during confirmation window — cancel alert
                    fall_pending = False
                    print(f"  [t={current_time:.1f}s] Person recovered. Fall cancelled.")
                
                elif fall_confirmed:
                    # Person recovered after confirmed fall
                    if prob < config.RECOVERY_THRESHOLD:
                        fall_confirmed = False
                        print(f"  [t={current_time:.1f}s] Person recovered from fall.")
        
        # 5. Visualization
        # Draw skeleton
        if show_skeleton and landmarks is not None:
            pose_estimator.draw_skeleton(display_frame, landmarks)
        
        # FPS calculation
        elapsed = time.time() - start_time
        fps_counter.append(elapsed)
        current_fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0
        
        # Status bar — 3 states: Normal, Pending, Confirmed
        if fall_confirmed:
            status_text = "!! FALL CONFIRMED !!"
            status_color = config.FALL_ALERT_COLOR
        elif fall_pending:
            wait_left = config.FALL_CONFIRMATION_SECONDS - (current_time - fall_pending_start)
            status_text = f"FALL SUSPECTED... confirming ({wait_left:.1f}s)"
            status_color = (0, 165, 255)  # Orange
        else:
            status_text = "Normal"
            status_color = config.NORMAL_COLOR
        
        cv2.rectangle(display_frame, (0, 0), (config.FRAME_WIDTH, 45), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Status: {status_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                    (config.FRAME_WIDTH - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Confidence bar
        if has_classifier and len(feature_buffer) == config.SEQUENCE_LENGTH:
            cv2.putText(display_frame, f"Conf: {fall_confidence:.2f}",
                        (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            bar_x, bar_w, bar_h, bar_y = 480, 120, 15, 18
            cv2.rectangle(display_frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
            fill_w = int(bar_w * fall_confidence)
            bar_color = (0, 0, 255) if fall_confidence >= config.FALL_THRESHOLD else (0, 255, 0)
            cv2.rectangle(display_frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        else:
            buffering_pct = len(feature_buffer) / config.SEQUENCE_LENGTH * 100
            cv2.putText(display_frame, f"Buffering: {buffering_pct:.0f}%",
                        (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        
        # Fall alert overlay
        if fall_confirmed:
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0),
                          (config.FRAME_WIDTH, config.FRAME_HEIGHT), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            cv2.putText(display_frame, "! FALL CONFIRMED !",
                        (int(config.FRAME_WIDTH * 0.1), int(config.FRAME_HEIGHT * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif fall_pending:
            # Orange border during confirmation wait
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0),
                          (config.FRAME_WIDTH, config.FRAME_HEIGHT), (0, 165, 255), -1)
            cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)
            cv2.putText(display_frame, "Verifying fall...",
                        (int(config.FRAME_WIDTH * 0.2), int(config.FRAME_HEIGHT * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        
        # Build display
        if show_depth:
            depth_colored = colorize_depth_map(depth_map)
            depth_colored = cv2.resize(depth_colored,
                                       (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            cv2.putText(depth_colored, "Depth Map (MiDaS)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display = np.hstack([display_frame, depth_colored])
        else:
            display = display_frame
        
        # Show window
        cv2.imshow("Fall Detection - Live Camera (Press Q to quit)", display)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # q or ESC
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose_estimator.release()
    
    print(f"\nStopped. Total frames processed: {frame_idx}")
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Run fall detection on video file or live camera"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input video file path")
    parser.add_argument("--live", action="store_true",
                        help="Use live camera (webcam) instead of video file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for live mode (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video file path")
    parser.add_argument("--model", type=str,
                        default=str(config.MODELS_DIR / "best_model.pth"),
                        help="Trained model checkpoint path")
    parser.add_argument("--device", type=str, default=config.MIDAS_DEVICE)
    parser.add_argument("--show-depth", action="store_true", default=True,
                        help="Show depth map side panel")
    parser.add_argument("--show-skeleton", action="store_true", default=True,
                        help="Show skeleton overlay")
    parser.add_argument("--no-depth", action="store_true",
                        help="Hide depth map panel")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="Hide skeleton overlay")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    if args.live:
        # Live camera mode
        process_live_camera(
            model_path=args.model,
            device=device,
            camera_id=args.camera,
            show_depth=not args.no_depth,
            show_skeleton=not args.no_skeleton,
        )
    elif args.input:
        # Video file mode
        if args.output is None:
            input_name = Path(args.input).stem
            args.output = str(config.RESULTS_DIR / f"{input_name}_output.mp4")
        
        process_video(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            device=device,
            show_depth=not args.no_depth,
            show_skeleton=not args.no_skeleton,
        )
    else:
        print("Please specify either --input <video_path> or --live")
        print("Examples:")
        print("  python -m src.inference --live")
        print("  python -m src.inference --input video.mp4")
        parser.print_help()


if __name__ == "__main__":
    main()
