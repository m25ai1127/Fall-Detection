"""
Data Preprocessing Pipeline
============================
Processes raw URFD dataset sequences through the depth estimation and
pose estimation pipeline to extract features (§4.2, §4.3).

Pipeline per sequence:
    1. Load RGB frames from image sequence
    2. Run MiDaS depth estimation on each frame
    3. Run MediaPipe pose estimation on each frame
    4. Extract combined feature vectors
    5. Save features + labels as .npy files
"""

import os
import sys
import cv2
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.models.depth_estimator import DepthEstimator
from src.models.pose_estimator import PoseEstimator
from src.models.feature_extractor import FeatureExtractor


def load_urfd_sequence(sequence_dir):
    """
    Load frames from a URFD sequence directory.
    Frames are stored as individual PNG images.
    
    Args:
        sequence_dir: Path to sequence directory containing PNG frames
        
    Returns:
        frames: list of BGR images
    """
    frame_paths = sorted(glob.glob(str(sequence_dir / "*.png")))
    
    if not frame_paths:
        # Try jpg format as fallback
        frame_paths = sorted(glob.glob(str(sequence_dir / "*.jpg")))
    
    if not frame_paths:
        print(f"  [WARN] No frames found in {sequence_dir}")
        return []
    
    frames = []
    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is not None:
            # Resize to standard dimensions (§4.2)
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            frames.append(frame)
    
    return frames


def get_urfd_sequences():
    """
    Discover all URFD sequences and their labels.
    
    Returns:
        sequences: list of (sequence_dir, label, name) tuples
                   label: 1 = fall, 0 = ADL (normal)
    """
    sequences = []
    
    if not config.URFD_RAW_DIR.exists():
        print(f"[ERROR] URFD data directory not found: {config.URFD_RAW_DIR}")
        print("Run: python -m src.data.download_dataset --dataset urfd")
        return sequences
    
    # Find fall RGB directories
    for i in range(1, config.URFD_NUM_FALL_SEQUENCES + 1):
        rgb_dir = config.URFD_RAW_DIR / f"fall-{i:02d}-cam0-rgb"
        if rgb_dir.exists():
            sequences.append((rgb_dir, 1, f"fall-{i:02d}"))
    
    # Find ADL RGB directories
    for i in range(1, config.URFD_NUM_ADL_SEQUENCES + 1):
        rgb_dir = config.URFD_RAW_DIR / f"adl-{i:02d}-cam0-rgb"
        if rgb_dir.exists():
            sequences.append((rgb_dir, 0, f"adl-{i:02d}"))
    
    return sequences


def preprocess_sequence(sequence_dir, label, name, feature_extractor, output_dir):
    """
    Process a single sequence through the full pipeline.
    
    Args:
        sequence_dir: Path to sequence frames
        label: 1=fall, 0=ADL
        name: Sequence identifier
        feature_extractor: FeatureExtractor instance
        output_dir: Path to save processed features
    """
    # Load frames
    frames = load_urfd_sequence(sequence_dir)
    
    if len(frames) == 0:
        print(f"  [SKIP] No frames in {name}")
        return False
    
    print(f"  Processing {name}: {len(frames)} frames, label={'Fall' if label else 'ADL'}")
    
    # Extract features for entire sequence
    feature_extractor.reset()
    all_features = []
    
    for frame in tqdm(frames, desc=f"    {name}", leave=False):
        feat = feature_extractor.extract_frame_features(frame)
        if feat is not None:
            all_features.append(feat)
        else:
            # Pad with zeros for frames without detected person
            all_features.append(np.zeros(config.FEATURE_DIM, dtype=np.float32))
    
    features_array = np.array(all_features, dtype=np.float32)
    
    # Save features and label
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / f"{name}_features.npy"), features_array)
    np.save(str(output_dir / f"{name}_label.npy"), np.array([label], dtype=np.int64))
    
    return True


def preprocess_all(dataset="urfd", max_sequences=None, dry_run=False):
    """
    Run preprocessing on all sequences.
    
    Args:
        dataset: Dataset to process ("urfd")
        max_sequences: Limit number of sequences (for testing)
        dry_run: If True, process only 2-3 frames per sequence
    """
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Output: {config.PROCESSED_DATA_DIR}")
    print()
    
    # Initialize models
    print("Loading models...")
    depth_estimator = DepthEstimator()
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor(depth_estimator, pose_estimator)
    
    # Get sequences
    if dataset == "urfd":
        sequences = get_urfd_sequences()
        output_dir = config.URFD_PROCESSED_DIR
    else:
        print(f"[ERROR] Unknown dataset: {dataset}")
        return
    
    if not sequences:
        print("[ERROR] No sequences found. Download the dataset first.")
        return
    
    if max_sequences:
        sequences = sequences[:max_sequences]
    
    print(f"\nFound {len(sequences)} sequences")
    print(f"Falls: {sum(1 for _, l, _ in sequences if l == 1)}")
    print(f"ADL:   {sum(1 for _, l, _ in sequences if l == 0)}")
    print()
    
    # Process each sequence
    processed = 0
    metadata = []
    
    for seq_dir, label, name in sequences:
        success = preprocess_sequence(
            seq_dir, label, name, feature_extractor, output_dir
        )
        if success:
            processed += 1
            feat_file = output_dir / f"{name}_features.npy"
            features = np.load(str(feat_file))
            metadata.append({
                "name": name,
                "label": int(label),
                "label_name": "fall" if label else "adl",
                "num_frames": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "file": str(feat_file),
            })
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(str(metadata_path), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print()
    print("=" * 60)
    print("Preprocessing Complete")
    print("=" * 60)
    print(f"Processed: {processed}/{len(sequences)} sequences")
    print(f"Feature dimension: {config.FEATURE_DIM}")
    print(f"Output: {output_dir}")
    print(f"Metadata: {metadata_path}")
    
    pose_estimator.release()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for fall detection training"
    )
    parser.add_argument(
        "--dataset", type=str, default="urfd",
        choices=["urfd"],
        help="Dataset to preprocess (default: urfd)"
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Max sequences to process (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only a few frames per sequence for testing"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output directory"
    )
    
    args = parser.parse_args()
    
    if args.output:
        config.PROCESSED_DATA_DIR = Path(args.output)
        config.URFD_PROCESSED_DIR = config.PROCESSED_DATA_DIR / "urfd"
    
    preprocess_all(
        dataset=args.dataset,
        max_sequences=args.max_sequences,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
