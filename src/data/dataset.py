"""
PyTorch Dataset for Fall Detection
====================================
Loads preprocessed feature sequences and creates sliding-window
samples for LSTM training (§4.4).
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class FallDetectionDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed .npy feature files
    and creates sliding-window sequences for temporal classification.
    
    Each sample is a (sequence, label) pair where:
        - sequence: (seq_len, feature_dim) tensor
        - label: 0 (normal) or 1 (fall)
    """
    
    def __init__(self, data_dir=None, seq_length=None, stride=None, transform=None):
        """
        Args:
            data_dir: Path to preprocessed data directory
            seq_length: Number of consecutive frames per sequence
            stride: Sliding window stride
            transform: Optional transform to apply to features
        """
        self.data_dir = Path(data_dir) if data_dir else config.URFD_PROCESSED_DIR
        self.seq_length = seq_length or config.SEQUENCE_LENGTH
        self.stride = stride or config.SEQUENCE_STRIDE
        self.transform = transform
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create sliding window samples
        self.samples = self._create_samples()
        
        print(f"[Dataset] Loaded {len(self.samples)} samples")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Stride: {self.stride}")
        print(f"  Falls: {sum(1 for _, l in self.samples if l == 1)}")
        print(f"  Normal: {sum(1 for _, l in self.samples if l == 0)}")
    
    def _load_metadata(self):
        """Load metadata JSON file."""
        metadata_path = self.data_dir / "metadata.json"
        
        if not metadata_path.exists():
            print(f"[Dataset] Metadata not found at {metadata_path}")
            print("[Dataset] Run preprocessing first: python -m src.data.preprocess")
            return []
        
        with open(str(metadata_path), 'r') as f:
            return json.load(f)
    
    def _create_samples(self):
        """
        Create sliding window samples from all sequences.
        
        Returns:
            list of (features_window, label) tuples
        """
        samples = []
        
        for entry in self.metadata:
            features_path = self.data_dir / f"{entry['name']}_features.npy"
            
            if not features_path.exists():
                continue
            
            features = np.load(str(features_path))
            label = entry['label']
            num_frames = features.shape[0]
            
            if num_frames < self.seq_length:
                # Pad shorter sequences with repetition
                pad_count = self.seq_length - num_frames
                padding = np.tile(features[-1:], (pad_count, 1))
                features = np.vstack([features, padding])
                samples.append((features[:self.seq_length], label))
            else:
                # Create sliding windows
                for start in range(0, num_frames - self.seq_length + 1, self.stride):
                    window = features[start:start + self.seq_length]
                    samples.append((window, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, label = self.samples[idx]
        
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.FloatTensor([label])
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
        
        return features_tensor, label_tensor
    
    def get_class_weights(self):
        """
        Compute class weights for imbalanced dataset handling.
        
        Returns:
            tensor: weights for [normal, fall] classes
        """
        labels = [l for _, l in self.samples]
        n_fall = sum(labels)
        n_normal = len(labels) - n_fall
        
        if n_fall == 0 or n_normal == 0:
            return torch.FloatTensor([1.0])
        
        # Inverse frequency weighting
        weight = n_normal / n_fall
        return torch.FloatTensor([weight])


def create_data_loaders(data_dir=None, batch_size=None, train_split=None,
                         val_split=None, seed=None):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to preprocessed data
        batch_size: Batch size
        train_split: Train set fraction
        val_split: Validation set fraction
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader, dataset
    """
    batch_size = batch_size or config.BATCH_SIZE
    train_split = train_split or config.TRAIN_SPLIT
    val_split = val_split or config.VAL_SPLIT
    seed = seed or config.RANDOM_SEED
    
    # Create dataset
    dataset = FallDetectionDataset(data_dir=data_dir)
    
    if len(dataset) == 0:
        print("[ERROR] No samples found in dataset!")
        return None, None, None, dataset
    
    # Split dataset
    total = len(dataset)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    test_size = total - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"\n[DataLoaders] Split: train={train_size}, val={val_size}, test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset


if __name__ == "__main__":
    print("Testing FallDetectionDataset...")
    
    dataset = FallDetectionDataset()
    
    if len(dataset) > 0:
        features, label = dataset[0]
        print(f"\nSample shape: {features.shape}")
        print(f"Label: {label.item()}")
        print(f"Class weights: {dataset.get_class_weights()}")
        
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader, _ = create_data_loaders()
        
        if train_loader:
            for batch_features, batch_labels in train_loader:
                print(f"Batch features: {batch_features.shape}")
                print(f"Batch labels: {batch_labels.shape}")
                break
    else:
        print("No data available. Run preprocessing first.")
    
    print("\nDataset test complete ✓")
