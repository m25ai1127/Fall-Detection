"""
Common Utility Functions
=========================
Logging, reproducibility, file I/O, and training helpers.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def set_seed(seed=None):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    seed = seed or config.RANDOM_SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


class AverageMeter:
    """
    Computes and stores the running average and current value.
    Used for tracking loss and other metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss and stops training when it stops improving.
    """
    
    def __init__(self, patience=10, delta=0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_device():
    """Get the best available torch device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def count_parameters(model):
    """Count trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}hr"


def print_model_summary(model, input_shape=None):
    """Print a summary of the model architecture."""
    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(model)
    
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    if input_shape:
        dummy = torch.randn(*input_shape)
        output = model(dummy)
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output.shape}")
    
    print("=" * 50)
