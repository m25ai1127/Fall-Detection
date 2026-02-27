"""
Visualization Utilities
========================
Helper functions for plotting training curves, confusion matrices,
ROC curves, depth map colorization, and video overlays.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def colorize_depth_map(depth_map, colormap=None):
    """
    Convert a normalized depth map to a colored visualization.
    
    Args:
        depth_map: (H, W) normalized depth map [0, 1]
        colormap: matplotlib colormap name
        
    Returns:
        colored: (H, W, 3) BGR colored depth image
    """
    colormap = colormap or config.DEPTH_COLORMAP
    
    # Use matplotlib colormap
    cm = plt.get_cmap(colormap)
    colored = cm(depth_map)[:, :, :3]  # Remove alpha
    colored = (colored * 255).astype(np.uint8)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    
    return colored


def plot_confusion_matrix(cm, classes, save_path=None, title="Confusion Matrix"):
    """
    Plot and save a confusion matrix.
    
    Args:
        cm: numpy confusion matrix
        classes: list of class names
        save_path: path to save the plot
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=ax, annot_kws={"size": 16}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None, title="ROC Curve"):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: ground truth labels
        y_scores: predicted probabilities
        save_path: path to save the plot
        title: plot title
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#2196F3', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
            label='Random classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        history: dict with train_loss, val_loss, train_acc, val_acc, etc.
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sensitivity & False Alarm Rate
    ax = axes[1, 0]
    if 'val_sensitivity' in history:
        ax.plot(epochs, history['val_sensitivity'], 'g-',
                label='Sensitivity (Recall)', linewidth=2)
    if 'val_false_alarm' in history:
        ax.plot(epochs, history['val_false_alarm'], 'r--',
                label='False Alarm Rate', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate')
    ax.set_title('Sensitivity & False Alarm Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'k-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def create_info_panel(width, height, info_dict):
    """
    Create an info panel image with key-value pairs.
    
    Args:
        width: panel width
        height: panel height
        info_dict: dict of {label: value} pairs
        
    Returns:
        panel: BGR image
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark background
    
    y = 30
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
    
    return panel


def create_side_by_side(frame, depth_colored, landmarks=None, label="",
                         confidence=0.0, pose_estimator=None):
    """
    Create a side-by-side visualization: RGB (with skeleton) | Depth map.
    
    Args:
        frame: BGR frame
        depth_colored: Colored depth map
        landmarks: Optional pose landmarks
        label: Classification label text
        confidence: Classification confidence
        pose_estimator: PoseEstimator for drawing skeleton
        
    Returns:
        combined: side-by-side BGR image
    """
    display = frame.copy()
    
    # Draw skeleton
    if landmarks is not None and pose_estimator is not None:
        pose_estimator.draw_skeleton(display, landmarks)
    
    # Draw label
    if label:
        color = config.FALL_ALERT_COLOR if "fall" in label.lower() else config.NORMAL_COLOR
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Conf: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Resize depth to match frame
    h, w = display.shape[:2]
    depth_resized = cv2.resize(depth_colored, (w, h))
    
    # Combine
    combined = np.hstack([display, depth_resized])
    
    return combined
