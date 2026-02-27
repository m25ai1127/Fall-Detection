"""
Training Script
================
Complete training loop for the LSTM fall detection classifier (§4.4).

Features:
- Train/validation split with class balancing
- Early stopping and learning rate scheduling
- Model checkpointing (saves best model)  
- Detailed per-epoch metrics logging
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.models.fall_detector import FallDetectorLSTM, build_model
from src.data.dataset import FallDetectionDataset, create_data_loaders
from src.utils.helpers import set_seed, AverageMeter, EarlyStopping


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        loss_meter.update(loss.item(), features.size(0))
        predictions = (outputs >= config.FALL_THRESHOLD).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return loss_meter.avg, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss_meter.update(loss.item(), features.size(0))
            predictions = (outputs >= config.FALL_THRESHOLD).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    accuracy = correct / total if total > 0 else 0
    
    # Compute sensitivity (recall) and false alarm rate
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return loss_meter.avg, accuracy, sensitivity, false_alarm_rate


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Fall Detection Model Training")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Set seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        batch_size=args.batch_size
    )
    
    if train_loader is None:
        print("[ERROR] Failed to create data loaders. Preprocess data first.")
        print("Run: python -m src.data.preprocess")
        return
    
    # Build model
    model = build_model(device)
    
    # Loss function with class weighting
    class_weights = dataset.get_class_weights().to(device)
    criterion = nn.BCELoss(weight=None)  # Could use pos_weight with BCEWithLogitsLoss
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.LR_SCHEDULER_PATIENCE,
        factor=config.LR_SCHEDULER_FACTOR,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        delta=0.001
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_sensitivity': [], 'val_false_alarm': [],
        'lr': [],
    }
    
    # Ensure models directory exists
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = config.MODELS_DIR / "best_model.pth"
    
    print()
    print("Training started...")
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Sens':>6} | {'FAR':>6} | {'LR':>10}")
    print("-" * 80)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, sensitivity, false_alarm = validate(
            model, val_loader, criterion, device
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.1%} | {sensitivity:>5.1%} | "
              f"{false_alarm:>5.1%} | {current_lr:>10.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_sensitivity'].append(sensitivity)
        history['val_false_alarm'].append(false_alarm)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'sensitivity': sensitivity,
                'false_alarm_rate': false_alarm,
                'config': {
                    'feature_dim': config.FEATURE_DIM,
                    'seq_length': config.SEQUENCE_LENGTH,
                    'hidden_size': config.LSTM_HIDDEN_SIZE,
                    'num_layers': config.LSTM_NUM_LAYERS,
                    'bidirectional': config.LSTM_BIDIRECTIONAL,
                },
            }, str(best_model_path))
            print(f"      -> Best model saved (val_loss: {val_loss:.4f})")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Training complete
    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved: {best_model_path}")
    
    # Save training history
    history_path = config.RESULTS_DIR / "training_history.npy"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(history_path), history)
    print(f"Training history: {history_path}")
    
    # Plot training curves
    try:
        from src.utils.visualization import plot_training_curves
        plot_training_curves(history, config.RESULTS_DIR / "training_curves.png")
        print(f"Training curves: {config.RESULTS_DIR / 'training_curves.png'}")
    except Exception as e:
        print(f"Could not plot training curves: {e}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM fall detection model"
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default=config.MIDAS_DEVICE)
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only N samples for quick testing")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
