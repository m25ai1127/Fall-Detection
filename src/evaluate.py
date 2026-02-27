"""
Evaluation Script
==================
Evaluates the trained fall detection model on the test set (§4.5).

Metrics computed:
- Accuracy: overall classification performance
- Sensitivity (Recall): ability to correctly detect falls
- Specificity: ability to correctly identify normal activities
- False Alarm Rate: incorrect fall detections during normal activities
- Precision, F1-Score, AUC-ROC
- Confusion Matrix visualization
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.models.fall_detector import FallDetectorLSTM
from src.data.dataset import create_data_loaders
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve
from src.utils.helpers import set_seed


def load_model(model_path, device="cpu"):
    """
    Load a saved model checkpoint.
    
    Args:
        model_path: Path to .pth checkpoint file
        device: torch device
        
    Returns:
        model: loaded FallDetectorLSTM
        checkpoint: full checkpoint dict
    """
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    
    # Get model config from checkpoint
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
    
    print(f"[Evaluate] Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"[Evaluate] Checkpoint val_loss: {checkpoint.get('val_loss', '?'):.4f}")
    
    return model, checkpoint


def evaluate(model, test_loader, device):
    """
    Run evaluation on test set.
    
    Returns:
        metrics: dict of all computed metrics
        all_probs: predicted probabilities
        all_labels: ground truth labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= config.FALL_THRESHOLD).astype(int)
    
    # Compute all metrics (§4.5)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    sensitivity = recall_score(all_labels, all_preds, zero_division=0)  # = Recall
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity_recall': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'false_alarm_rate': false_alarm_rate,
        'auc_roc': auc_roc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(all_labels),
        'total_falls': int(all_labels.sum()),
        'total_normal': int((1 - all_labels).sum()),
    }
    
    return metrics, all_probs, all_labels


def print_metrics(metrics):
    """Print formatted evaluation metrics."""
    print()
    print("=" * 50)
    print("EVALUATION RESULTS (S4.5)")
    print("=" * 50)
    print()
    print(f"  Total samples:     {metrics['total_samples']}")
    print(f"  Falls:             {metrics['total_falls']}")
    print(f"  Normal:            {metrics['total_normal']}")
    print()
    print("  Performance Metrics:")
    print(f"  -----------------------------------")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
    print(f"  Sensitivity:       {metrics['sensitivity_recall']:.4f} ({metrics['sensitivity_recall']:.1%})")
    print(f"  Specificity:       {metrics['specificity']:.4f} ({metrics['specificity']:.1%})")
    print(f"  Precision:         {metrics['precision']:.4f} ({metrics['precision']:.1%})")
    print(f"  F1-Score:          {metrics['f1_score']:.4f}")
    print(f"  False Alarm Rate:  {metrics['false_alarm_rate']:.4f} ({metrics['false_alarm_rate']:.1%})")
    print(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"  -----------------------------------")
    print(f"                  Predicted")
    print(f"                  Normal  Fall")
    print(f"  Actual Normal   {metrics['true_negatives']:>5}   {metrics['false_positives']:>5}")
    print(f"  Actual Fall     {metrics['false_negatives']:>5}   {metrics['true_positives']:>5}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fall detection model"
    )
    parser.add_argument(
        "--model", type=str, default=str(config.MODELS_DIR / "best_model.pth"),
        help="Path to model checkpoint"
    )
    parser.add_argument("--device", type=str, default=config.MIDAS_DEVICE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    set_seed(config.RANDOM_SEED)
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(args.model, device)
    
    # Create test data loader
    print("Loading test data...")
    _, _, test_loader, _ = create_data_loaders(batch_size=args.batch_size)
    
    if test_loader is None:
        print("[ERROR] Failed to load test data.")
        return
    
    # Run evaluation
    print("Running evaluation...")
    metrics, all_probs, all_labels = evaluate(model, test_loader, device)
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    import json
    metrics_path = config.RESULTS_DIR / "evaluation_metrics.json"
    with open(str(metrics_path), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Plot confusion matrix
    try:
        all_preds = (all_probs >= config.FALL_THRESHOLD).astype(int)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        plot_confusion_matrix(
            cm,
            classes=['Normal (ADL)', 'Fall'],
            save_path=config.RESULTS_DIR / "confusion_matrix.png"
        )
        print(f"Confusion matrix: {config.RESULTS_DIR / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")
    
    # Plot ROC curve
    try:
        plot_roc_curve(
            all_labels, all_probs,
            save_path=config.RESULTS_DIR / "roc_curve.png"
        )
        print(f"ROC curve: {config.RESULTS_DIR / 'roc_curve.png'}")
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")


if __name__ == "__main__":
    main()
