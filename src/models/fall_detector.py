"""
LSTM Fall Detection Classifier
================================
Sequence-based AI model for temporal motion analysis (§4.4).

Takes feature sequences extracted over consecutive frames and classifies
them as Fall or Normal Motion using a bidirectional LSTM network.
"""

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
import config


class FallDetectorLSTM(nn.Module):
    """
    Bidirectional LSTM-based fall detection classifier.
    
    Architecture:
        Input (seq_len × feature_dim)
        → Batch Normalization
        → 2-layer Bidirectional LSTM
        → Attention Layer
        → Fully Connected Layers
        → Sigmoid (fall probability)
    
    The model learns temporal motion transitions associated with 
    fall and non-fall activities (§4.4).
    """
    
    def __init__(self, input_dim=None, hidden_size=None, num_layers=None,
                 dropout=None, bidirectional=None):
        """
        Args:
            input_dim: Feature dimension per frame
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(FallDetectorLSTM, self).__init__()
        
        self.input_dim = input_dim or config.FEATURE_DIM
        self.hidden_size = hidden_size or config.LSTM_HIDDEN_SIZE
        self.num_layers = num_layers or config.LSTM_NUM_LAYERS
        self.dropout = dropout or config.LSTM_DROPOUT
        self.bidirectional = bidirectional if bidirectional is not None else config.LSTM_BIDIRECTIONAL
        self.num_directions = 2 if self.bidirectional else 1
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(self.input_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )
        
        # Attention mechanism for temporal weighting
        lstm_output_dim = self.hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
            
        Returns:
            output: Fall probability (batch_size, 1)
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # Apply batch normalization across features
        # Reshape: (B, T, F) → (B*T, F) → BatchNorm → (B, T, F)
        x_flat = x.contiguous().view(-1, feat_dim)
        x_flat = self.input_bn(x_flat)
        x = x_flat.view(batch_size, seq_len, feat_dim)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        
        # Attention-weighted aggregation
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size * num_directions)
        
        # Classification
        output = self.classifier(context)  # (batch, 1)
        
        return output
    
    def predict(self, x, threshold=None):
        """
        Predict fall/normal label.
        
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
            threshold: Float threshold for classification
            
        Returns:
            predictions: Binary tensor (batch_size,)
            probabilities: Float tensor (batch_size,)
        """
        threshold = threshold or config.FALL_THRESHOLD
        
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(x).squeeze(-1)
            predictions = (probabilities >= threshold).long()
        
        return predictions, probabilities
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(device=None):
    """
    Factory function to build and initialize the fall detector model.
    
    Args:
        device: torch device
        
    Returns:
        model: FallDetectorLSTM on specified device
    """
    device = device or config.MIDAS_DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    model = FallDetectorLSTM()
    model = model.to(device)
    
    print(f"[FallDetector] Model built on {device}")
    print(f"[FallDetector] Parameters: {model.count_parameters():,}")
    print(f"[FallDetector] Input dim: {model.input_dim}, Hidden: {model.hidden_size}")
    print(f"[FallDetector] Layers: {model.num_layers}, Bidirectional: {model.bidirectional}")
    
    return model


if __name__ == "__main__":
    print("Testing FallDetectorLSTM...")
    
    model = build_model("cpu")
    
    # Dummy input: batch=4, seq_len=30, features=177
    dummy_input = torch.randn(4, config.SEQUENCE_LENGTH, config.FEATURE_DIM)
    
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().detach().numpy()}")
    
    preds, probs = model.predict(dummy_input)
    print(f"Predictions: {preds.numpy()}")
    print(f"Probabilities: {probs.numpy()}")
    
    print("\nFallDetectorLSTM test PASSED ✓")
