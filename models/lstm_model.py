"""
LSTM Model for Cryptocurrency Price Prediction
Advanced neural network architecture for crypto forecasting
"""

import torch
import torch.nn as nn
import numpy as np

class CryptoPriceLSTM(nn.Module):
    """
    Advanced LSTM model for cryptocurrency price prediction
    Features attention mechanism and dropout for better generalization
    """
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2):
        super(CryptoPriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for prediction
        last_output = attn_out[:, -1, :]
        
        # Final prediction layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def predict_sequence(self, x, steps=30):
        """
        Predict multiple steps into the future
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            current_input = x.clone()
            
            for _ in range(steps):
                # Predict next value
                pred = self.forward(current_input)
                predictions.append(pred.item())
                
                # Update input sequence (sliding window)
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    pred.unsqueeze(0).unsqueeze(-1)
                ], dim=1)
        
        return predictions

class SimpleLSTM(nn.Module):
    """
    Simplified LSTM model for faster training
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

def create_model(model_type="advanced", **kwargs):
    """
    Factory function to create LSTM models
    """
    if model_type == "advanced":
        return CryptoPriceLSTM(**kwargs)
    elif model_type == "simple":
        return SimpleLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
