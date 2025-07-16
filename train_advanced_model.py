"""
Advanced AttentionLSTM Model Training Script for CryptoQuantum
Enhanced with Attention Mechanism and Asymmetric Loss for Superior Crypto Predictions
"""

import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AsymmetricLoss(nn.Module):
    """Custom loss that penalizes underestimation more than overestimation for crypto trading"""
    def __init__(self, underestimation_penalty=1.5):
        super(AsymmetricLoss, self).__init__()
        self.underestimation_penalty = underestimation_penalty

    def forward(self, predictions, targets):
        diff = predictions - targets
        loss = torch.mean(torch.where(diff < 0,
                                    self.underestimation_penalty * diff**2,
                                    diff**2))
        return loss

class AttentionLSTMModel(nn.Module):
    """Advanced LSTM with Attention Mechanism for Superior Crypto Predictions"""
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, dropout=0.3):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced LSTM with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers with regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization for better convergence"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # Ensure proper batch dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        
        # Output processing
        out = self.layer_norm(context)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        
        return self.fc2(out)

def fetch_crypto_data(symbol='BTC-USD', period='3y'):
    """Fetch cryptocurrency data with enhanced error handling"""
    print(f"Fetching {period} of data for {symbol}...")
    
    try:
        data = yf.download(symbol, period=period, progress=False)
        
        if data.empty:
            print(f"No data found for {symbol}, trying shorter period...")
            data = yf.download(symbol, period='2y', progress=False)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Handle multi-level columns
        if hasattr(data.columns, 'get_level_values'):
            data.columns = data.columns.get_level_values(0)
        
        # Use Close and Volume for training
        return data[['Close', 'Volume']].values
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def create_sequences(data, sequence_length=60):
    """Create sequences for LSTM training with attention"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # Target is Close price return
    return np.array(X), np.array(y)

def prepare_advanced_data(data, sequence_length=60, train_split=0.85):
    """Advanced data preparation with log returns and robust scaling"""
    if data is None or len(data) < sequence_length + 50:
        print("Insufficient data for training")
        return None
    
    # Compute log returns for price stationarity
    close = data[:, 0].reshape(-1, 1)
    volume = data[:, 1].reshape(-1, 1)
    
    # Handle edge cases
    close = np.maximum(close, 1e-8)
    volume = np.maximum(volume, 1e-8)
    
    # Log returns
    log_close = np.log(close)
    returns = np.diff(log_close, axis=0)
    volume = volume[1:]  # Align with returns
    
    # Combine features
    combined = np.hstack((returns, volume))
    
    # Split data
    train_size = int(len(combined) * train_split)
    raw_train = combined[:train_size]
    raw_test = combined[train_size:]
    
    # Separate scalers for better handling
    scaler_returns = MinMaxScaler(feature_range=(-0.8, 0.8))
    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    
    # Scale training data
    train_returns = raw_train[:, 0:1]
    train_volume = raw_train[:, 1:2]
    
    scaled_train = np.hstack((
        scaler_returns.fit_transform(train_returns),
        scaler_volume.fit_transform(train_volume)
    ))
    
    # Create sequences
    X_train, y_train = create_sequences(scaled_train, sequence_length)
    
    # Scale test data
    test_returns = raw_test[:, 0:1]
    test_volume = raw_test[:, 1:2]
    
    scaled_test = np.hstack((
        scaler_returns.transform(test_returns),
        scaler_volume.transform(test_volume)
    ))
    
    X_test, y_test = create_sequences(scaled_test, sequence_length)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Return log prices for evaluation
    test_log_prices = log_close[train_size + sequence_length:]
    
    return X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, test_log_prices

def train_attention_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.001):
    """Train the AttentionLSTM model with advanced techniques"""
    
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Initialize model
    model = AttentionLSTMModel(input_size=2, hidden_size=128, num_layers=3, dropout=0.3).to(device)
    
    # Loss function and optimizer
    criterion = AsymmetricLoss(underestimation_penalty=1.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)
    
    # Data loader for batch training
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Training tracking
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model = None
    patience_counter = 0
    patience = 20
    
    print(f"Starting AttentionLSTM training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            pred = model(batch_x)
            loss = criterion(pred.squeeze(), batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
        
        train_loss /= len(X_train)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred.squeeze(), y_test)
        
        test_losses.append(test_loss.item())
        scheduler.step(test_loss)
        
        # Progress logging
        if (epoch + 1) % 20 == 0 or epoch < 10:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss.item():.6f}')
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} with best test loss: {best_test_loss:.6f}")
                break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Move model back to CPU for saving
    model = model.cpu()
    return model, train_losses, test_losses

def evaluate_attention_model(model, X_test, y_test, scaler_returns, test_log_prices):
    """Evaluate the trained AttentionLSTM model"""
    model.eval()
    with torch.no_grad():
        pred_returns = model(X_test).squeeze().numpy()
        y_test_returns = y_test.numpy()
        
        # Inverse transform returns
        pred_returns = scaler_returns.inverse_transform(pred_returns.reshape(-1, 1)).flatten()
        y_test_returns = scaler_returns.inverse_transform(y_test_returns.reshape(-1, 1)).flatten()
        
        # Convert returns to prices
        pred_log_prices = test_log_prices[0] + np.cumsum(pred_returns)
        y_test_log_prices = test_log_prices[0] + np.cumsum(y_test_returns)
        
        predictions = np.exp(pred_log_prices)
        y_test_actual = np.exp(y_test_log_prices)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy metrics
        direction_accuracy = np.mean(np.sign(pred_returns[1:]) == np.sign(y_test_returns[1:])) * 100
        
        print("\n🎯 AttentionLSTM Model Evaluation:")
        print(f"MSE: ${mse:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"Direction Accuracy: {direction_accuracy:.1f}%")
        
        return predictions, y_test_actual

def predict_advanced_future(model, last_sequence, scaler_returns, scaler_volume, last_log_price, avg_volume, steps=30):
    """Generate future predictions with the AttentionLSTM model"""
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    current_log_price = float(last_log_price)
    
    with torch.no_grad():
        for step in range(steps):
            # Update volume in sequence
            scaled_volume = scaler_volume.transform([[avg_volume]])[0][0]
            current_seq[-1, 1] = scaled_volume
            
            # Make prediction
            input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
            pred_return = model(input_tensor).squeeze().item()
            
            # Convert to price
            pred_return_actual = scaler_returns.inverse_transform([[pred_return]])[0][0]
            current_log_price += pred_return_actual
            pred_price = np.exp(current_log_price)
            predictions.append(pred_price)
            
            # Update sequence
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1, 0] = pred_return
            
    return np.array(predictions)

def plot_advanced_results(train_losses, test_losses, predictions, y_test_actual, future_preds=None):
    """Plot comprehensive training and prediction results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(test_losses, label='Test Loss', alpha=0.8)
    ax1.set_title('🔥 AttentionLSTM Training Progress')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Predictions vs Actual
    recent_days = min(100, len(y_test_actual))
    ax2.plot(y_test_actual[-recent_days:], label='Actual', alpha=0.8, linewidth=2)
    ax2.plot(predictions[-recent_days:], label='Predicted', alpha=0.8, linewidth=2)
    if future_preds is not None:
        future_x = range(recent_days, recent_days + len(future_preds))
        ax2.plot(future_x, future_preds, label='Future Predictions', 
                alpha=0.8, linestyle='--', linewidth=2)
    ax2.set_title(f'🎯 Predictions vs Actual (Last {recent_days} days + Future)')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Prediction error analysis
    errors = predictions - y_test_actual
    ax3.hist(errors, bins=30, alpha=0.7, color='orange')
    ax3.set_title('📊 Prediction Error Distribution')
    ax3.set_xlabel('Prediction Error ($)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: ${np.mean(errors):.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Percentage error
    pct_errors = ((predictions - y_test_actual) / y_test_actual) * 100
    ax4.plot(pct_errors[-recent_days:], alpha=0.8, color='green')
    ax4.set_title('📈 Percentage Error Over Time')
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Percentage Error (%)')
    ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline for AttentionLSTM model"""
    print("🚀 ADVANCED AttentionLSTM Bitcoin Prediction Model Training")
    print("=" * 70)
    print(f"Device: {device}")
    print("Features: Attention Mechanism, Asymmetric Loss, Advanced Architecture")
    print("=" * 70)
    
    # Fetch and prepare data
    data = fetch_crypto_data('BTC-USD', '3y')
    if data is None:
        print("❌ Failed to fetch data")
        return
    
    prepared_data = prepare_advanced_data(data, sequence_length=60)
    if prepared_data is None:
        print("❌ Failed to prepare data")
        return
    
    X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, test_log_prices = prepared_data
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print("Features: Log Returns + Volume with Attention Mechanism")
    
    # Train model
    model, train_losses, test_losses = train_attention_model(
        X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.001
    )
    
    # Evaluate model
    predictions, y_test_actual = evaluate_attention_model(
        model, X_test, y_test, scaler_returns, test_log_prices
    )
    
    # Future predictions
    last_sequence = X_test[-1].numpy()
    last_log_price = test_log_prices[-1]
    avg_volume = np.mean(data[-60:, 1])
    
    future_preds = predict_advanced_future(
        model, last_sequence, scaler_returns, scaler_volume, last_log_price, avg_volume, steps=30
    )
    
    # Display key predictions
    current_price = y_test_actual[-1]
    print("\n🔮 Advanced AI Forecasts:")
    print(f"Current (last test): ${current_price:.2f}")
    print(f"7 days:  ${future_preds[6]:.2f} ({((future_preds[6]/current_price)-1)*100:+.1f}%)")
    print(f"14 days: ${future_preds[13]:.2f} ({((future_preds[13]/current_price)-1)*100:+.1f}%)")
    print(f"30 days: ${future_preds[29]:.2f} ({((future_preds[29]/current_price)-1)*100:+.1f}%)")
    
    # Plot results
    plot_advanced_results(train_losses, test_losses, predictions, y_test_actual, future_preds)
    
    # Save model and scalers
    torch.save(model.state_dict(), 'btc_attention_model.pth')
    print("\n✅ AttentionLSTM model saved as 'btc_attention_model.pth'")
    
    joblib.dump((scaler_returns, scaler_volume), 'attention_scalers.pkl')
    print("✅ Scalers saved as 'attention_scalers.pkl'")
    
    print("\n🎉 Advanced AttentionLSTM training completed successfully!")

if __name__ == "__main__":
    main()
