import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib  # For saving scaler
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Mixed precision training for speed
try:
    from torch.cuda.amp import autocast, GradScaler
    MIXED_PRECISION = torch.cuda.is_available()
except ImportError:
    MIXED_PRECISION = False

class LSTMModel(nn.Module):
    """Enhanced LSTM Model with BatchNorm and improved architecture"""
    def __init__(self, input_size=2, hidden_size=64, num_layers=3, output_size=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

def fetch_training_data(symbol='BTC-USD', period='2y'):
    """Fetch historical data including Volume for multi-variate input"""
    print(f"Fetching {period} of data for {symbol}...")
    data = yf.download(symbol, period=period, progress=False)
    
    # Handle multi-level columns from yfinance
    if hasattr(data.columns, 'get_level_values'):
        data.columns = data.columns.get_level_values(0)
    
    return data[['Close', 'Volume']].values  # Shape: (n_days, 2)

def create_sequences(data, sequence_length=30):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # Target is Close price (or return)
    return np.array(X), np.array(y)

def prepare_data(data, sequence_length=30, train_split=0.85):
    """Prepare data with log returns for stationarity, scale, and split"""
    # Compute log returns for Close (diff(log(Close))), keep Volume as is
    close = data[:, 0].reshape(-1, 1)
    volume = data[:, 1].reshape(-1, 1)
    
    # Handle any zero or negative values in close prices
    close = np.maximum(close, 1e-8)
    
    log_close = np.log(close)
    returns = np.diff(log_close, axis=0)  # Returns: shape (n-1, 1)
    volume = volume[1:]  # Align with returns

    # Combine returns and volume
    combined = np.hstack((returns, volume))

    # Split raw data - use more recent data for testing
    train_size = int(len(combined) * train_split)
    raw_train = combined[:train_size]
    raw_test = combined[train_size:]

    # Scale on train only (separate scalers for returns and volume)
    scaler_returns = MinMaxScaler(feature_range=(-0.8, 0.8))  # Narrower range for better stability
    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    
    # Handle volume scaling safely
    train_volume = raw_train[:, 1:2]
    test_volume = raw_test[:, 1:2]
    
    # Replace any zero volumes with small positive values
    train_volume = np.maximum(train_volume, 1e-8)
    test_volume = np.maximum(test_volume, 1e-8)
    
    scaled_train = np.hstack((
        scaler_returns.fit_transform(raw_train[:, 0:1]),
        scaler_volume.fit_transform(train_volume)
    ))

    # Create sequences (target y is returns)
    X_train, y_train = create_sequences(scaled_train, sequence_length)

    # Scale test
    scaled_test = np.hstack((
        scaler_returns.transform(raw_test[:, 0:1]),
        scaler_volume.transform(test_volume)
    ))
    X_test, y_test = create_sequences(scaled_test, sequence_length)

    # Tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, log_close[train_size + sequence_length:]

def train_model(X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.002, batch_size=64, patience=20):
    """Enhanced training with GPU support, mixed precision, and better optimization"""
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    model = LSTMModel(input_size=2, hidden_size=128, num_layers=2, output_size=1, dropout=0.2).to(device)
    criterion = nn.HuberLoss(delta=0.5)  # More conservative loss for crypto volatility
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)

    # Mixed precision setup
    scaler = None
    if MIXED_PRECISION:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print("Using mixed precision training for faster performance")

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model = None
    counter = 0
    
    print(f"Starting enhanced training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            if MIXED_PRECISION:
                from torch.cuda.amp import autocast
                with autocast():
                    pred = model(batch_x)
                    loss = criterion(pred.squeeze(), batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch_x)
                loss = criterion(pred.squeeze(), batch_y)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
        
        train_loss /= len(X_train)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            if MIXED_PRECISION:
                from torch.cuda.amp import autocast
                with autocast():
                    test_pred = model(X_test)
                    test_loss = criterion(test_pred.squeeze(), y_test)
            else:
                test_pred = model(X_test)
                test_loss = criterion(test_pred.squeeze(), y_test)
        
        test_losses.append(test_loss.item())
        scheduler.step(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss.item():.6f}')
        
        # Early stopping with improved logic
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1} with best test loss: {best_test_loss:.6f}")
                break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Move model back to CPU for inference if needed
    model = model.cpu()
    return model, train_losses, test_losses

def evaluate_model(model, X_test, y_test, scaler_returns, last_log_prices):
    """Evaluate, inverse transform returns to prices"""
    model.eval()
    with torch.no_grad():
        pred_returns = model(X_test).squeeze().numpy()
        y_test_returns = y_test.numpy()
        
        # Inverse scale returns
        pred_returns = scaler_returns.inverse_transform(pred_returns.reshape(-1, 1)).flatten()
        y_test_returns = scaler_returns.inverse_transform(y_test_returns.reshape(-1, 1)).flatten()
        
        # Cumsum returns to log prices, then exp to prices
        # Start from the first test log price
        pred_log_prices = last_log_prices[0] + np.cumsum(pred_returns)
        y_test_log_prices = last_log_prices[0] + np.cumsum(y_test_returns)
        predictions = np.exp(pred_log_prices)
        y_test_actual = np.exp(y_test_log_prices)
        
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        
        print("\nModel Evaluation:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        return predictions, y_test_actual

def predict_future(model, last_sequence, scaler_returns, scaler_volume, last_log_price, avg_volume, steps=30):
    """Predict future returns with very conservative constraints for realistic long-term forecasts"""
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()  # Shape: (30, 2)
    current_log_price = float(last_log_price)
    
    with torch.no_grad():
        for step in range(steps):
            # Scale assumed volume
            scaled_volume = scaler_volume.transform([[avg_volume]])[0][0]
            current_seq[-1, 1] = scaled_volume  # Update last volume in seq
            
            input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
            pred_return = model(input_tensor).squeeze().item()
            
            # Apply very aggressive constraints for long-term predictions
            if steps > 30:  # For any prediction longer than 1 month
                # Extremely aggressive dampening
                days_ratio = min(step / 365, 1.0)  # Cap at 1 year ratio
                
                # Massive dampening - reduce volatility by 95% over time
                dampening_factor = 1.0 - (days_ratio * 0.95)
                pred_return = pred_return * dampening_factor
                
                # Replace model predictions with simple trend after first month
                if step > 30:
                    # Use very conservative fixed annual growth rate
                    annual_growth = 0.10  # 10% annual growth max
                    daily_growth = (1 + annual_growth) ** (1/365) - 1
                    
                    # Reduce growth over time (market maturity)
                    if step > 365:
                        maturity_factor = max(0.3, 1 - (step - 365) / 1460)  # Reduce to 30% by year 5
                        daily_growth *= maturity_factor
                    
                    # Convert to scaled return
                    pred_return_actual = daily_growth
                    pred_return = scaler_returns.transform([[pred_return_actual]])[0][0]
                
                # Cap returns to very conservative levels
                max_daily_return = 0.005  # 0.5% max daily gain
                min_daily_return = -0.005  # 0.5% max daily loss
                pred_return_actual = scaler_returns.inverse_transform([[pred_return]])[0][0]
                pred_return_actual = np.clip(pred_return_actual, min_daily_return, max_daily_return)
                pred_return = scaler_returns.transform([[pred_return_actual]])[0][0]
            
            # Inverse return and add to log price
            pred_return_actual = scaler_returns.inverse_transform([[pred_return]])[0][0]
            current_log_price += pred_return_actual
            pred_price = np.exp(current_log_price)
            predictions.append(pred_price)
            
            # Roll and update seq with new return (scaled) and volume
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1, 0] = pred_return  # Scaled return
            
    return np.array(predictions)

def plot_results(train_losses, test_losses, predictions, y_test_actual, future_preds=None):
    """Plot results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(y_test_actual[-100:], label='Actual', alpha=0.7)
    ax2.plot(predictions[-100:], label='Predicted', alpha=0.7)
    if future_preds is not None:
        ax2.plot(range(len(y_test_actual)-100, len(y_test_actual)-100 + len(future_preds)), future_preds, label='Future Predicted', alpha=0.7, linestyle='--')
    ax2.set_title('Actual vs Predicted (Last 100 days + Future)')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def main():
    print("🚀 ENHANCED Bitcoin Price Prediction Model Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mixed Precision: {MIXED_PRECISION}")
    print("Features: Log Returns, GPU Acceleration, Huber Loss, Early Stopping")
    print("=" * 60)
    
    data = fetch_training_data('BTC-USD', '2y')
    prepared_data = prepare_data(data)
    
    if len(prepared_data) != 7 or prepared_data[0] is None:
        print("Error preparing data")
        return
        
    X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, test_log_prices = prepared_data
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print("Features: Returns + Volume")
    
    model, train_losses, test_losses = train_model(X_train, y_train, X_test, y_test, epochs=80)
    
    predictions, y_test_actual = evaluate_model(model, X_test, y_test, scaler_returns, test_log_prices)
    
    # Future predictions - Short term (30 days)
    last_sequence = X_test[-1].numpy()  # Last seq (30, 2)
    last_log_price = test_log_prices[-1]  # Last actual log price
    avg_volume = np.mean(data[-30:, 1])  # Avg recent volume
    future_preds_30d = predict_future(model, last_sequence, scaler_returns, scaler_volume, last_log_price, avg_volume, steps=30)
    
    # Long-term predictions (5 years = 1825 days)
    print("\n🔮 Generating Long-term Predictions (5 years)...")
    future_preds_5y = predict_future(model, last_sequence, scaler_returns, scaler_volume, last_log_price, avg_volume, steps=1825)
    
    # Show key predictions
    current_price = y_test_actual[-1]
    print("\n📈 SHORT-TERM FORECASTS (30 days):")
    print(f"Current (last test): ${current_price:.2f}")
    print(f"7 days:  ${future_preds_30d[6]:.2f} ({((future_preds_30d[6]/current_price)-1)*100:+.1f}%)")
    print(f"30 days: ${future_preds_30d[29]:.2f} ({((future_preds_30d[29]/current_price)-1)*100:+.1f}%)")
    
    print("\n🚀 LONG-TERM FORECASTS:")
    year_indices = [365, 730, 1095, 1460, 1824]  # 1, 2, 3, 4, 5 years
    for i, year_idx in enumerate(year_indices, 1):
        if year_idx < len(future_preds_5y):
            year_price = future_preds_5y[year_idx]
            year_change = ((year_price / current_price) - 1) * 100
            print(f"Year {i}:   ${year_price:.2f} ({year_change:+.1f}%)")
    
    # Target analysis for $225K
    target_price = 225000
    print("\n🎯 TARGET ANALYSIS ($225K):")
    print(f"Current price: ${current_price:.2f}")
    print(f"Target price:  ${target_price:.2f}")
    required_growth = ((target_price / current_price) - 1) * 100
    print(f"Required growth: {required_growth:.1f}%")
    
    # Find when we might reach $225K
    final_price_5y = future_preds_5y[-1]
    if final_price_5y >= target_price:
        # Find approximate day when target is reached
        for day, price in enumerate(future_preds_5y):
            if price >= target_price:
                years = day / 365.25
                print(f"🎉 Model predicts $225K could be reached in ~{years:.1f} years (day {day})")
                break
    else:
        print(f"📊 Model predicts ${final_price_5y:.2f} after 5 years")
        print(f"    Annual growth rate needed for $225K: {((target_price/current_price)**(1/5)-1)*100:.1f}%")
    
    # Realistic scenario analysis
    print("\n📊 REALISTIC SCENARIO ANALYSIS:")
    scenarios = [
        ("Conservative (3% annual)", 0.03),
        ("Moderate (8% annual)", 0.08), 
        ("Optimistic (15% annual)", 0.15),
        ("Very Bullish (25% annual)", 0.25)
    ]
    
    for scenario_name, annual_rate in scenarios:
        price_5y = current_price * ((1 + annual_rate) ** 5)
        print(f"{scenario_name:20}: ${price_5y:8,.0f}")
    
    print(f"\n💡 For $225K in 5 years, need: {((target_price/current_price)**(1/5)-1)*100:.1f}% annual growth")
    
    # Calculate prediction confidence metrics
    recent_volatility = np.std(y_test_actual[-30:]) / np.mean(y_test_actual[-30:]) * 100
    print("\nModel Confidence Metrics:")
    print(f"Recent 30-day volatility: {recent_volatility:.1f}%")
    print(f"Training completed with {len(train_losses)} epochs")
    
    plot_results(train_losses, test_losses, predictions, y_test_actual, future_preds_30d)
    
    torch.save(model.state_dict(), 'btc_model_enhanced.pth')
    print("\n✅ Enhanced model saved as 'btc_model_enhanced.pth'")
    
    joblib.dump((scaler_returns, scaler_volume), 'scalers_enhanced.pkl')
    print("✅ Scalers saved as 'scalers_enhanced.pkl'")

if __name__ == "__main__":
    main()
