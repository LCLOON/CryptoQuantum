"""
Multi-Cryptocurrency Portfolio Training Script for CryptoQuantum
Train models for multiple cryptocurrencies and create ensemble predictions
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
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Top cryptocurrencies for portfolio training
CRYPTO_SYMBOLS = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Binance Coin': 'BNB-USD',
    'Solana': 'SOL-USD',
    'XRP': 'XRP-USD',
    'Dogecoin': 'DOGE-USD',
    'Cardano': 'ADA-USD',
    'Avalanche': 'AVAX-USD',
    'Polygon': 'MATIC-USD',
    'Chainlink': 'LINK-USD'
}

class UniversalCryptoModel(nn.Module):
    """Universal LSTM model that works well across different cryptocurrencies"""
    def __init__(self, input_size=2, hidden_size=96, num_layers=2, dropout=0.25):
        super(UniversalCryptoModel, self).__init__()
        
        # Optimized architecture for multi-crypto training
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Batch normalization for better generalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Multi-layer perceptron with residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for optimal training across different crypto patterns"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias to 1
                n = param.size(0)
                if 'lstm' in name:
                    param.data[n//4:n//2].fill_(1)

    def forward(self, x):
        # LSTM processing
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last timestep
        
        # Batch normalization (handle single sample case)
        if out.size(0) > 1:
            out = self.batch_norm(out)
        
        # Multi-layer processing with residual connection
        identity = out
        out = self.gelu(self.fc1(out))
        out = self.dropout1(out)
        
        # Residual connection if dimensions match
        if identity.size(-1) == out.size(-1):
            out = out + identity
        
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        
        return self.fc3(out)

def fetch_crypto_data_safe(symbol, name, period='2y'):
    """Safely fetch cryptocurrency data with error handling"""
    try:
        print(f"📡 Fetching {name} ({symbol})...")
        data = yf.download(symbol, period=period, progress=False)
        
        if data.empty or len(data) < 100:
            print(f"⚠️ Insufficient data for {name}")
            return None, None
        
        # Handle multi-level columns
        if hasattr(data.columns, 'get_level_values'):
            data.columns = data.columns.get_level_values(0)
        
        # Return close and volume data
        crypto_data = data[['Close', 'Volume']].values
        print(f"✅ {name}: {len(crypto_data)} days of data")
        return crypto_data, name
        
    except Exception as e:
        print(f"❌ Error fetching {name}: {str(e)}")
        return None, None

def prepare_crypto_data(data, sequence_length=30, train_split=0.85):
    """Prepare individual cryptocurrency data for training"""
    if data is None or len(data) < sequence_length + 50:
        return None
    
    # Compute log returns for stationarity
    close = data[:, 0].reshape(-1, 1)
    volume = data[:, 1].reshape(-1, 1)
    
    # Handle edge cases
    close = np.maximum(close, 1e-8)
    volume = np.maximum(volume, 1e-8)
    
    # Calculate log returns
    log_close = np.log(close)
    returns = np.diff(log_close, axis=0)
    volume = volume[1:]  # Align with returns
    
    # Combine features
    combined = np.hstack((returns, volume))
    
    # Split data
    train_size = int(len(combined) * train_split)
    raw_train = combined[:train_size]
    raw_test = combined[train_size:]
    
    # Scale data
    scaler_returns = MinMaxScaler(feature_range=(-0.7, 0.7))
    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    
    # Scale training data
    scaled_train = np.hstack((
        scaler_returns.fit_transform(raw_train[:, 0:1]),
        scaler_volume.fit_transform(raw_train[:, 1:2])
    ))
    
    # Create sequences
    X_train, y_train = create_sequences(scaled_train, sequence_length)
    
    # Scale test data
    scaled_test = np.hstack((
        scaler_returns.transform(raw_test[:, 0:1]),
        scaler_volume.transform(raw_test[:, 1:2])
    ))
    
    X_test, y_test = create_sequences(scaled_test, sequence_length)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Additional info for evaluation
    test_log_prices = log_close[train_size + sequence_length:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler_returns': scaler_returns,
        'scaler_volume': scaler_volume,
        'test_log_prices': test_log_prices,
        'raw_data': data
    }

def create_sequences(data, sequence_length=30):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # Target is the return
    return np.array(X), np.array(y)

def train_universal_model(all_crypto_data, epochs=80, batch_size=64, learning_rate=0.001):
    """Train a universal model on all cryptocurrency data"""
    print("\n🤖 Training Universal Multi-Crypto Model...")
    
    # Combine all training data
    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []
    
    for crypto_name, crypto_data in all_crypto_data.items():
        if crypto_data is not None:
            all_X_train.append(crypto_data['X_train'])
            all_y_train.append(crypto_data['y_train'])
            all_X_test.append(crypto_data['X_test'])
            all_y_test.append(crypto_data['y_test'])
    
    # Concatenate all data
    X_train_combined = torch.cat(all_X_train, dim=0)
    y_train_combined = torch.cat(all_y_train, dim=0)
    X_test_combined = torch.cat(all_X_test, dim=0)
    y_test_combined = torch.cat(all_y_test, dim=0)
    
    print(f"Combined training data: {X_train_combined.shape}")
    print(f"Combined test data: {X_test_combined.shape}")
    
    # Move to device
    X_train_combined = X_train_combined.to(device)
    y_train_combined = y_train_combined.to(device)
    X_test_combined = X_test_combined.to(device)
    y_test_combined = y_test_combined.to(device)
    
    # Initialize model
    model = UniversalCryptoModel(input_size=2, hidden_size=96, num_layers=2, dropout=0.25).to(device)
    criterion = nn.SmoothL1Loss(beta=0.5)  # Robust loss for diverse crypto patterns
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=12)
    
    # Data loader
    dataset = TensorDataset(X_train_combined, y_train_combined)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Training tracking
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model = None
    patience_counter = 0
    patience = 25
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
        
        train_loss /= len(X_train_combined)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_combined)
            test_loss = criterion(test_pred.squeeze(), y_test_combined)
        
        test_losses.append(test_loss.item())
        scheduler.step(test_loss)
        
        # Progress logging
        if (epoch + 1) % 15 == 0 or epoch < 5:
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
    
    # Move back to CPU
    model = model.cpu()
    return model, train_losses, test_losses

def evaluate_multi_crypto_model(model, all_crypto_data):
    """Evaluate the universal model on each cryptocurrency"""
    print("\n📊 Evaluating Universal Model on Individual Cryptocurrencies...")
    
    results = {}
    
    for crypto_name, crypto_data in all_crypto_data.items():
        if crypto_data is None:
            continue
        
        model.eval()
        with torch.no_grad():
            # Make predictions
            pred_returns = model(crypto_data['X_test']).squeeze().numpy()
            y_test_returns = crypto_data['y_test'].numpy()
            
            # Inverse transform
            scaler_returns = crypto_data['scaler_returns']
            pred_returns = scaler_returns.inverse_transform(pred_returns.reshape(-1, 1)).flatten()
            y_test_returns = scaler_returns.inverse_transform(y_test_returns.reshape(-1, 1)).flatten()
            
            # Convert to prices
            test_log_prices = crypto_data['test_log_prices']
            pred_log_prices = test_log_prices[0] + np.cumsum(pred_returns)
            y_test_log_prices = test_log_prices[0] + np.cumsum(y_test_returns)
            
            predictions = np.exp(pred_log_prices)
            actual_prices = np.exp(y_test_log_prices)
            
            # Calculate metrics
            mse = mean_squared_error(actual_prices, predictions)
            mae = mean_absolute_error(actual_prices, predictions)
            rmse = np.sqrt(mse)
            
            # Direction accuracy
            direction_accuracy = np.mean(np.sign(pred_returns[1:]) == np.sign(y_test_returns[1:])) * 100
            
            results[crypto_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'direction_accuracy': direction_accuracy,
                'predictions': predictions,
                'actual': actual_prices
            }
            
            print(f"{crypto_name:12} | RMSE: ${rmse:8.2f} | MAE: ${mae:8.2f} | Dir Acc: {direction_accuracy:5.1f}%")
    
    return results

def create_portfolio_predictions(model, all_crypto_data, steps=30):
    """Create future predictions for the entire crypto portfolio"""
    print(f"\n🔮 Generating {steps}-day Portfolio Forecasts...")
    
    portfolio_predictions = {}
    
    for crypto_name, crypto_data in all_crypto_data.items():
        if crypto_data is None:
            continue
        
        # Get last sequence and data
        last_sequence = crypto_data['X_test'][-1].numpy()
        last_log_price = crypto_data['test_log_prices'][-1]
        avg_volume = np.mean(crypto_data['raw_data'][-60:, 1])
        
        # Generate predictions
        predictions = []
        current_seq = last_sequence.copy()
        current_log_price = float(last_log_price)
        
        model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Update volume
                scaled_volume = crypto_data['scaler_volume'].transform([[avg_volume]])[0][0]
                current_seq[-1, 1] = scaled_volume
                
                # Make prediction
                input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                pred_return = model(input_tensor).squeeze().item()
                
                # Convert to price
                pred_return_actual = crypto_data['scaler_returns'].inverse_transform([[pred_return]])[0][0]
                current_log_price += pred_return_actual
                pred_price = np.exp(current_log_price)
                predictions.append(pred_price)
                
                # Update sequence
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1, 0] = pred_return
        
        portfolio_predictions[crypto_name] = np.array(predictions)
    
    return portfolio_predictions

def plot_portfolio_results(evaluation_results, portfolio_predictions):
    """Create comprehensive portfolio visualization"""
    n_cryptos = len(evaluation_results)
    fig, axes = plt.subplots(2, (n_cryptos + 1) // 2, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (crypto_name, results) in enumerate(evaluation_results.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        # Plot recent actual vs predicted
        recent_days = min(60, len(results['actual']))
        actual = results['actual'][-recent_days:]
        predicted = results['predictions'][-recent_days:]
        
        ax.plot(actual, label='Actual', alpha=0.8, linewidth=2)
        ax.plot(predicted, label='Predicted', alpha=0.8, linewidth=2)
        
        # Add future predictions
        if crypto_name in portfolio_predictions:
            future_preds = portfolio_predictions[crypto_name]
            future_x = range(recent_days, recent_days + len(future_preds))
            ax.plot(future_x, future_preds, label='Future', alpha=0.8, linestyle='--', linewidth=2)
        
        ax.set_title(f'{crypto_name}\nRMSE: ${results["rmse"]:.2f} | Dir Acc: {results["direction_accuracy"]:.1f}%')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(evaluation_results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('multi_crypto_portfolio_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_portfolio_models(model, all_crypto_data):
    """Save the universal model and all scalers"""
    # Save universal model
    torch.save(model.state_dict(), 'universal_crypto_model.pth')
    print("✅ Universal model saved as 'universal_crypto_model.pth'")
    
    # Save all scalers
    scalers_dict = {}
    for crypto_name, crypto_data in all_crypto_data.items():
        if crypto_data is not None:
            scalers_dict[crypto_name] = {
                'scaler_returns': crypto_data['scaler_returns'],
                'scaler_volume': crypto_data['scaler_volume']
            }
    
    joblib.dump(scalers_dict, 'multi_crypto_scalers.pkl')
    print("✅ All scalers saved as 'multi_crypto_scalers.pkl'")

def main():
    """Main pipeline for multi-cryptocurrency training"""
    print("🚀 MULTI-CRYPTOCURRENCY PORTFOLIO TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Training Universal Model on Multiple Cryptocurrencies")
    print("=" * 60)
    
    # Fetch data for all cryptocurrencies
    all_crypto_data = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all download tasks
        future_to_crypto = {
            executor.submit(fetch_crypto_data_safe, symbol, name): name 
            for name, symbol in CRYPTO_SYMBOLS.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_crypto):
            name = future_to_crypto[future]
            try:
                data, crypto_name = future.result()
                if data is not None:
                    prepared_data = prepare_crypto_data(data)
                    all_crypto_data[crypto_name] = prepared_data
                else:
                    all_crypto_data[crypto_name] = None
            except Exception as e:
                print(f"❌ Error processing {name}: {e}")
                all_crypto_data[name] = None
    
    # Filter out None values
    valid_cryptos = {k: v for k, v in all_crypto_data.items() if v is not None}
    print(f"\n✅ Successfully prepared data for {len(valid_cryptos)} cryptocurrencies")
    
    if len(valid_cryptos) < 3:
        print("❌ Need at least 3 cryptocurrencies for meaningful training")
        return
    
    # Train universal model
    model, train_losses, test_losses = train_universal_model(valid_cryptos)
    
    # Evaluate on individual cryptocurrencies
    evaluation_results = evaluate_multi_crypto_model(model, valid_cryptos)
    
    # Create portfolio predictions
    portfolio_predictions = create_portfolio_predictions(model, valid_cryptos)
    
    # Display summary
    print("\n📈 PORTFOLIO SUMMARY:")
    total_rmse = np.mean([r['rmse'] for r in evaluation_results.values()])
    total_accuracy = np.mean([r['direction_accuracy'] for r in evaluation_results.values()])
    print(f"Average RMSE: ${total_rmse:.2f}")
    print(f"Average Direction Accuracy: {total_accuracy:.1f}%")
    
    # Show key future predictions
    print("\n🔮 30-Day Portfolio Forecasts:")
    for crypto_name, future_prices in portfolio_predictions.items():
        if crypto_name in evaluation_results:
            current_price = evaluation_results[crypto_name]['actual'][-1]
            future_30d = future_prices[29]
            change_pct = ((future_30d / current_price) - 1) * 100
            print(f"{crypto_name:12} | Current: ${current_price:8.2f} | 30d: ${future_30d:8.2f} | Change: {change_pct:+6.1f}%")
    
    # Plot results
    plot_portfolio_results(evaluation_results, portfolio_predictions)
    
    # Save models
    save_portfolio_models(model, valid_cryptos)
    
    print("\n🎉 Multi-Crypto Portfolio Training Completed!")
    print(f"Trained on {len(valid_cryptos)} cryptocurrencies")
    print("Ready for advanced portfolio analysis and trading strategies!")

if __name__ == "__main__":
    main()
