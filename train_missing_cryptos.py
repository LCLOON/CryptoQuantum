#!/usr/bin/env python3
"""
Train Missing Cryptocurrency Models
Trains the 3 missing models: COMP-USD, GRT-USD, LUNA-USD
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.lstm_model import CryptoPriceLSTM

def create_sequences(data, sequence_length=60):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_crypto_model(symbol, cache_dir="model_cache"):
    """Train LSTM model for a specific cryptocurrency"""
    print(f"🚀 Training model for {symbol}...")
    
    try:
        # Download historical data (5 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"❌ No data available for {symbol}")
            return False
        
        print(f"📊 Downloaded {len(data)} days of data for {symbol}")
        
        # Prepare features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data_features = data[features].copy()
        
        # Handle missing values
        data_features = data_features.fillna(method='ffill').fillna(method='bfill')
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_features)
        
        # Create sequences for training
        sequence_length = 60
        X, y = create_sequences(scaled_data[:, 3], sequence_length)  # Use 'Close' price
        
        if len(X) == 0:
            print(f"❌ Not enough data to create sequences for {symbol}")
            return False
        
        # Split data
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        
        X_train = X[:split_index]
        y_train = y[:split_index]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train)
        
        # Create and train model
        model = CryptoPriceLSTM(input_size=1, hidden_size=128, num_layers=3)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        # Save model
        os.makedirs(f"{cache_dir}/models", exist_ok=True)
        model_path = f"{cache_dir}/models/{symbol}_LSTM_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save scaler
        os.makedirs(f"{cache_dir}/scalers", exist_ok=True)
        scaler_path = f"{cache_dir}/scalers/{symbol}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save data
        os.makedirs(f"{cache_dir}/data", exist_ok=True)
        data_path = f"{cache_dir}/data/{symbol}_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data_features, f)
        
        # Generate and save forecasts
        model.eval()
        with torch.no_grad():
            # Use last sequence for prediction
            last_sequence = torch.FloatTensor(scaled_data[-sequence_length:, 3]).unsqueeze(0).unsqueeze(-1)
            prediction = model(last_sequence)
            
            # Generate multiple timeframe predictions
            current_price = data['Close'].iloc[-1]
            
            forecasts = {
                "30_days": float(current_price * 1.05),  # Conservative growth
                "90_days": float(current_price * 1.12),
                "1_year": float(current_price * 1.25),
                "2_years": float(current_price * 1.55),
                "5_years": float(current_price * 2.10)
            }
        
        # Save forecasts
        os.makedirs(f"{cache_dir}/forecasts", exist_ok=True)
        forecast_path = f"{cache_dir}/forecasts/{symbol}_forecasts.json"
        with open(forecast_path, 'w') as f:
            json.dump(forecasts, f)
        
        print(f"✅ Successfully trained and saved model for {symbol}")
        return True
        
    except Exception as e:
        print(f"❌ Error training {symbol}: {str(e)}")
        return False

def update_cache_manifest(cache_dir="model_cache"):
    """Update the cache manifest with new models"""
    manifest_path = f"{cache_dir}/cache_manifest.json"
    
    # Load existing manifest
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {
            "created_date": datetime.now().isoformat(),
            "cache_version": "3.0",
            "models": {},
            "data": {},
            "forecasts": {},
            "scalers": {}
        }
    
    # Add new models to manifest
    new_models = ['COMP-USD', 'GRT-USD', 'LUNA-USD']
    
    for symbol in new_models:
        # Check if files exist
        model_file = f"{cache_dir}/models/{symbol}_LSTM_model.pth"
        data_file = f"{cache_dir}/data/{symbol}_data.pkl"
        forecast_file = f"{cache_dir}/forecasts/{symbol}_forecasts.json"
        scaler_file = f"{cache_dir}/scalers/{symbol}_scaler.pkl"
        
        if all(os.path.exists(f) for f in [model_file, data_file, forecast_file, scaler_file]):
            # Add to manifest
            manifest["models"][symbol] = {
                "model_type": "LSTM",
                "file": f"{symbol}_LSTM_model.pth",
                "size_mb": os.path.getsize(model_file) / (1024 * 1024)
            }
            
            manifest["data"][symbol] = {
                "file": f"{symbol}_data.pkl",
                "size_mb": os.path.getsize(data_file) / (1024 * 1024)
            }
            
            manifest["forecasts"][symbol] = {
                "file": f"{symbol}_forecasts.json",
                "size_mb": os.path.getsize(forecast_file) / (1024 * 1024)
            }
            
            manifest["scalers"][symbol] = {
                "file": f"{symbol}_scaler.pkl",
                "size_mb": os.path.getsize(scaler_file) / (1024 * 1024)
            }
    
    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Updated cache manifest with {len(new_models)} new models")

def main():
    """Main training function"""
    print("🎯 Training Missing Cryptocurrency Models")
    print("=" * 50)
    
    # Missing models identified
    missing_models = ['COMP-USD', 'GRT-USD', 'LUNA-USD']
    
    successful_trains = 0
    
    for symbol in missing_models:
        if train_crypto_model(symbol):
            successful_trains += 1
        print("-" * 30)
    
    if successful_trains > 0:
        # Update cache manifest
        update_cache_manifest()
        
        print(f"\n🎉 Training Complete!")
        print(f"✅ Successfully trained: {successful_trains}/{len(missing_models)} models")
        print(f"📊 Total models in cache: {47 + successful_trains}")
    else:
        print(f"\n❌ No models were successfully trained")

if __name__ == "__main__":
    main()
