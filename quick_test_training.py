#!/usr/bin/env python3
"""
Quick test training with forecast fix
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.linear(out[:, -1, :])
        return out

class QuickTrainer:
    def __init__(self):
        self.cache_dir = Path("cache")
        self.models_dir = self.cache_dir / 'models'
        self.forecasts_dir = self.cache_dir / 'forecasts'
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.forecasts_dir.mkdir(exist_ok=True)
        
        # Timeframes for forecasting
        self.timeframes = {
            "30_days": 30,
            "90_days": 90,
            "180_days": 180,
            "365_days": 365,
            "730_days": 730
        }
    
    def prepare_data(self, symbol):
        """Prepare data for training"""
        try:
            logger.info(f"📊 Preparing data for {symbol}")
            
            # Download data
            end_date = "2025-09-06"
            start_date = "2023-09-07"
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if data.empty:
                return None, None, None
            
            # Use closing prices
            prices = data['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(prices)
            
            # Create sequences
            sequence_length = 60
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            logger.info(f"✅ Data prepared for {symbol}: {len(X)} sequences")
            return X, y, scaler
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return None, None, None
    
    def train_model(self, symbol, X, y):
        """Train the LSTM model"""
        try:
            logger.info(f"🧠 Training model for {symbol}")
            
            model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).view(-1, 1)
            
            # Quick training - just 20 epochs for testing
            for epoch in range(20):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/20, Loss: {loss.item():.6f}")
            
            logger.info(f"✅ Model trained for {symbol}")
            return model
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return None
    
    def generate_forecasts(self, model, scaler, symbol, last_value):
        """Generate forecasts for all timeframes"""
        try:
            logger.info(f"🔮 Generating forecasts for {symbol}")
            
            forecasts = {}
            model.eval()
            
            with torch.no_grad():
                current_value = last_value
                
                for timeframe, days in self.timeframes.items():
                    predictions = []
                    current_pred = current_value
                    
                    for _ in range(days):
                        # Create a sequence of length 60 filled with current prediction
                        sequence = np.full((1, 60, 1), current_pred, dtype=np.float32)
                        input_tensor = torch.FloatTensor(sequence)
                        
                        # Make prediction
                        prediction = model(input_tensor).item()
                        predictions.append(prediction)
                        
                        # Update for next prediction
                        current_pred = prediction
                    
                    # Transform back to original scale
                    final_prediction = scaler.inverse_transform([[predictions[-1]]])[0][0]
                    forecasts[timeframe] = float(final_prediction)
            
            logger.info(f"✅ Forecasts generated for {symbol}: {forecasts}")
            return forecasts
            
        except Exception as e:
            logger.error(f"❌ Error generating forecasts for {symbol}: {e}")
            return {}
    
    def train_crypto(self, symbol):
        """Train and cache one cryptocurrency"""
        try:
            # 1. Prepare data
            X, y, scaler = self.prepare_data(symbol)
            if X is None:
                return False
            
            # 2. Train model
            model = self.train_model(symbol, X, y)
            if model is None:
                return False
            
            # 3. Save model
            model_file = f"{symbol}_LSTM_model.pth"
            model_path = self.models_dir / model_file
            torch.save(model.state_dict(), model_path)
            
            # 4. Generate forecasts using the last scaled value
            last_value = scaler.transform([[y[-1]]])[0][0]  # Use last y value
            forecasts = self.generate_forecasts(model, scaler, symbol, last_value)
            
            if forecasts:
                # 5. Save forecasts
                forecast_data = {
                    'symbol': symbol,
                    'generated_date': '2025-09-06',
                    'forecasts': forecasts
                }
                
                forecast_file = f"{symbol}_forecasts.json"
                forecast_path = self.forecasts_dir / forecast_file
                
                with open(forecast_path, 'w') as f:
                    json.dump(forecast_data, f, indent=2)
                
                logger.info(f"✅ Successfully cached {symbol}")
                return True
            else:
                logger.warning(f"Failed to generate forecasts for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error caching {symbol}: {e}")
            return False

def main():
    trainer = QuickTrainer()
    
    # Test with just 3 popular cryptos
    test_symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
    
    print("🚀 Quick Test Training with Forecast Fix")
    print("="*50)
    
    success_count = 0
    
    for symbol in test_symbols:
        print(f"\n📊 Training {symbol}...")
        if trainer.train_crypto(symbol):
            success_count += 1
            print(f"✅ {symbol} completed successfully!")
        else:
            print(f"❌ {symbol} failed!")
    
    print(f"\n📈 Results: {success_count}/{len(test_symbols)} successful")
    
    # Test cache loading
    print("\n🔍 Testing cache loading...")
    from cache_loader import CacheLoader
    cache_loader = CacheLoader()
    
    for symbol in test_symbols:
        is_available = cache_loader.is_cache_available(symbol)
        print(f"  {symbol}: Cache available = {is_available}")
        
        if is_available:
            forecasts = cache_loader.get_all_cached_forecasts(symbol)
            if forecasts and 'forecasts' in forecasts:
                forecast_data = forecasts['forecasts']
                print(f"    Forecast keys: {list(forecast_data.keys())}")
                if '30_days' in forecast_data:
                    print(f"    30-day forecast: ${forecast_data['30_days']:.2f}")

if __name__ == "__main__":
    main()
