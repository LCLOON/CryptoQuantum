"""
CryptoQuantum - Complete Cache Generation
========================================

Trains and caches models for ALL 50 supported cryptocurrencies.
Creates comprehensive cache for ultra-fast predictions.

Features:
- Trains LSTM models for all 50 cryptos
- Generates forecasts for multiple timeframes
- Creates scalers and data caches
- Updates manifest with complete coverage

Usage:
    python train_all_cryptos.py

Developed by Lewis Loon | © 2025 Lewis Loon Analytics
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import logging

# Set deterministic training for consistent results
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CRYPTO_SYMBOLS
from ai_models import LSTMModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteCacheGenerator:
    """Generates complete cache for all supported cryptocurrencies"""
    
    def __init__(self, cache_dir='./model_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / 'models').mkdir(exist_ok=True)
        (self.cache_dir / 'data').mkdir(exist_ok=True)
        (self.cache_dir / 'forecasts').mkdir(exist_ok=True)
        (self.cache_dir / 'scalers').mkdir(exist_ok=True)
        
        self.manifest = {
            'created_date': datetime.now().isoformat(),
            'cache_version': '3.0',
            'models': {},
            'data': {},
            'forecasts': {},
            'scalers': {}
        }
        
        # Timeframes for forecasting
        self.timeframes = {
            '30_days': 30,
            '90_days': 90,
            '180_days': 180,
            '365_days': 365,
            '730_days': 730
        }
        
    def prepare_data(self, symbol, sequence_length=60):
        """Prepare training data for LSTM model"""
        try:
            logger.info(f"📊 Preparing data for {symbol}")
            
            # Get historical data (2 years for better training)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"❌ No data available for {symbol}")
                return None, None, None
            
            # Use closing prices
            prices = data['Close'].values.reshape(-1, 1)
            
            if len(prices) < sequence_length + 30:  # Need minimum data
                logger.warning(f"❌ Insufficient data for {symbol} ({len(prices)} days)")
                return None, None, None
            
            # Create and fit scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            logger.info(f"✅ Data prepared for {symbol}: {X.shape[0]} sequences")
            return X, y, scaler
            
        except Exception as e:
            logger.error(f"❌ Error preparing data for {symbol}: {e}")
            return None, None, None
    
    def train_model(self, symbol, X, y):
        """Train LSTM model for cryptocurrency"""
        try:
            logger.info(f"🧠 Training model for {symbol}")
            
            # Create model
            model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            # Training parameters
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            epochs = 50
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            
            logger.info(f"✅ Model trained for {symbol}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training model for {symbol}: {e}")
            return None
    
    def generate_forecasts(self, model, scaler, symbol, last_sequence):
        """Generate forecasts for all timeframes"""
        try:
            logger.info(f"🔮 Generating forecasts for {symbol}")
            
            forecasts = {}
            model.eval()
            
            with torch.no_grad():
                # Fix: Use only the last value of the sequence for prediction
                current_value = last_sequence[-1] if len(last_sequence) > 0 else 0.0
                
                for timeframe, days in self.timeframes.items():
                    predictions = []
                    current_pred = current_value
                    
                    for _ in range(days):
                        # Prepare input - use single value with correct shape
                        input_tensor = torch.FloatTensor([[current_pred]]).unsqueeze(0)  # Shape: [1, 1, 1]
                        
                        # Make prediction
                        prediction = model(input_tensor).item()
                        predictions.append(prediction)
                        
                        # Update for next prediction
                        current_pred = prediction
                    
                    # Transform back to original scale
                    final_prediction = scaler.inverse_transform([[predictions[-1]]])[0][0]
                    forecasts[timeframe] = float(final_prediction)
            
            logger.info(f"✅ Forecasts generated for {symbol}")
            return forecasts
            
        except Exception as e:
            logger.error(f"❌ Error generating forecasts for {symbol}: {e}")
            # Return empty dict but don't crash the whole process
            return {}
    
    def cache_crypto(self, display_name, symbol):
        """Cache complete data for one cryptocurrency"""
        try:
            logger.info(f"🚀 Caching {display_name} ({symbol})")
            
            # 1. Prepare data
            X, y, scaler = self.prepare_data(symbol)
            if X is None:
                logger.warning(f"⚠️ Skipping {symbol} - no data available")
                return False
            
            # 2. Train model
            model = self.train_model(symbol, X, y)
            if model is None:
                logger.warning(f"⚠️ Skipping {symbol} - training failed")
                return False
            
            # 3. Save model
            model_file = f"{symbol}_LSTM_model.pth"
            model_path = self.cache_dir / 'models' / model_file
            torch.save(model.state_dict(), model_path)
            
            # 4. Save scaler
            scaler_file = f"{symbol}_scaler.pkl"
            scaler_path = self.cache_dir / 'scalers' / scaler_file
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 5. Save training data  
            data_file = f"{symbol}_data.pkl"
            data_path = self.cache_dir / 'data' / data_file
            with open(data_path, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)
            
            # 5b. Save historical data for app usage
            self.save_historical_data(symbol)
            
            # 6. Generate and save forecasts
            last_sequence = X[-1].flatten()
            forecasts = self.generate_forecasts(model, scaler, symbol, last_sequence)
            
            forecast_file = f"{symbol}_forecasts.json"
            forecast_path = self.cache_dir / 'forecasts' / forecast_file
            
            forecast_data = {
                'symbol': symbol,
                'generated_date': datetime.now().isoformat(),
                'forecasts': forecasts
            }
            
            with open(forecast_path, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            
            # 7. Update manifest
            self.manifest['models'][symbol] = {
                'model_type': 'LSTM',
                'file': model_file,
                'size_mb': model_path.stat().st_size / (1024 * 1024)
            }
            
            self.manifest['scalers'][symbol] = {
                'file': scaler_file,
                'size_mb': scaler_path.stat().st_size / (1024 * 1024)
            }
            
            self.manifest['data'][symbol] = {
                'file': data_file,
                'size_mb': data_path.stat().st_size / (1024 * 1024)
            }
            
            self.manifest['forecasts'][symbol] = {
                'file': forecast_file,
                'size_mb': forecast_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"✅ Successfully cached {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error caching {symbol}: {e}")
            return False
    
    def save_historical_data(self, symbol):
        """Save historical data in the format expected by the app"""
        try:
            logger.info(f"💾 Saving historical data for {symbol}")
            
            # Download fresh historical data with technical indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"❌ No historical data for {symbol}")
                return False
            
            # Add technical indicators
            hist = self.add_technical_indicators(hist)
            
            # Save as pickle (DataFrame format that app expects)
            data_file = self.cache_dir / 'data' / f'{symbol}_historical.pkl'
            hist.to_pickle(data_file)
            
            # Also overwrite the main data file with historical data for app compatibility
            main_data_file = self.cache_dir / 'data' / f'{symbol}_data.pkl'
            hist.to_pickle(main_data_file)
            
            # Save as JSON for web compatibility
            data_json_file = self.cache_dir / 'data' / f'{symbol}_data.json'
            hist_json = hist.reset_index()
            hist_json['Date'] = hist_json['Date'].dt.strftime('%Y-%m-%d')
            hist_json.to_json(data_json_file, orient='records')
            
            logger.info(f"✅ Saved {len(hist)} days of historical data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving historical data for {symbol}: {e}")
            return False
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        try:
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI (simplified)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicator
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def generate_complete_cache(self):
        """Generate complete cache for all cryptocurrencies"""
        logger.info("🚀 Starting complete cache generation for all 50 cryptocurrencies")
        
        successful = 0
        failed = 0
        
        # Process all cryptocurrencies
        crypto_list = list(CRYPTO_SYMBOLS.items())
        
        with tqdm(crypto_list, desc="Caching Cryptocurrencies") as pbar:
            for display_name, symbol in pbar:
                pbar.set_description(f"Caching {symbol}")
                
                if self.cache_crypto(display_name, symbol):
                    successful += 1
                else:
                    failed += 1
                
                pbar.set_postfix(success=successful, failed=failed)
        
        # Save final manifest
        manifest_path = self.cache_dir / 'cache_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        # Summary
        total_models = len(self.manifest['models'])
        total_forecasts = len(self.manifest['forecasts'])
        
        logger.info("🎉 Cache generation complete!")
        logger.info(f"📊 Total cryptocurrencies: {len(CRYPTO_SYMBOLS)}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"🧠 Models cached: {total_models}")
        logger.info(f"🔮 Forecasts generated: {total_forecasts}")
        
        if total_models == len(CRYPTO_SYMBOLS):
            logger.info("🎯 100% SUCCESS - All cryptocurrencies cached!")
        else:
            logger.warning(f"⚠️ Only {(total_models/len(CRYPTO_SYMBOLS)*100):.1f}% cached")

def main():
    """Main execution function"""
    print("🚀 CryptoQuantum Complete Cache Generator")
    print("=" * 50)
    print(f"📊 Target: {len(CRYPTO_SYMBOLS)} cryptocurrencies")
    print("🎯 Goal: 100% cache coverage")
    print()
    
    # Create generator
    generator = CompleteCacheGenerator()
    
    # Generate complete cache
    generator.generate_complete_cache()

if __name__ == "__main__":
    main()
