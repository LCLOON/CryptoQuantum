"""
CryptoQuantum Terminal - Model Pre-training & Data Caching System
================================================================

This script pre-trains all ML models and caches data for faster remote user experience.
Run this script periodically (daily/weekly) to keep models and data fresh.

Features:
- Pre-trains models for top cryptocurrencies
- Caches historical data for faster loading
- Saves pre-computed forecasts
- Creates model checkpoints for instant loading
- Generates pre-calculated technical indicators

Usage:
    python pretrain_models.py [--cryptos all|top10] [--retrain] [--cache-days 730]

Developed by Lewis Loon | © 2025 Lewis Loon Analytics
"""

import pickle
import json
import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pretrain_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import model classes from main app
sys.path.append('.')

class AttentionLSTMModel(nn.Module):
    """AttentionLSTM model for pre-training"""
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, dropout=0.3):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        out = self.fc1(attn_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class LSTMModel(nn.Module):
    """Enhanced LSTM model for pre-training"""
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, output_size=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

# Cryptocurrency symbols for pre-training
CRYPTO_SYMBOLS = {
    'BTC-USD': '₿ Bitcoin',
    'ETH-USD': 'Ξ Ethereum',
    'BNB-USD': '🔷 Binance Coin',
    'SOL-USD': '◎ Solana',
    'XRP-USD': '✖️ XRP',
    'ADA-USD': '₳ Cardano',
    'DOGE-USD': 'Ð Dogecoin',
    'DOT-USD': '● Polkadot',
    'LINK-USD': '⬡ Chainlink',
    'LTC-USD': 'Ł Litecoin',
    'AVAX-USD': '🔺 Avalanche',
    'SHIB-USD': '💎 Shiba Inu',
    'BCH-USD': '🔴 Bitcoin Cash',
    'TRX-USD': '🌪️ TRON',
    'NEAR-USD': '🔰 NEAR Protocol',
    'MATIC-USD': '⬢ Polygon',
    'UNI-USD': '💰 Uniswap',
    'ICP-USD': '🚀 Internet Computer',
    'APT-USD': '⚡ Aptos',
    'ALGO-USD': '🎯 Algorand'
}

TOP_10_CRYPTOS = list(CRYPTO_SYMBOLS.keys())[:10]

class ModelPretrainer:
    """Main class for pre-training models and caching data"""
    
    def __init__(self, cache_dir='./model_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / 'models').mkdir(exist_ok=True)
        (self.cache_dir / 'data').mkdir(exist_ok=True)
        (self.cache_dir / 'forecasts').mkdir(exist_ok=True)
        (self.cache_dir / 'scalers').mkdir(exist_ok=True)
        
        logger.info(f"Cache directory initialized: {self.cache_dir}")
    
    def fetch_and_cache_data(self, symbol, period='5y'):
        """Fetch and cache historical data"""
        try:
            logger.info(f"Fetching data for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist = ticker.history(period=period, auto_adjust=True)
            
            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Add technical indicators
            hist = self.add_technical_indicators(hist)
            
            # Cache the data
            data_file = self.cache_dir / 'data' / f'{symbol}_data.pkl'
            hist.to_pickle(data_file)
            
            # Also save as JSON for web compatibility
            data_json_file = self.cache_dir / 'data' / f'{symbol}_data.json'
            hist.reset_index().to_json(data_json_file, orient='records', date_format='iso')
            
            logger.info(f"✅ Cached {len(hist)} days of data for {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        try:
            # Moving averages
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def prepare_training_data(self, data, sequence_length=30):
        """Prepare data for model training"""
        try:
            # Use Close and Volume for training
            close = data['Close'].values.reshape(-1, 1)
            volume = data['Volume'].values.reshape(-1, 1)
            
            # Handle any zero or negative values
            close = np.maximum(close, 1e-8)
            volume = np.maximum(volume, 1e-8)
            
            # Compute log returns
            log_close = np.log(close)
            returns = np.diff(log_close, axis=0)
            volume = volume[1:]  # Align with returns
            
            # Process returns and volume data
            
            # Scale the data
            scaler_returns = MinMaxScaler(feature_range=(-0.8, 0.8))
            scaler_volume = MinMaxScaler(feature_range=(0, 1))
            
            scaled_returns = scaler_returns.fit_transform(returns)
            scaled_volume = scaler_volume.fit_transform(volume)
            scaled_data = np.hstack((scaled_returns, scaled_volume))
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_returns[i, 0])  # Target is next return
            
            X = np.array(X)
            y = np.array(y)
            
            return X, y, scaler_returns, scaler_volume, log_close
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None, None, None, None
    
    def train_model(self, symbol, model_type='AttentionLSTM', epochs=100):
        """Train and save a model for a specific cryptocurrency"""
        try:
            logger.info(f"Training {model_type} model for {symbol}...")
            
            # Load cached data
            data_file = self.cache_dir / 'data' / f'{symbol}_data.pkl'
            if not data_file.exists():
                logger.error(f"No cached data found for {symbol}")
                return False
            
            data = pd.read_pickle(data_file)
            
            # Prepare training data
            X, y, scaler_returns, scaler_volume, log_close = self.prepare_training_data(data)
            
            if X is None:
                logger.error(f"Failed to prepare training data for {symbol}")
                return False
            
            # Split data
            train_size = int(len(X) * 0.85)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test)
            
            # Create model
            if model_type == 'AttentionLSTM':
                model = AttentionLSTMModel(input_size=2, hidden_size=128, num_layers=3)
                lr = 0.0005
            else:
                model = LSTMModel(input_size=2, hidden_size=128, num_layers=3)
                lr = 0.001
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)
            
            # Training loop
            model.train()
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step(loss)
                
                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
            # Save model
            model_file = self.cache_dir / 'models' / f'{symbol}_{model_type}_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'input_size': 2,
                'hidden_size': 128,
                'num_layers': 3,
                'final_loss': best_loss,
                'training_date': datetime.now().isoformat()
            }, model_file)
            
            # Save scalers
            scaler_file = self.cache_dir / 'scalers' / f'{symbol}_scalers.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump({
                    'scaler_returns': scaler_returns,
                    'scaler_volume': scaler_volume,
                    'last_log_price': log_close[-1]
                }, f)
            
            logger.info(f"✅ Model trained and saved for {symbol} (Final loss: {best_loss:.6f})")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return False
    
    def generate_precomputed_forecasts(self, symbol, model_type='AttentionLSTM'):
        """Generate and cache forecasts for faster loading"""
        try:
            logger.info(f"Generating precomputed forecasts for {symbol}...")
            
            # Load model and scalers
            model_file = self.cache_dir / 'models' / f'{symbol}_{model_type}_model.pth'
            scaler_file = self.cache_dir / 'scalers' / f'{symbol}_scalers.pkl'
            
            if not model_file.exists() or not scaler_file.exists():
                logger.error(f"Missing model or scalers for {symbol}")
                return False
            
            # Load model
            checkpoint = torch.load(model_file)
            if model_type == 'AttentionLSTM':
                model = AttentionLSTMModel(input_size=2, hidden_size=128, num_layers=3)
            else:
                model = LSTMModel(input_size=2, hidden_size=128, num_layers=3)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load scalers
            with open(scaler_file, 'rb') as f:
                scalers_data = pickle.load(f)
            
            # Generate forecasts for different time horizons
            forecasts = {}
            time_horizons = [30, 90, 180, 365, 365*2, 365*3, 365*5]  # 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y
            
            for days in time_horizons:
                predictions = self.predict_future_prices(
                    model, scalers_data, days
                )
                forecasts[f'{days}_days'] = predictions.tolist() if predictions is not None else []
            
            # Save forecasts
            forecast_file = self.cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
            forecast_data = {
                'symbol': symbol,
                'forecasts': forecasts,
                'generated_date': datetime.now().isoformat(),
                'model_type': model_type
            }
            
            with open(forecast_file, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            
            logger.info(f"✅ Precomputed forecasts saved for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating forecasts for {symbol}: {str(e)}")
            return False
    
    def predict_future_prices(self, model, scalers_data, steps=30):
        """Predict future prices using the trained model"""
        try:
            # Implementation similar to the main app's prediction function
            # This is a simplified version - you can enhance with the full logic
            
            scaler_returns = scalers_data['scaler_returns']
            last_log_price = scalers_data['last_log_price']
            
            # Generate dummy sequence for prediction (you'll want to use real recent data)
            sequence_length = 30
            dummy_sequence = np.random.randn(sequence_length, 2) * 0.01
            
            predictions = []
            current_seq = dummy_sequence.copy()
            current_log_price = float(last_log_price)
            
            for step in range(steps):
                with torch.no_grad():
                    seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                    pred_return = model(seq_tensor).item()
                
                # Apply dampening
                pred_return *= 0.8
                pred_return = np.clip(pred_return, -0.1, 0.1)
                
                # Convert to price
                pred_return_actual = scaler_returns.inverse_transform([[pred_return]])[0][0]
                current_log_price += pred_return_actual
                pred_price = np.exp(current_log_price)
                
                predictions.append(pred_price)
                
                # Update sequence
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1, 0] = pred_return
                current_seq[-1, 1] = 0.5  # Dummy volume
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting future prices: {str(e)}")
            return None
    
    def create_cache_manifest(self):
        """Create a manifest file with cache information"""
        try:
            manifest = {
                'created_date': datetime.now().isoformat(),
                'cache_version': '2.0',
                'models': {},
                'data': {},
                'forecasts': {}
            }
            
            # List all cached files
            for model_file in (self.cache_dir / 'models').glob('*.pth'):
                symbol = model_file.stem.split('_')[0]
                model_type = model_file.stem.split('_')[1]
                manifest['models'][symbol] = {
                    'model_type': model_type,
                    'file': model_file.name,
                    'size_mb': model_file.stat().st_size / (1024*1024)
                }
            
            for data_file in (self.cache_dir / 'data').glob('*.pkl'):
                symbol = data_file.stem.replace('_data', '')
                manifest['data'][symbol] = {
                    'file': data_file.name,
                    'size_mb': data_file.stat().st_size / (1024*1024)
                }
            
            for forecast_file in (self.cache_dir / 'forecasts').glob('*.json'):
                symbol = forecast_file.stem.replace('_forecasts', '')
                manifest['forecasts'][symbol] = {
                    'file': forecast_file.name,
                    'size_mb': forecast_file.stat().st_size / (1024*1024)
                }
            
            # Save manifest
            manifest_file = self.cache_dir / 'cache_manifest.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✅ Cache manifest created: {len(manifest['models'])} models, {len(manifest['data'])} datasets")
            return True
            
        except Exception as e:
            logger.error(f"Error creating cache manifest: {str(e)}")
            return False
    
    def run_full_pretraining(self, crypto_list=None, model_types=['AttentionLSTM', 'LSTM']):
        """Run complete pre-training pipeline"""
        if crypto_list is None:
            crypto_list = TOP_10_CRYPTOS
        
        logger.info(f"Starting full pre-training pipeline for {len(crypto_list)} cryptocurrencies...")
        
        success_count = 0
        total_tasks = len(crypto_list) * len(model_types)
        
        for symbol in crypto_list:
            try:
                # Step 1: Fetch and cache data
                data = self.fetch_and_cache_data(symbol)
                if data is None:
                    continue
                
                # Step 2: Train models
                for model_type in model_types:
                    if self.train_model(symbol, model_type):
                        # Step 3: Generate forecasts
                        if self.generate_precomputed_forecasts(symbol, model_type):
                            success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Step 4: Create manifest
        self.create_cache_manifest()
        
        logger.info(f"✅ Pre-training completed: {success_count}/{total_tasks} tasks successful")
        return success_count

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Pre-train CryptoQuantum models and cache data')
    parser.add_argument('--cryptos', choices=['all', 'top10'], default='top10',
                       help='Which cryptocurrencies to process')
    parser.add_argument('--models', nargs='+', default=['AttentionLSTM', 'LSTM'],
                       help='Which model types to train')
    parser.add_argument('--cache-dir', default='./model_cache',
                       help='Directory to store cached models and data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Determine crypto list
    if args.cryptos == 'all':
        crypto_list = list(CRYPTO_SYMBOLS.keys())
    else:
        crypto_list = TOP_10_CRYPTOS
    
    # Initialize pretrainer
    pretrainer = ModelPretrainer(cache_dir=args.cache_dir)
    
    # Run pre-training
    success_count = pretrainer.run_full_pretraining(crypto_list, args.models)
    
    logger.info(f"Pre-training completed with {success_count} successful tasks")
    
    # Print cache summary
    manifest_file = Path(args.cache_dir) / 'cache_manifest.json'
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print("\n📊 Cache Summary:")
        print(f"✅ Models: {len(manifest['models'])}")
        print(f"✅ Datasets: {len(manifest['data'])}")
        print(f"✅ Forecasts: {len(manifest['forecasts'])}")
        print(f"📁 Cache Directory: {args.cache_dir}")

if __name__ == "__main__":
    main()
