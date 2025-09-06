"""
CryptoQuantum Terminal - Fast Cache Loader
==========================================

This module provides fast loading capabilities for pre-trained models and cached data.
Significantly improves performance for remote users by loading pre-computed results.

Features:
- Instant model loading from cache
- Pre-computed forecast retrieval
- Cached data access
- Fallback to live training if cache unavailable

Usage:
    from cache_loader import CacheLoader
    loader = CacheLoader()
    forecasts = loader.get_cached_forecasts('BTC-USD')

Developed by Lewis Loon | © 2025 Lewis Loon Analytics
"""

import json
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLoader:
    """Fast cache loader for pre-trained models and data"""
    
    def __init__(self, cache_dir='./model_cache'):
        self.cache_dir = Path(cache_dir)
        self.manifest = self.load_manifest()
        
        if self.manifest:
            logger.info(f"✅ Cache loaded: {len(self.manifest.get('models', {}))} models available")
        else:
            logger.warning("⚠️ No cache manifest found - falling back to live training")
    
    def load_manifest(self):
        """Load cache manifest"""
        try:
            manifest_file = self.cache_dir / 'cache_manifest.json'
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading manifest: {str(e)}")
            return None
    
    def is_cache_available(self, symbol, model_type='AttentionLSTM'):
        """Check if cached forecasts are available (prioritizes forecasts over models)"""
        # Check for actual files rather than relying on potentially outdated manifest
        forecast_file = self.cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
        data_file = self.cache_dir / 'data' / f'{symbol}_data.pkl'
        data_json_file = self.cache_dir / 'data' / f'{symbol}_data.json'
        
        has_forecasts = forecast_file.exists()
        has_data = data_file.exists() or data_json_file.exists()
        
        # For ultra-fast mode, we only need forecasts and data
        return has_forecasts and has_data
    
    def get_cached_data(self, symbol):
        """Load cached historical data"""
        try:
            data_file = self.cache_dir / 'data' / f'{symbol}_data.pkl'
            if data_file.exists():
                data = pd.read_pickle(data_file)
                logger.info(f"✅ Loaded cached data for {symbol}: {len(data)} records")
                return data
            
            # Try JSON fallback
            json_file = self.cache_dir / 'data' / f'{symbol}_data.json'
            if json_file.exists():
                data = pd.read_json(json_file)
                data.set_index('Date', inplace=True)
                logger.info(f"✅ Loaded cached JSON data for {symbol}: {len(data)} records")
                return data
                
            return None
            
        except Exception as e:
            logger.error(f"Error loading cached data for {symbol}: {str(e)}")
            return None
    
    def load_cached_model(self, symbol, model_type='AttentionLSTM'):
        """Load pre-trained model from cache"""
        try:
            # Import model classes
            from pretrain_models import AttentionLSTMModel, LSTMModel
            
            model_file = self.cache_dir / 'models' / f'{symbol}_{model_type}_model.pth'
            if not model_file.exists():
                logger.warning(f"No cached model found for {symbol} ({model_type})")
                return None
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Create model
            if model_type == 'AttentionLSTM':
                model = AttentionLSTMModel(
                    input_size=checkpoint.get('input_size', 2),
                    hidden_size=checkpoint.get('hidden_size', 128),
                    num_layers=checkpoint.get('num_layers', 3)
                )
            else:
                model = LSTMModel(
                    input_size=checkpoint.get('input_size', 2),
                    hidden_size=checkpoint.get('hidden_size', 128),
                    num_layers=checkpoint.get('num_layers', 3)
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"✅ Loaded cached {model_type} model for {symbol}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading cached model for {symbol}: {str(e)}")
            return None
    
    def load_cached_scalers(self, symbol):
        """Load cached scalers for data preprocessing"""
        try:
            scaler_file = self.cache_dir / 'scalers' / f'{symbol}_scalers.pkl'
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    scalers = pickle.load(f)
                logger.info(f"✅ Loaded cached scalers for {symbol}")
                return scalers
            return None
            
        except Exception as e:
            logger.error(f"Error loading cached scalers for {symbol}: {str(e)}")
            return None
    
    def get_cached_forecasts(self, symbol, time_horizon_days=365):
        """Get pre-computed forecasts"""
        try:
            forecast_file = self.cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
            if not forecast_file.exists():
                logger.warning(f"No cached forecasts found for {symbol}")
                return None
            
            with open(forecast_file, 'r') as f:
                forecast_data = json.load(f)
            
            forecasts = forecast_data.get('forecasts', {})
            
            # Find closest time horizon
            available_horizons = [int(k.replace('_days', '')) for k in forecasts.keys()]
            closest_horizon = min(available_horizons, key=lambda x: abs(x - time_horizon_days))
            
            forecast_key = f'{closest_horizon}_days'
            predictions = forecasts.get(forecast_key, [])
            
            if predictions:
                logger.info(f"✅ Loaded cached forecasts for {symbol}: {len(predictions)} predictions")
                return {
                    'predictions': predictions,
                    'time_horizon_days': closest_horizon,
                    'generated_date': forecast_data.get('generated_date'),
                    'model_type': forecast_data.get('model_type', 'AttentionLSTM')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading cached forecasts for {symbol}: {str(e)}")
            return None
    
    def get_all_cached_forecasts(self, symbol):
        """Get all available forecast horizons for a symbol"""
        try:
            forecast_file = self.cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
            if not forecast_file.exists():
                return None
            
            with open(forecast_file, 'r') as f:
                forecast_data = json.load(f)
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error loading all forecasts for {symbol}: {str(e)}")
            return None
    
    def is_cache_fresh(self, symbol, max_age_hours=24):
        """Check if cached data is fresh enough"""
        try:
            forecast_file = self.cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
            if not forecast_file.exists():
                return False
            
            with open(forecast_file, 'r') as f:
                forecast_data = json.load(f)
            
            generated_date = datetime.fromisoformat(forecast_data.get('generated_date', ''))
            age_hours = (datetime.now() - generated_date).total_seconds() / 3600
            
            return age_hours <= max_age_hours
            
        except Exception as e:
            logger.error(f"Error checking cache freshness for {symbol}: {str(e)}")
            return False
    
    def get_cache_stats(self):
        """Get cache statistics based on actual files"""
        try:
            # Count actual files
            available_symbols = self.get_available_symbols()
            
            # Calculate total cache size
            total_size_mb = 0
            
            # Check forecasts size
            forecasts_dir = self.cache_dir / 'forecasts'
            if forecasts_dir.exists():
                for file in forecasts_dir.glob('*.json'):
                    total_size_mb += file.stat().st_size / (1024 * 1024)
            
            # Check data size
            data_dir = self.cache_dir / 'data'
            if data_dir.exists():
                for file in data_dir.glob('*'):
                    total_size_mb += file.stat().st_size / (1024 * 1024)
            
            # Check models size
            models_dir = self.cache_dir / 'models'
            if models_dir.exists():
                for file in models_dir.glob('*.pth'):
                    total_size_mb += file.stat().st_size / (1024 * 1024)
            
            stats = {
                'models_count': len(available_symbols),
                'data_count': len(available_symbols),
                'forecasts_count': len(available_symbols),
                'cache_version': 'Ultra-Fast v2.1',
                'created_date': '2025-07-16',
                'total_size_mb': round(total_size_mb, 1)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return None
    
    def get_available_symbols(self):
        """Get list of symbols with cached forecasts (based on actual files)"""
        try:
            forecasts_dir = self.cache_dir / 'forecasts'
            if not forecasts_dir.exists():
                return []
            
            # Extract symbols from forecast filenames
            symbols = []
            for forecast_file in forecasts_dir.glob('*_forecasts.json'):
                symbol = forecast_file.stem.replace('_forecasts', '')
                # Verify both forecast and data exist
                if self.is_cache_available(symbol):
                    symbols.append(symbol)
            
            logger.info(f"✅ Found {len(symbols)} cached symbols with ultra-fast predictions")
            return sorted(symbols)
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
    
    def fast_predict(self, symbol, time_horizon_days=365, model_type='AttentionLSTM'):
        """Fast prediction using cached models and data"""
        try:
            # Try to get cached forecasts first
            cached_forecasts = self.get_cached_forecasts(symbol, time_horizon_days)
            if cached_forecasts and self.is_cache_fresh(symbol):
                logger.info(f"🚀 Using cached forecasts for {symbol}")
                return cached_forecasts['predictions']
            
            # If no fresh cache, try to use cached model for live prediction
            model = self.load_cached_model(symbol, model_type)
            scalers = self.load_cached_scalers(symbol)
            data = self.get_cached_data(symbol)
            
            if model and scalers and data is not None:
                logger.info(f"🔄 Using cached model for live prediction: {symbol}")
                # Here you would implement the prediction logic
                # This is a placeholder - you'd integrate with your actual prediction function
                return self._predict_with_cached_model(model, scalers, data, time_horizon_days)
            
            logger.warning(f"⚠️ Cache miss for {symbol} - falling back to live training")
            return None
            
        except Exception as e:
            logger.error(f"Error in fast prediction for {symbol}: {str(e)}")
            return None
    
    def _predict_with_cached_model(self, model, scalers, data, steps):
        """Use cached model to make predictions"""
        try:
            # This is a simplified implementation
            # You would integrate this with the actual prediction logic from your main app
            
            # Prepare recent data for prediction
            close = data['Close'].values[-60:]  # Use last 60 days
            volume = data['Volume'].values[-60:]
            
            # Apply same preprocessing as in training
            log_returns = np.diff(np.log(close))
            volume = volume[1:]  # Align with returns
            
            # Scale data
            scaler_returns = scalers['scaler_returns']
            scaler_volume = scalers['scaler_volume']
            
            scaled_returns = scaler_returns.transform(log_returns.reshape(-1, 1))
            scaled_volume = scaler_volume.transform(volume.reshape(-1, 1))
            
            # Create sequence for prediction
            sequence_length = 30
            if len(scaled_returns) >= sequence_length:
                recent_sequence = np.hstack((
                    scaled_returns[-sequence_length:],
                    scaled_volume[-sequence_length:]
                ))
                
                predictions = []
                current_seq = recent_sequence.copy()
                current_log_price = np.log(close[-1])
                
                for step in range(steps):
                    with torch.no_grad():
                        seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                        pred_return = model(seq_tensor).item()
                    
                    # Apply dampening and clipping
                    pred_return *= 0.8
                    pred_return = np.clip(pred_return, -0.1, 0.1)
                    
                    # Convert to price
                    pred_return_actual = scaler_returns.inverse_transform([[pred_return]])[0][0]
                    current_log_price += pred_return_actual
                    pred_price = np.exp(current_log_price)
                    
                    predictions.append(float(pred_price))
                    
                    # Update sequence
                    current_seq = np.roll(current_seq, -1, axis=0)
                    current_seq[-1, 0] = pred_return
                    current_seq[-1, 1] = 0.5  # Dummy volume
                
                return predictions
            
            return None
            
        except Exception as e:
            logger.error(f"Error in cached model prediction: {str(e)}")
            return None

# Convenience functions for easy integration
def quick_forecast(symbol, time_horizon_days=365, cache_dir='./model_cache'):
    """Quick forecast function using cache"""
    loader = CacheLoader(cache_dir)
    return loader.fast_predict(symbol, time_horizon_days)

def get_cached_data_quick(symbol, cache_dir='./model_cache'):
    """Quick data access function"""
    loader = CacheLoader(cache_dir)
    return loader.get_cached_data(symbol)

def is_symbol_cached(symbol, cache_dir='./model_cache'):
    """Quick check if symbol is cached"""
    loader = CacheLoader(cache_dir)
    return loader.is_cache_available(symbol)

if __name__ == "__main__":
    # Test the cache loader
    loader = CacheLoader()
    stats = loader.get_cache_stats()
    
    if stats:
        print("📊 Cache Statistics:")
        print(f"   Models: {stats['models_count']}")
        print(f"   Datasets: {stats['data_count']}")
        print(f"   Forecasts: {stats['forecasts_count']}")
        print(f"   Total Size: {stats['total_size_mb']:.1f} MB")
        print(f"   Cache Version: {stats['cache_version']}")
    else:
        print("❌ No cache available")
