#!/usr/bin/env python3
"""
Fix forecasts for already trained models
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from ai_models import LSTMModel
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastFixer:
    def __init__(self):
        self.cache_dir = Path("cache")
        self.models_dir = self.cache_dir / 'models'
        self.forecasts_dir = self.cache_dir / 'forecasts'
        
        # Timeframes for forecasting
        self.timeframes = {
            "30_days": 30,
            "90_days": 90,
            "180_days": 180,
            "365_days": 365,
            "730_days": 730
        }
    
    def prepare_data(self, symbol):
        """Prepare data for a cryptocurrency symbol"""
        try:
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
            
            # Get last value for forecasting
            last_value = scaled_data[-1][0]
            
            return last_value, scaler, len(scaled_data)
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return None, None, None
    
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
                        # Prepare input with correct shape [batch_size, seq_len, features]
                        input_tensor = torch.FloatTensor([[[current_pred]]])  # Shape: [1, 1, 1]
                        
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
            return {}
    
    def fix_crypto_forecasts(self, symbol):
        """Fix forecasts for one cryptocurrency"""
        try:
            # Check if model exists
            model_file = f"{symbol}_LSTM_model.pth"
            model_path = self.models_dir / model_file
            
            if not model_path.exists():
                logger.warning(f"No model found for {symbol}")
                return False
            
            # Prepare data
            last_value, scaler, data_len = self.prepare_data(symbol)
            if last_value is None:
                logger.warning(f"No data available for {symbol}")
                return False
            
            # Load model
            model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # Generate forecasts
            forecasts = self.generate_forecasts(model, scaler, symbol, last_value)
            
            if forecasts:
                # Save forecasts
                forecast_data = {
                    'symbol': symbol,
                    'generated_date': '2025-09-06',
                    'forecasts': forecasts
                }
                
                forecast_file = f"{symbol}_forecasts.json"
                forecast_path = self.forecasts_dir / forecast_file
                
                with open(forecast_path, 'w') as f:
                    json.dump(forecast_data, f, indent=2)
                
                logger.info(f"✅ Fixed forecasts for {symbol}")
                return True
            else:
                logger.warning(f"Failed to generate forecasts for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error fixing forecasts for {symbol}: {e}")
            return False
    
    def fix_all_forecasts(self):
        """Fix forecasts for all cached models"""
        logger.info("🚀 Starting forecast fix for all cached models")
        
        success_count = 0
        total_count = 0
        
        # Get all symbols with models
        if self.models_dir.exists():
            for model_file in self.models_dir.glob("*_LSTM_model.pth"):
                symbol = model_file.stem.replace("_LSTM_model", "")
                total_count += 1
                
                logger.info(f"Fixing forecasts for {symbol} ({total_count})")
                
                if self.fix_crypto_forecasts(symbol):
                    success_count += 1
        
        logger.info(f"🎉 Forecast fix complete!")
        logger.info(f"✅ Fixed: {success_count}/{total_count}")
        
        return success_count, total_count

def main():
    fixer = ForecastFixer()
    success, total = fixer.fix_all_forecasts()
    
    print(f"\n📊 Summary:")
    print(f"✅ Successfully fixed: {success}")
    print(f"📈 Total processed: {total}")
    print(f"💯 Success rate: {(success/total*100):.1f}%" if total > 0 else "No models found")

if __name__ == "__main__":
    main()
