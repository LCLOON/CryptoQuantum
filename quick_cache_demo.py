"""
Quick Cache Builder - Fast demo cache for testing
=================================================

This script builds a minimal cache quickly for testing the performance improvements.
Creates basic cached data and forecasts without full model training.

Usage:
    python quick_cache_demo.py

Developed by Lewis Loon | © 2025 Lewis Loon Analytics
"""

import json
import yfinance as yf
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

def create_demo_cache():
    """Create a minimal cache for testing"""
    print("🚀 Building Quick Demo Cache...")
    print("=" * 40)
    
    # Create cache directories
    cache_dir = Path('./model_cache')
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / 'data').mkdir(exist_ok=True)
    (cache_dir / 'forecasts').mkdir(exist_ok=True)
    (cache_dir / 'models').mkdir(exist_ok=True)
    (cache_dir / 'scalers').mkdir(exist_ok=True)
    
    # Top 10 cryptocurrencies for comprehensive cache
    demo_cryptos = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
        'ADA-USD', 'DOGE-USD', 'DOT-USD', 'LINK-USD', 'LTC-USD'
    ]
    
    for symbol in demo_cryptos:
        try:
            print(f"📊 Processing {symbol}...")
            
            # Fetch and cache data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5y', auto_adjust=True)
            
            if hist.empty:
                print(f"⚠️ No data for {symbol}")
                continue
            
            # Add basic technical indicators
            hist['MA_20'] = hist['Close'].rolling(window=20).mean()
            hist['MA_50'] = hist['Close'].rolling(window=50).mean()
            
            # Cache data
            data_file = cache_dir / 'data' / f'{symbol}_data.pkl'
            hist.to_pickle(data_file)
            
            json_file = cache_dir / 'data' / f'{symbol}_data.json'
            hist.reset_index().to_json(json_file, orient='records', date_format='iso')
            
            print(f"   ✅ Cached {len(hist)} days of data")
            
            # Create dummy forecasts
            current_price = hist['Close'].iloc[-1]
            
            # Generate simple projections (not ML-based, just for demo)
            forecasts = {}
            time_horizons = [30, 90, 180, 365, 730, 1095, 1825]  # Various periods
            
            for days in time_horizons:
                # Simple trend-based forecast for demo
                daily_return = hist['Close'].pct_change().mean()
                predictions = []
                
                for day in range(days):
                    # Add some randomness to make it realistic
                    trend_factor = 1 + (daily_return * 0.5)  # Dampened trend
                    noise = np.random.normal(0, 0.02)  # Small random noise
                    
                    if day == 0:
                        pred_price = current_price * trend_factor * (1 + noise)
                    else:
                        pred_price = predictions[-1] * trend_factor * (1 + noise)
                    
                    predictions.append(float(pred_price))
                
                forecasts[f'{days}_days'] = predictions
            
            # Save forecasts
            forecast_file = cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
            forecast_data = {
                'symbol': symbol,
                'forecasts': forecasts,
                'generated_date': datetime.now().isoformat(),
                'model_type': 'Demo'
            }
            
            with open(forecast_file, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            
            print(f"   ✅ Created forecasts for {len(time_horizons)} time horizons")
            
            # Create dummy scalers
            from sklearn.preprocessing import MinMaxScaler
            scaler_returns = MinMaxScaler(feature_range=(-0.8, 0.8))
            scaler_volume = MinMaxScaler(feature_range=(0, 1))
            
            # Fit on some sample data
            returns = hist['Close'].pct_change().dropna().values.reshape(-1, 1)
            volume = hist['Volume'].values.reshape(-1, 1)
            
            scaler_returns.fit(returns)
            scaler_volume.fit(volume)
            
            scaler_file = cache_dir / 'scalers' / f'{symbol}_scalers.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump({
                    'scaler_returns': scaler_returns,
                    'scaler_volume': scaler_volume,
                    'last_log_price': np.log(current_price)
                }, f)
            
            print("   ✅ Created scalers")
            
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {str(e)}")
            continue
    
    # Create manifest
    manifest = {
        'created_date': datetime.now().isoformat(),
        'cache_version': '2.0-demo',
        'models': {},
        'data': {},
        'forecasts': {}
    }
    
    # List cached files
    for data_file in (cache_dir / 'data').glob('*.pkl'):
        symbol = data_file.stem.replace('_data', '')
        manifest['data'][symbol] = {
            'file': data_file.name,
            'size_mb': data_file.stat().st_size / (1024*1024)
        }
    
    for forecast_file in (cache_dir / 'forecasts').glob('*.json'):
        symbol = forecast_file.stem.replace('_forecasts', '')
        manifest['forecasts'][symbol] = {
            'file': forecast_file.name,
            'size_mb': forecast_file.stat().st_size / (1024*1024)
        }
    
    # Models (empty for demo)
    for symbol in demo_cryptos:
        manifest['models'][symbol] = {
            'model_type': 'Demo',
            'file': f'{symbol}_demo_model.pth',
            'size_mb': 0
        }
    
    # Save manifest
    manifest_file = cache_dir / 'cache_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n✅ Demo Cache Built Successfully!")
    print(f"📊 Cached data for {len(demo_cryptos)} cryptocurrencies")
    print(f"📁 Cache location: {cache_dir}")
    print("\n🚀 Now you can test the performance improvements!")
    print("   Run: python performance_demo.py")

if __name__ == "__main__":
    create_demo_cache()
