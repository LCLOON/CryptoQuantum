#!/usr/bin/env python3
"""
Direct test of forecast loading
"""

import json
from pathlib import Path

def direct_test():
    """Test loading forecasts directly"""
    cache_dir = Path("cache")
    forecast_file = cache_dir / "forecasts" / "BTC-USD_forecasts.json"
    
    print(f"Testing direct file loading...")
    print(f"File exists: {forecast_file.exists()}")
    
    if forecast_file.exists():
        with open(forecast_file, 'r') as f:
            data = json.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Data keys: {list(data.keys())}")
        
        if 'forecasts' in data:
            forecasts = data['forecasts']
            print(f"Forecasts type: {type(forecasts)}")
            print(f"Forecasts keys: {list(forecasts.keys())}")
            print(f"30-day forecast: {forecasts.get('30_days')}")
        else:
            print("No 'forecasts' key found")
    
    # Now test through cache loader
    print("\n" + "="*50)
    print("Testing through cache loader...")
    
    from cache_loader import CacheLoader
    cache_loader = CacheLoader()
    
    forecasts = cache_loader.get_all_cached_forecasts('BTC-USD')
    print(f"Cache loader result: {type(forecasts)}")
    
    if forecasts:
        print(f"Keys: {list(forecasts.keys())}")
        if 'forecasts' in forecasts:
            print(f"Forecast data: {forecasts['forecasts']}")
        else:
            print("No forecasts key in result")

if __name__ == "__main__":
    direct_test()
