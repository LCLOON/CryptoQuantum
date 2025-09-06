#!/usr/bin/env python3
"""
Simulate exactly what the Streamlit app does when you select BTC
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CRYPTO_SYMBOLS
import cache_loader

def simulate_app_flow():
    print("=== Simulating Streamlit App Flow ===")
    print()
    
    # This is what happens when you select a crypto in the dropdown
    crypto_name = "₿ BTC/USD"  # This is what you'd see in the dropdown
    symbol = CRYPTO_SYMBOLS[crypto_name]  # This gets BTC-USD
    
    print(f"Selected cryptocurrency: {crypto_name}")
    print(f"Symbol mapping: {symbol}")
    print()
    
    # This is what the app does
    loader = cache_loader.CacheLoader()
    cache_available = loader.is_cache_available(symbol)
    
    print(f"Cache available check: {cache_available}")
    print()
    
    if cache_available:
        print("✅ App should show: '⚡ Ultra-Fast Prediction Ready!'")
        
        # Get cached forecasts (this is what the app does)
        cached_forecasts = loader.get_all_cached_forecasts(symbol)
        
        if cached_forecasts and 'forecasts' in cached_forecasts:
            forecasts = cached_forecasts['forecasts']
            
            # Test different prediction timeframes
            for days in [30, 90, 180, 365]:
                time_key = f"{days}_days"
                if time_key in forecasts:
                    predicted_price = forecasts[time_key]
                    print(f"   {days} days: ${predicted_price:,.2f}")
                else:
                    print(f"   {days} days: ❌ Missing")
        else:
            print("❌ No forecasts found in cached data")
    else:
        print("❌ App should show: 'No cached model' message")
    
    print()
    print("=== Testing Other Major Cryptos ===")
    
    test_cryptos = [
        "Ξ ETH/USD",
        "Ð DOGE/USD", 
        "◎ SOL/USD",
        "₳ ADA/USD"
    ]
    
    for crypto_name in test_cryptos:
        if crypto_name in CRYPTO_SYMBOLS:
            symbol = CRYPTO_SYMBOLS[crypto_name]
            is_available = loader.is_cache_available(symbol)
            status = "✅ WORKS" if is_available else "❌ BROKEN"
            print(f"   {crypto_name:<15} -> {status}")

if __name__ == "__main__":
    simulate_app_flow()
