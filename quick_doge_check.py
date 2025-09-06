"""
Quick DOGE Test - Check cached data structure
"""

import sys
sys.path.append(r"c:\Users\lcloo\OneDrive\Desktop\CryptoQuantum-Clone")

from cache_loader import CacheLoader
from market_data import get_crypto_info

def check_doge_cache():
    """Check DOGE cached data structure"""
    
    print("🔍 CHECKING DOGE CACHE STRUCTURE")
    print("=" * 40)
    
    # Get current DOGE data
    crypto_info = get_crypto_info("DOGE-USD")
    current_price = crypto_info['current_price']
    print(f"📊 Current DOGE Price: ${current_price:.8f}")
    
    # Load cached predictions
    cache_loader = CacheLoader()
    cached_forecasts = cache_loader.get_all_cached_forecasts("DOGE-USD")
    
    if cached_forecasts:
        print(f"\n📦 Cached forecasts keys: {list(cached_forecasts.keys())}")
        
        if 'forecasts' in cached_forecasts:
            forecast_data = cached_forecasts['forecasts']
            print(f"📊 Forecast data type: {type(forecast_data)}")
            if isinstance(forecast_data, dict):
                print(f"📊 Forecast keys: {list(forecast_data.keys())}")
                
                # Find the predictions
                if 'predictions' in forecast_data:
                    predictions = forecast_data['predictions']
                    print(f"📊 Predictions length: {len(predictions)}")
                    print(f"📊 First 5 predictions: {predictions[:5]}")
                    
                    # Test with first 5 predictions
                    test_periods = [30, 90, 365]
                    for days in test_periods:
                        if days <= len(predictions):
                            raw_price = predictions[days-1]
                            growth = ((raw_price / current_price) - 1) * 100
                            annual_growth = (((raw_price / current_price) ** (365.25 / days)) - 1) * 100
                            
                            print(f"\n📅 {days} days: ${raw_price:.8f} ({growth:+.1f}% total, {annual_growth:+.1f}% annual)")
                            
                            # Check if this is extreme
                            if annual_growth > 500:
                                print(f"   ⚠️  EXTREME: {annual_growth:.1f}% annual growth!")
                            elif annual_growth > 100:
                                print(f"   🔶 HIGH: {annual_growth:.1f}% annual growth")
                            else:
                                print(f"   ✅ REASONABLE: {annual_growth:.1f}% annual growth")
                else:
                    print(f"📊 No 'predictions' key found in forecast data")
            elif isinstance(forecast_data, list):
                print(f"📊 Forecast length: {len(forecast_data)}")
                print(f"📊 First 5 predictions: {forecast_data[:5]}")
                
                # Test with first 5 predictions
                test_periods = [30, 90, 365]
                for days in test_periods:
                    if days <= len(forecast_data):
                        raw_price = forecast_data[days-1]
                        growth = ((raw_price / current_price) - 1) * 100
                        annual_growth = (((raw_price / current_price) ** (365.25 / days)) - 1) * 100
                        
                        print(f"\n📅 {days} days: ${raw_price:.8f} ({growth:+.1f}% total, {annual_growth:+.1f}% annual)")
                        
                        # Check if this is extreme
                        if annual_growth > 500:
                            print(f"   ⚠️  EXTREME: {annual_growth:.1f}% annual growth!")
                        elif annual_growth > 100:
                            print(f"   🔶 HIGH: {annual_growth:.1f}% annual growth")
                        else:
                            print(f"   ✅ REASONABLE: {annual_growth:.1f}% annual growth")
    else:
        print("❌ No cached forecasts found")

if __name__ == "__main__":
    check_doge_cache()
