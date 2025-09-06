#!/usr/bin/env python3
"""
Quick Cache vs Live Training Speed Comparison
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cache_speed_comparison():
    """Compare cache loading vs estimated live training time"""
    
    try:
        from cache_loader import CacheLoader
        cache_loader = CacheLoader()
    except ImportError:
        print("❌ Cache loader not available")
        return
    
    # Test parameters
    test_symbol = 'BTC-USD'
    forecast_days = 1095  # 3 years
    
    print("🚀 SPEED COMPARISON TEST")
    print("=" * 50)
    print(f"📊 Symbol: {test_symbol}")
    print(f"📅 Forecast: {forecast_days} days (3 years)")
    print()
    
    # Test cache loading
    print("⚡ Testing CACHE loading speed...")
    
    if cache_loader.is_cache_available(test_symbol):
        # Time the cache loading
        start_time = time.time()
        cached_forecasts = cache_loader.get_cached_forecasts(test_symbol, forecast_days)
        end_time = time.time()
        
        cache_time_ms = (end_time - start_time) * 1000
        
        if cached_forecasts:
            predictions = cached_forecasts['predictions'][:forecast_days]
            print(f"✅ CACHE: {len(predictions)} predictions loaded in {cache_time_ms:.1f}ms")
            print(f"   📈 Price range: ${predictions[0]:.2f} → ${predictions[-1]:.2f}")
        else:
            print("❌ CACHE: No predictions available")
            return
    else:
        print("❌ CACHE: Not available for this symbol")
        return
    
    # Estimated live training time
    estimated_training_time_ms = 180000  # 3 minutes average
    print(f"🔄 LIVE TRAINING: Estimated ~{estimated_training_time_ms/1000:.0f} seconds")
    
    # Calculate speed improvement
    speed_improvement = estimated_training_time_ms / cache_time_ms
    time_saved_seconds = (estimated_training_time_ms - cache_time_ms) / 1000
    
    print()
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"⚡ Cache Loading:     {cache_time_ms:.1f}ms")
    print(f"🔄 Live Training:     ~{estimated_training_time_ms:,.0f}ms")
    print(f"🚀 Speed Improvement: {speed_improvement:,.0f}x faster")
    print(f"💾 Time Saved:       {time_saved_seconds:.1f} seconds")
    print()
    print("💡 CONCLUSION: Cache system provides near-instant predictions!")
    print("   Perfect for real-time trading analysis and quick decision making.")

if __name__ == "__main__":
    test_cache_speed_comparison()
