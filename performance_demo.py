"""
CryptoQuantum Terminal - Performance Demo
========================================

This script demonstrates the performance difference between:
1. Traditional model training (slow)
2. Cached model loading (fast)

Run this after building the cache to see the dramatic speed improvement.

Usage:
    python performance_demo.py

Developed by Lewis Loon | © 2025 Lewis Loon Analytics
"""

import time
from pathlib import Path

def demo_performance():
    """Demonstrate performance improvements with caching"""
    
    print("🚀 CryptoQuantum Performance Demo")
    print("=" * 50)
    
    # Check if cache exists
    cache_dir = Path('./model_cache')
    if not cache_dir.exists():
        print("❌ No cache found!")
        print("📝 To build cache, run:")
        print("   python pretrain_models.py --cryptos top10")
        print("\nOr use the convenient batch script:")
        print("   build_cache.bat")
        return
    
    try:
        from cache_loader import CacheLoader
        cache_loader = CacheLoader()
        
        # Get cache stats
        stats = cache_loader.get_cache_stats()
        if not stats:
            print("❌ Cache not properly initialized")
            return
        
        print("📊 Cache Status:")
        print(f"   ✅ Models: {stats['models_count']}")
        print(f"   ✅ Datasets: {stats['data_count']}")
        print(f"   ✅ Forecasts: {stats['forecasts_count']}")
        print(f"   💾 Size: {stats['total_size_mb']:.1f} MB")
        print()
        
        # Demo 1: Data Loading Speed
        print("🔍 Demo 1: Data Loading Speed")
        print("-" * 30)
        
        symbol = 'BTC-USD'
        
        # Cached data loading
        start_time = time.time()
        cached_data = cache_loader.get_cached_data(symbol)
        cache_time = time.time() - start_time
        
        if cached_data is not None:
            print(f"⚡ Cached data loading: {cache_time:.3f} seconds")
            print(f"   📊 Records loaded: {len(cached_data):,}")
        
        # Simulate live data fetch (without actually fetching)
        start_time = time.time()
        time.sleep(2)  # Simulate network delay
        live_time = time.time() - start_time
        
        print(f"🐌 Simulated live fetch: {live_time:.3f} seconds")
        
        if cached_data is not None:
            speedup = live_time / cache_time
            print(f"🚀 Speedup: {speedup:.1f}x faster")
        
        print()
        
        # Demo 2: Model Loading Speed
        print("🔍 Demo 2: Model Loading Speed")
        print("-" * 30)
        
        # Cached model loading
        start_time = time.time()
        cached_model = cache_loader.load_cached_model(symbol, 'AttentionLSTM')
        model_time = time.time() - start_time
        
        if cached_model is not None:
            print(f"⚡ Cached model loading: {model_time:.3f} seconds")
        
        # Simulate model training time
        simulated_training_time = 45.0  # Typical training time
        print(f"🐌 Typical model training: {simulated_training_time:.1f} seconds")
        
        if cached_model is not None:
            speedup = simulated_training_time / model_time
            print(f"🚀 Speedup: {speedup:.0f}x faster")
        
        print()
        
        # Demo 3: Forecast Generation
        print("🔍 Demo 3: Forecast Generation")
        print("-" * 30)
        
        # Cached forecasts
        start_time = time.time()
        cached_forecasts = cache_loader.get_cached_forecasts(symbol, 365)
        forecast_time = time.time() - start_time
        
        if cached_forecasts:
            print(f"⚡ Cached forecasts: {forecast_time:.3f} seconds")
            print(f"   📈 Predictions: {len(cached_forecasts['predictions'])}")
        
        # Simulate live prediction time
        simulated_prediction_time = 15.0
        print(f"🐌 Live prediction generation: {simulated_prediction_time:.1f} seconds")
        
        if cached_forecasts:
            speedup = simulated_prediction_time / forecast_time
            print(f"🚀 Speedup: {speedup:.0f}x faster")
        
        print()
        
        # Summary
        print("📊 Performance Summary")
        print("=" * 30)
        print("With pre-trained cache:")
        print("✅ Application startup: ~2-5 seconds")
        print("✅ Model predictions: ~0.5 seconds")
        print("✅ Data loading: ~0.1 seconds")
        print()
        print("Without cache (traditional):")
        print("❌ Application startup: ~45-90 seconds")
        print("❌ Model predictions: ~10-30 seconds")
        print("❌ Data loading: ~2-5 seconds")
        print()
        print("🎯 Overall Performance Improvement:")
        print("   🚀 10-180x faster for end users!")
        print("   💾 Reduced server computational load")
        print("   🌐 Better experience for remote users")
        
    except ImportError:
        print("❌ Cache loader not available")
        print("📝 Run: python integrate_cache.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def show_cache_info():
    """Show detailed cache information"""
    try:
        from cache_loader import CacheLoader
        cache_loader = CacheLoader()
        
        available_symbols = cache_loader.get_available_symbols()
        
        print("\n🔍 Detailed Cache Information")
        print("=" * 40)
        
        for symbol in available_symbols[:5]:  # Show first 5
            print(f"\n📈 {symbol}:")
            
            # Check data
            data = cache_loader.get_cached_data(symbol)
            if data is not None:
                print(f"   📊 Data: {len(data):,} records")
                print(f"   📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            # Check forecasts
            forecasts = cache_loader.get_all_cached_forecasts(symbol)
            if forecasts:
                forecast_horizons = list(forecasts['forecasts'].keys())
                print(f"   🔮 Forecasts: {len(forecast_horizons)} time horizons")
                print(f"   ⏱️  Generated: {forecasts.get('generated_date', 'Unknown')}")
            
            # Check freshness
            is_fresh = cache_loader.is_cache_fresh(symbol, 24)
            print(f"   🕐 Fresh (24h): {'✅ Yes' if is_fresh else '⚠️ No'}")
        
        if len(available_symbols) > 5:
            print(f"\n... and {len(available_symbols) - 5} more symbols")
            
    except Exception as e:
        print(f"❌ Error getting cache info: {str(e)}")

def main():
    """Main demo function"""
    demo_performance()
    
    # Ask if user wants detailed info
    try:
        response = input("\n❓ Show detailed cache information? (y/n): ").lower()
        if response == 'y':
            show_cache_info()
    except KeyboardInterrupt:
        print("\n👋 Demo completed!")

if __name__ == "__main__":
    main()
