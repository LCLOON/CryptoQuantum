"""
Quick check of cached prediction values across different cryptocurrencies
"""
import json
import os

def check_cached_predictions():
    """Check a sample of cached predictions to identify any extreme values"""
    forecasts_dir = "model_cache/forecasts"
    
    # Test a few different cryptocurrencies
    test_symbols = ["BTC-USD", "ETH-USD", "DOGE-USD", "SHIB-USD", "ADA-USD"]
    
    print("🔍 Checking Cached Predictions for Extreme Values")
    print("=" * 60)
    
    for symbol in test_symbols:
        try:
            file_path = os.path.join(forecasts_dir, f"{symbol}_forecasts.json")
            if not os.path.exists(file_path):
                print(f"❌ {symbol}: File not found")
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check 30-day predictions
            predictions_30 = data['forecasts'].get('30_days', [])
            predictions_365 = data['forecasts'].get('365_days', [])
            
            if predictions_30:
                start_price = predictions_30[0]
                end_price_30 = predictions_30[-1] if len(predictions_30) >= 30 else predictions_30[-1]
                growth_30 = ((end_price_30 / start_price - 1) * 100) if start_price > 0 else 0
                
                print(f"📊 {symbol}:")
                print(f"   Start Price: ${start_price:.6f}")
                print(f"   30-Day Price: ${end_price_30:.6f}")
                print(f"   30-Day Growth: {growth_30:.1f}%")
                
                if predictions_365:
                    end_price_365 = predictions_365[364] if len(predictions_365) > 364 else predictions_365[-1]
                    growth_365 = ((end_price_365 / start_price - 1) * 100) if start_price > 0 else 0
                    annual_rate = ((end_price_365 / start_price) ** (365.25/365) - 1) * 100
                    
                    print(f"   1-Year Price: ${end_price_365:.6f}")
                    print(f"   1-Year Growth: {growth_365:.1f}%")
                    print(f"   Annual Rate: {annual_rate:.1f}%")
                    
                    # Flag extreme values
                    if annual_rate > 1000:  # More than 1000% annual growth
                        print(f"   🚨 EXTREME: {annual_rate:.1f}% annual growth detected!")
                    elif annual_rate > 500:  # More than 500% annual growth
                        print(f"   ⚠️  HIGH: {annual_rate:.1f}% annual growth")
                    elif annual_rate < -90:  # More than 90% decline
                        print(f"   ⚠️  DECLINE: {annual_rate:.1f}% annual decline")
                    else:
                        print(f"   ✅ REASONABLE: {annual_rate:.1f}% annual growth")
                
                print()
                
        except Exception as e:
            print(f"❌ {symbol}: Error reading data - {str(e)}")
            print()

if __name__ == "__main__":
    check_cached_predictions()
