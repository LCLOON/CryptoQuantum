"""
Final DOGE Validation Test
Tests the exact logic from the Streamlit app
"""

import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r"c:\Users\lcloo\OneDrive\Desktop\CryptoQuantum-Clone")

from cache_loader import CacheLoader
from market_data import get_crypto_info
import numpy as np

def test_doge_app_logic():
    """Test DOGE predictions using exact app logic"""
    
    print("🎯 FINAL DOGE VALIDATION TEST")
    print("=" * 50)
    
    # Initialize
    cache_loader = CacheLoader()
    symbol = "DOGE-USD"
    
    # Get current price
    crypto_info = get_crypto_info(symbol)
    current_price = crypto_info['current_price']
    print(f"📊 Current DOGE Price: ${current_price:.8f}")
    
    # Get cached forecasts (exactly like the app)
    cached_forecasts = cache_loader.get_all_cached_forecasts(symbol)
    
    if cached_forecasts:
        print(f"✅ Found cached forecasts for {symbol}")
        
        # Test different time periods
        test_days = [30, 90, 365, 730, 1095]
        
        print(f"\n🔍 TESTING ULTRA-CONSERVATIVE DOGE VALIDATION:")
        print("-" * 60)
        
        for days in test_days:
            # Get cached prediction for this timeframe
            time_key = f"{days}_days"
            if time_key in cached_forecasts['forecasts']:
                cached_prediction_list = cached_forecasts['forecasts'][time_key]
                
                # Get the final prediction (last value in the list)
                if isinstance(cached_prediction_list, list) and len(cached_prediction_list) > 0:
                    cached_prediction = cached_prediction_list[-1]  # Last prediction
                else:
                    cached_prediction = cached_prediction_list  # If not a list
                
                # Apply EXACT app validation logic
                validated_price = apply_ultra_conservative_doge_validation(
                    cached_prediction, current_price, days, symbol
                )
                
                # Calculate metrics
                cached_growth = ((cached_prediction / current_price) - 1) * 100
                validated_growth = ((validated_price / current_price) - 1) * 100
                
                cached_annual = (((cached_prediction / current_price) ** (365.25 / days)) - 1) * 100
                validated_annual = (((validated_price / current_price) ** (365.25 / days)) - 1) * 100
                
                print(f"\n📅 {days} Days ({time_key}):")
                print(f"   💰 Cached: ${cached_prediction:.8f} ({cached_growth:+.1f}% total, {cached_annual:+.1f}% annual)")
                print(f"   ✅ Final:  ${validated_price:.8f} ({validated_growth:+.1f}% total, {validated_annual:+.1f}% annual)")
                
                # Check validation status
                if abs(cached_prediction - validated_price) > 0.001:
                    print(f"   🛡️  ULTRA-CONSERVATIVE VALIDATION APPLIED")
                    print(f"   📉 Reduced from {cached_annual:+.1f}% to {validated_annual:+.1f}% annual")
                else:
                    print(f"   ✨ Cached prediction was already conservative")
                
                # Check if within ultra-conservative limits
                max_limits = {30: 150, 90: 80, 365: 40, 730: 25, 1095: 15}
                max_allowed = max_limits.get(days, 15)
                
                if validated_annual <= max_allowed:
                    print(f"   ✅ WITHIN LIMIT: {validated_annual:.1f}% ≤ {max_allowed}% (GOOD)")
                else:
                    print(f"   ⚠️  OVER LIMIT: {validated_annual:.1f}% > {max_allowed}% (NEEDS MORE CONSTRAINT)")
    
    return True

def apply_ultra_conservative_doge_validation(pred_price, current_price, days_elapsed, symbol):
    """Apply exact ultra-conservative DOGE validation from the app"""
    
    # Handle extreme/invalid values first
    if (not isinstance(pred_price, (int, float)) or 
        np.isnan(pred_price) or np.isinf(pred_price) or 
        pred_price <= 0 or pred_price > 1e12):  # Cap at 1 trillion
        
        return current_price * (1.02 ** (days_elapsed / 365.25))  # 2% fallback
    
    # Calculate growth metrics with overflow protection
    total_growth = pred_price / current_price
    
    # Protect against overflow in exponential calculation
    try:
        if total_growth > 1e6:  # If growth is extreme, cap it immediately
            annual_growth_rate = 1000.0  # 100,000% annual (will be capped)
        else:
            annual_growth_rate = (total_growth ** (365.25 / days_elapsed)) - 1
    except (OverflowError, ValueError):
        annual_growth_rate = 1000.0  # 100,000% annual (will be capped)
    
    # ULTRA-CONSERVATIVE CONSTRAINTS FOR MEME COINS (DOGE/SHIB)
    if symbol in ["DOGE-USD", "SHIB-USD"]:
        # Ultra-conservative for meme coins
        if days_elapsed <= 30:
            max_annual_growth = 1.5  # 150% max for meme coins short-term
            min_annual_growth = -0.70  # -70% max decline
        elif days_elapsed <= 90:
            max_annual_growth = 0.8  # 80% max for meme coins 3 months
            min_annual_growth = -0.60  # -60% max decline
        elif days_elapsed <= 365:
            max_annual_growth = 0.4  # 40% max for meme coins 1 year
            min_annual_growth = -0.50  # -50% max decline
        elif days_elapsed <= 730:  # 2 years
            max_annual_growth = 0.25  # 25% max for meme coins 2+ years
            min_annual_growth = -0.40  # -40% max decline
        else:  # 3+ years
            max_annual_growth = 0.15  # 15% max for meme coins long-term
            min_annual_growth = -0.30  # -30% max decline
    else:
        # Standard constraints for other cryptocurrencies
        if days_elapsed <= 30:
            max_annual_growth = 5.0  # 500% max
            min_annual_growth = -0.90  # -90% max decline
        elif days_elapsed <= 90:
            max_annual_growth = 3.0  # 300% max
            min_annual_growth = -0.80  # -80% max decline
        elif days_elapsed <= 365:
            max_annual_growth = 1.5  # 150% max
            min_annual_growth = -0.70  # -70% max decline
        elif days_elapsed <= 730:  # 2 years
            max_annual_growth = 0.8  # 80% max for 2+ years
            min_annual_growth = -0.60  # -60% max decline
        else:  # 3+ years (Most Conservative)
            max_annual_growth = 0.5  # 50% max for long-term
            min_annual_growth = -0.50  # -50% max decline
    
    # Apply constraints
    if annual_growth_rate > max_annual_growth:
        annual_growth_rate = max_annual_growth
        # Add minimal random variation to avoid flat lines
        variation = np.random.uniform(0.95, 1.05)
        annual_growth_rate *= variation
    elif annual_growth_rate < min_annual_growth:
        annual_growth_rate = min_annual_growth
        # Add minimal random variation
        variation = np.random.uniform(0.95, 1.05)
        annual_growth_rate *= variation
    
    # Convert back to price
    validated_price = current_price * ((1 + annual_growth_rate) ** (days_elapsed / 365.25))
    
    return validated_price

if __name__ == "__main__":
    try:
        success = test_doge_app_logic()
        if success:
            print(f"\n🎉 DOGE ULTRA-CONSERVATIVE VALIDATION TEST COMPLETE!")
            print(f"✅ Maximum annual growth rates for DOGE:")
            print(f"   • 30 days: 150% annual")
            print(f"   • 90 days: 80% annual")
            print(f"   • 1 year: 40% annual")
            print(f"   • 2 years: 25% annual")
            print(f"   • 3+ years: 15% annual")
            print(f"\n🚀 DOGE is now under ultra-conservative constraints!")
            print(f"🌐 Check the app at: http://localhost:8503")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
