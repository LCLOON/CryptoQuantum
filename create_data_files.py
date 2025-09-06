#!/usr/bin/env python3
"""
Generate missing data files for cache
"""

import json
import yfinance as yf
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_data_files():
    """Create data files for cached cryptos"""
    cache_dir = Path("cache")
    data_dir = cache_dir / "data"
    forecasts_dir = cache_dir / "forecasts"
    
    # Get all cryptocurrencies with forecasts
    if not forecasts_dir.exists():
        print("No forecasts directory found")
        return
    
    for forecast_file in forecasts_dir.glob("*_forecasts.json"):
        symbol = forecast_file.stem.replace("_forecasts", "")
        print(f"Creating data file for {symbol}...")
        
        try:
            # Download recent data
            data = yf.download(symbol, period="1y")
            
            if not data.empty:
                # Create simple data structure
                current_price = float(data['Close'].iloc[-1])
                
                # Calculate 24h change if we have enough data
                if len(data) >= 2:
                    prev_price = float(data['Close'].iloc[-2])
                    price_change_24h = ((current_price - prev_price) / prev_price) * 100
                else:
                    price_change_24h = 0.0
                
                # Get market cap if available (approximate)
                market_cap = current_price * 1000000  # Simplified
                
                data_content = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "price_change_24h": price_change_24h,
                    "market_cap": market_cap,
                    "last_updated": "2025-09-06"
                }
                
                # Save data file
                data_file = data_dir / f"{symbol}_data.json"
                with open(data_file, 'w') as f:
                    json.dump(data_content, f, indent=2)
                
                print(f"✅ Created data file for {symbol}")
            else:
                print(f"❌ No data available for {symbol}")
                
        except Exception as e:
            print(f"❌ Error creating data for {symbol}: {e}")

if __name__ == "__main__":
    create_data_files()
