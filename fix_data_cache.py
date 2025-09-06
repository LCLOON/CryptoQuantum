#!/usr/bin/env python3
"""
Fix the data cache by re-saving historical data in the correct format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import json
from config import CRYPTO_SYMBOLS

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    try:
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI (simplified)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicator
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        return df
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        return df

def fix_data_cache():
    print("=== Fixing Data Cache ===")
    print("Re-downloading and saving historical data in correct format...")
    print()
    
    cache_dir = Path('model_cache')
    data_dir = cache_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with a few major cryptocurrencies first
    test_symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SOL-USD', 'ADA-USD']
    
    for symbol in test_symbols:
        print(f"Processing {symbol}...")
        
        try:
            # Download fresh data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")  # 2 years of data
            
            if hist.empty:
                print(f"  ❌ No data available for {symbol}")
                continue
            
            # Add technical indicators
            hist = add_technical_indicators(hist)
            
            # Save as pickle (DataFrame format)
            data_file = data_dir / f'{symbol}_data.pkl'
            hist.to_pickle(data_file)
            
            # Save as JSON for web compatibility
            data_json_file = data_dir / f'{symbol}_data.json'
            hist_json = hist.reset_index()
            hist_json['Date'] = hist_json['Date'].dt.strftime('%Y-%m-%d')
            hist_json.to_json(data_json_file, orient='records')
            
            print(f"  ✅ Saved {len(hist)} days of data")
            
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
    
    print()
    print("✅ Data cache fix completed!")
    print("Now test the app again with BTC, ETH, DOGE, SOL, or ADA")

if __name__ == "__main__":
    fix_data_cache()
