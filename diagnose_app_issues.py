#!/usr/bin/env python3
"""
Diagnostic script for mobile crypto app issues
Tests: NaN prices, volume calculations, total return calculations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_crypto_data(symbol='BTC-USD'):
    """Test cryptocurrency data retrieval and calculations"""
    print(f"🔍 Testing {symbol} data...")
    
    try:
        # Test current price data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        print(f"📊 Ticker Info Keys: {list(info.keys())[:10]}...")
        
        # Test current price methods
        current_price = info.get('currentPrice', 'N/A')
        regular_market_price = info.get('regularMarketPrice', 'N/A')
        previous_close = info.get('previousClose', 'N/A')
        
        print(f"💰 Current Price: {current_price}")
        print(f"💰 Regular Market Price: {regular_market_price}")
        print(f"💰 Previous Close: {previous_close}")
        
        # Test historical data
        end_date = datetime.now()
        start_date_1m = end_date - timedelta(days=30)
        start_date_6m = end_date - timedelta(days=180)
        start_date_1y = end_date - timedelta(days=365)
        
        # Get recent data
        recent_data = yf.download(symbol, start=start_date_1m, end=end_date, progress=False)
        print(f"📈 Recent data shape: {recent_data.shape}")
        print(f"📈 Recent data columns: {recent_data.columns.tolist()}")
        
        if not recent_data.empty:
            latest_price = recent_data['Close'].iloc[-1]
            print(f"💰 Latest Close Price: {latest_price}")
            
            # Test volume calculations
            if 'Volume' in recent_data.columns:
                avg_volume = recent_data['Volume'].mean()
                latest_volume = recent_data['Volume'].iloc[-1]
                print(f"📊 Average Volume (30d): {avg_volume:,.0f}")
                print(f"📊 Latest Volume: {latest_volume:,.0f}")
            else:
                print("❌ No Volume data available")
            
            # Test price change calculations
            if len(recent_data) >= 7:
                price_7d_ago = recent_data['Close'].iloc[-7]
                price_change_7d = ((latest_price - price_7d_ago) / price_7d_ago * 100)
                print(f"📈 7-day price change: {price_change_7d:.2f}%")
            
            if len(recent_data) >= 30:
                price_30d_ago = recent_data['Close'].iloc[0]
                price_change_30d = ((latest_price - price_30d_ago) / price_30d_ago * 100)
                print(f"📈 30-day price change: {price_change_30d:.2f}%")
            
            # Test 6-month data for charts
            hist_data_6m = yf.download(symbol, start=start_date_6m, end=end_date, progress=False)
            print(f"📊 6-month data shape: {hist_data_6m.shape}")
            
            if not hist_data_6m.empty:
                start_price = hist_data_6m['Close'].iloc[0]
                end_price = hist_data_6m['Close'].iloc[-1]
                total_return = ((end_price - start_price) / start_price * 100)
                print(f"📈 6-month total return: {total_return:.2f}%")
            
        else:
            print("❌ No recent data available")
            
    except Exception as e:
        print(f"❌ Error testing {symbol}: {str(e)}")

def test_multiple_cryptos():
    """Test multiple cryptocurrencies"""
    symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
    
    for symbol in symbols:
        print("=" * 50)
        test_crypto_data(symbol)
        print()

if __name__ == "__main__":
    print("🔧 CryptoQuantum Mobile App Diagnostics")
    print("=" * 50)
    test_multiple_cryptos()
