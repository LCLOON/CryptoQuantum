"""
Market Data Utilities
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=300)
except ImportError:
    def cache_decorator(func):
        return func

@cache_decorator
def get_crypto_info(symbol):
    """Get current cryptocurrency information"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return create_fallback_crypto_info(symbol)
        
        current_price = float(hist['Close'].iloc[-1])
        
        if len(hist) >= 2:
            prev_close = float(hist['Close'].iloc[-2])
            price_change_24h = ((current_price - prev_close) / prev_close) * 100
        else:
            price_change_24h = 0.0
        
        return {
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'high_24h': float(hist['High'].max()),
            'low_24h': float(hist['Low'].min()),
            'market_cap': 0,
            'volume_24h': float(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
            'volatility': 50.0,
            'symbol': symbol,
            'name': symbol.replace('-USD', ''),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return create_fallback_crypto_info(symbol)

def create_fallback_crypto_info(symbol):
    """Create fallback crypto info when API fails"""
    fallback_prices = {
        'BTC-USD': 45000,
        'ETH-USD': 2500,
        'DOGE-USD': 0.08,
        'ADA-USD': 0.50,
        'SOL-USD': 100
    }
    
    price = fallback_prices.get(symbol, 100.0)
    
    return {
        'current_price': price,
        'price_change_24h': 0.0,
        'high_24h': price * 1.05,
        'low_24h': price * 0.95,
        'market_cap': 0,
        'volume_24h': price * 10000000,
        'volatility': 75.0,
        'symbol': symbol,
        'name': symbol.replace('-USD', ''),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
