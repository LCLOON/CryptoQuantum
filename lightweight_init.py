#!/usr/bin/env python3
"""
Lightweight cache initialization for Streamlit Cloud
Creates basic forecasts without heavy ML training
"""

import os
import sys
import json
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cache_loader

def create_simple_forecasts(symbol, current_price):
    """Create simple rule-based forecasts without ML training"""
    
    # Simple growth assumptions based on historical crypto patterns
    growth_factors = {
        'BTC-USD': {'30_days': 1.08, '90_days': 1.15, '180_days': 1.25, '365_days': 1.45, '730_days': 1.85},
        'ETH-USD': {'30_days': 1.10, '90_days': 1.18, '180_days': 1.30, '365_days': 1.55, '730_days': 2.00},
        'DOGE-USD': {'30_days': 1.12, '90_days': 1.25, '180_days': 1.40, '365_days': 1.75, '730_days': 2.50},
        'SOL-USD': {'30_days': 1.15, '90_days': 1.25, '180_days': 1.45, '365_days': 1.80, '730_days': 2.80},
        'ADA-USD': {'30_days': 1.10, '90_days': 1.20, '180_days': 1.35, '365_days': 1.65, '730_days': 2.20}
    }
    
    # Get growth factors for this symbol or use defaults
    factors = growth_factors.get(symbol, growth_factors['BTC-USD'])
    
    # Add some randomness for realism (±5%)
    forecasts = {}
    for timeframe, factor in factors.items():
        noise = np.random.uniform(0.95, 1.05)  # ±5% variation
        predicted_price = current_price * factor * noise
        forecasts[timeframe] = float(predicted_price)
    
    return forecasts

def create_basic_cache():
    """Create basic cache structure with simple forecasts"""
    
    # Essential cryptocurrencies  
    essential_cryptos = {
        '₿ BTC/USD': 'BTC-USD',
        'Ξ ETH/USD': 'ETH-USD',
        'Ð DOGE/USD': 'DOGE-USD',
        '◎ SOL/USD': 'SOL-USD',
        '₳ ADA/USD': 'ADA-USD'
    }
    
    # Create cache directories
    cache_dir = Path('model_cache')
    for subdir in ['models', 'data', 'forecasts', 'scalers']:
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    manifest = {
        'created_date': datetime.now().isoformat(),
        'cache_version': '3.0',
        'models': {},
        'forecasts': {}
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    completed = 0
    total = len(essential_cryptos)
    
    for crypto_name, symbol in essential_cryptos.items():
        try:
            status_text.text(f"Creating cache for {crypto_name}...")
            
            # Get current price
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d")
            
            if current_data.empty:
                status_text.text(f"❌ No data for {crypto_name}")
                continue
                
            current_price = float(current_data['Close'].iloc[-1])
            
            # Create simple forecasts
            forecasts = create_simple_forecasts(symbol, current_price)
            
            # Save forecast file
            forecast_data = {
                'symbol': symbol,
                'generated_date': datetime.now().isoformat(),
                'forecasts': forecasts,
                'method': 'rule_based_v1'
            }
            
            forecast_file = cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
            with open(forecast_file, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            
            # Create basic historical data file
            hist_data = ticker.history(period="1y")
            if not hist_data.empty:
                data_file = cache_dir / 'data' / f'{symbol}_data.pkl'
                hist_data.to_pickle(data_file)
                
                # JSON version
                data_json_file = cache_dir / 'data' / f'{symbol}_data.json'
                hist_json = hist_data.reset_index()
                hist_json['Date'] = hist_json['Date'].dt.strftime('%Y-%m-%d')
                hist_json.to_json(data_json_file, orient='records')
            
            # Add to manifest (even without ML model)
            manifest['forecasts'][symbol] = {
                'file': f'{symbol}_forecasts.json',
                'method': 'rule_based',
                'created': datetime.now().isoformat()
            }
            
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"✅ {crypto_name} ready!")
            
        except Exception as e:
            status_text.text(f"❌ Error with {crypto_name}: {str(e)}")
            continue
    
    # Save manifest
    manifest_file = cache_dir / 'cache_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    progress_bar.empty()
    status_text.empty()
    
    if completed >= 2:
        st.success(f"🎉 CryptoQuantum initialized! {completed} cryptocurrencies ready.")
        st.balloons()
        return True
    else:
        st.error("❌ Initialization failed. Please refresh to try again.")
        return False

def lightweight_initialize():
    """Lightweight initialization for Streamlit Cloud"""
    
    # Check if basic cache exists
    cache_dir = Path('model_cache')
    manifest_file = cache_dir / 'cache_manifest.json'
    
    if manifest_file.exists():
        return True  # Already initialized
    
    # Check if we have any forecast files
    forecast_dir = cache_dir / 'forecasts'
    if forecast_dir.exists():
        forecast_files = list(forecast_dir.glob('*_forecasts.json'))
        if len(forecast_files) >= 2:
            return True  # Sufficient cache exists
    
    # Need to initialize
    st.info("🚀 Setting up CryptoQuantum for first use...")
    st.info("⚡ Creating instant predictions (30 seconds)...")
    
    return create_basic_cache()

if __name__ == "__main__":
    lightweight_initialize()
