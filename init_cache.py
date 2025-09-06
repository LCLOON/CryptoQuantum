#!/usr/bin/env python3
"""
Initialize cache for Streamlit Cloud deployment
This script will be run on first app startup to generate essential models
"""

import os
import sys
import time
import streamlit as st
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_all_cryptos import CompleteCacheGenerator
from config import CRYPTO_SYMBOLS
import cache_loader

def initialize_cloud_cache():
    """Initialize essential models for Streamlit Cloud"""
    
    # Check if cache already exists
    loader = cache_loader.CacheLoader()
    
    # Essential cryptocurrencies to train first (top 2 for cloud efficiency)
    essential_cryptos = {
        '₿ BTC/USD': 'BTC-USD',
        'Ξ ETH/USD': 'ETH-USD'
    }
    
    # Check how many essential cryptos are cached
    cached_essentials = 0
    for crypto_name, symbol in essential_cryptos.items():
        if loader.is_cache_available(symbol):
            cached_essentials += 1
    
    # If less than 2 essential cryptos are cached, initialize
    if cached_essentials < 2:
        st.info("🚀 Initializing CryptoQuantum cache for first time...")
        st.info("⏳ Training essential cryptocurrency models (BTC & ETH - takes ~1 minute)...")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        generator = CompleteCacheGenerator()
        
        total_essential = len(essential_cryptos)
        completed = 0
        
        for crypto_name, symbol in essential_cryptos.items():
            if not loader.is_cache_available(symbol):
                status_text.text(f"Training {crypto_name}...")
                
                # Train this cryptocurrency
                success = generator.cache_crypto(crypto_name, symbol)
                
                if success:
                    completed += 1
                    status_text.text(f"✅ {crypto_name} trained successfully!")
                else:
                    status_text.text(f"❌ {crypto_name} training failed")
                
                # Update progress
                progress_bar.progress(completed / total_essential)
                time.sleep(1)  # Brief pause for UI
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if completed >= 2:
            st.success("🎉 CryptoQuantum is ready! Essential models (BTC & ETH) trained successfully.")
            st.balloons()
            time.sleep(2)
            st.rerun()  # Refresh the app
        else:
            st.error("❌ Failed to initialize essential models. Please try refreshing the page.")
    
    return cached_essentials >= 2 or completed >= 2

if __name__ == "__main__":
    initialize_cloud_cache()
