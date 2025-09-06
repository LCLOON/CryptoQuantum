"""
CryptoQuantum Mobile - iPhone Optimized Crypto Predictions
Mobile-first design for cryptocurrency price predictions
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import time
from config import CRYPTO_SYMBOLS
from market_data import get_crypto_info
from cache_loader import CacheLoader

def load_mobile_css():
    """Load enhanced mobile-optimized CSS styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main > div {
        padding-top: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom App Header */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #581c87 100%);
        border-radius: 25px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 
            0 15px 50px rgba(30, 58, 138, 0.4),
            0 5px 20px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: shimmer 4s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 3px 15px rgba(0,0,0,0.4);
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
        font-family: 'Inter', sans-serif;
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.95);
        margin: 0.8rem 0 0 0;
        font-weight: 500;
        position: relative;
        z-index: 1;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Enhanced Card Design */
    .crypto-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 
            0 8px 32px rgba(30, 58, 138, 0.15),
            0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(30, 58, 138, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .crypto-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 12px 48px rgba(30, 58, 138, 0.25),
            0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Special Crypto Selection Card */
    .crypto-selection-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e6f0ff 100%);
        border-radius: 25px;
        padding: 2rem 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 12px 40px rgba(30, 58, 138, 0.2),
            0 4px 16px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border: 2px solid rgba(30, 58, 138, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .crypto-selection-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(30, 58, 138, 0.05), transparent);
        animation: slide 3s infinite;
    }
    
    @keyframes slide {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .crypto-selection-card:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 
            0 18px 60px rgba(30, 58, 138, 0.3),
            0 6px 24px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border-color: rgba(30, 58, 138, 0.25);
    }
    
    .crypto-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 20px 20px 0 0;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .crypto-selection-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -0.3px;
    }
    
    .section-emoji {
        font-size: 1.8rem;
        filter: drop-shadow(0 3px 6px rgba(0,0,0,0.15));
    }
    
    .crypto-selection-emoji {
        font-size: 2.2rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        width: 100%;
        height: 3.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #581c87 100%);
        color: white;
        box-shadow: 
            0 6px 20px rgba(30, 58, 138, 0.4),
            0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 8px 30px rgba(30, 58, 138, 0.5),
            0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.01);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Period Selection Buttons */
    .period-button {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        margin: 0.25rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .period-button:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .period-button.selected {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Selectbox */
    .stSelectbox > div > div {
        font-size: 1.2rem;
        padding: 1.2rem;
        border-radius: 16px;
        border: 2px solid rgba(30, 58, 138, 0.2);
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(30, 58, 138, 0.1);
    }
    
    .stSelectbox > div > div:focus {
        border-color: #1e3a8a;
        box-shadow: 
            0 0 0 4px rgba(30, 58, 138, 0.15),
            0 6px 24px rgba(30, 58, 138, 0.2);
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(30, 58, 138, 0.3);
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.15);
        transform: translateY(-2px);
    }
    
    /* Prediction Results Card */
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #581c87 100%);
        color: white;
        border-radius: 25px;
        padding: 2.5rem 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 15px 50px rgba(30, 58, 138, 0.5),
            0 6px 24px rgba(0, 0, 0, 0.15);
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    
    .prediction-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .price-display {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .change-display {
        font-size: 1.3rem;
        font-weight: 600;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 
            0 6px 20px rgba(102, 126, 234, 0.1),
            0 2px 8px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 8px 30px rgba(102, 126, 234, 0.15),
            0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    /* Status Messages */
    .status-success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
    }
    
    .status-info {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
    }
    
    /* Loading Spinner */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
        padding: 2rem 1rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 20px 20px 0 0;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .app-title {
            font-size: 1.8rem;
        }
        
        .price-display {
            font-size: 2rem;
        }
        
        .crypto-card {
            margin: 0.75rem 0;
            padding: 1.25rem;
        }
        
        .stButton > button {
            height: 3rem;
            font-size: 1rem;
        }
    }
    
    /* Smooth Transitions */
    * {
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Mobile-optimized main application"""
    
    # Page setup for mobile
    st.set_page_config(
        page_title="CryptoQuantum Mobile",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed"  # Collapsed by default on mobile
    )
    
    # Load mobile CSS
    load_mobile_css()
    
    # Enhanced Mobile header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">� CryptoQuantum</div>
        <div class="app-subtitle">Professional Crypto Predictions</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize cache with force check for Streamlit Cloud
    cache_loader = CacheLoader()
    
    # Check if initialization is needed
    cache_manifest_path = Path('model_cache/cache_manifest.json')
    needs_init = not cache_manifest_path.exists()
    
    # Force Smart ML initialization on Streamlit Cloud if needed
    if needs_init:
        st.info("🚀 First-time initialization - Setting up CryptoQuantum...")
        try:
            from smart_ml_init import smart_initialize
            success = smart_initialize()
            if success:
                st.success("✅ Smart ML system initialized!")
                time.sleep(2)  # Brief pause before reload
                st.rerun()
        except Exception as e:
            st.warning(f"Smart ML failed ({e}), using lightweight system...")
            try:
                from lightweight_init import lightweight_initialize
                success = lightweight_initialize()
                if success:
                    st.success("✅ Lightweight system initialized!")
                    time.sleep(2)
                    st.rerun()
            except Exception as e2:
                st.error(f"All initialization failed: {e2}")
                st.stop()
    
    # Cryptocurrency Selection Section
    st.markdown("""
    <div class="crypto-selection-card">
        <div class="crypto-selection-header">
            <span class="crypto-selection-emoji">🚀</span>
            Choose Your Cryptocurrency
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional selection prompt
    st.markdown("""
    <div style="
        text-align: center; 
        color: #1e3a8a; 
        font-size: 1.1rem; 
        font-weight: 600; 
        margin: 1rem 0 1.5rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f0f4ff 0%, #e6f0ff 100%);
        border-radius: 12px;
        border-left: 4px solid #1e3a8a;
    ">
        💡 Select your cryptocurrency below to get professional AI-powered predictions
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Cryptocurrency selection with mobile-friendly display
    crypto_options = []
    crypto_mapping = {}
    
    for display_name, symbol in CRYPTO_SYMBOLS.items():
        crypto_options.append(display_name)
        crypto_mapping[display_name] = symbol
    
    selected_crypto = st.selectbox(
        "🎯 Select Cryptocurrency:",
        crypto_options,
        key="crypto_select",
        help="Choose from 50+ supported cryptocurrencies for AI analysis"
    )
    
    # Extract symbol using mapping
    symbol = crypto_mapping[selected_crypto]
    crypto_name = symbol.replace('-USD', '')
    
    # 2. Prediction period with enhanced selection
    st.markdown("""
    <div class="crypto-card">
        <div class="section-header">
            <span class="section-emoji">⏰</span>
            Prediction Timeframe
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional timeframe selection
    st.markdown("""
    <div style="
        text-align: center; 
        color: #1e3a8a; 
        font-size: 1rem; 
        font-weight: 500; 
        margin: 0.5rem 0 1rem 0;
        padding: 0.75rem;
        background: linear-gradient(135deg, #f0f4ff 0%, #e6f0ff 100%);
        border-radius: 10px;
        border-left: 3px solid #3730a3;
    ">
        ⏰ Choose your prediction horizon for AI analysis
    </div>
    """, unsafe_allow_html=True)
    
    # Timeframe options with better organization
    timeframe_options = {
        "📅 1 Month (30 days)": 30,
        "📆 3 Months (90 days)": 90,
        "🗓️ 6 Months (180 days)": 180,
        "📊 1 Year (365 days)": 365,
        "📈 2 Years (730 days)": 730
    }
    
    selected_timeframe = st.selectbox(
        "🎯 Select Prediction Period:",
        list(timeframe_options.keys()),
        index=1,  # Default to 3 months
        key="timeframe_select",
        help="Choose how far into the future you want the AI to predict"
    )
    
    # Get the actual days value
    prediction_days = timeframe_options[selected_timeframe]
    
    # Display confirmation
    st.info(f"✅ **Selected Timeframe:** {selected_timeframe.split(' ', 1)[1]}")
    
    # 3. Analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    
    analyze_btn = st.button(
        "🚀 GET PREDICTION",
        key="analyze_mobile",
        help="Tap to get your crypto prediction!"
    )
    
    # Results section
    if analyze_btn:
        with st.spinner("🔮 Analyzing..."):
            # Get current price
            crypto_info = get_crypto_info(symbol)
            current_price = crypto_info['current_price']
            
            # Check cache availability
            cache_available = cache_loader.is_cache_available(symbol)
            
            if cache_available:
                # Success indicator
                st.markdown("""
                <div class="mobile-card">
                    <div class="status-good">⚡ Ultra-Fast Prediction Ready!</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Get cached prediction
                cached_forecasts = cache_loader.get_all_cached_forecasts(symbol)
                
                if cached_forecasts and 'forecasts' in cached_forecasts:
                    forecasts = cached_forecasts['forecasts']
                    time_key = f"{prediction_days}_days"
                    
                    if time_key in forecasts:
                        predicted_price = forecasts[time_key]
                        validated_price = validate_prediction(predicted_price, current_price, prediction_days, symbol)
                        
                        # Calculate metrics
                        total_change = ((validated_price / current_price) - 1) * 100
                        change_color = "#48bb78" if total_change >= 0 else "#f56565"
                        change_icon = "📈" if total_change >= 0 else "📉"
                        
                        # Get period name for display
                        period_name = selected_timeframe.split(' ', 1)[1].replace(' (', ' (').replace(')', '')
                        
                        # Display prediction in mobile-friendly card
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-title">{selected_crypto} Prediction</div>
                            <div class="price-display">${validated_price:.4f}</div>
                            <div class="change-display" style="color: white;">
                                {change_icon} {total_change:+.1f}% in {period_name}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Current vs Predicted comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="color: #718096; font-size: 0.9rem;">Current Price</div>
                                <div style="font-size: 1.3rem; font-weight: 600; color: #2d3748;">
                                    ${current_price:.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="color: #718096; font-size: 0.9rem;">Predicted Price</div>
                                <div style="font-size: 1.3rem; font-weight: 600; color: {change_color};">
                                    ${validated_price:.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Market info
                        if crypto_info:
                            st.markdown("""
                            <div class="mobile-card">
                                <div class="prediction-title">📊 Market Info</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            info_items = []
                            if 'market_cap' in crypto_info:
                                info_items.append(f"💰 Market Cap: ${crypto_info['market_cap']:,.0f}")
                            if 'price_change_24h' in crypto_info:
                                change_24h = crypto_info['price_change_24h']
                                change_icon_24h = "🟢" if change_24h >= 0 else "🔴"
                                info_items.append(f"{change_icon_24h} 24h Change: {change_24h:+.2f}%")
                            
                            for item in info_items:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div style="font-size: 1rem; color: #2d3748;">{item}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # 📈 ENHANCED CHARTS SECTION
                        st.markdown("""
                        <div class="mobile-card">
                            <div class="prediction-title">📈 Professional Price Analysis</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            # Get comprehensive historical data
                            end_date = datetime.now()
                            start_date_6m = end_date - timedelta(days=180)  # 6 months
                            start_date_1y = end_date - timedelta(days=365)  # 1 year
                            
                            # Download 6-month data for main chart
                            hist_data_6m = yf.download(symbol, start=start_date_6m, end=end_date, auto_adjust=True, progress=False)
                            hist_data_1y = yf.download(symbol, start=start_date_1y, end=end_date, auto_adjust=True, progress=False)
                            
                            # Fix multi-level column index if present
                            if isinstance(hist_data_6m.columns, pd.MultiIndex):
                                hist_data_6m.columns = hist_data_6m.columns.droplevel(1)
                            if isinstance(hist_data_1y.columns, pd.MultiIndex):
                                hist_data_1y.columns = hist_data_1y.columns.droplevel(1)
                            
                            if not hist_data_6m.empty:
                                # === MAIN PRICE CHART WITH CANDLESTICKS ===
                                fig_main = make_subplots(
                                    rows=2, cols=1,
                                    subplot_titles=[f'{selected_crypto} - 6 Month Candlestick Chart', 'Volume'],
                                    vertical_spacing=0.1,
                                    row_heights=[0.7, 0.3]
                                )
                                
                                # Candlestick chart
                                fig_main.add_trace(go.Candlestick(
                                    x=hist_data_6m.index,
                                    open=hist_data_6m['Open'],
                                    high=hist_data_6m['High'],
                                    low=hist_data_6m['Low'],
                                    close=hist_data_6m['Close'],
                                    name=f'{selected_crypto} Price',
                                    increasing_line_color='#00C851',
                                    decreasing_line_color='#FF4444'
                                ), row=1, col=1)
                                
                                # Add moving averages
                                if len(hist_data_6m) >= 20:
                                    ma_20 = hist_data_6m['Close'].rolling(window=20).mean()
                                    fig_main.add_trace(go.Scatter(
                                        x=hist_data_6m.index,
                                        y=ma_20,
                                        mode='lines',
                                        name='MA 20',
                                        line=dict(color='#FF8C00', width=1),
                                        opacity=0.8
                                    ), row=1, col=1)
                                
                                if len(hist_data_6m) >= 50:
                                    ma_50 = hist_data_6m['Close'].rolling(window=50).mean()
                                    fig_main.add_trace(go.Scatter(
                                        x=hist_data_6m.index,
                                        y=ma_50,
                                        mode='lines',
                                        name='MA 50',
                                        line=dict(color='#9C27B0', width=1),
                                        opacity=0.8
                                    ), row=1, col=1)
                                
                                # Current price marker
                                fig_main.add_trace(go.Scatter(
                                    x=[hist_data_6m.index[-1]],
                                    y=[current_price],
                                    mode='markers',
                                    name='Current Price',
                                    marker=dict(color='#1e3a8a', size=15, symbol='circle', 
                                               line=dict(color='white', width=2)),
                                    hovertemplate='Current: $%{y:,.4f}<extra></extra>'
                                ), row=1, col=1)
                                
                                # Prediction point and trend
                                future_date = end_date + timedelta(days=prediction_days)
                                fig_main.add_trace(go.Scatter(
                                    x=[future_date],
                                    y=[validated_price],
                                    mode='markers',
                                    name=f'{period_name} Target',
                                    marker=dict(color=change_color, size=18, symbol='star',
                                               line=dict(color='white', width=2)),
                                    hovertemplate=f'{period_name} Target: $%{{y:,.4f}}<br>Change: {total_change:+.1f}%<extra></extra>'
                                ), row=1, col=1)
                                
                                # Prediction trend line with confidence band
                                prediction_x = [hist_data_6m.index[-1], future_date]
                                prediction_y = [current_price, validated_price]
                                
                                # Main prediction line
                                fig_main.add_trace(go.Scatter(
                                    x=prediction_x,
                                    y=prediction_y,
                                    mode='lines',
                                    name='Forecast Trend',
                                    line=dict(color=change_color, width=3, dash='dash'),
                                    hovertemplate='Forecast: $%{y:,.4f}<extra></extra>'
                                ), row=1, col=1)
                                
                                # Confidence bands (±20%)
                                upper_band = [current_price, validated_price * 1.2]
                                lower_band = [current_price, validated_price * 0.8]
                                
                                # Convert hex color to RGB for confidence band
                                hex_color = change_color.strip('#')
                                if len(hex_color) == 6:
                                    r = int(hex_color[0:2], 16)
                                    g = int(hex_color[2:4], 16)
                                    b = int(hex_color[4:6], 16)
                                    fillcolor = f'rgba({r}, {g}, {b}, 0.1)'
                                else:
                                    fillcolor = 'rgba(72, 187, 120, 0.1)'  # Default green
                                
                                fig_main.add_trace(go.Scatter(
                                    x=prediction_x + prediction_x[::-1],
                                    y=upper_band + lower_band[::-1],
                                    fill='toself',
                                    fillcolor=fillcolor,
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='Confidence Band',
                                    showlegend=False,
                                    hovertemplate='Confidence Range<extra></extra>'
                                ), row=1, col=1)
                                
                                # Volume chart
                                if 'Volume' in hist_data_6m.columns:
                                    colors = ['#00C851' if close >= open else '#FF4444' 
                                             for close, open in zip(hist_data_6m['Close'], hist_data_6m['Open'])]
                                    
                                    fig_main.add_trace(go.Bar(
                                        x=hist_data_6m.index,
                                        y=hist_data_6m['Volume'],
                                        name='Volume',
                                        marker_color=colors,
                                        opacity=0.7,
                                        hovertemplate='Volume: %{y:,.0f}<extra></extra>'
                                    ), row=2, col=1)
                                
                                # Enhanced styling
                                fig_main.update_layout(
                                    title=dict(
                                        text=f'🚀 {selected_crypto} Professional Analysis - {period_name} Forecast',
                                        font=dict(size=16, color='#1e3a8a')
                                    ),
                                    height=600,
                                    template='plotly_white',
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="center",
                                        x=0.5,
                                        font=dict(size=10)
                                    ),
                                    margin=dict(l=20, r=20, t=60, b=20),
                                    font=dict(size=11),
                                    hovermode='x unified',
                                    dragmode='pan'
                                )
                                
                                # Styling for axes
                                fig_main.update_xaxes(
                                    showgrid=True, 
                                    gridwidth=1, 
                                    gridcolor='rgba(128,128,128,0.1)',
                                    title_text="Date",
                                    row=2, col=1
                                )
                                fig_main.update_yaxes(
                                    showgrid=True, 
                                    gridwidth=1, 
                                    gridcolor='rgba(128,128,128,0.1)',
                                    title_text="Price (USD)",
                                    row=1, col=1
                                )
                                fig_main.update_yaxes(
                                    title_text="Volume",
                                    row=2, col=1
                                )
                                
                                # Remove rangeslider for cleaner look
                                fig_main.update_layout(xaxis_rangeslider_visible=False)
                                
                                st.plotly_chart(fig_main, use_container_width=True)
                                
                                # === PRICE PERFORMANCE COMPARISON CHART ===
                                if not hist_data_1y.empty and len(hist_data_1y) >= 30:
                                    st.markdown("""
                                    <div class="mobile-card">
                                        <div class="prediction-title">📊 Performance Timeline</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Calculate normalized returns (percentage change from start)
                                    normalized_data = (hist_data_1y['Close'] / hist_data_1y['Close'].iloc[0] - 1) * 100
                                    
                                    fig_perf = go.Figure()
                                    
                                    # Performance line with gradient fill
                                    fig_perf.add_trace(go.Scatter(
                                        x=hist_data_1y.index,
                                        y=normalized_data,
                                        mode='lines',
                                        name='Performance %',
                                        line=dict(color='#1e3a8a', width=3),
                                        fill='tonexty',
                                        fillcolor='rgba(30, 58, 138, 0.1)',
                                        hovertemplate='Date: %{x}<br>Return: %{y:.1f}%<extra></extra>'
                                    ))
                                    
                                    # Add zero line
                                    fig_perf.add_hline(
                                        y=0, 
                                        line_dash="dash", 
                                        line_color="gray",
                                        annotation_text="Break Even"
                                    )
                                    
                                    # Add prediction point
                                    future_return = ((validated_price / current_price) - 1) * 100
                                    fig_perf.add_trace(go.Scatter(
                                        x=[future_date],
                                        y=[future_return],
                                        mode='markers',
                                        name=f'{period_name} Target',
                                        marker=dict(color=change_color, size=15, symbol='star'),
                                        hovertemplate=f'{period_name} Target: {future_return:+.1f}%<extra></extra>'
                                    ))
                                    
                                    fig_perf.update_layout(
                                        title='📈 Total Return Performance (%)',
                                        xaxis_title='Date',
                                        yaxis_title='Total Return (%)',
                                        height=350,
                                        template='plotly_white',
                                        showlegend=False,
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        font=dict(size=11)
                                    )
                                    
                                    fig_perf.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                    fig_perf.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                    
                                    st.plotly_chart(fig_perf, use_container_width=True)
                                
                                # === TECHNICAL INDICATORS SUMMARY ===
                                if len(hist_data_6m) >= 20:
                                    st.markdown("""
                                    <div class="mobile-card">
                                        <div class="prediction-title">📊 Technical Indicators</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Calculate indicators
                                    recent_data = hist_data_6m.tail(20)
                                    
                                    # Safe price change calculations
                                    try:
                                        if len(recent_data) >= 7:
                                            price_change_7d = ((current_price / recent_data['Close'].iloc[-7]) - 1) * 100
                                            if isinstance(price_change_7d, pd.Series):
                                                price_change_7d = price_change_7d.iloc[0] if not price_change_7d.empty else 0
                                        else:
                                            price_change_7d = 0
                                            
                                        if len(hist_data_6m) >= 30:
                                            price_change_30d = ((current_price / hist_data_6m['Close'].iloc[-30]) - 1) * 100
                                            if isinstance(price_change_30d, pd.Series):
                                                price_change_30d = price_change_30d.iloc[0] if not price_change_30d.empty else 0
                                        else:
                                            price_change_30d = 0
                                    except:
                                        price_change_7d = 0
                                        price_change_30d = 0
                                    
                                    volatility = recent_data['Close'].pct_change().std() * 100
                                    
                                    # RSI calculation (simplified)
                                    delta = recent_data['Close'].diff()
                                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                                    rs = gain / loss
                                    
                                    # Safe RSI calculation
                                    try:
                                        if not rs.empty and len(rs.dropna()) > 0:
                                            rs_value = rs.iloc[-1]
                                            if not np.isnan(rs_value) and rs_value != 0:
                                                rsi = 100 - (100 / (1 + rs_value))
                                            else:
                                                rsi = 50
                                        else:
                                            rsi = 50
                                    except:
                                        rsi = 50
                                    
                                    # Display indicators in columns
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div style="color: #718096; font-size: 0.8rem;">7-Day Change</div>
                                            <div style="font-size: 1.1rem; font-weight: 600; color: {'#00C851' if price_change_7d >= 0 else '#FF4444'};">
                                                {price_change_7d:+.1f}%
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div style="color: #718096; font-size: 0.8rem;">30-Day Change</div>
                                            <div style="font-size: 1.1rem; font-weight: 600; color: {'#00C851' if price_change_30d >= 0 else '#FF4444'};">
                                                {price_change_30d:+.1f}%
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col3:
                                        rsi_color = '#FF4444' if rsi > 70 else '#00C851' if rsi < 30 else '#FFA500'
                                        rsi_status = 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div style="color: #718096; font-size: 0.8rem;">RSI Signal</div>
                                            <div style="font-size: 1.1rem; font-weight: 600; color: {rsi_color};">
                                                {rsi_status}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            else:
                                st.warning("⚠️ Unable to load chart data")
                                
                        except Exception as e:
                            st.error(f"📊 Chart temporarily unavailable: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                
            else:
                st.markdown(f"""
                <div class="mobile-card">
                    <div class="status-warning">⚠️ No cached model for {crypto_name}</div>
                    <div style="color: #718096; margin-top: 0.5rem;">
                        Live training would take 30-60 seconds
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.9rem; padding: 1rem;">
        📱 Optimized for iPhone • 💡 Educational purposes only • 🚫 Not financial advice
    </div>
    """, unsafe_allow_html=True)

def get_crypto_emoji(crypto_name):
    """Get emoji for cryptocurrency"""
    emoji_map = {
        'BTC': '₿', 'ETH': '⟠', 'ADA': '🔷', 'DOT': '🔴', 'LINK': '🔗',
        'XRP': '💧', 'LTC': '⚡', 'BCH': '💰', 'XLM': '⭐', 'UNI': '🦄',
        'DOGE': '🐕', 'SHIB': '🐶', 'MATIC': '🔮', 'AVAX': '⛷️', 'SOL': '☀️',
        'ATOM': '⚛️', 'ICP': '∞', 'FIL': '📁', 'TRX': '🎭', 'ETC': '💎',
        'VET': '✅', 'THETA': 'θ', 'FTT': '🚀', 'ALGO': '🔄', 'XMR': '🔒',
        'AAVE': '👻', 'MKR': '🏭', 'COMP': '⚖️', 'SUSHI': '🍣', 'YFI': '🌾'
    }
    return emoji_map.get(crypto_name, '💰')

def validate_prediction(pred_price, current_price, days, symbol):
    """Apply simple validation for mobile app"""
    
    if not isinstance(pred_price, (int, float)) or pred_price <= 0:
        return current_price * 1.1
    
    total_growth = pred_price / current_price
    try:
        annual_growth = (total_growth ** (365 / days)) - 1
    except Exception:
        annual_growth = 0.1
    
    # Conservative mobile limits
    if symbol in ["DOGE-USD", "SHIB-USD"]:
        max_annual = 0.3 if days <= 365 else 0.15
    else:
        max_annual = 1.0 if days <= 365 else 0.5
    
    if annual_growth > max_annual:
        annual_growth = max_annual
    elif annual_growth < -0.8:
        annual_growth = -0.8
    
    return current_price * ((1 + annual_growth) ** (days / 365))

if __name__ == "__main__":
    main()
