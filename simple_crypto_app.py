"""
CryptoQuantum - Simplified Cryptocurrency Prediction App
Clean, simple interface for crypto price predictions
"""

import streamlit as st
from config import CRYPTO_SYMBOLS
from market_data import get_crypto_info
from cache_loader import CacheLoader

def main():
    """Simplified main application"""
    
    # Page setup
    st.set_page_config(
        page_title="CryptoQuantum - Simple",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title
    st.title("💰 CryptoQuantum - Simple Crypto Predictions")
    st.markdown("Get quick, reliable cryptocurrency price predictions")
    
    # Initialize cache
    cache_loader = CacheLoader()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # 1. Select Cryptocurrency
        crypto_options = [f"{symbol.replace('-USD', '')} ({symbol})" for symbol in CRYPTO_SYMBOLS]
        selected_option = st.selectbox("Select Cryptocurrency:", crypto_options)
        symbol = selected_option.split('(')[1].replace(')', '')
        
        # 2. Prediction Period
        prediction_days = st.selectbox(
            "Prediction Period:",
            [30, 90, 180, 365, 730],
            index=2,
            format_func=lambda x: f"{x} days ({x//30} months)" if x < 365 else f"{x} days ({x//365} years)"
        )
        
        # 3. Run Analysis
        analyze_btn = st.button("🚀 Get Prediction", type="primary", use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    if analyze_btn:
        with col1:
            # Get current price
            crypto_info = get_crypto_info(symbol)
            current_price = crypto_info['current_price']
            
            st.subheader(f"📊 Analysis for {symbol}")
            st.write(f"**Current Price:** ${current_price:.4f}")
            
            # Check if cache is available
            cache_available = cache_loader.is_cache_available(symbol)
            
            if cache_available:
                st.success("⚡ Using cached model for instant prediction!")
                
                # Get cached prediction
                cached_forecasts = cache_loader.get_all_cached_forecasts(symbol)
                
                if cached_forecasts and 'forecasts' in cached_forecasts:
                    forecasts = cached_forecasts['forecasts']
                    
                    # Get prediction for selected period
                    time_key = f"{prediction_days}_days"
                    if time_key in forecasts:
                        predicted_price = forecasts[time_key]
                        
                        # Apply validation for realistic predictions
                        validated_price = validate_prediction(predicted_price, current_price, prediction_days, symbol)
                        
                        # Calculate metrics
                        total_change = ((validated_price / current_price) - 1) * 100
                        annual_change = (((validated_price / current_price) ** (365 / prediction_days)) - 1) * 100
                        
                        # Display results
                        st.markdown("### 🎯 Prediction Results")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Predicted Price", 
                                f"${validated_price:.4f}",
                                f"{total_change:+.1f}%"
                            )
                        with col_b:
                            st.metric(
                                "Annualized Return",
                                f"{annual_change:+.1f}%",
                                help="Expected yearly return rate"
                            )
                        
                        # Simple chart
                        chart_data = {
                            'Day': [0, prediction_days],
                            'Price': [current_price, validated_price]
                        }
                        st.line_chart(chart_data, x='Day', y='Price')
                        
                    else:
                        st.error(f"No cached prediction available for {prediction_days} days")
                else:
                    st.error("No cached forecasts found")
            else:
                st.warning(f"⚠️ No cached model available for {symbol}")
                st.info("This would require live training (30-60 seconds)")
        
        with col2:
            # Market info
            if crypto_info:
                st.subheader("📈 Market Info")
                
                if 'market_cap' in crypto_info:
                    st.write(f"**Market Cap:** ${crypto_info['market_cap']:,.0f}")
                if 'volume_24h' in crypto_info:
                    st.write(f"**24h Volume:** ${crypto_info['volume_24h']:,.0f}")
                if 'price_change_24h' in crypto_info:
                    change_24h = crypto_info['price_change_24h']
                    color = "🟢" if change_24h >= 0 else "🔴"
                    st.write(f"**24h Change:** {color} {change_24h:+.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** Predictions are for educational purposes only. Not financial advice.")

def validate_prediction(pred_price, current_price, days, symbol):
    """Apply simple validation to ensure realistic predictions"""
    
    if not isinstance(pred_price, (int, float)) or pred_price <= 0:
        return current_price * 1.1  # 10% growth fallback
    
    # Calculate annual growth rate
    total_growth = pred_price / current_price
    try:
        annual_growth = (total_growth ** (365 / days)) - 1
    except:
        annual_growth = 0.1  # 10% fallback
    
    # Conservative limits based on timeframe and asset type
    if symbol in ["DOGE-USD", "SHIB-USD"]:  # Meme coins
        if days <= 90:
            max_annual = 0.5  # 50% max annual
        elif days <= 365:
            max_annual = 0.3  # 30% max annual
        else:
            max_annual = 0.15  # 15% max annual
    else:  # Other cryptos
        if days <= 90:
            max_annual = 2.0  # 200% max annual
        elif days <= 365:
            max_annual = 1.0  # 100% max annual
        else:
            max_annual = 0.5  # 50% max annual
    
    # Apply limits
    if annual_growth > max_annual:
        annual_growth = max_annual
    elif annual_growth < -0.8:  # Max 80% decline
        annual_growth = -0.8
    
    # Convert back to price
    validated_price = current_price * ((1 + annual_growth) ** (days / 365))
    
    return validated_price

if __name__ == "__main__":
    main()
