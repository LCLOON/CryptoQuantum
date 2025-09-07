import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Crypto Tracker",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
.main > div { padding: 1rem; }
.stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 10px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# Crypto symbols
CRYPTOS = {
    "Bitcoin": "BTC-USD",
    "Dogecoin": "DOGE-USD", 
    "Pepe": "PEPE-USD",
    "Shiba Inu": "SHIB-USD"
}

def get_crypto_data(symbol, period="1d"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        info = ticker.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None

def format_price(price):
    if price < 0.001:
        return f"${price:.8f}"
    elif price < 1:
        return f"${price:.6f}"
    else:
        return f"${price:,.2f}"

def main():
    st.title("🚀 Crypto Tracker")
    st.markdown("### Track Bitcoin, Dogecoin, Pepe & Shiba Inu")
    
    period = st.selectbox("Select Time Period", ["1d", "7d", "30d", "90d", "1y"], index=1)
    
    cols = st.columns(2)
    
    for idx, (name, symbol) in enumerate(CRYPTOS.items()):
        col = cols[idx % 2]
        
        with col:
            st.markdown(f"### {name}")
            data, info = get_crypto_data(symbol, period)
            
            if data is not None and not data.empty:
                current_price = data["Close"].iloc[-1]
                prev_close = data["Close"].iloc[0] if len(data) > 1 else current_price
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                st.metric(
                    label="Current Price",
                    value=format_price(current_price),
                    delta=f"{change_pct:.2f}%"
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name=name,
                    line=dict(color="#00ff88", width=2)
                ))
                
                fig.update_layout(
                    title=f"{name} Price Chart",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not load data for {name}")
            
            st.markdown("---")
    
    st.markdown("🚀 Built with Streamlit | Data from Yahoo Finance")

if __name__ == "__main__":
    main()
