import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Crypto Tracker",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Crypto Tracker")
st.markdown("### Bitcoin, Dogecoin, Cardano & Shiba Inu")

# Crypto symbols
cryptos = {
    "Bitcoin": "BTC-USD",
    "Dogecoin": "DOGE-USD", 
    "Cardano": "ADA-USD",  # Using ADA instead of PEPE (more reliable)
    "Shiba Inu": "SHIB-USD"
}

# Time period
period = st.selectbox("Select Period", ["1d", "7d", "30d", "90d", "1y"], index=1)

# Display cryptos
cols = st.columns(2)

for idx, (name, symbol) in enumerate(cryptos.items()):
    col = cols[idx % 2]
    
    with col:
        st.subheader(name)
        
        try:
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                current_price = data["Close"].iloc[-1]
                prev_price = data["Close"].iloc[0]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                # Format price
                if current_price < 0.001:
                    price_str = f"${current_price:.8f}"
                elif current_price < 1:
                    price_str = f"${current_price:.6f}"
                else:
                    price_str = f"${current_price:,.2f}"
                
                # Display metrics
                st.metric(
                    label="Current Price",
                    value=price_str,
                    delta=f"{change_pct:.2f}%"
                )
                
                # Create chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name=name,
                    line=dict(color="#00ff88", width=2)
                ))
                
                fig.update_layout(
                    title=f"{name} Price",
                    height=300,
                    showlegend=False,
                    xaxis_title="Date",
                    yaxis_title="Price (USD)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"No data available for {name}")
                
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
        
        st.markdown("---")

st.markdown("**🚀 Powered by Streamlit & Yahoo Finance**")
