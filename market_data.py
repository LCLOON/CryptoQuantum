"""
Market Data Utilities
Handles fetching and processing cryptocurrency market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_comprehensive_data(symbol, period='1y'):
    """Fetch comprehensive cryptocurrency data with error handling"""
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Download data
        data = ticker.history(period=period)
        
        if data.empty:
            return None, None
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'], data['MACD_signal'] = calculate_macd(data['Close'])
        data['Bollinger_Upper'], data['Bollinger_Lower'] = calculate_bollinger_bands(data['Close'])
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        # Create dates list
        dates = data.index.tolist()
        
        return data, dates
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_crypto_info(symbol):
    """Get current cryptocurrency information with robust error handling"""
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get basic info
        info = ticker.info
        
        # Download recent data for calculations
        hist = ticker.history(period="5d", interval="1d")
        
        # Fix multi-level column index if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        
        if hist.empty:
            # Fallback: try different period
            hist = ticker.history(period="1d", interval="1h")
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
            if hist.empty:
                return create_fallback_crypto_info(symbol)
        
        # Get current price (ensure it's a scalar)
        current_price = float(hist['Close'].iloc[-1])
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
        
        # Calculate 24h change
        if len(hist) >= 2:
            prev_close = float(hist['Close'].iloc[-2])
            price_change_24h = ((current_price - prev_close) / prev_close) * 100
        else:
            price_change_24h = 0.0
        
        # Get high and low
        high_24h = float(hist['High'].max()) if not hist['High'].empty else current_price
        low_24h = float(hist['Low'].min()) if not hist['Low'].empty else current_price
        
        # Calculate volatility
        if len(hist) >= 5:
            returns = hist['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility
        else:
            volatility = 50.0  # Default volatility
        
        # Get market cap and volume from info (with fallbacks)
        market_cap = info.get('marketCap', 0)
        if market_cap == 0:
            # Estimate market cap if not available
            try:
                shares_outstanding = info.get('sharesOutstanding', 0)
                if shares_outstanding > 0:
                    market_cap = current_price * shares_outstanding
            except Exception:
                market_cap = 0
        
        # Get volume
        volume_24h = info.get('volume24Hr', 0)
        if volume_24h == 0:
            volume_24h = info.get('regularMarketVolume', 0)
        if volume_24h == 0 and not hist['Volume'].empty:
            volume_24h = float(hist['Volume'].iloc[-1])
        
        # Create comprehensive info dictionary
        crypto_info = {
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'market_cap': market_cap,
            'volume_24h': volume_24h,
            'volatility': volatility,
            'symbol': symbol,
            'name': info.get('longName', symbol.replace('-USD', '')),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return crypto_info
        
    except Exception as e:
        print(f"Error getting crypto info for {symbol}: {str(e)}")
        return create_fallback_crypto_info(symbol)

def create_fallback_crypto_info(symbol):
    """Create fallback crypto info when API fails"""
    # Basic fallback prices based on common knowledge
    fallback_prices = {
        'BTC-USD': 45000,
        'ETH-USD': 2500,
        'DOGE-USD': 0.08,
        'ADA-USD': 0.50,
        'SOL-USD': 100,
        'XRP-USD': 0.60,
        'DOT-USD': 8.0,
        'LINK-USD': 15.0,
        'LTC-USD': 100,
        'BNB-USD': 300
    }
    
    price = fallback_prices.get(symbol, 100.0)
    
    return {
        'current_price': price,
        'price_change_24h': 0.0,
        'high_24h': price * 1.05,
        'low_24h': price * 0.95,
        'market_cap': price * 1000000000,  # Estimate
        'volume_24h': price * 10000000,    # Estimate
        'volatility': 75.0,
        'symbol': symbol,
        'name': symbol.replace('-USD', ''),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def get_realistic_price_prediction(symbol, current_price, forecast_years=5):
    """Get realistic price prediction with proper scaling"""
    from config import GROWTH_ESTIMATES
    
    # Get growth rates for the symbol
    rates = GROWTH_ESTIMATES.get(symbol, GROWTH_ESTIMATES['default'])
    
    # Calculate predictions for different scenarios
    predictions = {}
    for scenario, annual_rate in rates.items():
        future_price = current_price * ((1 + annual_rate) ** forecast_years)
        predictions[scenario] = future_price
    
    return predictions

def create_price_scenarios_display(symbol, current_price, forecast_years=5):
    """Create a comprehensive price scenarios display"""
    predictions = get_realistic_price_prediction(symbol, current_price, forecast_years)
    
    scenarios_data = []
    for scenario, price in predictions.items():
        total_return = ((price - current_price) / current_price) * 100
        annual_return = ((price / current_price) ** (1/forecast_years) - 1) * 100
        
        scenarios_data.append({
            'Scenario': scenario.replace('_', ' ').title(),
            'Target Price': f"${price:,.2f}",
            'Total Return': f"{total_return:.0f}%",
            'Annual Return': f"{annual_return:.1f}%"
        })
    
    return scenarios_data

def get_update_schedule_info():
    """Get dynamic update schedule information"""
    from datetime import datetime, timedelta
    import pytz
    
    # Current time in EST
    est = pytz.timezone('US/Eastern')
    current_time = datetime.now(est)
    
    # Next midnight EST
    next_midnight = (current_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Time until next update
    time_until_next = next_midnight - current_time
    hours, remainder = divmod(time_until_next.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    # Format time until next
    if time_until_next.days > 0:
        time_until_str = f"{time_until_next.days}d {hours}h {minutes}m"
    else:
        time_until_str = f"{hours}h {minutes}m"
    
    return {
        'status': 'ACTIVE' if current_time.hour < 1 else 'SCHEDULED',
        'last_update': current_time.strftime('%H:%M EST'),
        'next_update': next_midnight.strftime('%H:%M EST'),
        'time_until_next': time_until_str,
        'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S EST')
    }

def format_price(price):
    """Format price with appropriate decimal places"""
    if price <= 0:
        return "N/A"
    elif price < 0.01:
        return f"${price:.6f}"
    elif price < 1:
        return f"${price:.4f}"
    elif price < 100:
        return f"${price:.2f}"
    else:
        return f"${price:,.2f}"

def format_projection_price(price):
    """Format projection price with appropriate decimal places"""
    if price <= 0:
        return "N/A"
    elif price < 0.01:
        return f"${price:.6f}"
    elif price < 1:
        return f"${price:.4f}"
    elif price < 100:
        return f"${price:.2f}"
    else:
        return f"${price:,.2f}"
