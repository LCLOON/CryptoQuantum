import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

CRYPTOS = {
    'Bitcoin': 'BTC-USD',
    'Dogecoin': 'DOGE-USD', 
    'Pepe': 'PEPE-USD',
    'Shiba Inu': 'SHIB-USD'
}

def get_crypto_price(symbol):
    '''Get current crypto price'''
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

def get_crypto_info(symbol):
    '''Get crypto information'''
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        data = ticker.history(period='1d')
        
        current_price = data['Close'].iloc[-1] if not data.empty else None
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'market_cap': info.get('marketCap', 'N/A'),
            'name': info.get('longName', symbol)
        }
    except Exception as e:
        return {'error': str(e)}
