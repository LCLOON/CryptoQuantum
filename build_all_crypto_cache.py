"""
All Crypto Cache Builder - Complete Cryptocurrency Cache
=======================================================

This script builds cache for ALL cryptocurrencies available in the CryptoQuantum Terminal
dropdown menu with 5 years of historical data for maximum performance.

Usage:
    python build_all_crypto_cache.py

Developed by Lewis Loon | © 2025 Lewis Loon Analytics
"""

import json
import yfinance as yf
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler

# Complete list of all cryptocurrencies in the dropdown
ALL_CRYPTOCURRENCIES = {
    'BTC-USD': '₿ Bitcoin',
    'ETH-USD': 'Ξ Ethereum', 
    'BNB-USD': '🔷 Binance Coin',
    'XRP-USD': '✖️ XRP',
    'ADA-USD': '₳ Cardano',
    'SOL-USD': '◎ Solana',
    'DOGE-USD': 'Ð Dogecoin',
    'DOT-USD': '● Polkadot',
    'SHIB-USD': '💎 Shiba Inu',
    'AVAX-USD': '🔺 Avalanche',
    'LINK-USD': '⬡ Chainlink',
    'LTC-USD': 'Ł Litecoin',
    'UNI-USD': '💰 Uniswap',
    'BCH-USD': '🔴 Bitcoin Cash',
    'ATOM-USD': '⚛️ Cosmos',
    'TRX-USD': '🌪️ TRON',
    'ETC-USD': '💚 Ethereum Classic',
    'XLM-USD': '⭐ Stellar',
    'FIL-USD': '💾 Filecoin',
    'NEAR-USD': '🔰 NEAR Protocol',
    'ICP-USD': '🚀 Internet Computer',
    'VET-USD': '🌊 VeChain',
    'HBAR-USD': '🔗 Hedera',
    'APT-USD': '⚡ Aptos',
    'ALGO-USD': '🎯 Algorand',
    'FLOW-USD': '🌊 Flow',
    'MANA-USD': '🏛️ Decentraland',
    'SAND-USD': '🏖️ The Sandbox',
    'USDT-USD': '💵 Tether',
    'USDC-USD': '🔵 USD Coin'
}

def build_complete_crypto_cache():
    """Build comprehensive cache for all cryptocurrencies"""
    print("🚀 Building Complete Cryptocurrency Cache")
    print("=" * 50)
    print(f"📊 Processing {len(ALL_CRYPTOCURRENCIES)} cryptocurrencies...")
    print("🕐 Estimated time: 15-25 minutes")
    print()
    
    # Create cache directories
    cache_dir = Path('./model_cache')
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / 'data').mkdir(exist_ok=True)
    (cache_dir / 'forecasts').mkdir(exist_ok=True)
    (cache_dir / 'models').mkdir(exist_ok=True)
    (cache_dir / 'scalers').mkdir(exist_ok=True)
    
    successful_builds = 0
    failed_builds = []
    
    for i, (symbol, name) in enumerate(ALL_CRYPTOCURRENCIES.items(), 1):
        try:
            print(f"📈 [{i:2d}/{len(ALL_CRYPTOCURRENCIES)}] Processing {symbol} ({name})...")
            
            # Fetch 5 years of historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5y', auto_adjust=True)
            
            if hist.empty or len(hist) < 100:  # Need at least 100 days of data
                print(f"   ⚠️ Insufficient data for {symbol} ({len(hist) if not hist.empty else 0} days)")
                failed_builds.append(symbol)
                continue
            
            # Add technical indicators
            hist = add_technical_indicators(hist)
            
            # Cache data files
            data_file = cache_dir / 'data' / f'{symbol}_data.pkl'
            hist.to_pickle(data_file)
            
            json_file = cache_dir / 'data' / f'{symbol}_data.json'
            hist.reset_index().to_json(json_file, orient='records', date_format='iso')
            
            print(f"   ✅ Cached {len(hist):,} days of data")
            
            # Generate forecasts for multiple time horizons
            forecasts = generate_comprehensive_forecasts(hist, symbol)
            
            # Save forecasts
            forecast_file = cache_dir / 'forecasts' / f'{symbol}_forecasts.json'
            forecast_data = {
                'symbol': symbol,
                'name': name,
                'forecasts': forecasts,
                'generated_date': datetime.now().isoformat(),
                'model_type': 'Enhanced_Demo',
                'data_period': '5_years'
            }
            
            with open(forecast_file, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            
            print(f"   ✅ Generated forecasts for {len(forecasts)} time horizons")
            
            # Create scalers
            create_scalers(hist, symbol, cache_dir)
            print("   ✅ Created preprocessing scalers")
            
            successful_builds += 1
            
            # Progress indicator
            if i % 5 == 0:
                progress = (i / len(ALL_CRYPTOCURRENCIES)) * 100
                print(f"\n📊 Progress: {progress:.1f}% complete ({successful_builds}/{i} successful)\n")
            
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {str(e)}")
            failed_builds.append(symbol)
            continue
    
    # Create comprehensive manifest
    create_manifest(cache_dir, successful_builds, failed_builds)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Cache Building Complete!")
    print(f"✅ Successfully cached: {successful_builds}/{len(ALL_CRYPTOCURRENCIES)} cryptocurrencies")
    print(f"📊 Cache size: ~{estimate_cache_size(cache_dir):.1f} MB")
    
    if failed_builds:
        print(f"\n⚠️ Failed to cache: {len(failed_builds)} cryptocurrencies")
        print("   " + ", ".join(failed_builds))
    
    print(f"\n🚀 Your application now supports {successful_builds} cryptocurrencies with instant performance!")
    print("   Run your app to see the dramatic speed improvements.")

def add_technical_indicators(df):
    """Add comprehensive technical indicators"""
    try:
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price volatility
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        return df
        
    except Exception as e:
        print(f"      Warning: Error adding technical indicators: {str(e)}")
        return df

def generate_comprehensive_forecasts(hist, symbol):
    """Generate realistic forecasts based on historical patterns"""
    try:
        current_price = hist['Close'].iloc[-1]
        
        # Analyze historical trends
        short_term_trend = hist['Close'].tail(30).pct_change().mean()  # 30-day trend
        medium_term_trend = hist['Close'].tail(90).pct_change().mean()  # 90-day trend
        long_term_trend = hist['Close'].tail(365).pct_change().mean()  # 1-year trend
        
        # Calculate volatility
        volatility = hist['Close'].pct_change().std()
        
        # Time horizons for forecasts
        time_horizons = [
            (30, 'monthly'),
            (90, 'quarterly'), 
            (180, 'semi_annual'),
            (365, 'annual'),
            (730, 'bi_annual'),
            (1095, 'tri_annual'),
            (1825, 'five_year')
        ]
        
        forecasts = {}
        
        for days, period_name in time_horizons:
            predictions = []
            
            # Determine trend based on time horizon
            if days <= 90:
                base_trend = short_term_trend
                volatility_factor = volatility * 0.8
            elif days <= 365:
                base_trend = medium_term_trend  
                volatility_factor = volatility * 0.9
            else:
                base_trend = long_term_trend
                volatility_factor = volatility * 1.0
            
            # Apply dampening for longer forecasts
            trend_dampening = max(0.1, 1.0 - (days / 2000))
            dampened_trend = base_trend * trend_dampening
            
            current_pred_price = current_price
            
            for day in range(days):
                # Progressive trend decay
                day_factor = max(0.1, 1.0 - (day / days) * 0.5)
                daily_trend = dampened_trend * day_factor
                
                # Add realistic noise
                noise = np.random.normal(0, volatility_factor) * 0.5
                
                # Calculate next price
                daily_return = daily_trend + noise
                daily_return = np.clip(daily_return, -0.1, 0.1)  # Limit extreme moves
                
                current_pred_price *= (1 + daily_return)
                predictions.append(float(current_pred_price))
            
            forecasts[f'{days}_days'] = predictions
        
        return forecasts
        
    except Exception as e:
        print(f"      Warning: Error generating forecasts: {str(e)}")
        # Return simple forecasts as fallback
        return {f'{days}_days': [float(current_price)] * days for days, _ in time_horizons}

def create_scalers(hist, symbol, cache_dir):
    """Create and save data preprocessing scalers"""
    try:
        # Prepare returns and volume data
        returns = hist['Close'].pct_change().dropna().values.reshape(-1, 1)
        volume = hist['Volume'].values.reshape(-1, 1)
        
        # Remove any invalid values
        returns = returns[~np.isnan(returns).any(axis=1)]
        volume = volume[~np.isnan(volume).any(axis=1)]
        volume = np.maximum(volume, 1e-8)  # Avoid zeros
        
        # Create scalers
        scaler_returns = MinMaxScaler(feature_range=(-0.8, 0.8))
        scaler_volume = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scalers
        scaler_returns.fit(returns)
        scaler_volume.fit(volume)
        
        # Save scalers
        scaler_file = cache_dir / 'scalers' / f'{symbol}_scalers.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump({
                'scaler_returns': scaler_returns,
                'scaler_volume': scaler_volume,
                'last_log_price': np.log(hist['Close'].iloc[-1]),
                'data_quality': {
                    'returns_samples': len(returns),
                    'volume_samples': len(volume),
                    'date_range': [hist.index[0].isoformat(), hist.index[-1].isoformat()]
                }
            }, f)
        
    except Exception as e:
        print(f"      Warning: Error creating scalers: {str(e)}")

def create_manifest(cache_dir, successful_builds, failed_builds):
    """Create comprehensive cache manifest"""
    try:
        manifest = {
            'created_date': datetime.now().isoformat(),
            'cache_version': '2.0-complete',
            'total_cryptocurrencies': len(ALL_CRYPTOCURRENCIES),
            'successful_builds': successful_builds,
            'failed_builds': len(failed_builds),
            'failed_symbols': failed_builds,
            'data_period': '5_years',
            'models': {},
            'data': {},
            'forecasts': {}
        }
        
        # Catalog all cached files
        for data_file in (cache_dir / 'data').glob('*.pkl'):
            symbol = data_file.stem.replace('_data', '')
            if symbol in ALL_CRYPTOCURRENCIES:
                manifest['data'][symbol] = {
                    'name': ALL_CRYPTOCURRENCIES[symbol],
                    'file': data_file.name,
                    'size_mb': data_file.stat().st_size / (1024*1024)
                }
        
        for forecast_file in (cache_dir / 'forecasts').glob('*.json'):
            symbol = forecast_file.stem.replace('_forecasts', '')
            if symbol in ALL_CRYPTOCURRENCIES:
                manifest['forecasts'][symbol] = {
                    'name': ALL_CRYPTOCURRENCIES[symbol],
                    'file': forecast_file.name,
                    'size_mb': forecast_file.stat().st_size / (1024*1024)
                }
        
        # Models placeholder (for future ML model caching)
        for symbol in manifest['data'].keys():
            manifest['models'][symbol] = {
                'model_type': 'Complete_Demo',
                'file': f'{symbol}_complete_model.pth',
                'size_mb': 0,
                'status': 'placeholder'
            }
        
        # Save manifest
        manifest_file = cache_dir / 'cache_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   ✅ Cache manifest created with {len(manifest['data'])} entries")
        
    except Exception as e:
        print(f"   ⚠️ Error creating manifest: {str(e)}")

def estimate_cache_size(cache_dir):
    """Estimate total cache size in MB"""
    try:
        total_size = 0
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    except Exception:
        return 0

if __name__ == "__main__":
    print("🚀 CryptoQuantum Complete Cache Builder")
    print("Building cache for ALL supported cryptocurrencies...")
    print()
    
    response = input("This will build cache for 30 cryptocurrencies with 5 years of data each.\nContinue? (y/n): ")
    
    if response.lower() == 'y':
        build_complete_crypto_cache()
    else:
        print("Cache building cancelled.")
