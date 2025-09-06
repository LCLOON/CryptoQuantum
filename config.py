"""
CryptoQuantum Configuration File
Contains all configuration constants and settings
"""

# Professional Cryptocurrency Terminal Symbols - Top 50 Cryptos
CRYPTO_SYMBOLS = {
    '₿ BTC/USD': 'BTC-USD',
    'Ξ ETH/USD': 'ETH-USD', 
    '₮ USDT/USD': 'USDT-USD',
    '🔷 BNB/USD': 'BNB-USD',
    '◎ SOL/USD': 'SOL-USD',
    '🔵 USDC/USD': 'USDC-USD',
    '✖️ XRP/USD': 'XRP-USD',
    'Ð DOGE/USD': 'DOGE-USD',
    '🌕 LUNA/USD': 'LUNA-USD',
    '₳ ADA/USD': 'ADA-USD',
    '🔺 AVAX/USD': 'AVAX-USD',
    '💎 SHIB/USD': 'SHIB-USD',
    '● DOT/USD': 'DOT-USD',
    '⬡ LINK/USD': 'LINK-USD',
    '🔴 BCH/USD': 'BCH-USD',
    '🌪️ TRX/USD': 'TRX-USD',
    '🔰 NEAR/USD': 'NEAR-USD',
    '⬢ MATIC/USD': 'MATIC-USD',
    'Ł LTC/USD': 'LTC-USD',
    '💰 UNI/USD': 'UNI-USD',
    '🚀 ICP/USD': 'ICP-USD',
    '⚡ APT/USD': 'APT-USD',
    '📈 FTT/USD': 'FTT-USD',
    '🌟 ETC/USD': 'ETC-USD',
    '🔸 XLM/USD': 'XLM-USD',
    '⚖️ ATOM/USD': 'ATOM-USD',
    '🏦 CRO/USD': 'CRO-USD',
    '🌊 APE/USD': 'APE-USD',
    '🎯 ALGO/USD': 'ALGO-USD',
    '🔥 MANA/USD': 'MANA-USD',
    '⚔️ AXS/USD': 'AXS-USD',
    '🎮 SAND/USD': 'SAND-USD',
    '💸 VET/USD': 'VET-USD',
    '🔗 FIL/USD': 'FIL-USD',
    '🌈 FLOW/USD': 'FLOW-USD',
    '🎨 CHZ/USD': 'CHZ-USD',
    '💎 GRT/USD': 'GRT-USD',
    '🌍 THETA/USD': 'THETA-USD',
    '⚙️ ENJ/USD': 'ENJ-USD',
    '📱 BAT/USD': 'BAT-USD',
    '🔮 CRV/USD': 'CRV-USD',
    '⭐ XTZ/USD': 'XTZ-USD',
    '🏛️ MKR/USD': 'MKR-USD',
    '📊 COMP/USD': 'COMP-USD',
    '🎪 SUSHI/USD': 'SUSHI-USD',
    '🔋 YFI/USD': 'YFI-USD',
    '🌟 SNX/USD': 'SNX-USD',
    '🎯 AAVE/USD': 'AAVE-USD',
    '🔄 1INCH/USD': '1INCH-USD',
    '🚀 RUNE/USD': 'RUNE-USD'
}

# Top 10 Cryptocurrency Data Storage and Update System
TOP_10_CRYPTOS = [
    ('₿ BTC/USD', 'BTC-USD'),
    ('Ξ ETH/USD', 'ETH-USD'), 
    ('🔷 BNB/USD', 'BNB-USD'),
    ('◎ SOL/USD', 'SOL-USD'),
    ('✖️ XRP/USD', 'XRP-USD'),
    ('₳ ADA/USD', 'ADA-USD'),
    ('Ð DOGE/USD', 'DOGE-USD'),
    ('● DOT/USD', 'DOT-USD'),
    ('⬡ LINK/USD', 'LINK-USD'),
    ('Ł LTC/USD', 'LTC-USD')
]

# Market-based growth estimates (more conservative and realistic)
GROWTH_ESTIMATES = {
    'BTC-USD': {
        'conservative': 0.12,
        'moderate': 0.18,
        'optimistic': 0.25,
        'bull_case': 0.35
    },
    'ETH-USD': {
        'conservative': 0.15,
        'moderate': 0.22,
        'optimistic': 0.30,
        'bull_case': 0.45
    },
    'DOGE-USD': {
        'conservative': 0.08,
        'moderate': 0.15,
        'optimistic': 0.35,
        'bull_case': 0.60
    },
    'default': {
        'conservative': 0.10,
        'moderate': 0.15,
        'optimistic': 0.25,
        'bull_case': 0.40
    }
}

# AI Model Options
AI_MODEL_OPTIONS = [
    "⚡ Ultra-Fast Cache Mode (Recommended)",
    "🎯 Advanced AttentionLSTM + Market Analysis",
    "📊 Multi-Model Ensemble (AttentionLSTM + XGBoost)"
]

# Chart Theme Options
CHART_THEMES = [
    "Professional Dark", 
    "Terminal Green", 
    "Trading Blue", 
    "Classic"
]

# Chart Color Schemes
CHART_COLORS = {
    "Terminal Green": {
        'main_color': '#00ff88',
        'forecast_color': '#00cc6a',
        'sma20_color': '#00dd77',
        'sma50_color': '#00bb66'
    },
    "Trading Blue": {
        'main_color': '#00bfff',
        'forecast_color': '#0080ff',
        'sma20_color': '#00aaff',
        'sma50_color': '#0099ee'
    },
    "Classic": {
        'main_color': '#000000',
        'forecast_color': '#ff6600',
        'sma20_color': '#666666',
        'sma50_color': '#999999'
    },
    "Professional Dark": {
        'main_color': '#00ff88',
        'forecast_color': '#ffd700',
        'sma20_color': '#00dd77',
        'sma50_color': '#ffcc00'
    }
}

# UI Configuration
CONFIDENCE_LEVELS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
DEFAULT_CONFIDENCE = 0.85
DEFAULT_FORECAST_YEARS = 3
MAX_FORECAST_YEARS = 5
MIN_FORECAST_YEARS = 1

# Cache Configuration
CACHE_TTL = 86400  # 24 hours in seconds
CACHE_DIRECTORY = "model_cache"
DATA_DIRECTORY = "data"
MODELS_DIRECTORY = "models"

# Performance thresholds
HIGH_VOLATILITY_THRESHOLD = 100
MEDIUM_VOLATILITY_THRESHOLD = 50

# File paths
REQUIREMENTS_FILE = "requirements.txt"
CACHE_MANIFEST_FILE = "cache_manifest.json"
