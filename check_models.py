#!/usr/bin/env python3
"""
Check which cryptocurrency models are missing vs cached
"""

import os
import json
from config import CRYPTO_SYMBOLS

def check_model_status():
    print("=== Cryptocurrency Model Status ===")
    print()
    
    cache_file = 'model_cache/cache_manifest.json'
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        
        cached_models = set(cache['models'].keys())
        config_symbols = set(CRYPTO_SYMBOLS.values())
        
        print(f"Total cryptocurrencies in config: {len(config_symbols)}")
        print(f"Total models cached: {len(cached_models)}")
        print()
        
        missing = config_symbols - cached_models
        extra = cached_models - config_symbols
        
        if missing:
            print(f"⚠️  Missing models ({len(missing)}):")
            for symbol in sorted(missing):
                # Find the display name for this symbol
                name = "Unknown"
                for k, v in CRYPTO_SYMBOLS.items():
                    if v == symbol:
                        name = k
                        break
                print(f"   {symbol} ({name})")
            print()
        
        if extra:
            print(f"ℹ️  Extra models not in config ({len(extra)}):")
            for symbol in sorted(extra):
                print(f"   {symbol}")
            print()
        
        if not missing:
            print("✅ All cryptocurrency models are cached!")
        else:
            print(f"Need to train {len(missing)} missing models.")
            
    else:
        print("❌ No cache manifest found - all models need training")

if __name__ == "__main__":
    check_model_status()
