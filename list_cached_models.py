#!/usr/bin/env python3
"""
List all cached models to see what's actually available
"""

import json
import os

def list_cached_models():
    cache_file = 'model_cache/cache_manifest.json'
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        
        print("=== Models in Cache Manifest ===")
        print(f"Total models: {len(cache['models'])}")
        print()
        
        for symbol in sorted(cache['models'].keys()):
            model_info = cache['models'][symbol]
            print(f"  {symbol:<12} -> {model_info['file']}")
    else:
        print("No cache manifest found!")

if __name__ == "__main__":
    list_cached_models()
