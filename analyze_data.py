#!/usr/bin/env python3
"""
Analyze the BTC data files to understand the format issue
"""

import pickle
import json
import os
import pandas as pd

def analyze_btc_data():
    print("=== BTC Data File Analysis ===")
    print()
    
    pkl_file = 'model_cache/data/BTC-USD_data.pkl'
    json_file = 'model_cache/data/BTC-USD_data.json'
    
    # Check PKL file
    if os.path.exists(pkl_file):
        print(f"PKL file exists: {os.path.getsize(pkl_file)} bytes")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"PKL data type: {type(data)}")
            
            if isinstance(data, pd.DataFrame):
                print(f"PKL DataFrame shape: {data.shape}")
                print(f"PKL DataFrame columns: {list(data.columns)}")
                print(f"PKL DataFrame index: {data.index[:3]}...")
            elif isinstance(data, dict):
                print(f"PKL data keys: {list(data.keys())}")
                for key, value in data.items():
                    print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else value}")
            else:
                print(f"PKL data content: {data}")
                
        except Exception as e:
            print(f"PKL load error: {e}")
    else:
        print("PKL file does not exist")
    
    print()
    
    # Check JSON file
    if os.path.exists(json_file):
        print(f"JSON file exists: {os.path.getsize(json_file)} bytes")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"JSON data type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"JSON data keys: {list(data.keys())}")
                for key, value in data.items():
                    print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else value}")
            else:
                print(f"JSON data content: {data}")
                
        except Exception as e:
            print(f"JSON load error: {e}")
    else:
        print("JSON file does not exist")

if __name__ == "__main__":
    analyze_btc_data()
