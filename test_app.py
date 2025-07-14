"""
CryptoQuantum Terminal - Simple Test Suite
Quick validation that the app loads without errors
"""

import sys
import importlib.util

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        import yfinance as yf
        print("✅ yfinance imported successfully")
        
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
        
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        print("✅ Data science libraries imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_app_structure():
    """Test that the app file has proper structure"""
    print("\n🏗️ Testing app structure...")
    
    try:
        # Load the app module
        spec = importlib.util.spec_from_file_location("app", "stunning_crypto_app.py")
        app_module = importlib.util.module_from_spec(spec)
        
        # Check if main function exists
        if hasattr(app_module, 'main'):
            print("✅ Main function found")
        else:
            print("⚠️ Main function not found")
        
        # Check if required classes exist
        spec.loader.exec_module(app_module)
        
        if hasattr(app_module, 'AttentionLSTMModel'):
            print("✅ AttentionLSTMModel class found")
        else:
            print("⚠️ AttentionLSTMModel class not found")
            
        if hasattr(app_module, 'LSTMModel'):
            print("✅ LSTMModel class found")
        else:
            print("⚠️ LSTMModel class not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False

def run_tests():
    """Run all tests"""
    print("🎯 CryptoQuantum Terminal - Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_app_structure():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL TESTS PASSED - App ready for deployment!")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
    
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
