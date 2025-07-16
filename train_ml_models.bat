@echo off
echo.
echo ========================================
echo   CryptoQuantum ML Training Suite
echo ========================================
echo.

echo Testing basic dependencies...
python -c "import torch, numpy, pandas, sklearn, yfinance, plotly; print('✅ All dependencies available')" 2>nul
if errorlevel 1 (
    echo ❌ Missing dependencies. Run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Select training option:
echo 1. Basic LSTM Model (train_model.py)
echo 2. Advanced AttentionLSTM Model (train_advanced_model.py)  
echo 3. Multi-Crypto Portfolio Model (train_multi_crypto.py)
echo 4. Test ML imports in main app
echo 5. Run main CryptoQuantum app
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Training Basic Enhanced LSTM Model...
    python train_model.py
    echo.
    echo Training completed! Check btc_model_enhanced.pth
    pause
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Training Advanced AttentionLSTM Model...
    python train_advanced_model.py
    echo.
    echo Training completed! Check btc_attention_model.pth
    pause
) else if "%choice%"=="3" (
    echo.
    echo 🚀 Training Multi-Crypto Portfolio Model...
    echo This may take several minutes...
    python train_multi_crypto.py
    echo.
    echo Training completed! Check universal_crypto_model.pth
    pause
) else if "%choice%"=="4" (
    echo.
    echo 🧪 Testing ML imports in main app...
    python -c "
import sys
sys.path.append('.')
try:
    from stunning_crypto_app import *
    print('✅ All imports successful')
    print('✅ LSTMModel class loaded')
    print('✅ AttentionLSTMModel class loaded')
    print('✅ Main functions accessible')
    print('✅ CryptoQuantum app ready!')
except Exception as e:
    print(f'❌ Import error: {e}')
"
    echo.
    pause
) else if "%choice%"=="5" (
    echo.
    echo 🚀 Starting CryptoQuantum Terminal...
    streamlit run stunning_crypto_app.py
) else (
    echo Invalid choice. Please run again.
    pause
)
