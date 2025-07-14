@echo off
echo ========================================
echo  🚀 CRYPTOQUANTUM TERMINAL LAUNCHER
echo ========================================
echo.

echo ✨ STUNNING VISUALS for Large Screens!
echo 🧠 DUAL AI ENGINES with Advanced Forecasting!
echo 💎 50+ Cryptocurrencies Supported!
echo 📊 Real-time Market Data with Accurate Market Caps!
echo.

echo 🔍 Checking system status...

REM Kill any existing Streamlit processes
taskkill /f /im "streamlit.exe" 2>nul
taskkill /f /im "python.exe" /fi "WINDOWTITLE eq streamlit*" 2>nul

echo ⚡ Clearing any port conflicts...
timeout /t 2 /nobreak >nul

echo 🚀 Launching CryptoQuantum Terminal...
echo 📱 Browser will open automatically at: http://localhost:8501
echo.

REM Start Streamlit with explicit browser opening
start /min streamlit run stunning_crypto_app.py --server.headless=false --server.port=8501 --browser.gatherUsageStats=false

echo ✅ CryptoQuantum Terminal is starting...
echo 🌐 If browser doesn't open, manually go to: http://localhost:8501
echo.
echo Press any key to close this window (app will keep running)...
pause >nul
