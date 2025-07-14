@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  🚀 CRYPTOQUANTUM ADVANCED LAUNCHER
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    echo 📥 Download from: https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Check if required packages are installed
echo 🔍 Checking dependencies...
python -c "import streamlit, torch, yfinance, plotly" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Missing dependencies detected!
    echo 📦 Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies!
        pause
        exit /b 1
    )
)

echo ✅ All dependencies ready
echo.

REM Kill existing processes
echo 🛑 Stopping any running instances...
taskkill /f /im "streamlit.exe" >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501" ^| find "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul

REM Start the application
echo 🚀 Starting CryptoQuantum Terminal...
echo 📊 Features: 50+ Cryptos, Dual AI Engines, Real-time Data
echo 🌐 Opening browser at: http://localhost:8501
echo.

start /b streamlit run stunning_crypto_app.py --server.headless=false --server.port=8501 --browser.gatherUsageStats=false

REM Wait a moment then try to open browser manually if needed
timeout /t 3 /nobreak >nul
start http://localhost:8501

echo ✅ CryptoQuantum Terminal is now running!
echo.
echo 💡 Tips:
echo    - Keep this window open while using the app
echo    - Close this window to stop the application
echo    - If browser doesn't show the app, try refreshing
echo.
echo Press CTRL+C to stop the application, or close this window
echo ========================================

REM Keep the window open and monitor the process
:wait
timeout /t 5 /nobreak >nul
tasklist /fi "imagename eq python.exe" /fo csv | find "python.exe" >nul
if errorlevel 1 (
    echo ⚠️  Application stopped unexpectedly
    pause
    exit /b 1
)
goto wait