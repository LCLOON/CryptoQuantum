@echo off
echo ========================================
echo  Crypto Future Forecast - Setup Script
echo ========================================
echo.

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Training the AI model (this may take 10-15 minutes)...
python train_model.py

echo.
echo Setup complete! You can now run the app with:
echo streamlit run enhanced_app.py
echo.
pause
