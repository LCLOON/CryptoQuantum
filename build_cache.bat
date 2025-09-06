@echo off
echo Building CryptoQuantum Cache...
echo This may take 30-60 minutes depending on your hardware.
echo.

python pretrain_models.py --cryptos top10 --epochs 100

echo.
echo Cache building completed!
echo Run the main app for faster performance.
pause
