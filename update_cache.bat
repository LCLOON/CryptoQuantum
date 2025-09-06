@echo off
echo Updating CryptoQuantum Cache...
echo This will refresh all cached models and data.
echo.

python pretrain_models.py --cryptos top10 --epochs 50

echo.
echo Cache update completed!
pause
