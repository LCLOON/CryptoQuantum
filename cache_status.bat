@echo off
echo CryptoQuantum Cache Status
echo ==========================
echo.

python -c "from cache_loader import CacheLoader; loader = CacheLoader(); stats = loader.get_cache_stats(); print(f'Models: {stats["models_count"] if stats else 0}'); print(f'Data: {stats["data_count"] if stats else 0}'); print(f'Size: {stats["total_size_mb"]:.1f} MB' if stats else 'No cache')"

echo.
pause
