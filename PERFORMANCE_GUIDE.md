# CryptoQuantum Terminal - Performance Optimization Guide

## 🚀 Pre-training Models for Faster Performance

This guide explains how to pre-train models and cache data for significantly faster performance, especially beneficial for remote users or production deployments.

## Quick Setup

### 1. Build Cache (First Time)
```bash
# For Windows
build_cache.bat

# For Linux/Mac
python pretrain_models.py --cryptos top10 --epochs 100
```

### 2. Run Application
```bash
# Application will automatically use cache for faster predictions
python -m streamlit run stunning_crypto_app.py --server.port 8507
```

## Performance Benefits

| Feature | Without Cache | With Cache | Improvement |
|---------|---------------|------------|-------------|
| Model Loading | 30-60 seconds | 0.1 seconds | **300-600x faster** |
| Predictions | 10-30 seconds | 0.5 seconds | **20-60x faster** |
| Initial Load | 45-90 seconds | 2-5 seconds | **10-20x faster** |

## Cache Management

### Check Cache Status
```bash
# Windows
cache_status.bat

# Linux/Mac
python -c "from cache_loader import CacheLoader; CacheLoader().get_cache_stats()"
```

### Update Cache (Weekly Recommended)
```bash
# Windows
update_cache.bat

# Linux/Mac
python pretrain_models.py --cryptos top10 --epochs 50
```

### Cache Structure
```
model_cache/
├── models/           # Pre-trained ML models (.pth files)
├── data/            # Historical cryptocurrency data (.pkl, .json)
├── forecasts/       # Pre-computed predictions (.json)
├── scalers/         # Data preprocessing scalers (.pkl)
└── cache_manifest.json  # Cache metadata and index
```

## Advanced Usage

### Full Cache Build (All Cryptocurrencies)
```bash
python pretrain_models.py --cryptos all --epochs 100
```

### Custom Cache Directory
```bash
python pretrain_models.py --cache-dir /path/to/cache --cryptos top10
```

### Model Types
```bash
# Train specific model types
python pretrain_models.py --models AttentionLSTM LSTM --cryptos top10
```

## Production Deployment

### 1. Server Setup
```bash
# Build cache on server
python pretrain_models.py --cryptos top10 --epochs 100

# Deploy with cache
python -m streamlit run stunning_crypto_app.py --server.port 8507 --server.headless true
```

### 2. Scheduled Updates
```bash
# Add to crontab for weekly updates
0 2 * * 0 cd /path/to/app && python pretrain_models.py --cryptos top10 --epochs 50
```

### 3. Docker Deployment
```dockerfile
# Example Dockerfile with cache
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Build cache during image build (optional)
RUN python pretrain_models.py --cryptos top10 --epochs 50

EXPOSE 8507
CMD ["streamlit", "run", "stunning_crypto_app.py", "--server.port=8507", "--server.headless=true"]
```

## Cache Optimization Tips

### 1. Regular Updates
- Update cache weekly for best accuracy
- Use fewer epochs (50) for updates vs initial build (100)

### 2. Storage Management
- Cache size: ~100-500 MB for top 10 cryptos
- Cache size: ~1-5 GB for all supported cryptos

### 3. Network Deployment
- Pre-build cache on powerful local machine
- Transfer cache directory to production server
- Reduces server computational requirements

## Troubleshooting

### Cache Not Loading
1. Check if `model_cache/` directory exists
2. Verify `cache_manifest.json` is present
3. Run `python cache_loader.py` to test cache

### Performance Issues
1. Ensure cache is up-to-date (check timestamps)
2. Verify sufficient disk space for cache
3. Consider rebuilding cache if corrupted

### Memory Issues
1. Cache loads models into memory as needed
2. Reduce number of cached models if memory limited
3. Use `--cryptos top10` instead of `--cryptos all`

## Integration Details

The application automatically:
- Detects available cache on startup
- Falls back to live training if cache unavailable
- Shows cache status in sidebar
- Uses cached forecasts when fresh (<24 hours)

## Support

For issues with caching or performance optimization:
1. Check cache status with provided scripts
2. Review logs in `pretrain_models.log`
3. Rebuild cache if corruption suspected

---

**Developed by Lewis Loon | © 2025 Lewis Loon Analytics**
