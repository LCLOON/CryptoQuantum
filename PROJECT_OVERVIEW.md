# CryptoQuantum Terminal - Project Structure

## 🎯 Quick Start
```bash
# Method 1: One-click launch (Windows)
double-click run_app.bat

# Method 2: Python setup + launch
python setup.py
python stunning_crypto_app.py

# Method 3: Direct Streamlit
streamlit run stunning_crypto_app.py
```

## 📁 Core Files

### Main Application
- `stunning_crypto_app.py` - Main trading terminal with dual AI engines
- `requirements.txt` - All dependencies for the project

### AI Training Scripts
- `train_advanced_model.py` - Train AttentionLSTM model
- `train_model.py` - Train legacy LSTM model  
- `train_multi_crypto.py` - Train models for multiple cryptocurrencies

### Deployment & Setup
- `setup.py` - Quick setup script for local development
- `run_app.bat` - Windows one-click launcher
- `launch_advanced.bat` - Advanced launcher with dependency checking
- `deploy.py` - Streamlit Cloud deployment automation
- `test_app.py` - Application validation script

### Configuration
- `.streamlit/config.toml` - Clean Streamlit configuration
- `.gitignore` - Comprehensive Git ignore rules

### Documentation
- `README.md` - Complete project documentation
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - MIT License

## 🤖 AI Architecture

### Dual Engine System
1. **AttentionLSTM Model** (Primary)
   - Advanced attention mechanism for better pattern recognition
   - AsymmetricLoss for handling volatile crypto markets
   - Superior performance on recent data

2. **Improved Legacy LSTM** (Fallback)
   - Enhanced traditional LSTM with SmoothL1Loss
   - Stable predictions for historical patterns
   - Backup system for reliability

### Features
- 50 cryptocurrency support with real market cap data
- Real-time price predictions and trend analysis
- Professional trading terminal interface
- Advanced technical indicators and charts
- Risk management tools and disclaimers

## 🚀 Deployment Options

### Local Development
```bash
git clone <repository>
cd CryptoQuantum-Terminal
python setup.py
python stunning_crypto_app.py
```

### Streamlit Cloud
```bash
python deploy.py  # Automated deployment
```

### Manual Streamlit Cloud
1. Connect GitHub repository
2. Set main file: `stunning_crypto_app.py`
3. Python version: 3.8+
4. Dependencies auto-detected from requirements.txt

## 🔧 Technical Stack
- **Frontend**: Streamlit with custom CSS/HTML
- **Backend**: Python 3.8+
- **AI/ML**: PyTorch, NumPy, Pandas
- **Data**: yfinance API for real-time crypto data
- **Charts**: Plotly for interactive visualizations
- **Deployment**: Streamlit Cloud ready

## 📊 Performance
- Real-time data fetching and processing
- Dual AI engine redundancy for reliability
- Professional Wall Street-style interface
- Mobile responsive design
- Error handling and graceful fallbacks

## 🛡️ Security & Risk
- Comprehensive risk disclaimers
- No financial advice provided
- Educational and research purposes only
- User responsibility for trading decisions

---
**Status**: Production Ready ✅
**Last Updated**: January 2025
**Maintainer**: CryptoQuantum Terminal Team
