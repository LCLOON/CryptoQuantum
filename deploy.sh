#!/bin/bash
# Deployment script for CryptoQuantum Terminal
echo "🚀 Starting CryptoQuantum Terminal deployment..."

# Check Python version
python --version

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run the app
echo "🎯 Starting Streamlit application..."
streamlit run stunning_crypto_app.py
