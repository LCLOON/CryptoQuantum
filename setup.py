#!/usr/bin/env python3
"""
CryptoQuantum Terminal Setup Script
Quick setup for local development
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.8+")
        return False

def create_models_directory():
    """Create models directory if it doesn't exist"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("📁 Created models directory")
    else:
        print("📁 Models directory already exists")

def main():
    """Main setup function"""
    print("🚀 CryptoQuantum Terminal Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create necessary directories
    create_models_directory()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Run: python stunning_crypto_app.py")
    print("2. Or use: streamlit run stunning_crypto_app.py")
    print("3. Or double-click: run_app.bat (Windows)")
    print("\n🌐 The app will open at: http://localhost:8501")

if __name__ == "__main__":
    main()
