"""
CryptoQuantum Terminal - Deployment Script
Automated deployment and health check system
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'torch', 'numpy', 'pandas', 
        'yfinance', 'scikit-learn', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {missing_packages}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
    else:
        print("✅ All dependencies satisfied!")

def health_check():
    """Perform basic health checks"""
    print("\n🏥 Running health checks...")
    
    # Check if app file exists
    if os.path.exists("stunning_crypto_app.py"):
        print("✅ App file found")
    else:
        print("❌ App file missing!")
        return False
    
    # Check if config exists
    if os.path.exists(".streamlit/config.toml"):
        print("✅ Config file found")
    else:
        print("⚠️ Config file missing - creating default...")
        os.makedirs(".streamlit", exist_ok=True)
        with open(".streamlit/config.toml", "w") as f:
            f.write("""[server]
enableCORS = true
enableXsrfProtection = false
headless = true
port = 8501

[theme]
primaryColor = "#00ff88"
backgroundColor = "#0a0e1a"
secondaryBackgroundColor = "#1a202c"
textColor = "#ffffff"
font = "monospace"

[browser]
gatherUsageStats = false""")
    
    return True

def deploy():
    """Deploy the CryptoQuantum Terminal"""
    print("🚀 Deploying CryptoQuantum Terminal...")
    
    if not health_check():
        print("❌ Health check failed. Cannot deploy.")
        return
    
    check_dependencies()
    
    print("\n🎯 Starting Streamlit application...")
    print("📊 Access your CryptoQuantum Terminal at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "stunning_crypto_app.py",
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")

if __name__ == "__main__":
    deploy()
