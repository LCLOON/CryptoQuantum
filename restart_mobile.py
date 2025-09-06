"""
Quick script to restart the mobile app after training completion
"""
import subprocess
import time
import sys

def restart_mobile_app():
    """Restart the mobile crypto app"""
    print("🚀 Restarting CryptoQuantum Mobile App...")
    
    try:
        # Kill any existing streamlit processes
        subprocess.run("taskkill /f /im streamlit.exe 2>nul", shell=True, capture_output=True)
        time.sleep(2)
        
        # Start the mobile app
        print("📱 Starting mobile app on port 8507...")
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "mobile_crypto_app.py", "--server.port", "8507"
        ])
        
        print("✅ Mobile app restarted!")
        print("🌐 Access at: http://localhost:8507")
        
    except Exception as e:
        print(f"❌ Error restarting app: {e}")

if __name__ == "__main__":
    restart_mobile_app()
