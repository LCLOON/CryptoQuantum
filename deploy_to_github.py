#!/usr/bin/env python3
"""
GitHub Deployment Script for CryptoQuantum
Prepares and commits the enhanced mobile crypto app for deployment
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {str(e)}")
        return False

def prepare_deployment():
    """Prepare the repository for deployment"""
    print("🚀 Preparing CryptoQuantum for GitHub Deployment")
    print("=" * 60)
    
    # 1. Copy mobile app as the main app
    print("📱 Setting up mobile app as primary application...")
    if os.path.exists("mobile_crypto_app.py"):
        if os.path.exists("app.py"):
            os.remove("app.py")
        os.rename("mobile_crypto_app.py", "app.py")
        print("✅ Mobile app set as main app.py")
    
    # 2. Update README
    print("📝 Updating README...")
    if os.path.exists("README_NEW.md"):
        if os.path.exists("README.md"):
            os.rename("README.md", "README_OLD.md")
        os.rename("README_NEW.md", "README.md")
        print("✅ README updated")
    
    # 3. Clean up unnecessary files
    cleanup_files = [
        "diagnose_app_issues.py",
        "test_final_fixes.py",
        "train_missing_cryptos.py",
        "README_OLD.md"
    ]
    
    print("🧹 Cleaning up development files...")
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   Removed: {file}")
    
    print("✅ Repository prepared for deployment")

def commit_and_push():
    """Commit changes and push to GitHub"""
    print("\n📤 Committing and pushing to GitHub...")
    
    # Add all changes
    if not run_command("git add .", "Adding all changes"):
        return False
    
    # Create commit message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_message = f"🚀 Deploy Enhanced Mobile Crypto App - Professional Trading Platform ({timestamp})\n\n✨ Features:\n- Professional candlestick charts with volume analysis\n- Mobile-optimized responsive design\n- AI-powered predictions with 47 pre-trained models\n- Technical indicators (RSI, MA20, MA50)\n- Real-time price data and performance tracking\n- Ultra-fast caching system\n\n🔧 Fixes:\n- Resolved price display issues (NaN errors)\n- Fixed volume calculations and chart rendering\n- Corrected total return calculations\n- Enhanced error handling and data validation\n\n📱 Mobile-First Design:\n- Optimized for iPhone and mobile trading\n- Professional trading platform interface\n- Interactive charts with zoom and pan\n- Responsive layout for all screen sizes"
    
    # Commit changes
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        return False
    
    # Push to GitHub
    if not run_command("git push origin main", "Pushing to GitHub"):
        return False
    
    print("✅ Successfully deployed to GitHub!")
    return True

def create_streamlit_config():
    """Create Streamlit configuration for deployment"""
    config_content = """[general]
email = ""

[server]
headless = true
enableCORS = false
port = $PORT

[theme]
primaryColor = "#1e3a8a"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    # Create .streamlit directory
    os.makedirs(".streamlit", exist_ok=True)
    
    # Write config file
    with open(".streamlit/config.toml", "w") as f:
        f.write(config_content)
    
    print("✅ Streamlit configuration created")

def create_deployment_files():
    """Create necessary deployment files"""
    
    # Create Procfile for Heroku deployment
    procfile_content = "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    # Create runtime.txt for Python version
    runtime_content = "python-3.9.18"
    with open("runtime.txt", "w") as f:
        f.write(runtime_content)
    
    print("✅ Deployment files created (Procfile, runtime.txt)")

def main():
    """Main deployment function"""
    try:
        # Check if we're in a git repository
        if not os.path.exists(".git"):
            print("❌ Not in a git repository. Please run this from the CryptoQuantum directory.")
            return
        
        # Prepare for deployment
        prepare_deployment()
        
        # Create deployment configuration
        create_streamlit_config()
        create_deployment_files()
        
        # Get user confirmation
        print("\n🎯 Ready to deploy to GitHub!")
        print("This will:")
        print("1. Set mobile_crypto_app.py as the main app.py")
        print("2. Update README with mobile-first documentation")
        print("3. Commit and push all changes to GitHub")
        print("4. Make the repository ready for Streamlit Cloud deployment")
        
        response = input("\nProceed with deployment? (y/N): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            if commit_and_push():
                print("\n🎉 Deployment Successful!")
                print("=" * 60)
                print("📱 Your enhanced mobile crypto app is now on GitHub!")
                print("🌐 Repository: https://github.com/LCLOON/CryptoQuantum")
                print("\n🚀 Next Steps:")
                print("1. Visit https://share.streamlit.io")
                print("2. Connect your GitHub repository")
                print("3. Deploy app.py (your mobile crypto app)")
                print("4. Share your professional crypto trading platform!")
                print("\n✨ Features deployed:")
                print("- Professional candlestick charts")
                print("- Mobile-optimized interface")
                print("- AI-powered predictions")
                print("- Technical indicators")
                print("- Real-time data")
            else:
                print("\n❌ Deployment failed. Please check the errors above.")
        else:
            print("\n⏸️ Deployment cancelled.")
    
    except Exception as e:
        print(f"\n❌ Deployment failed with error: {str(e)}")

if __name__ == "__main__":
    main()
