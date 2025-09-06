"""
DOGE-Specific Ultra-Conservative Validation Fix
Implements meme coin specific ultra-conservative constraints
"""

import sys
import os

def apply_doge_fix():
    """Apply DOGE-specific ultra-conservative validation"""
    
    app_file = r"c:\Users\lcloo\OneDrive\Desktop\CryptoQuantum-Clone\stunning_crypto_app.py"
    
    # Read the current file
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the ultra-conservative validation replacement
    old_validation = '''                        # ENHANCED PROGRESSIVE CONSTRAINTS (Much Stricter)
                        if days_elapsed <= 30:
                            max_annual_growth = 5.0  # 500% max (reduced from 800%)
                            min_annual_growth = -0.90  # -90% max decline
                        elif days_elapsed <= 90:
                            max_annual_growth = 3.0  # 300% max (reduced from 600%)
                            min_annual_growth = -0.80  # -80% max decline
                        elif days_elapsed <= 365:
                            max_annual_growth = 1.5  # 150% max (reduced from 300%)
                            min_annual_growth = -0.70  # -70% max decline
                        elif days_elapsed <= 730:  # 2 years
                            max_annual_growth = 0.8  # 80% max for 2+ years
                            min_annual_growth = -0.60  # -60% max decline
                        else:  # 3+ years (Most Conservative)
                            max_annual_growth = 0.5  # 50% max for long-term
                            min_annual_growth = -0.50  # -50% max decline'''
    
    new_validation = '''                        # ENHANCED PROGRESSIVE CONSTRAINTS (Much Stricter)
                        # Special ultra-conservative handling for meme coins
                        if symbol in ["DOGE-USD", "SHIB-USD"]:
                            # Ultra-conservative for meme coins
                            if days_elapsed <= 30:
                                max_annual_growth = 1.5  # 150% max for meme coins short-term
                                min_annual_growth = -0.70  # -70% max decline
                            elif days_elapsed <= 90:
                                max_annual_growth = 0.8  # 80% max for meme coins 3 months
                                min_annual_growth = -0.60  # -60% max decline
                            elif days_elapsed <= 365:
                                max_annual_growth = 0.4  # 40% max for meme coins 1 year
                                min_annual_growth = -0.50  # -50% max decline
                            elif days_elapsed <= 730:  # 2 years
                                max_annual_growth = 0.25  # 25% max for meme coins 2+ years
                                min_annual_growth = -0.40  # -40% max decline
                            else:  # 3+ years
                                max_annual_growth = 0.15  # 15% max for meme coins long-term
                                min_annual_growth = -0.30  # -30% max decline
                        else:
                            # Standard constraints for other cryptocurrencies
                            if days_elapsed <= 30:
                                max_annual_growth = 5.0  # 500% max (reduced from 800%)
                                min_annual_growth = -0.90  # -90% max decline
                            elif days_elapsed <= 90:
                                max_annual_growth = 3.0  # 300% max (reduced from 600%)
                                min_annual_growth = -0.80  # -80% max decline
                            elif days_elapsed <= 365:
                                max_annual_growth = 1.5  # 150% max (reduced from 300%)
                                min_annual_growth = -0.70  # -70% max decline
                            elif days_elapsed <= 730:  # 2 years
                                max_annual_growth = 0.8  # 80% max for 2+ years
                                min_annual_growth = -0.60  # -60% max decline
                            else:  # 3+ years (Most Conservative)
                                max_annual_growth = 0.5  # 50% max for long-term
                                min_annual_growth = -0.50  # -50% max decline'''
    
    # Count occurrences to replace all three validation sections
    occurrences = content.count(old_validation)
    print(f"Found {occurrences} validation sections to update...")
    
    if occurrences > 0:
        # Replace all occurrences
        content = content.replace(old_validation, new_validation)
        
        # Write back to file
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated {occurrences} validation sections with DOGE-specific ultra-conservative constraints")
        print("✅ DOGE/SHIB now limited to: 150% (30d), 80% (90d), 40% (1y), 25% (2y), 15% (3y+)")
    else:
        print("❌ Could not find validation sections to update")
        return False
    
    return True

if __name__ == "__main__":
    success = apply_doge_fix()
    if success:
        print("\n🚀 DOGE-specific validation fix applied successfully!")
        print("Now testing DOGE predictions in the application...")
    else:
        print("\n❌ Failed to apply DOGE fix")
        sys.exit(1)
