"""
DOGE Ultra-Strict Final Fix
Makes DOGE constraints even more conservative
"""

import sys

def apply_ultra_strict_doge_fix():
    """Apply ultra-strict DOGE validation constraints"""
    
    app_file = r"c:\Users\lcloo\OneDrive\Desktop\CryptoQuantum-Clone\stunning_crypto_app.py"
    
    # Read the current file
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the meme coin constraints with even stricter ones
    old_doge_constraints = '''        # Special ultra-conservative handling for meme coins
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
                min_annual_growth = -0.30  # -30% max decline'''
    
    new_doge_constraints = '''        # Special ultra-strict handling for meme coins (DOGE/SHIB)
        if symbol in ["DOGE-USD", "SHIB-USD"]:
            # Ultra-strict for meme coins - more realistic expectations
            if days_elapsed <= 30:
                max_annual_growth = 1.2  # 120% max for meme coins short-term
                min_annual_growth = -0.70  # -70% max decline
            elif days_elapsed <= 90:
                max_annual_growth = 0.65  # 65% max for meme coins 3 months
                min_annual_growth = -0.60  # -60% max decline
            elif days_elapsed <= 365:
                max_annual_growth = 0.3  # 30% max for meme coins 1 year
                min_annual_growth = -0.50  # -50% max decline
            elif days_elapsed <= 730:  # 2 years
                max_annual_growth = 0.2  # 20% max for meme coins 2+ years
                min_annual_growth = -0.40  # -40% max decline
            else:  # 3+ years
                max_annual_growth = 0.12  # 12% max for meme coins long-term
                min_annual_growth = -0.30  # -30% max decline'''
    
    # Count occurrences 
    occurrences = content.count(old_doge_constraints)
    print(f"Found {occurrences} DOGE constraint sections to update...")
    
    if occurrences > 0:
        # Replace all occurrences
        content = content.replace(old_doge_constraints, new_doge_constraints)
        
        # Write back to file
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated {occurrences} DOGE constraint sections with ultra-strict limits")
        print("📉 New DOGE/SHIB limits:")
        print("   • 30 days: 120% annual (was 150%)")
        print("   • 90 days: 65% annual (was 80%)")
        print("   • 1 year: 30% annual (was 40%)")
        print("   • 2 years: 20% annual (was 25%)")
        print("   • 3+ years: 12% annual (was 15%)")
    else:
        print("❌ Could not find DOGE constraint sections to update")
        return False
    
    return True

if __name__ == "__main__":
    success = apply_ultra_strict_doge_fix()
    if success:
        print("\n🎯 ULTRA-STRICT DOGE CONSTRAINTS APPLIED!")
        print("🔥 DOGE predictions are now extremely conservative")
        print("✨ This should fix the 'Doge dont look right at all' issue")
        print("🌐 Refresh the app at: http://localhost:8503")
    else:
        print("\n❌ Failed to apply ultra-strict DOGE fix")
        sys.exit(1)
