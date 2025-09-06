"""
Force Initialization System for Streamlit Cloud
Creates a guaranteed initialization that bypasses cache checks
"""

import streamlit as st
import time
from datetime import datetime
from pathlib import Path
import json
import os

def force_initialize_app():
    """
    Guaranteed initialization that works regardless of cache state
    """
    st.info("🚀 **FORCE INITIALIZING CRYPTOQUANTUM APP...**")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Create cache directory structure
        progress_bar.progress(10)
        status_text.text("📁 Creating cache directories...")
        
        cache_dir = Path("model_cache")
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "models").mkdir(exist_ok=True)
        (cache_dir / "data").mkdir(exist_ok=True)
        (cache_dir / "forecasts").mkdir(exist_ok=True)
        (cache_dir / "scalers").mkdir(exist_ok=True)
        
        # Step 2: Create cache manifest
        progress_bar.progress(20)
        status_text.text("📋 Creating cache manifest...")
        
        manifest = {
            "created_date": datetime.now().isoformat(),
            "cache_version": "5.0_force_init",
            "models": {},
            "forecasts": {},
            "ml_trained": [],
            "rule_based": [],
            "force_initialized": True
        }
        
        manifest_file = cache_dir / "cache_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Step 3: Initialize Smart ML System
        progress_bar.progress(40)
        status_text.text("🧠 Initializing Smart ML System...")
        
        from smart_ml_init import smart_ml_initialize
        result = smart_ml_initialize()
        
        progress_bar.progress(80)
        status_text.text("✅ Initialization complete!")
        
        # Step 4: Final validation
        progress_bar.progress(100)
        time.sleep(1)
        
        st.success(f"🎉 **FORCE INITIALIZATION SUCCESSFUL!**")
        st.success(f"📊 Trained Models: {len(result.get('ml_trained', []))}")
        st.success(f"📈 Rule-based Models: {len(result.get('rule_based', []))}")
        st.success(f"⏱️ Total Time: {result.get('total_time', 'Unknown')}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ **FORCE INITIALIZATION FAILED:**")
        st.error(f"Error: {str(e)}")
        st.error("Please refresh the page to try again.")
        return False

def check_if_force_init_needed():
    """
    Check if force initialization is required - Streamlit Cloud compatible
    """
    try:
        # Check 1: Session state override
        if hasattr(st, 'session_state') and getattr(st.session_state, 'force_init_needed', True):
            return True
        
        # Check 2: Cache directory and manifest
        cache_dir = Path("model_cache")
        manifest_file = cache_dir / "cache_manifest.json"
        
        # If no cache directory or manifest, force init needed
        if not cache_dir.exists() or not manifest_file.exists():
            return True
            
        # Check 3: If manifest exists but is empty, force init needed
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            
        if not manifest.get('models') and not manifest.get('forecasts'):
            return True
        
        # Check 4: Force init flag in manifest
        if manifest.get('force_initialized') != True:
            return True
            
        return False
        
    except Exception as e:
        # If any error occurs, force init is needed
        print(f"Force init check error: {e}")
        return True

def display_force_init_button():
    """
    Display force initialization button for debugging
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Debug Tools")
    
    if st.sidebar.button("🚀 FORCE INITIALIZE", type="primary"):
        st.sidebar.info("Force initialization started...")
        force_initialize_app()
        st.rerun()
    
    # Show cache status
    try:
        cache_dir = Path("model_cache")
        manifest_file = cache_dir / "cache_manifest.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            st.sidebar.text(f"Cache Version: {manifest.get('cache_version', 'Unknown')}")
            st.sidebar.text(f"Models: {len(manifest.get('models', {}))}")
            st.sidebar.text(f"Forecasts: {len(manifest.get('forecasts', {}))}")
        else:
            st.sidebar.text("❌ No cache manifest found")
            
    except Exception as e:
        st.sidebar.text(f"❌ Cache error: {str(e)}")
