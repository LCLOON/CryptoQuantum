# -*- coding: utf-8 -*-
"""
CryptoQuantum - Streamlit Cloud Entry Point
Mobile-optimized cryptocurrency predictions app
"""

try:
    # Import and execute the main mobile app directly
    from app import main
    
    # Execute the main app (Streamlit executes the entire file)
    main()
    
except ImportError as e:
    import streamlit as st
    st.error(f"Import Error: {e}")
    st.info("Please ensure all dependencies are properly installed.")
    st.code("pip install -r requirements.txt")
    
except Exception as e:
    import streamlit as st
    st.error(f"Runtime Error: {e}")
    st.write("**CryptoQuantum Mobile App**")
    st.write("Professional cryptocurrency analysis and AI predictions")
    st.write("Optimized for mobile devices")
    st.info("Please refresh the page or check the deployment logs.")

# Entry point for Streamlit Cloud deployment
