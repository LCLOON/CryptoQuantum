import streamlit as st

# Simple redirect to main app
st.title("🚀 CryptoQuantum")
st.write("Loading crypto tracker...")

# Import and run the main app
try:
    import app
    app.main()
except Exception as e:
    st.error(f"Error loading app: {e}")
    st.write("Please check the application logs.")
