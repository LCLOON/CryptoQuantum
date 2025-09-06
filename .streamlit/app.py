# This file tells Streamlit Cloud which app to run
# Redirect to main app.py
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from app import *
