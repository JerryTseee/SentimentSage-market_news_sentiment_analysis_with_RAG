"""
Deployment Guide for Financial Sentiment Analyzer

This file contains instructions and helper functions for deploying your Streamlit app online.
"""

import streamlit as st

def show_deployment_guide():
    st.title("🚀 Deployment Guide")
    
    st.markdown("""
    ## Option 1: Streamlit Cloud (Recommended - Free)
    
    ### Steps:
    1. **Push to GitHub:**
       - Create a new repository on GitHub
       - Upload your files: `streamlit_app.py`, `test.py`, `requirements.txt`
       - Make sure your repository is public
    
    2. **Deploy on Streamlit Cloud:**
       - Go to [share.streamlit.io](https://share.streamlit.io)
       - Sign in with GitHub
       - Click "New app"
       - Select your repository
       - Set main file path: `streamlit_app.py`
       - Click "Deploy"
    
    3. **Environment Variables:**
       - In Streamlit Cloud, go to app settings
       - Add secrets in the "Secrets" section:
       ```toml
       OPENAI_API_KEY = "your-openai-key"
       NEWS_API_KEY = "your-news-api-key"
