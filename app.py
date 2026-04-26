import streamlit as st
from vistes.chatbot import render_chatbot
from src.styles import apply_custom_css

st.set_page_config(page_title="Smadex Hackathon", layout="wide")

# Apply custom styles
apply_custom_css()

# Simple navigation logic
if 'page' not in st.session_state:
    st.session_state.page = 'HOME'

if st.session_state.page == 'HOME':
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.markdown("## Campaign Analysis Platform")
        st.markdown("""
        **Intelligent Ad Campaign Analyzer**
        
        Advanced data extraction and optimization powered by machine learning.
        """)
        
        st.markdown("### Key Features:")
        st.markdown("""
        - Campaign data extraction and analysis
        - Creative asset evaluation
        - Performance optimization
        - Data-driven insights
        """)
        
        st.info("Begin by clicking the button to start analyzing your campaigns.")
        
    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        if st.button("Start Analysis", use_container_width=True, type="primary", key="start_btn"):
            st.session_state.page = 'CHATBOT'
            st.rerun()
else:
    render_chatbot()