import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Fons blanc pur */
        .stApp { background-color: white !important; }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F8F9FA !important;
            border-right: 1px solid #EEE !important;
        }

        /* Estil per als botons de navegació del radio */
        div[role="radiogroup"] label {
            background-color: transparent !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin-bottom: 5px !important;
        }

        div[role="radiogroup"] label:has(input:checked) {
            background-color: #EDE9FE !important;
            color: #6D28D9 !important;
        }
        </style>
    """, unsafe_allow_html=True)