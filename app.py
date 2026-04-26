import streamlit as st
from vistes.chatbot import render_chatbot

st.set_page_config(page_title="Smadex Hackathon", layout="wide")

# Lògica simple de navegació
if 'page' not in st.session_state:
    st.session_state.page = 'HOME'

if st.session_state.page == 'HOME':
    st.title("Benvingut a Smadex Ad-Gen")
    if st.button("Començar Nova Campanya"):
        st.session_state.page = 'CHATBOT'
        st.rerun()
else:
    render_chatbot()