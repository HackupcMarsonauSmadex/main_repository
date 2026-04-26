import streamlit as st

def render_inici():
    st.title("🏠 Inici")
    st.write("Benvingut a AdAnalyzer. Utilitza el Chatbot per començar.")
    
    # Si ja existeix un CSV, el podríem mostrar aquí com a resum
    if 'input_csv' in st.session_state:
        st.info("Tens una configuració activa carregada.")