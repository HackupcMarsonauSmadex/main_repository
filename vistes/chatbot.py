import streamlit as st
import pandas as pd
import json
import os
from src.gemini_ai_service import analyze_full_campaign

def render_chatbot():
    # 1. Configuració inicial i rutes
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'src', 'utils', 'ranges.json')
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Definició de KPIs
    kpi_details = {
        "CPA": "🎯 **Cost Per Action**: Pagues per venda/registre.",
        "ROAS": "💰 **Return on Ad Spend**: Retorn econòmic de la inversió.",
        "CTR": "🖱️ **Click Through Rate**: Atractiu de l'anunci (clics).",
        "IPM": "🎮 **Installs Per Mille**: Eficàcia en Gaming (installs/1k).",
        "CVR": "📈 **Conversion Rate**: Eficàcia de l'embut de conversió."
    }

    if 'step' not in st.session_state:
        st.session_state.step = 'CHAT'

    if st.session_state.step == 'CHAT':
        st.title("🚀 Smadex Smart Auditor")
        st.caption("Eina professional d'extracció i auditoria de campanyes AdTech.")
        
        # --- SECCIÓ 1: CAMPANYA ---
        with st.container(border=True):
            st.subheader("📋 Configuració Global")
            campaign_desc = st.text_area(
                "Context de la campanya:",
                placeholder="Escriu aquí el vertical i l'estratègia...",
                height=100,
                label_visibility="collapsed"
            )
            
            col_kpi, col_info = st.columns([1, 1])
            
            with col_kpi:
                kpi_choice = st.selectbox(
                    "🎯 Escull el KPI objectiu:",
                    options=config['kpi_goals'],
                    index=None,
                    placeholder="Selecciona mètrica..."
                )
            
            with col_info:
                # La info només s'ensenya si l'usuari vol (Expander discret)
                with st.expander("❓ Què hauria d'escollir?"):
                    for k, v in kpi_details.items():
                        st.markdown(f"- {v}")

        st.markdown("---")

        # --- SECCIÓ 2: CREATIVES (HORITZONTALS) ---
        st.subheader("🖼️ Slots de Creativitats")
        
        c_texts, c_files = [], []
        
        # Creem 3 columnes per a la primera fila d'slots
        row1 = st.columns(3)
        for i in range(3):
            with row1[i]:
                with st.container(border=True):
                    st.markdown(f"**Slot {i+1}**")
                    c_texts.append(st.text_area(f"Descripció S{i+1}", key=f"t{i}", height=80, label_visibility="collapsed", placeholder="Descripció..."))
                    c_files.append(st.file_uploader(f"Imatge {i+1}", key=f"f{i}", label_visibility="collapsed"))

        # Creem 3 columnes per a la segona fila d'slots
        row2 = st.columns(3)
        for i in range(3, 6):
            with row2[i-3]:
                with st.container(border=True):
                    st.markdown(f"**Slot {i+1}**")
                    c_texts.append(st.text_area(f"Descripció S{i+1}", key=f"t{i}", height=80, label_visibility="collapsed", placeholder="Descripció..."))
                    c_files.append(st.file_uploader(f"Imatge {i+1}", key=f"f{i}", label_visibility="collapsed"))

        st.markdown("###") # Espaiat
        
        if st.button("🚀 ANALITZAR CAMPANYA", use_container_width=True, type="primary"):
            if not campaign_desc or kpi_choice is None:
                st.error("Si us plau, omple la descripció i el KPI.")
            else:
                with st.spinner("Auditant coherència..."):
                    res, avis_b = analyze_full_campaign(campaign_desc, c_texts, c_files)
                    res['campaign']['kpi_goal'] = kpi_choice
                    st.session_state.full_data = res
                    st.session_state.avis_borrat = avis_b
                    
                    if not res['campaign'].get('vertical'):
                        st.session_state.step = 'MISSING_DATA'
                    else:
                        st.session_state.step = 'REVIEW'
                    st.rerun()

    # --- PANTALLA MISSING_DATA ---
    elif st.session_state.step == 'MISSING_DATA':
        st.info("💡 L'IA necessita que li donis un cop de mà amb el vertical.")
        with st.form("fix"):
            v = st.selectbox("Vertical real:", options=config['verticals'])
            if st.form_submit_button("Confirmar"):
                st.session_state.full_data['campaign']['vertical'] = v
                for c in st.session_state.full_data['creatives']: c['vertical'] = v
                st.session_state.step = 'REVIEW'
                st.rerun()

    # --- PANTALLA REVIEW ---
    elif st.session_state.step == 'REVIEW':
        st.title("🔍 Resultats de l'Anàlisi")
        
        if st.session_state.avis_borrat:
            st.error("🚨 **COHERÈNCIA VIOLADA**: S'han detectat slots que no corresponen al vertical i s'han buidat automàticament.")

        with st.expander("📊 Veure Dades de Campanya", expanded=True):
            st.table(pd.DataFrame([st.session_state.full_data['campaign']]))
        
        st.subheader("🎨 Creativitats Extractades")
        st.dataframe(pd.DataFrame(st.session_state.full_data['creatives']), use_container_width=True)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.download_button("📥 Baixar Campaign.csv", pd.DataFrame([st.session_state.full_data['campaign']]).to_csv(index=False), "camp.csv", use_container_width=True)
        c2.download_button("📥 Baixar Creatives.csv", pd.DataFrame(st.session_state.full_data['creatives']).to_csv(index=False), "crea.csv", use_container_width=True)
        if c3.button("⬅️ Tornar", use_container_width=True):
            st.session_state.step = 'CHAT'
            st.rerun()