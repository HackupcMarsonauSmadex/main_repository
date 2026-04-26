import streamlit as st
import pandas as pd
import json
import os
from src.gemini_ai_service import analyze_full_campaign
from src.hack_en import run_xgboost_pipeline, FEATURES, CATEGORICAL_FEATURES, COLS_HAS

def render_chatbot():
    # 1. Configuració inicial i rutes
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'src', 'utils', 'ranges.json')
    HISTORIC_PATH = os.path.join(BASE_DIR, 'data', 'creative_summary.csv')
    DAILY_PATH    = os.path.join(BASE_DIR, 'data', 'creative_daily_country_os_stats.csv')

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    kpi_details = {
        "CPA": "🎯 **Cost Per Action**: Pagues per venda/registre.",
        "ROAS": "💰 **Return on Ad Spend**: Retorn econòmic de la inversió.",
        "CTR": "🖱️ **Click Through Rate**: Atractiu de l'anunci (clics).",
        "IPM": "🎮 **Installs Per Mille**: Eficàcia en Gaming (installs/1k).",
        "CVR": "📈 **Conversion Rate**: Eficàcia de l'embut de conversió."
    }

    if 'step' not in st.session_state:
        st.session_state.step = 'CHAT'

    # ==========================================================================
    # PAS 1: FORMULARI D'ENTRADA
    # ==========================================================================
    if st.session_state.step == 'CHAT':
        st.title("🚀 Smadex Smart Auditor")
        st.caption("Eina professional d'extracció i auditoria de campanyes AdTech.")

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
                    options=list(config['kpi_goals'].keys()),
                    index=None,
                    placeholder="Selecciona mètrica..."
                )
            with col_info:
                with st.expander("❓ Què hauria d'escollir?"):
                    for k, v in kpi_details.items():
                        st.markdown(f"- {v}")

        st.markdown("---")
        st.subheader("🖼️ Slots de Creativitats")

        c_texts, c_files = [], []
        row1 = st.columns(3)
        for i in range(3):
            with row1[i]:
                with st.container(border=True):
                    st.markdown(f"**Slot {i+1}**")
                    c_texts.append(st.text_area(f"Descripció S{i+1}", key=f"t{i}", height=80, label_visibility="collapsed", placeholder="Descripció..."))
                    c_files.append(st.file_uploader(f"Imatge {i+1}", key=f"f{i}", label_visibility="collapsed"))

        row2 = st.columns(3)
        for i in range(3, 6):
            with row2[i-3]:
                with st.container(border=True):
                    st.markdown(f"**Slot {i+1}**")
                    c_texts.append(st.text_area(f"Descripció S{i+1}", key=f"t{i}", height=80, label_visibility="collapsed", placeholder="Descripció..."))
                    c_files.append(st.file_uploader(f"Imatge {i+1}", key=f"f{i}", label_visibility="collapsed"))

        st.markdown("###")

        if st.button("🚀 ANALITZAR CAMPANYA", use_container_width=True, type="primary"):
            if not campaign_desc or kpi_choice is None:
                st.error("Si us plau, omple la descripció i el KPI.")
            else:
                with st.spinner("🤖 Extraient dades amb Gemini..."):
                    res, avis_b = analyze_full_campaign(campaign_desc, c_texts, c_files)

                    if "error" in res:
                        st.error(f"❌ Error de l'IA: {res['error']}")
                        st.stop()

                    res['campaign']['kpi_goal'] = kpi_choice
                    st.session_state.full_data = res
                    st.session_state.avis_borrat = avis_b

                    if not res['campaign'].get('vertical'):
                        st.session_state.step = 'MISSING_DATA'
                    else:
                        st.session_state.step = 'REVIEW'
                    st.rerun()

    # ==========================================================================
    # PAS 2 (OPCIONAL): VERTICAL DESCONEGUT
    # ==========================================================================
    elif st.session_state.step == 'MISSING_DATA':
        st.info("💡 L'IA necessita que li donis un cop de mà amb el vertical.")
        with st.form("fix"):
            v = st.selectbox("Vertical real:", options=config['verticals'])
            if st.form_submit_button("Confirmar"):
                st.session_state.full_data['campaign']['vertical'] = v
                for c in st.session_state.full_data['creatives']:
                    c['vertical'] = v
                st.session_state.step = 'REVIEW'
                st.rerun()

    # ==========================================================================
    # PAS 3: REVISIÓ DE L'EXTRACCIÓ DE GEMINI
    # ==========================================================================
    elif st.session_state.step == 'REVIEW':
        st.title("🔍 Resultats de l'Extracció (Gemini)")

        if st.session_state.avis_borrat:
            st.error("🚨 **COHERÈNCIA VIOLADA**: S'han detectat slots que no corresponen al vertical i s'han buidat automàticament.")

        with st.expander("📊 Dades de Campanya Extretes", expanded=True):
            st.table(pd.DataFrame([st.session_state.full_data['campaign']]))

        st.subheader("🎨 Creativitats Extretes")
        st.dataframe(pd.DataFrame(st.session_state.full_data['creatives']), use_container_width=True)

        st.divider()
        col_download, col_next, col_back = st.columns(3)

        col_download.download_button(
            "📥 Baixar Campaign.csv",
            pd.DataFrame([st.session_state.full_data['campaign']]).to_csv(index=False),
            "camp.csv",
            use_container_width=True
        )
        col_download.download_button(
            "📥 Baixar Creatives.csv",
            pd.DataFrame(st.session_state.full_data['creatives']).to_csv(index=False),
            "crea.csv",
            use_container_width=True
        )

        if col_next.button("🤖 OPTIMITZAR AMB IA →", use_container_width=True, type="primary"):
            # Carreguem el CSV històric
            if not os.path.exists(HISTORIC_PATH):
                st.error(f"❌ No s'ha trobat el CSV d'entrenament a: `{HISTORIC_PATH}`")
                st.stop()

            with st.spinner("⚙️ Entrenant model XGBoost i optimitzant creativitats..."):
                try:
                    df_historic = pd.read_csv(HISTORIC_PATH)
                    df_daily = pd.read_csv(DAILY_PATH) if os.path.exists(DAILY_PATH) else None

                    if df_daily is None:
                        st.warning("⚠️ No s'ha trobat `creative_daily_country_os_stats.csv`. El mòdul de fatiga no s'activarà.")

                    df_orig, df_opt, importancias, kpi, mse_kpi, mse_fatigue = run_xgboost_pipeline(
                        campaign_data=st.session_state.full_data['campaign'],
                        creatives_data=st.session_state.full_data['creatives'],
                        df_historic=df_historic,
                        df_daily=df_daily
                    )
                    st.session_state.df_original    = df_orig
                    st.session_state.df_optimitzat  = df_opt
                    st.session_state.importancias   = importancias
                    st.session_state.kpi_optimitzat = kpi
                    st.session_state.mse_model      = mse_kpi
                    st.session_state.mse_fatigue    = mse_fatigue
                    st.session_state.step           = 'RESULTS'
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al model XGBoost: {e}")

        if col_back.button("⬅️ Tornar", use_container_width=True):
            st.session_state.step = 'CHAT'
            st.rerun()

    elif st.session_state.step == 'RESULTS':
        kpi        = st.session_state.kpi_optimitzat
        df_or      = st.session_state.df_original
        df_op      = st.session_state.df_optimitzat
        imp        = st.session_state.importancias
        mse_kpi    = st.session_state.mse_model
        mse_fat    = st.session_state.mse_fatigue

        has_fatigue = 'Prediction_Days_to_Fatigue' in df_op.columns

        st.title(f"✨ Optimització per {kpi}")

        # MSE info
        mse_cols = st.columns(2)
        mse_cols[0].caption(f"📉 KPI Model MSE: `{mse_kpi:.6f}`")
        if mse_fat is not None:
            mse_cols[1].caption(f"⏳ Fatigue Model MSE: `{mse_fat:.4f}`")

        # --- MÈTRIQUES GLOBALS ---
        pred_col = f'Prediction_{kpi}'
        preds = df_op[pred_col]

        if has_fatigue:
            m1, m2, m3, m4 = st.columns(4)
            fat_preds = df_op['Prediction_Days_to_Fatigue']
            m4.metric("⏳ Fatiga Mitjana", f"{fat_preds.mean():.0f} dies")
        else:
            m1, m2, m3 = st.columns(3)

        m1.metric("KPI Objectiu", kpi)
        m2.metric(f"{kpi} Mitjà Predit", f"{preds.mean():.4f}")
        m3.metric("🏆 Millor Slot", f"Slot {preds.idxmax() + 1} → {preds.max():.4f}")

        st.divider()

        # --- COMPARACIÓ SLOT A SLOT ---
        st.subheader("🔍 Comparació de Canvis per Slot")
        st.caption("🟡 = camp imputat per la IA  |  ✅ = ja estava informat")

        all_cols = FEATURES + CATEGORICAL_FEATURES

        for i in range(len(df_or)):
            pred_val = df_op[pred_col].iloc[i]
            fat_label = f" | ⏳ Fatiga: {df_op['Prediction_Days_to_Fatigue'].iloc[i]} dies" if has_fatigue else ""
            with st.expander(
                f"**Slot {i+1}** — {kpi} predit: `{pred_val:.4f}`{fat_label}",
                expanded=(i == preds.idxmax())
            ):
                rows = []
                for col in all_cols:
                    val_orig = df_or[col].iloc[i] if col in df_or.columns else None
                    val_opt  = df_op[col].iloc[i]  if col in df_op.columns  else None

                    was_null = pd.isna(val_orig) or val_orig is None or str(val_orig).strip() == ''
                    changed  = was_null and not pd.isna(val_opt)

                    rows.append({
                        "Atribut":           col,
                        "Valor Original":    "—" if was_null else str(val_orig),
                        "Valor Optimitzat":  str(val_opt),
                        "Canvi IA":          "🟡 Imputat" if changed else "✅ Mantingut"
                    })

                df_comp = pd.DataFrame(rows)

                def highlight_row(row):
                    if row['Canvi IA'] == '🟡 Imputat':
                        return ['background-color: #fff8e1'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_comp.style.apply(highlight_row, axis=1),
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()

        # --- TOP ATRIBUTS MÉS INFLUENTS ---
        st.subheader("📊 Top 10 Atributs més Influents")
        top_imp = imp.head(10).copy()
        # Columnes en anglès (nom actualitzat del service)
        top_imp['Direction'] = top_imp['Correlation'].apply(
            lambda c: "⬆️ Augmenta KPI" if c > 0 else "⬇️ Redueix KPI"
        )
        st.dataframe(
            top_imp[['XGB_Column', 'Importance', 'Correlation', 'Direction']],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # --- EXPORTACIÓ ---
        st.subheader("📥 Exportar Resultats")
        c1, c2, c3, c4 = st.columns(4)

        c1.download_button(
            "📄 Creatius Originals",
            df_or.to_csv(index=False),
            "creatives_original.csv",
            use_container_width=True
        )
        c2.download_button(
            "✨ Creatius Optimitzats",
            df_op.to_csv(index=False),
            "output_creatives.csv",
            use_container_width=True
        )
        c3.download_button(
            "📊 Importàncies Model",
            imp.to_csv(index=False),
            "Import_Corr.csv",
            use_container_width=True
        )

        if c4.button("⬅️ Tornar a l'Inici", use_container_width=True):
            for key in ['df_original', 'df_optimitzat', 'importancias', 'kpi_optimitzat',
                        'mse_model', 'mse_fatigue', 'full_data', 'avis_borrat']:
                st.session_state.pop(key, None)
            st.session_state.step = 'CHAT'
            st.rerun()