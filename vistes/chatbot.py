import streamlit as st
import pandas as pd
import json
import os
from src.gemini_ai_service import analyze_full_campaign
from src.hack_en import run_xgboost_pipeline, FEATURES, CATEGORICAL_FEATURES, COLS_HAS

def render_chatbot():
    # Initial configuration and paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'src', 'utils', 'ranges.json')
    HISTORIC_PATH = os.path.join(BASE_DIR, 'data', 'creative_summary.csv')
    DAILY_PATH    = os.path.join(BASE_DIR, 'data', 'creative_daily_country_os_stats.csv')

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    kpi_details = {
        "CPA": "**Cost Per Action**: You pay per sale/registration.",
        "ROAS": "**Return on Ad Spend**: Economic return on investment.",
        "CTR": "**Click Through Rate**: Ad attractiveness (clicks).",
        "IPM": "**Installs Per Mille**: Gaming effectiveness (installs/1k).",
        "CVR": "**Conversion Rate**: Conversion funnel effectiveness."
    }

    if 'step' not in st.session_state:
        st.session_state.step = 'CHAT'

    # ==========================================================================
    # STEP 1: INPUT FORM
    # ==========================================================================
    if st.session_state.step == 'CHAT':
        st.title("Campaign Analysis")
        st.caption("Professional extraction and auditing of AdTech campaigns.")
        
        st.markdown("---")

        with st.container(border=True):
            st.subheader("Campaign Configuration")
            st.markdown("Provide campaign details and select your optimization target.")
            
            campaign_desc = st.text_area(
                "Campaign context:",
                placeholder="Describe the vertical and strategy...",
                height=100,
                label_visibility="collapsed"
            )
            col_kpi, col_info = st.columns([1, 1])
            with col_kpi:
                kpi_choice = st.selectbox(
                    "Select the target KPI:",
                    options=list(config['kpi_goals'].keys()),
                    index=None,
                    placeholder="Choose a metric..."
                )
            with col_info:
                with st.expander("KPI Definitions"):
                    for k, v in kpi_details.items():
                        st.markdown(f"**{k}**: {v}")

        st.markdown("---")
        st.subheader("Creative Assets")
        st.markdown("Upload up to 6 creative variations for analysis.")

        c_texts, c_files = [], []
        row1 = st.columns(3)
        for i in range(3):
            with row1[i]:
                with st.container(border=True):
                    st.markdown(f"**Slot {i+1}**")
                    c_texts.append(st.text_area(f"Description {i+1}", key=f"t{i}", height=80, label_visibility="collapsed", placeholder="Enter description..."))
                    c_files.append(st.file_uploader(f"Image {i+1}", key=f"f{i}", label_visibility="collapsed"))

        row2 = st.columns(3)
        for i in range(3, 6):
            with row2[i-3]:
                with st.container(border=True):
                    st.markdown(f"**Slot {i+1}**")
                    c_texts.append(st.text_area(f"Description {i+1}", key=f"t{i}", height=80, label_visibility="collapsed", placeholder="Enter description..."))
                    c_files.append(st.file_uploader(f"Image {i+1}", key=f"f{i}", label_visibility="collapsed"))

        st.markdown("###")

        if st.button("Analyze Campaign", use_container_width=True, type="primary"):
            if not campaign_desc or kpi_choice is None:
                st.error("Please complete all required fields.")
            else:
                with st.spinner("Processing data..."):
                    res, avis_b = analyze_full_campaign(campaign_desc, c_texts, c_files)

                    if "error" in res:
                        st.error(f"Error: {res['error']}")
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
    # STEP 2 (OPTIONAL): UNKNOWN VERTICAL
    # ==========================================================================
    elif st.session_state.step == 'MISSING_DATA':
        st.info("Please confirm the campaign vertical.")
        with st.form("fix"):
            v = st.selectbox("Campaign vertical:", options=config['verticals'])
            if st.form_submit_button("Confirm"):
                st.session_state.full_data['campaign']['vertical'] = v
                for c in st.session_state.full_data['creatives']:
                    c['vertical'] = v
                st.session_state.step = 'REVIEW'
                st.rerun()

    # ==========================================================================
    # STEP 3: REVIEW OF EXTRACTION RESULTS
    # ==========================================================================
    elif st.session_state.step == 'REVIEW':
        st.title("Analysis Results")
        st.markdown("---")

        if st.session_state.avis_borrat:
            st.error("Validation Alert: Some assets were removed due to vertical inconsistency.")

        with st.expander("Campaign Data", expanded=True):
            st.table(pd.DataFrame([st.session_state.full_data['campaign']]))

        st.subheader("Extracted Creatives")
        st.dataframe(pd.DataFrame(st.session_state.full_data['creatives']), use_container_width=True)

        st.markdown("---")
        col_download, col_next, col_back = st.columns(3)

        col_download.download_button(
            "Download Campaign Data",
            pd.DataFrame([st.session_state.full_data['campaign']]).to_csv(index=False),
            "campaign_data.csv",
            use_container_width=True
        )
        col_download.download_button(
            "Download Creatives Data",
            pd.DataFrame(st.session_state.full_data['creatives']).to_csv(index=False),
            "creatives_data.csv",
            use_container_width=True
        )

        if col_next.button("Optimize Campaign", use_container_width=True, type="primary"):
            # Load historic CSV
            if not os.path.exists(HISTORIC_PATH):
                st.error(f"Training data not found at: {HISTORIC_PATH}")
                st.stop()

            with st.spinner("Processing optimization..."):
                try:
                    df_historic = pd.read_csv(HISTORIC_PATH)
                    df_daily = pd.read_csv(DAILY_PATH) if os.path.exists(DAILY_PATH) else None

                    if df_daily is None:
                        st.warning("Daily performance data not found. Proceeding with available data.")

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
                    st.error(f"Optimization error: {e}")

        if col_back.button("Back", use_container_width=True):
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

        st.title(f"Optimization Results: {kpi}")

        # Campaign info at the top
        campaign_data = st.session_state.full_data.get('campaign', {})
        with st.expander("Campaign Configuration", expanded=True):
            st.table(pd.DataFrame([campaign_data]))

        # Model metrics
        mse_cols = st.columns(2)
        mse_cols[0].caption(f"Model MSE: {mse_kpi:.6f}")
        if mse_fat is not None:
            mse_cols[1].caption(f"Fatigue MSE: {mse_fat:.4f}")

        # --- GLOBAL METRICS ---
        pred_col = f'Prediction_{kpi}'
        preds = df_op[pred_col]

        if has_fatigue:
            m1, m2, m3, m4 = st.columns(4)
            fat_preds = df_op['Prediction_Days_to_Fatigue']
            m4.metric("Average Fatigue (days)", f"{fat_preds.mean():.0f}")
        else:
            m1, m2, m3 = st.columns(3)

        m1.metric("Target KPI", kpi)
        m2.metric(f"Predicted {kpi} (avg)", f"{preds.mean():.4f}")
        m3.metric("Best Performing Slot", f"Slot {preds.idxmax() + 1}")

        st.markdown("---")

        # Slot-by-slot comparison
        st.subheader("Creative Performance Analysis")
        st.caption("Comparison of original and optimized values")

        all_cols = FEATURES + CATEGORICAL_FEATURES

        for i in range(len(df_or)):
            pred_val = df_op[pred_col].iloc[i]
            fat_label = f" | Fatigue: {df_op['Prediction_Days_to_Fatigue'].iloc[i]:.0f} days" if has_fatigue else ""
            with st.expander(
                f"Slot {i+1} - Predicted {kpi}: {pred_val:.4f}{fat_label}",
                expanded=(i == preds.idxmax())
            ):
                rows = []
                for col in all_cols:
                    val_orig = df_or[col].iloc[i] if col in df_or.columns else None
                    val_opt  = df_op[col].iloc[i]  if col in df_op.columns  else None

                    was_null = pd.isna(val_orig) or val_orig is None or str(val_orig).strip() == ''
                    changed  = was_null and not pd.isna(val_opt)

                    rows.append({
                        "Attribute":           col,
                        "Original":           "—" if was_null else str(val_orig),
                        "Optimized":          str(val_opt),
                        "Status":             "Imputed" if changed else "Maintained"
                    })

                df_comp = pd.DataFrame(rows)

                def highlight_row(row):
                    if row['Status'] == 'Imputed':
                        return ['background-color: #fff8e1'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_comp.style.apply(highlight_row, axis=1),
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown("---")

        # Top influential attributes
        st.subheader("Key Performance Drivers")
        top_imp = imp.head(10).copy()
        top_imp['Impact Direction'] = top_imp['Correlation'].apply(
            lambda c: "Increases KPI" if c > 0 else "Reduces KPI"
        )
        st.dataframe(
            top_imp[['XGB_Column', 'Importance', 'Correlation', 'Impact Direction']],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Export section
        st.subheader("Export Results")
        c1, c2, c3, c4 = st.columns(4)

        c1.download_button(
            "Original Data",
            df_or.to_csv(index=False),
            "creatives_original.csv",
            use_container_width=True
        )
        c2.download_button(
            "Optimized Data",
            df_op.to_csv(index=False),
            "creatives_optimized.csv",
            use_container_width=True
        )
        c3.download_button(
            "Performance Drivers",
            imp.to_csv(index=False),
            "importance_analysis.csv",
            use_container_width=True
        )

        if c4.button("Return to Start", use_container_width=True):
            for key in ['df_original', 'df_optimitzat', 'importancias', 'kpi_optimitzat',
                        'mse_model', 'mse_fatigue', 'full_data', 'avis_borrat']:
                st.session_state.pop(key, None)
            st.session_state.step = 'CHAT'
            st.rerun()