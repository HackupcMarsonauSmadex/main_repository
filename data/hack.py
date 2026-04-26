import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================================================================
# ATRIBUTS DEL MODEL
# ==============================================================================
FEATURES = [
    'total_days_active', 'total_spend_usd', 'area', 'duration_sec',
    'text_density', 'copy_length_chars', 'faces_count', 'product_count'
]

CATEGORICAL_FEATURES = [
    'vertical', 'format', 'language', 'theme', 'hook_type', 'dominant_color',
    'emotional_tone', 'advertiser_name', 'app_name', 'cta_text', 'headline',
    'subhead', 'has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style'
]

MAPA_TARGETS = {
    'CTR': 'overall_ctr',
    'CVR': 'overall_cvr',
    'IPM': 'overall_ipm',
    'ROAS': 'overall_roas',
    'CPA': 'overall_cpa'
}

COLS_HAS = ['has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style']


# ==============================================================================
# MOTOR D'IMPUTACIÓ INTEL·LIGENT
# ==============================================================================

def get_interval(nom_atribut, df_import_corr, df_sencer, kpi_solicitado):
    if nom_atribut in CATEGORICAL_FEATURES:
        prefix = nom_atribut + "_"
        df_filtre = df_import_corr[df_import_corr['Columna_XGB'].str.startswith(prefix, na=False)].copy()
        if df_filtre.empty:
            return []
        if kpi_solicitado == 'CPA':
            df_filtre['Score'] = df_filtre['Importancia'] * (df_filtre['Correlacion'] * -1)
        else:
            df_filtre['Score'] = df_filtre['Importancia'] * df_filtre['Correlacion']
        df_filtre = df_filtre.sort_values(by='Score', ascending=False)
        return df_filtre.head(5)['Columna_XGB'].apply(lambda x: x[len(prefix):]).tolist()

    elif nom_atribut in FEATURES:
        fila = df_import_corr[df_import_corr['Columna_XGB'] == nom_atribut]
        corr_original = fila['Correlacion'].values[0] if not fila.empty else 0
        corr_ajustada = corr_original * -1 if kpi_solicitado == 'CPA' else corr_original

        dades_reals = df_sencer[nom_atribut].dropna()
        if dades_reals.empty:
            return [0, 0, 0, 0, 0]

        minim, maxim = dades_reals.min(), dades_reals.max()
        q25, q75 = dades_reals.quantile(0.25), dades_reals.quantile(0.75)
        rang_min, rang_max = (q75, maxim) if corr_ajustada > 0 else (minim, q25)

        valors_generats = []
        es_enter = pd.api.types.is_integer_dtype(dades_reals)
        for _ in range(5):
            if es_enter:
                valors_generats.append(random.randint(int(rang_min), int(rang_max)))
            else:
                valors_generats.append(round(random.uniform(rang_min, rang_max), 2))
        return valors_generats

    return []


def omplir_forats_inteligentment(df_entrada, importancias_brutas, X, kpi_solicitado):
    df_omplir = df_entrada.copy()

    for col in df_omplir.columns:
        if col not in FEATURES and col not in CATEGORICAL_FEATURES:
            continue

        nans_idx = df_omplir[df_omplir[col].isna()].index.tolist()
        if not nans_idx:
            continue

        possibles_solucions = get_interval(col, importancias_brutas, X, kpi_solicitado)
        if not possibles_solucions:
            continue

        set_solucions = list(set(possibles_solucions)) if col in CATEGORICAL_FEATURES else possibles_solucions

        if len(set_solucions) >= len(nans_idx):
            valors_escollits = random.sample(set_solucions, len(nans_idx))
        else:
            valors_escollits = random.choices(set_solucions, k=len(nans_idx))

        for i, idx in enumerate(nans_idx):
            df_omplir.at[idx, col] = valors_escollits[i]

    return df_omplir


# ==============================================================================
# FUNCIÓ PRINCIPAL: REP ELS DICTS DE GEMINI I RETORNA ELS RESULTATS
# ==============================================================================

def run_xgboost_pipeline(campaign_data: dict, creatives_data: list, df_historic: pd.DataFrame):
    """
    Paràmetres:
      - campaign_data:  dict amb les dades de campanya (output de Gemini)
      - creatives_data: llista de dicts amb les creativitats (output de Gemini)
      - df_historic:    DataFrame amb el CSV d'entrenament (creative_summary.csv)

    Retorna:
      - df_original:    DataFrame amb els creatius tal com han entrat
      - df_optimitzat:  DataFrame amb els creatius optimitzats + prediccions
      - importancias:   DataFrame amb importàncies i correlacions del model
      - kpi:            String amb el KPI optimitzat
      - mse:            Float amb l'error del model
    """
    df = df_historic.copy()

    # --- Mètriques derivades ---
    df['area'] = df['width'] * df['height']
    df['overall_cpa'] = np.where(
        df['total_conversions'] > 0,
        df['total_spend_usd'] / df['total_conversions'],
        0
    )

    # --- KPI objectiu ---
    kpi_solicitado = campaign_data.get('kpi_goal', 'CTR')
    target_col = MAPA_TARGETS.get(kpi_solicitado, 'overall_ctr')

    if target_col not in df.columns:
        raise ValueError(f"Columna '{target_col}' no trobada al CSV d'entrenament.")

    # --- Preparació X ---
    X = df[FEATURES + CATEGORICAL_FEATURES].copy()
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False).astype(float)
    y = df[target_col]

    # --- Entrenament XGBoost ---
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    mse = mean_squared_error(y_test, preds_test)

    # --- Importàncies i correlacions ---
    importancias = pd.DataFrame({
        'Columna_XGB': X_train.columns,
        'Importancia': model.feature_importances_
    })
    correlacions = X_train.apply(lambda col: col.corr(y_train))
    importancias['Correlacion'] = importancias['Columna_XGB'].map(correlacions)
    importancias = importancias.sort_values(by='Importancia', ascending=False)

    # --- Construïm df_creatius des dels dicts de Gemini ---
    df3 = pd.DataFrame(creatives_data)

    # Afegim àrea si tenim width/height; si no, posem NaN
    if 'width' in df3.columns and 'height' in df3.columns:
        df3['area'] = df3['width'] * df3['height']
    else:
        df3['area'] = np.nan

    # Assegurem que totes les columnes necessàries existeixen
    for col in FEATURES + CATEGORICAL_FEATURES:
        if col not in df3.columns:
            df3[col] = np.nan

    # Guardem còpia original ABANS d'imputar
    df_original = df3[FEATURES + CATEGORICAL_FEATURES].copy()

    # Normalitzem booleans
    for col in COLS_HAS:
        if col in df3.columns:
            df3[col] = df3[col].replace({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})

    # --- Imputació intel·ligent ---
    df_input_omplert = omplir_forats_inteligentment(
        df_entrada=df3[FEATURES + CATEGORICAL_FEATURES],
        importancias_brutas=importancias,
        X=X,
        kpi_solicitado=kpi_solicitado
    )

    # Normalitzem booleans post-imputació
    for col in COLS_HAS:
        if col in df_input_omplert.columns:
            df_input_omplert[col] = df_input_omplert[col].replace({True: 1, False: 0, 'True': 1, 'False': 0})

    # --- Encoding i predicció ---
    X_input_encoded = pd.get_dummies(df_input_omplert, columns=CATEGORICAL_FEATURES)
    X_input_encoded = X_input_encoded.loc[:, ~X_input_encoded.columns.duplicated()]
    columnes_train = X_train.columns[~X_train.columns.duplicated()]
    X_input_encoded = X_input_encoded.reindex(columns=columnes_train, fill_value=0).astype(float)

    prediccions = model.predict(X_input_encoded)

    # --- Construïm df_optimitzat ---
    df_optimitzat = df_input_omplert.copy()
    df_optimitzat[f'Prediccio_{kpi_solicitado}'] = prediccions

    return df_original, df_optimitzat, importancias, kpi_solicitado, mse