import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================

# has_* are 0/1 integers in the historical CSV → treat as numeric, NOT categorical
COLS_HAS = ['has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style']

FEATURES = [
    'total_days_active', 'total_spend_usd', 'area', 'duration_sec',
    'text_density', 'copy_length_chars', 'faces_count', 'product_count',
    'has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style'
]

CATEGORICAL_FEATURES = [
    'vertical', 'format', 'language', 'theme', 'hook_type', 'dominant_color',
    'emotional_tone', 'advertiser_name', 'app_name', 'cta_text', 'headline', 'subhead'
]

TARGET_MAPPING = {
    'CTR': 'overall_ctr',
    'CVR': 'overall_cvr',
    'IPM': 'overall_ipm',
    'ROAS': 'overall_roas',
    'CPA': 'overall_cpa'
}


# ==============================================================================
# FATIGUE MODULE
# ==============================================================================

def compute_fatigue(df_historical: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Crosses CTR + CPA with rolling averages to detect the day a creative fatigues.
    Returns df_historical with a new 'days_to_fatigue' column merged in.
    """
    df_daily = df_daily.copy()
    df_daily['ctr'] = np.where(
        df_daily['impressions'] > 0,
        df_daily['clicks'] / df_daily['impressions'],
        0
    )
    # Penalized CPA: spend with 0 conversions → artificially high CPA
    df_daily['cpa'] = np.where(
        df_daily['conversions'] > 0,
        df_daily['spend_usd'] / df_daily['conversions'],
        np.where(df_daily['spend_usd'] > 0, 9999.00, 0)
    )

    fatigue_records = []

    for creative_id, group in df_daily.groupby('creative_id'):
        group = group.sort_values('days_since_launch').copy()

        avg_ctr = group['ctr'].mean()
        avg_cpa = group[group['cpa'] < 9999]['cpa'].mean()
        if pd.isna(avg_cpa):
            avg_cpa = group['cpa'].mean()

        # Thresholds: CTR drops 30%, CPA rises 30%
        ctr_threshold = avg_ctr * 0.70
        cpa_threshold = avg_cpa * 1.30

        # 3-day rolling average to smooth daily anomalies
        group['rolling_ctr'] = group['ctr'].rolling(window=3, min_periods=1).mean()
        group['rolling_cpa'] = group['cpa'].rolling(window=3, min_periods=1).mean()

        # Ignore first 3 days (maturation / learning phase)
        valid_days = group[group['days_since_launch'] > 3]

        drops = valid_days[
            (valid_days['rolling_ctr'] < ctr_threshold) &
            (valid_days['rolling_cpa'] > cpa_threshold)
        ]

        fatigue_day = drops.iloc[0]['days_since_launch'] if not drops.empty \
            else (group['days_since_launch'].max() if not group.empty else 0)

        fatigue_records.append({'creative_id': creative_id, 'days_to_fatigue': fatigue_day})

    df_fatigue = pd.DataFrame(fatigue_records)
    df_out = df_historical.merge(df_fatigue, on='creative_id', how='left')
    fatigue_median = df_out['days_to_fatigue'].median()
    df_out['days_to_fatigue'] = df_out['days_to_fatigue'].fillna(fatigue_median)

    return df_out


# ==============================================================================
# SMART IMPUTATION ENGINE
# ==============================================================================

def get_imputation_interval(attribute_name, df_import_corr, df_full, target_kpi):
    if attribute_name in CATEGORICAL_FEATURES:
        prefix = attribute_name + "_"
        df_filtered = df_import_corr[
            df_import_corr['XGB_Column'].str.startswith(prefix, na=False)
        ].copy()

        if df_filtered.empty:
            return []

        if target_kpi == 'CPA':
            df_filtered['Score'] = df_filtered['Importance'] * (df_filtered['Correlation'] * -1)
        else:
            df_filtered['Score'] = df_filtered['Importance'] * df_filtered['Correlation']

        df_filtered = df_filtered.sort_values(by='Score', ascending=False)
        return df_filtered.head(5)['XGB_Column'].apply(lambda x: x[len(prefix):]).tolist()

    elif attribute_name in FEATURES:
        row = df_import_corr[df_import_corr['XGB_Column'] == attribute_name]
        original_corr = row['Correlation'].values[0] if not row.empty else 0
        adjusted_corr = original_corr * -1 if target_kpi == 'CPA' else original_corr

        real_data = df_full[attribute_name].dropna()
        if real_data.empty:
            return [0, 0, 0, 0, 0]

        min_val, max_val = real_data.min(), real_data.max()
        q25, q75 = real_data.quantile(0.25), real_data.quantile(0.75)
        range_min, range_max = (q75, max_val) if adjusted_corr > 0 else (min_val, q25)

        generated_values = []
        is_integer = pd.api.types.is_integer_dtype(real_data)
        for _ in range(5):
            if is_integer:
                generated_values.append(random.randint(int(range_min), int(range_max)))
            else:
                generated_values.append(round(random.uniform(range_min, range_max), 2))
        return generated_values

    return []


def smart_nan_imputation(df_input, raw_importances, X_full, target_kpi):
    df_fill = df_input.copy()

    for col in df_fill.columns:
        if col not in FEATURES and col not in CATEGORICAL_FEATURES:
            continue

        nan_indices = df_fill[df_fill[col].isna()].index.tolist()
        if not nan_indices:
            continue

        possible_solutions = get_imputation_interval(col, raw_importances, X_full, target_kpi)
        if not possible_solutions:
            continue

        unique_solutions = list(set(possible_solutions)) if col in CATEGORICAL_FEATURES else possible_solutions

        if len(unique_solutions) >= len(nan_indices):
            chosen_values = random.sample(unique_solutions, len(nan_indices))
        else:
            chosen_values = random.choices(unique_solutions, k=len(nan_indices))

        for i, idx in enumerate(nan_indices):
            df_fill.at[idx, col] = chosen_values[i]

    return df_fill


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_xgboost_pipeline(
    campaign_data: dict,
    creatives_data: list,
    df_historic: pd.DataFrame,
    df_daily: pd.DataFrame = None   # Optional: needed for fatigue module
):
    """
    Parameters:
      - campaign_data:  dict with campaign data (Gemini output)
      - creatives_data: list of dicts with creatives (Gemini output)
      - df_historic:    DataFrame from creative_summary.csv
      - df_daily:       DataFrame from creative_daily_country_os_stats.csv (optional)

    Returns:
      - df_original:     DataFrame with creatives as received from Gemini
      - df_optimized:    DataFrame with optimized creatives + predictions
      - raw_importances: DataFrame with XGBoost importances and correlations
      - target_kpi:      String with the optimized KPI
      - mse_kpi:         Float with KPI model MSE
      - mse_fatigue:     Float with fatigue model MSE (None if no daily data)
    """
    df = df_historic.copy()

    # --- Derived metrics ---
    df['area'] = df['width'] * df['height']
    df['overall_cpa'] = np.where(
        df['total_conversions'] > 0,
        df['total_spend_usd'] / df['total_conversions'],
        0
    )

    # --- Target KPI ---
    target_kpi = campaign_data.get('kpi_goal', 'CTR')
    target_col = TARGET_MAPPING.get(target_kpi, 'overall_ctr')
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in training CSV.")

    # --- Fatigue module (only if daily CSV is provided) ---
    fatigue_model = None
    mse_fatigue = None
    if df_daily is not None:
        df = compute_fatigue(df, df_daily)

    # --- Feature matrix ---
    X = df[FEATURES + CATEGORICAL_FEATURES].copy()

    # Normalize has_* booleans from historical CSV
    for col in COLS_HAS:
        if col in X.columns:
            X[col] = X[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}).astype(float)

    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False).astype(float)

    # --- Train/test split (shared split index for both models) ---
    X_train, X_test, y_kpi_train, y_kpi_test = train_test_split(
        X_encoded, df[target_col], test_size=0.2, random_state=42
    )

    # --- Main KPI model ---
    kpi_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    kpi_model.fit(X_train, y_kpi_train)
    mse_kpi = mean_squared_error(y_kpi_test, kpi_model.predict(X_test))

    # --- Fatigue model (only if daily data available) ---
    if df_daily is not None:
        y_fatigue_train = df.loc[X_train.index, 'days_to_fatigue']
        y_fatigue_test  = df.loc[X_test.index,  'days_to_fatigue']

        fatigue_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=-1
        )
        fatigue_model.fit(X_train, y_fatigue_train)
        mse_fatigue = mean_squared_error(y_fatigue_test, fatigue_model.predict(X_test))

    # --- Importances and correlations ---
    raw_importances = pd.DataFrame({
        'XGB_Column': X_train.columns,
        'Importance': kpi_model.feature_importances_
    })
    correlations = X_train.apply(lambda col: col.corr(y_kpi_train))
    raw_importances['Correlation'] = raw_importances['XGB_Column'].map(correlations)
    raw_importances = raw_importances.sort_values(by='Importance', ascending=False)

    # --- Build creatives DataFrame from Gemini dicts ---
    df3 = pd.DataFrame(creatives_data)

    # Area: Gemini returns null for width/height → NaN
    if 'width' in df3.columns and 'height' in df3.columns:
        df3['area'] = df3['width'] * df3['height']
    else:
        df3['area'] = np.nan

    # Ensure all required columns exist
    for col in FEATURES + CATEGORICAL_FEATURES:
        if col not in df3.columns:
            df3[col] = np.nan

    # Normalize has_* coming from Gemini (can be None, 0, 1, "True", "False")
    for col in COLS_HAS:
        if col in df3.columns:
            df3[col] = df3[col].replace({True: 1, False: 0, 'True': 1, 'False': 0, None: np.nan})
            df3[col] = pd.to_numeric(df3[col], errors='coerce')

    # Save original BEFORE imputation
    df_original = df3[FEATURES + CATEGORICAL_FEATURES].copy()

    # --- Smart imputation ---
    df_imputed = smart_nan_imputation(
        df_input=df3[FEATURES + CATEGORICAL_FEATURES],
        raw_importances=raw_importances,
        X_full=X,
        target_kpi=target_kpi
    )

    # Normalize has_* post-imputation
    for col in COLS_HAS:
        if col in df_imputed.columns:
            df_imputed[col] = pd.to_numeric(
                df_imputed[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}),
                errors='coerce'
            ).fillna(0)

    # --- Encode and align with training columns ---
    X_input_encoded = pd.get_dummies(df_imputed, columns=CATEGORICAL_FEATURES)
    X_input_encoded = X_input_encoded.loc[:, ~X_input_encoded.columns.duplicated()]
    training_columns = X_train.columns[~X_train.columns.duplicated()]
    X_input_encoded = X_input_encoded.reindex(columns=training_columns, fill_value=0).astype(float)

    # --- Predictions ---
    kpi_predictions = kpi_model.predict(X_input_encoded)

    df_optimized = df_imputed.copy()
    df_optimized[f'Prediction_{target_kpi}'] = kpi_predictions

    if fatigue_model is not None:
        fatigue_predictions = np.maximum(
            0, np.round(fatigue_model.predict(X_input_encoded))
        ).astype(int)
        df_optimized['Prediction_Days_to_Fatigue'] = fatigue_predictions

    return df_original, df_optimized, raw_importances, target_kpi, mse_kpi, mse_fatigue