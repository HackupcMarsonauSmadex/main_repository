import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
df_historical = pd.read_csv('creative_summary.csv')       # Historical data for training
df_campaign = pd.read_csv('input_campaign.csv')           # Campaign data (Target KPI)
df_creatives = pd.read_csv('input_creatives.csv')         # New creatives to evaluate

# ==============================================================================
# 2. DERIVED METRICS CALCULATION
# ==============================================================================
# Geometric area calculation
df_historical['area'] = df_historical['width'] * df_historical['height']
df_creatives['area'] = df_creatives['width'] * df_creatives['height']

# Historical CPA calculation (avoiding division by zero)
df_historical['overall_cpa'] = np.where(
    df_historical['total_conversions'] > 0, 
    df_historical['total_spend_usd'] / df_historical['total_conversions'], 
    0
)

# ==============================================================================
# 3. FEATURE PREPARATION
# ==============================================================================
features = [
    'total_days_active', 'total_spend_usd', 'area', 'duration_sec',
    'text_density', 'copy_length_chars', 'faces_count', 'product_count'    
]

categorical_features = [
    'vertical', 'format', 'language', 'theme', 'hook_type', 'dominant_color',
    'emotional_tone', 'advertiser_name', 'app_name', 'cta_text', 'headline',
    'subhead', 'has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style'  
]

# Preparing the main X matrix (Training Data)
X = df_historical[features + categorical_features]
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
X_encoded = X_encoded.astype(float)

# ==============================================================================
# 4. TARGET KPI IDENTIFICATION
# ==============================================================================
target_kpi = df_campaign['kpi_goal'].iloc[0]

target_mapping = {
    'CTR': 'overall_ctr',
    'CVR': 'overall_cvr',
    'IPM': 'overall_ipm',
    'ROAS': 'overall_roas',
    'CPA': 'overall_cpa'
}
final_target = target_mapping.get(target_kpi)

print(f"KPI TO OPTIMIZE: {target_kpi} (Target Column: {final_target})")

# ==============================================================================
# 4.5. FATIGUE MODULE: CROSSING CTR + CPA WITH ROLLING AVERAGES
# ==============================================================================
print("Calculating fatigue metrics by crossing CTR and CPA...")

# 1. Load daily logs
df_daily = pd.read_csv('creative_daily_country_os_stats.csv')

# 2. Base daily calculations
# Standard CTR
df_daily['ctr'] = np.where(df_daily['impressions'] > 0, 
                           df_daily['clicks'] / df_daily['impressions'], 0)

# Penalized CPA: If there is spend but 0 conversions, assign an artificially high CPA
# so the model understands that this specific day was highly inefficient.
df_daily['cpa'] = np.where(df_daily['conversions'] > 0, 
                           df_daily['spend_usd'] / df_daily['conversions'], 
                           np.where(df_daily['spend_usd'] > 0, 9999.00, 0))

# 3. Robust Fatigue Detection Logic per Creative
fatigue_records = []

for creative_id, group in df_daily.groupby('creative_id'):
    # Ensure chronological order
    group = group.sort_values('days_since_launch').copy()
    
    # Creative's historical averages (Baseline)
    avg_ctr = group['ctr'].mean()
    # For the average CPA, filter out the 9999 days to avoid distorting the real baseline
    avg_cpa = group[group['cpa'] < 9999]['cpa'].mean() 
    if pd.isna(avg_cpa): avg_cpa = group['cpa'].mean()
    
    # =========================================================================
    #  FATIGUE THRESHOLDS APPLIED:
    # - CTR: < 0.70 (Fatigue triggered if CTR drops 30% below its average)
    # - CPA: > 1.30 (Fatigue triggered if CPA rises 30% above its average)
    # - Time window: 3-day rolling average to smooth out daily anomalies
    # - Maturation: The first 3 days of the ad's life are ignored
    # =========================================================================
    ctr_threshold = avg_ctr * 0.70  
    cpa_threshold = avg_cpa * 1.30  
    
    # Smoothing: 3-day rolling average to prevent outliers from breaking the model
    group['rolling_ctr'] = group['ctr'].rolling(window=3, min_periods=1).mean()
    group['rolling_cpa'] = group['cpa'].rolling(window=3, min_periods=1).mean()
    
    # Maturation filter: Discard the first 3 days of the learning phase
    valid_days = group[group['days_since_launch'] > 3]
    
    # FATIGUE CONDITION: Engagement drops AND cost spikes simultaneously
    drops = valid_days[(valid_days['rolling_ctr'] < ctr_threshold) & 
                       (valid_days['rolling_cpa'] > cpa_threshold)]
    
    # Assign the exact day fatigue occurred
    if not drops.empty:
        fatigue_day = drops.iloc[0]['days_since_launch']
    else:
        # If it never fatigued, assign the maximum days it was active
        fatigue_day = group['days_since_launch'].max() if not group.empty else 0
        
    fatigue_records.append({'creative_id': creative_id, 'days_to_fatigue': fatigue_day})

df_fatigue = pd.DataFrame(fatigue_records)

# 4. Merge with the training dataset
df_historical = df_historical.merge(df_fatigue, on='creative_id', how='left')

# Impute potential NaNs with the global median
fatigue_median = df_historical['days_to_fatigue'].median()
df_historical['days_to_fatigue'] = df_historical['days_to_fatigue'].fillna(fatigue_median)

# 5. TRAINING THE XGBOOST FATIGUE MODEL
print("Training predictive XGBoost model for Ad Fatigue...")
y_fatigue = df_historical['days_to_fatigue']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_encoded, y_fatigue, test_size=0.2, random_state=42)

xgb_fatigue_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,             
    learning_rate=0.05,            
    max_depth=4,                  
    random_state=42,
    n_jobs=-1
)

xgb_fatigue_model.fit(X_train_f, y_train_f)
fatigue_mse = mean_squared_error(y_test_f, xgb_fatigue_model.predict(X_test_f))
print(f"Fatigue Test MSE: {fatigue_mse:.4f}")

# ==============================================================================
# 5. TRAINING THE MAIN KPI MODEL (XGBOOST)
# ==============================================================================
y = df_historical[final_target]

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,             
    learning_rate=0.1,            
    max_depth=5,                  
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
test_predictions = xgb_model.predict(X_test)
mse_error = mean_squared_error(y_test, test_predictions)
print(f"Main Model Test MSE: {mse_error:.9f}")

# ==============================================================================
# 6. EXPORT IMPORTANCES AND CORRELATIONS
# ==============================================================================
raw_importances = pd.DataFrame({
    'XGB_Column': X_train.columns,
    'Importance': xgb_model.feature_importances_
})

correlations = X_train.apply(lambda col: col.corr(y_train))
raw_importances['Correlation'] = raw_importances['XGB_Column'].map(correlations)
raw_importances = raw_importances.sort_values(by='Importance', ascending=False)

raw_importances.to_csv('Import_Corr.csv', index=False)
print(" 'Import_Corr.csv' successfully generated.")

# ==============================================================================
# 7. SMART IMPUTATION ENGINE (PRESCRIPTIVE)
# ==============================================================================

def get_imputation_interval(attribute_name, df_import_corr, df_full, features, categorical_features, target_kpi):
    """
    Calculates the 5 best possible values for an empty attribute 
    based on XGBoost feature importance and its correlation with the KPI.
    """
    if attribute_name in categorical_features:
        prefix = attribute_name + "_"
        df_filtered = df_import_corr[df_import_corr['XGB_Column'].str.startswith(prefix, na=False)].copy()
        
        if df_filtered.empty: return []
            
        # Correlation adjustment: If we aim to lower CPA, we invert the impact
        if target_kpi == 'CPA':
            df_filtered['Score'] = df_filtered['Importance'] * (df_filtered['Correlation'] * -1)
        else:
            df_filtered['Score'] = df_filtered['Importance'] * df_filtered['Correlation']
        
        df_filtered = df_filtered.sort_values(by='Score', ascending=False)
        return df_filtered.head(5)['XGB_Column'].apply(lambda x: x[len(prefix):]).tolist()

    elif attribute_name in features:
        row = df_import_corr[df_import_corr['XGB_Column'] == attribute_name]
        original_corr = row['Correlation'].values[0] if not row.empty else 0
        adjusted_corr = original_corr * -1 if target_kpi == 'CPA' else original_corr
        
        real_data = df_full[attribute_name].dropna()
        if real_data.empty: return [0, 0, 0, 0, 0]
            
        # Limit and quartile calculations
        min_val, max_val = real_data.min(), real_data.max()
        q25, q75 = real_data.quantile(0.25), real_data.quantile(0.75) 
        
        # Optimal interval selection
        range_min, range_max = (q75, max_val) if adjusted_corr > 0 else (min_val, q25)
            
        # Random generation within the ideal range
        generated_values = []
        is_integer = pd.api.types.is_integer_dtype(real_data) 
        
        for _ in range(5):
            if is_integer:
                generated_values.append(random.randint(int(range_min), int(range_max)))
            else:
                generated_values.append(round(random.uniform(range_min, range_max), 2))
                
        return generated_values
    
    return []


def smart_nan_imputation(df_input, raw_importances, X_full, features, categorical_features, target_kpi):
    """
    Fills NaNs in new creatives by crossing the model's correlation 
    to suggest the best possible design decisions.
    """
    df_fill = df_input.copy()
    df_import_corr = raw_importances.copy()
    df_full = X_full.copy()
    
    for col in df_fill.columns:
        if col not in features and col not in categorical_features:
            continue 
            
        nan_indices = df_fill[df_fill[col].isna()].index.tolist()
        num_missing = len(nan_indices)
        
        if num_missing == 0:
            continue 
            
        possible_solutions = get_imputation_interval(col, df_import_corr, df_full, features, categorical_features, target_kpi)
        if not possible_solutions:
            continue 
            
        # Remove duplicates for categoricals and assign values intelligently
        unique_solutions = list(set(possible_solutions)) if col in categorical_features else possible_solutions
        
        if len(unique_solutions) >= num_missing:
            chosen_values = random.sample(unique_solutions, num_missing)
        else:
            chosen_values = random.choices(unique_solutions, k=num_missing)
            
        for i, idx in enumerate(nan_indices):
            df_fill.at[idx, col] = chosen_values[i]
            
    print(f" Smart imputation process to optimize {target_kpi} successfully completed!")
    return df_fill


# Execute the prescriptive engine
X_input_final = smart_nan_imputation(
     df_input=df_creatives[features + categorical_features],
     raw_importances=raw_importances,
     X_full=X,
     features=features,
     categorical_features=categorical_features,
     target_kpi=target_kpi
 )

# ==============================================================================
# 8. FINAL PREPARATION AND PREDICTION (KPI + FATIGUE)
# ==============================================================================
# Fix booleans for compatibility
cols_has = ['has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style']
for col in cols_has:
    if col in X_input_final.columns:
        X_input_final[col] = X_input_final[col].replace({True: 1, False: 0, 'True': 1, 'False': 0})

# Dummy encoding and anti-duplicate cleanup (Prevents XGBoost ValueError)
X_input_encoded = pd.get_dummies(X_input_final, columns=categorical_features)
X_input_encoded = X_input_encoded.loc[:, ~X_input_encoded.columns.duplicated()]
training_columns = X_train.columns[~X_train.columns.duplicated()]

# Align exact dimensions with the trained model
X_input_encoded = X_input_encoded.reindex(columns=training_columns, fill_value=0)
X_input_encoded = X_input_encoded.astype(float)

# Main KPI Prediction
kpi_predictions = xgb_model.predict(X_input_encoded)

# Days to Fatigue Prediction
fatigue_predictions = xgb_fatigue_model.predict(X_input_encoded)
# Round to whole days and ensure no negative predictions
fatigue_predictions = np.maximum(0, np.round(fatigue_predictions)).astype(int)

# ==============================================================================
# 9. SAVING AND EXPORTING
# ==============================================================================
# Inject the AI suggestions back into the original dataset
df_creatives[features + categorical_features] = X_input_final[features + categorical_features]
df_creatives[f'Prediction_{target_kpi}'] = kpi_predictions
df_creatives['Prediction_Days_to_Fatigue'] = fatigue_predictions  # <-- New column added

df_creatives.to_csv('output_creatives.csv', index=False)

print("\n" + "="*50)
print(f" PROCESS COMPLETED!")
print(f" Generated file: 'output_creatives.csv'")
print("-" * 50)
print(" PREDICTIONS:")
for index, (pred_kpi, pred_fat) in enumerate(zip(kpi_predictions, fatigue_predictions)):
    print(f" - Creative {index + 1}: {target_kpi} = {pred_kpi:.6f} | Fatigue in: {pred_fat} days")
print("="*50)