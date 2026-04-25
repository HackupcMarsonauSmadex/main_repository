import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
df = pd.read_csv('creative_summary.csv')      # Histórico para entrenar
df2 = pd.read_csv('input_campaign.csv')       # Datos de la campaña (KPI)
df3 = pd.read_csv('input_creatives.csv')      # Nuevos creativos a evaluar

# ==============================================================================
# ✨ CÁLCULO DE MÉTRICAS DERIVADAS
# ==============================================================================
# Área geométrica
df['area'] = df['width'] * df['height']
df3['area'] = df3['width'] * df3['height']

# Cálculo del CPA histórico (evitando división por cero)
df['overall_cpa'] = np.where(
    df['total_conversions'] > 0, 
    df['total_spend_usd'] / df['total_conversions'], 
    0
)

# ==============================================================================
# 2. DEFINICIÓN Y PREPARACIÓN DE ATRIBUTOS
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

# Preparamos la X principal (Datos de entrenamiento)
X = df[features + categorical_features]
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

# 🔥 CORRECCIÓN: Convertimos todo a numérico para evitar el error de XGBoost
X_encoded = X_encoded.astype(float)

# ==============================================================================
# 3. IDENTIFICAR EL KPI DE LA CAMPAÑA
# ==============================================================================
kpi_solicitado = df2['kpi_goal'].iloc[0]

mapa_targets = {
    'CTR': 'overall_ctr',
    'CVR': 'overall_cvr',
    'IPM': 'overall_ipm',
    'ROAS': 'overall_roas',
    'CPA': 'overall_cpa'
}
target_final = mapa_targets.get(kpi_solicitado)

print(f"🎯 KPI A OPTIMIZAR: {kpi_solicitado} (Columna: {target_final})")

# ==============================================================================
# 4. ENTRENAMIENTO DEL MODELO
# ==============================================================================
y = df[target_final]

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

modelo_xgb = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,             
    learning_rate=0.1,            
    max_depth=5,                  
    random_state=42,
    n_jobs=-1
)

modelo_xgb.fit(X_train, y_train)
predicciones_prueba = modelo_xgb.predict(X_test)
error = mean_squared_error(y_test, predicciones_prueba)
print(f" > MSE en prueba: {error:.9f}")

# ==============================================================================
# 5. EXPORTAR IMPORTANCIAS Y CORRELACIONES
# ==============================================================================
importancias_brutas = pd.DataFrame({
    'Columna_XGB': X_train.columns,
    'Importancia': modelo_xgb.feature_importances_
})

correlaciones = X_train.apply(lambda col: col.corr(y_train))
importancias_brutas['Correlacion'] = importancias_brutas['Columna_XGB'].map(correlaciones)
importancias_brutas = importancias_brutas.sort_values(by='Importancia', ascending=False)

importancias_brutas.to_csv('Import_Corr.csv', index=False)
print("✅ Archivo 'Import_Corr.csv' generado con éxito.")

# ==============================================================================
# 6. ANÁLISIS DE NUEVOS CREATIVOS (df3) Y PREDICCIÓN
# ==============================================================================
# Extraemos atributos del archivo de creativos
X_input_raw = df3[features + categorical_features]
X_input_encoded = pd.get_dummies(X_input_raw, columns=categorical_features)

# Alineamos las columnas con las de entrenamiento (rellenando con 1 lo que falte)
X_input_final = X_input_encoded.reindex(columns=X_encoded.columns, fill_value=1)

# 🔥 CORRECCIÓN: Aseguramos que todo sea numérico antes de predecir
X_input_final = X_input_final.astype(float)

# Hacemos la predicción sobre los nuevos creativos
pred_input = modelo_xgb.predict(X_input_final)

# Guardamos la predicción en el propio dataframe por si quieres usarlo
df3[f'Prediccion_{kpi_solicitado}'] = pred_input

print("\n" + "="*40)
print("🚀 PREDICCIONES PARA LOS NUEVOS CREATIVOS:")
for index, pred in enumerate(pred_input):
    print(f" - Creativo {index + 1}: {kpi_solicitado} estimado = {pred:.6f}")
print("="*40)