import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np

# 1. Carga de datos
df = pd.read_csv('creative_summary.csv')
df2 = pd.read_csv('input_campaign.csv')
df3 = pd.read_csv('input_creatives.csv')

# ✨ CÁLCULO DE MÉTRICAS DERIVADAS (Área y CPA)
# Área geométrica
df['area'] = df['width'] * df['height']
df2['area'] = df2['width'] * df2['height']

# Cálculo del CPA (Cost Per Action) = Gasto / Conversiones
# Usamos np.where para evitar el error de "división por cero" si no hay conversiones
df['overall_cpa'] = np.where(
    df['total_conversions'] > 0, 
    df['total_spend_usd'] / df['total_conversions'], 
    0
)

# 2. DEFINICIÓN DE ATRIBUTOS (Features)
features = [
    'total_days_active',
    'total_spend_usd',
    'area',
    'duration_sec',
    'text_density',
    'copy_length_chars',
    'faces_count',
    'product_count'     
]

categorical_features = [
    'vertical', 'format', 'language', 'theme', 'hook_type', 'dominant_color',
    'emotional_tone', 'advertiser_name', 'app_name', 'cta_text', 'headline',
    'subhead', 'has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style'  
]

# Preparamos la X principal
X = df[features + categorical_features]
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

# Preparamos el input final para predicciones
X_input_raw = df2[features + categorical_features]
X_input_encoded = pd.get_dummies(X_input_raw, columns=categorical_features)
X_input_final = X_input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

kpi_solicitado = df2['kpi_goal'].iloc[0]

# Diccionario de traducción (Perf Score eliminado)
mapa_targets = {
    'CTR': 'overall_ctr',
    'CVR': 'overall_cvr',
    'IPM': 'overall_ipm',
    'ROAS': 'overall_roas',
    'CPA': 'overall_cpa'
}

# Obtenemos el nombre real de la columna a predecir
target_final = mapa_targets.get(kpi_solicitado)

print(f"🎯 CAMPAÑA DETECTADA: {id_campaña}")
print(f"🎯 KPI A OPTIMIZAR: {kpi_solicitado} (Columna: {target_final})")

# Asignamos 'y' solo a la métrica que nos interesa
y = df[target_final]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Inicializar modelo
modelo_xgb = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,             
    learning_rate=0.1,            
    max_depth=5,                  
    random_state=42,
    n_jobs=-1
)

# Entrenar
modelo_xgb.fit(X_train, y_train)

# Evaluar
predicciones = modelo_xgb.predict(X_test)
error = mean_squared_error(y_test, predicciones)
print(f" > MSE en prueba: {error:.9f}")


# Predecir sobre input.csv
#pred_input = modelo_xgb.predict(X_input_final)
#print(f" - {kpi_solicitado} estimado: {pred_input[0]:.6f}")

# Calculamos las importancias del modelo
importancias_brutas = pd.DataFrame({
    'Columna_XGB': X_train.columns,
    'Importancia': modelo_xgb.feature_importances_
})

# Calculamos las correlaciones directas con el target seleccionado
correlaciones = X_train.apply(lambda col: col.corr(y_train))
importancias_brutas['Correlacion'] = importancias_brutas['Columna_XGB'].map(correlaciones)

# Ordenamos por importancia (de mayor a menor)
importancias_brutas = importancias_brutas.sort_values(by='Importancia', ascending=False)

# Guardamos el archivo CSV con el nombre del KPI analizado
importancias_brutas.to_csv('Import_Corr.csv', index=False)