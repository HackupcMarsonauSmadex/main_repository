import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# 1. Carga de datos
df = pd.read_csv('creative_summary.csv')
df2 = pd.read_csv('input.csv')
df_campaigns = pd.read_csv('campaigns.csv')

df['area'] = df['width'] * df['height']
df2['area'] = df2['width'] * df2['height']

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


# ==============================================================================
# ✨ NUEVA LÓGICA: IDENTIFICAR EL OBJETIVO ÚNICO
# ==============================================================================

# 1. Sacamos el ID del anuncio a evaluar
id_campaña = df2['campaign_id'].iloc[0]

# 2. Buscamos ese ID en el archivo de campañas y sacamos su kpi_goal
kpi_solicitado = df_campaigns.loc[df_campaigns['campaign_id'] == id_campaña, 'kpi_goal'].values[0]

# 3. Diccionario traductor (de lo que dice la campaña a tu columna del CSV)
mapa_targets = {
    'CTR': 'overall_ctr',
    'CVR': 'overall_cvr',
    'IPM': 'overall_ipm',
    'ROAS': 'overall_roas',
    'Perf Score': 'perf_score'
}

# 4. Obtenemos el nombre real de la columna a predecir
target_final = mapa_targets.get(kpi_solicitado)

print("="*40)
print(f"🎯 CAMPAÑA DETECTADA: {id_campaña}")
print(f"🎯 KPI A OPTIMIZAR: {kpi_solicitado} (Columna: {target_final})")
print("="*40)

# ==============================================================================
# ENTRENAMIENTO Y PREDICCIÓN (SOLO UNA VEZ)
# ==============================================================================

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
print("Entrenando modelo especialista...")
modelo_xgb.fit(X_train, y_train)

# Evaluar
predicciones = modelo_xgb.predict(X_test)
error = mean_squared_error(y_test, predicciones)
print(f" > MSE en prueba: {error:.9f}")

# Predecir sobre input.csv
pred_input = modelo_xgb.predict(X_input_final)

print("\n" + "="*40)
print("🚀 RESULTADO DE LA PREDICCIÓN:")
print(f" - {kpi_solicitado} estimado: {pred_input[0]:.6f}")
print("="*40)


#Analisis of the correlations of the model
importancias_brutas = pd.DataFrame({
    'Columna_XGB': X_train.columns,
    'Importancia': modelo_xgb.feature_importances_
})
correlaciones = X_train.apply(lambda col: col.corr(y_train))
importancias_brutas['Correlacion'] = importancias_brutas['Columna_XGB'].map(correlaciones)

# 3. Lógica para desempaquetar Nombre y Valor
nombres = []
valores = []
tipos = [] # Para saber a qué CSV va

for col in importancias_brutas['Columna_XGB']:
    es_categorica = False
    
    # Comprobamos si la columna viene de un One-Hot Encoding (ej. format_banner)
    for cat in categorical_features:
        if col.startswith(cat + '_'):
            nombres.append(cat)
            # Quitamos el prefijo (ej. 'format_') para quedarnos solo con 'banner'
            valores.append(col.replace(cat + '_', '', 1)) 
            tipos.append('categorica')
            es_categorica = True
            break
            
    # Si no es categórica, comprobamos si es numérica
    if not es_categorica:
        if col in features:
            nombres.append(col)
            valores.append('-') # Los valores numéricos no tienen categoría
            tipos.append('numerica')
        else:
            nombres.append(col)
            valores.append('-')
            tipos.append('desconocido')

# 4. Asignamos las nuevas columnas al DataFrame
importancias_brutas['Nombre del atributo'] = nombres
importancias_brutas['Valor del atributo'] = valores
importancias_brutas['Tipo'] = tipos

# 5. Ordenamos por importancia y filtramos solo las columnas que quieres
importancias_brutas = importancias_brutas.sort_values(by='Importancia', ascending=False)
columnas_exportar = ['Nombre del atributo', 'Valor del atributo', 'Importancia', 'Correlacion']

# 6. Partimos en dos DataFrames distintos
df_numericas = importancias_brutas[importancias_brutas['Tipo'] == 'numerica'][columnas_exportar]
df_categoricas = importancias_brutas[importancias_brutas['Tipo'] == 'categorica'][columnas_exportar]

# 7. Guardamos los dos CSV
df_numericas.to_csv('importancia_features_numericas.csv', index=False)
df_categoricas.to_csv('importancia_features_categoricas.csv', index=False)

print("¡Archivos 'importancia_features_numericas.csv' e 'importancia_features_categoricas.csv' creados con éxito!")

def calidad():
    
    return 

Q = calidad()