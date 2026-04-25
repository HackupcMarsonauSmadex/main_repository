import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

df = pd.read_csv('creative_summary.csv')

df['area'] = df['width'] * df['height']

# df2 = pd.read_csv('input.csv') # Comentado si no lo usas
features = [
    'total_days_active', # 🚨 Cuidado: Data leakage
    'total_spend_usd',   # 🚨 Cuidado: Data leakage
    'area',
    'duration_sec',
    'text_density',
    'copy_length_chars',
    'faces_count',
    'product_count'     
]

categorical_features = [
    'vertical',
    'format',
    'language',
    'theme',
    'hook_type',
    'dominant_color',
    'emotional_tone',
    'advertiser_name', 
    'app_name',
    'cta_text',
    'headline',
    'subhead',
    'has_price',          
    'has_discount_badge', 
    'has_gameplay',       
    'has_ugc_style'  
]

target = 'perf_score'

X = df[features + categorical_features]
y = df[target]

# 4. Hot-Encoding
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

# 4. Dividir los datos en conjunto de Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Inicializar y configurar el modelo XGBoost
modelo_xgb = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,             
    learning_rate=0.1,            
    max_depth=5,                  
    random_state=42,
    n_jobs=-1
)

# Model training
modelo_xgb.fit(X_train, y_train)

# Evaluate de model
predicciones = modelo_xgb.predict(X_test)
error = mean_squared_error(y_test, predicciones)

print(f"Error Cuadrático Medio (MSE) en prueba: {error:.9f}")

# ==============================================================================
# ✨ LA NUEVA MAGIA: SEPARAR Y CREAR LOS DOS CSV ✨
# ==============================================================================

# 1. Obtenemos las importancias brutas
importancias_brutas = pd.DataFrame({
    'Columna_XGB': X_train.columns,
    'Importancia': modelo_xgb.feature_importances_
})

# 2. Calculamos las correlaciones
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

print("\n✅ ¡Archivos 'importancia_features_numericas.csv' e 'importancia_features_categoricas.csv' creados con éxito!")

# ==============================================================================

# PDP Múltiple
X_train_float = X_train.astype(float)
features_to_plot = features 

print(f"\nGenerando {len(features_to_plot)} gráficos de dependencia parcial...")

fig, ax = plt.subplots(figsize=(15, 12)) 

display = PartialDependenceDisplay.from_estimator(
    modelo_xgb, 
    X_train_float, 
    features=features_to_plot, 
    ax=ax,
    grid_resolution=50,
    n_cols=4  
)

plt.suptitle('Análisis de Dependencia Parcial: Impacto de las variables en Perf_Score', fontsize=16)
plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

# Ojo: esto secuestrará tu terminal. Si prefieres guardarlo, cambia show() por savefig()
plt.show()

def calidad():
    
    return 

Q = calidad()