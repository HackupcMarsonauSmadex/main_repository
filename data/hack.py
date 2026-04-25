import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

df = pd.read_csv('creative_summary.csv')

df['area'] = df['width'] * df['height']

df2 = pd.read_csv('input.csv')
features = [
    'total_days_active',
    'total_spend_usd',
    'area',
    'duration_sec',
    'text_density',
    'copy_length_chars',
    'faces_count',
    'product_count',
    'has_price',          
    'has_discount_badge', 
    'has_gameplay',       
    'has_ugc_style'       
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
    'subhead'
]

target = 'perf_score'

X = df[features + categorical_features]
y = df[target]

# 4. Hot-Encoding
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

# 4. Dividir los datos en conjunto de Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Inicializar y configurar el modelo XGBoost
# Usamos XGBRegressor porque el CTR es un número continuo, no una categoría.
modelo_xgb = xgb.XGBRegressor(
    objective='reg:squarederror', # Función de pérdida estándar para regresión
    n_estimators=100,             # Número de árboles
    learning_rate=0.1,            # Tasa de aprendizaje
    max_depth=5,                  # Profundidad de cada árbol
    random_state=42
)

#Model training
modelo_xgb.fit(X_train, y_train)

# Evaluate de model
predicciones = modelo_xgb.predict(X_test)
error = mean_squared_error(y_test, predicciones)

print(f"Error Cuadrático Medio (MSE) en prueba: {error:.9f}")

# 8. Extraer qué parámetros de diseño son los más importantes
importancias = pd.DataFrame({
    'Atributo': X_train.columns,
    'Importancia': modelo_xgb.feature_importances_
}).sort_values(by='Importancia', ascending=False)

#Correlación
# Calculamos la dirección (+ o -) usando la correlación lineal básica
correlaciones = X_train.apply(lambda col: col.corr(y_train))
importancias['Correlacion'] = importancias['Atributo'].map(correlaciones)
importancias = importancias.sort_values(by='Importancia', ascending=False)

importancias.to_csv('importancia_diseno_ctr.csv', index=False)

print("\n--- Importancia y Dirección de los Parámetros ---")
print(importancias.head(10))

# 1. Aseguramos que todos los datos de entrenamiento sean float para evitar errores
X_train_float = X_train.astype(float)

# 2. Definimos la lista de variables a graficar (tus features numéricas)
# Nota: He quitado las que puedan ser binarias o constantes si prefieres, 
# pero aquí están todas las de tu lista 'features'
features_to_plot = features 

print(f"\nGenerando {len(features_to_plot)} gráficos de dependencia parcial...")

# 3. Configuramos el tamaño de la figura
# Usamos una rejilla. Scikit-learn la gestiona automáticamente si le damos un tamaño grande.
fig, ax = plt.subplots(figsize=(15, 12)) 

# 4. Creamos el PDP múltiple
display = PartialDependenceDisplay.from_estimator(
    modelo_xgb, 
    X_train_float, 
    features=features_to_plot, 
    ax=ax,
    grid_resolution=50,
    n_cols=4  # Definimos 4 columnas de gráficos para que sea legible
)

# Ajustes estéticos
plt.suptitle('Análisis de Dependencia Parcial: Impacto de las variables en Perf_Score', fontsize=16)
plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
plt.show()