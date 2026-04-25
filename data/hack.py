import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

df = pd.read_csv('creative_summary.csv')

df['area'] = df['width'] * df['height']

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

#Visualizar el efecto real de las variables críticas (PDP)
print("\nGenerando gráficos de dependencia parcial...")
features_to_plot = ['text_density'] 
fig, ax = plt.subplots(figsize=(8, 6))

display = PartialDependenceDisplay.from_estimator(
    modelo_xgb, 
    X_train,
    features=features_to_plot, 
    ax=ax,
    grid_resolution=50
)

plt.title('Efecto del Área del Anuncio en el Performance Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()