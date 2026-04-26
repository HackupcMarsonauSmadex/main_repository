import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
df = pd.read_csv('creative_summary.csv')      # Histórico para entrenar
df2 = pd.read_csv('input_campaign.csv')       # Datos de la campaña (KPI)
df3 = pd.read_csv('input_creatives.csv')      # Nuevos creativos a evaluar

# ==============================================================================
# 2. CÁLCULO DE MÉTRICAS DERIVADAS
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
# 3. PREPARACIÓN DE ATRIBUTOS
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
X_encoded = X_encoded.astype(float)

# ==============================================================================
# 4. IDENTIFICACIÓN DEL KPI OBJETIVO
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
# 4.5. NUEVO MÓDULO: CÁLCULO DE FATIGA (CTR + CPA CON MEDIA MÓVIL)
# ==============================================================================
print("⏳ Calculando métricas de fatiga cruzando CTR y CPA...")

# 1. Cargar los logs diarios
df_daily = pd.read_csv('creative_daily_country_os_stats.csv')

# 2. Cálculos base diarios
# CTR normal
df_daily['ctr'] = np.where(df_daily['impressions'] > 0, 
                           df_daily['clicks'] / df_daily['impressions'], 0)

# CPA con penalización: Si hay gasto pero 0 conversiones, asignamos un CPA artificialmente alto
# para que el modelo entienda que ese día fue muy ineficiente.
df_daily['cpa'] = np.where(df_daily['conversions'] > 0, 
                           df_daily['spend_usd'] / df_daily['conversions'], 
                           np.where(df_daily['spend_usd'] > 0, 9999.00, 0))

# 3. Lógica de Detección de Fatiga Robusta por Creativo
fatigue_records = []

for creative_id, group in df_daily.groupby('creative_id'):
    # Asegurar el orden cronológico
    group = group.sort_values('days_since_launch').copy()
    
    # Promedios históricos del creativo (Baseline)
    avg_ctr = group['ctr'].mean()
    # Para el CPA medio, filtramos los días de 9999 para no distorsionar el baseline real
    avg_cpa = group[group['cpa'] < 9999]['cpa'].mean() 
    if pd.isna(avg_cpa): avg_cpa = group['cpa'].mean()
    
    # =========================================================================
    # 📌 THRESHOLDS (UMBRALES) DE FATIGA UTILIZADOS:
    # - CTR: < 0.70 (Consideramos fatiga si el CTR cae un 30% bajo su media)
    # - CPA: > 1.30 (Consideramos fatiga si el CPA sube un 30% sobre su media)
    # - Ventana temporal: Media móvil de 3 días para suavizar picos diarios
    # - Maduración: Se ignoran los 3 primeros días de vida del anuncio
    # =========================================================================
    ctr_threshold = avg_ctr * 0.70  
    cpa_threshold = avg_cpa * 1.30  
    
    # Suavizado: Media móvil de 3 días para evitar que un día atípico rompa el modelo
    group['rolling_ctr'] = group['ctr'].rolling(window=3, min_periods=1).mean()
    group['rolling_cpa'] = group['cpa'].rolling(window=3, min_periods=1).mean()
    
    # Filtro de maduración: Descartamos los primeros 3 días de aprendizaje
    valid_days = group[group['days_since_launch'] > 3]
    
    # CONDICIÓN DE FATIGA: Cae el engagement Y se dispara el coste simultáneamente
    drops = valid_days[(valid_days['rolling_ctr'] < ctr_threshold) & 
                       (valid_days['rolling_cpa'] > cpa_threshold)]
    
    # Asignamos el día de fatiga
    if not drops.empty:
        fatigue_day = drops.iloc[0]['days_since_launch']
    else:
        # Si nunca se fatigó, asignamos el máximo día que estuvo vivo
        fatigue_day = group['days_since_launch'].max() if not group.empty else 0
        
    fatigue_records.append({'creative_id': creative_id, 'days_to_fatigue': fatigue_day})

df_fatigue = pd.DataFrame(fatigue_records)

# 4. Cruzar con el dataset de entrenamiento
df = df.merge(df_fatigue, on='creative_id', how='left')

# Imputar posibles NaNs con la mediana global
mediana_fatiga = df['days_to_fatigue'].median()
df['days_to_fatigue'] = df['days_to_fatigue'].fillna(mediana_fatiga)

# 5. ENTRENAMIENTO DEL MODELO XGBOOST DE FATIGA
print("🤖 Entrenando modelo XGBoost predictivo de Ad Fatigue...")
y_fatigue = df['days_to_fatigue']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_encoded, y_fatigue, test_size=0.2, random_state=42)

modelo_xgb_fatigue = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,             
    learning_rate=0.05,            
    max_depth=4,                  
    random_state=42,
    n_jobs=-1
)

modelo_xgb_fatigue.fit(X_train_f, y_train_f)
error_fatigue = mean_squared_error(y_test_f, modelo_xgb_fatigue.predict(X_test_f))
print(f" > MSE de Fatiga en prueba: {error_fatigue:.4f}")

# ==============================================================================

# ==============================================================================
# 5. ENTRENAMIENTO DEL MODELO (XGBOOST)
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
# 6. EXPORTAR IMPORTANCIAS Y CORRELACIONES
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
# 7. MOTOR DE IMPUTACIÓN INTELIGENTE (PRESCRIPTIVO)
# ==============================================================================

def get_interval(nom_atribut, df_import_corr, df_sencer, features, categorical_features, kpi_solicitado):
    """
    Calcula los 5 mejores valores posibles para un atributo vacío
    basándose en la importancia de XGBoost y su correlación con el KPI.
    """
    if nom_atribut in categorical_features:
        prefix = nom_atribut + "_"
        df_filtre = df_import_corr[df_import_corr['Columna_XGB'].str.startswith(prefix, na=False)].copy()
        
        if df_filtre.empty: return []
            
        # Ajuste de correlación: Si buscamos bajar el CPA, invertimos el impacto
        if kpi_solicitado == 'CPA':
            df_filtre['Score'] = df_filtre['Importancia'] * (df_filtre['Correlacion'] * -1)
        else:
            df_filtre['Score'] = df_filtre['Importancia'] * df_filtre['Correlacion']
        
        df_filtre = df_filtre.sort_values(by='Score', ascending=False)
        return df_filtre.head(5)['Columna_XGB'].apply(lambda x: x[len(prefix):]).tolist()

    elif nom_atribut in features:
        fila = df_import_corr[df_import_corr['Columna_XGB'] == nom_atribut]
        corr_original = fila['Correlacion'].values[0] if not fila.empty else 0
        corr_ajustada = corr_original * -1 if kpi_solicitado == 'CPA' else corr_original
        
        dades_reals = df_sencer[nom_atribut].dropna()
        if dades_reals.empty: return [0, 0, 0, 0, 0]
            
        # Cálculo de límites y cuartiles
        minim, maxim = dades_reals.min(), dades_reals.max()
        q25, q75 = dades_reals.quantile(0.25), dades_reals.quantile(0.75) 
        
        # Selección del intervalo óptimo
        rang_min, rang_max = (q75, maxim) if corr_ajustada > 0 else (minim, q25)
            
        # Generación aleatoria dentro del rango ideal
        valors_generats = []
        es_enter = pd.api.types.is_integer_dtype(dades_reals) 
        
        for _ in range(5):
            if es_enter:
                valors_generats.append(random.randint(int(rang_min), int(rang_max)))
            else:
                valors_generats.append(round(random.uniform(rang_min, rang_max), 2))
                
        return valors_generats
    
    return []


def omplir_forats_inteligentment(df_entrada, importancias_brutas, X, features, categorical_features, kpi_solicitado):
    """
    Rellena los NaNs de nuevos creativos cruzando la correlación del modelo
    para sugerir las mejores decisiones de diseño posibles.
    """
    df_omplir = df_entrada.copy()
    df_import_corr = importancias_brutas.copy()
    df_sencer = X.copy()
    
    for col in df_omplir.columns:
        if col not in features and col not in categorical_features:
            continue 
            
        nans_idx = df_omplir[df_omplir[col].isna()].index.tolist()
        num_forats = len(nans_idx)
        
        if num_forats == 0:
            continue 
            
        possibles_solucions = get_interval(col, df_import_corr, df_sencer, features, categorical_features, kpi_solicitado)
        if not possibles_solucions:
            continue 
            
        # Eliminamos duplicados y asignamos valores inteligentemente
        set_solucions = list(set(possibles_solucions)) if col in categorical_features else possibles_solucions
        
        if len(set_solucions) >= num_forats:
            valors_escollits = random.sample(set_solucions, num_forats)
        else:
            valors_escollits = random.choices(set_solucions, k=num_forats)
            
        for i, idx in enumerate(nans_idx):
            df_omplir.at[idx, col] = valors_escollits[i]
            
    print(f"✨ Procés d'imputació intel·ligent per optimitzar {kpi_solicitado} acabat amb èxit!")
    return df_omplir


# Ejecución del motor prescriptivo
x_input_final = omplir_forats_inteligentment(
     df_entrada=df3[features + categorical_features],
     importancias_brutas=importancias_brutas,
     X=X,
     features=features,
     categorical_features=categorical_features,
     kpi_solicitado=kpi_solicitado
 )

# ==============================================================================
# 8. PREPARACIÓN FINAL Y PREDICCIÓN (KPI + FATIGA)
# ==============================================================================
# Arreglamos los booleanos para compatibilidad
cols_has = ['has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style']
for col in cols_has:
    if col in x_input_final.columns:
        x_input_final[col] = x_input_final[col].replace({True: 1, False: 0, 'True': 1, 'False': 0})

# Dummy encoding y limpieza anti-duplicados (Evita ValueError de XGBoost)
X_input_encoded = pd.get_dummies(x_input_final, columns=categorical_features)
X_input_encoded = X_input_encoded.loc[:, ~X_input_encoded.columns.duplicated()]
columnas_entrenamiento = X_train.columns[~X_train.columns.duplicated()]

# Alineamos dimensiones exactas con el modelo entrenado
X_input_encoded = X_input_encoded.reindex(columns=columnas_entrenamiento, fill_value=0)
X_input_encoded = X_input_encoded.astype(float)

# 1️⃣ Predicción del KPI principal
pred_input = modelo_xgb.predict(X_input_encoded)

# 2️⃣ Predicción de los Días hasta la Fatiga
pred_fatigue = modelo_xgb_fatigue.predict(X_input_encoded)
# Redondeamos a días enteros y evitamos que prediga números negativos
pred_fatigue = np.maximum(0, np.round(pred_fatigue)).astype(int)

# ==============================================================================
# 9. GUARDADO Y EXPORTACIÓN
# ==============================================================================
# Volcamos las sugerencias de la IA en el dataset original
df3[features + categorical_features] = x_input_final[features + categorical_features]
df3[f'Prediccion_{kpi_solicitado}'] = pred_input
df3['Prediccion_Dias_Fatiga'] = pred_fatigue  # <-- Nueva columna añadida

df3.to_csv('output_creatives.csv', index=False)

print("\n" + "="*50)
print(f"✅ ¡PROCESO COMPLETADO!")
print(f"📂 Archivo generado: 'output_creatives.csv'")
print("-" * 50)
print("🚀 PREDICCIONES:")
for index, (pred_kpi, pred_fat) in enumerate(zip(pred_input, pred_fatigue)):
    print(f" - Creativo {index + 1}: {kpi_solicitado} = {pred_kpi:.6f} | Fatiga en: {pred_fat} días")
print("="*50)