import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import random

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
#==============================================================================

def get_interval(nom_atribut, df_import_corr, df_sencer, features, categorical_features):
    """
    Retorna una llista de 5 valors 'ideals' per un atribut concret,
    basant-se en la importància i la correlació.
    """
    # Canvi: ara comprovem directament contra el paràmetre 'categorical_features'
    if nom_atribut in categorical_features:
        # 1. CATEGÒRICS: Filtrar les files que comencen per "nom_atribut_" (ex: "format_")
        prefix = nom_atribut + "_"
        df_filtre = df_import_corr[df_import_corr['Columna_XGB'].str.startswith(prefix, na=False)].copy()
        
        if df_filtre.empty:
            return [] # Si no hi ha dades, retornem buit
            
        # Calcular el SCORE: Importància * Correlació
        df_filtre['Score'] = df_filtre['Importancia'] * df_filtre['Correlacion']
        
        # Ordenar de millor a pitjor score
        df_filtre = df_filtre.sort_values(by='Score', ascending=False)
        
        # Agafar el TOP 5 dels millors valors. 
        # Tallem el prefix per quedar-nos només amb el valor (ex: "format_banner" -> "banner")
        millors_candidats = df_filtre.head(5)['Columna_XGB'].apply(lambda x: x[len(prefix):]).tolist()
        
        return millors_candidats

    # Canvi: ara comprovem directament contra el paràmetre 'features' (els numèrics)
    elif nom_atribut in features:
        # 2. NUMÈRICS: Busquem la correlació
        fila = df_import_corr[df_import_corr['Columna_XGB'] == nom_atribut]
        corr = fila['Correlacion'].values[0] if not fila.empty else 0
        
        # Necessitem el df_sencer (el creatiu original) per saber el Mínim i el Màxim real d'aquest atribut
        dades_reals = df_sencer[nom_atribut].dropna()
        if dades_reals.empty:
            return [0, 0, 0, 0, 0]
            
        minim = dades_reals.min()
        maxim = dades_reals.max()
        q25 = dades_reals.quantile(0.25) # El valor que marca el 25% més baix
        q75 = dades_reals.quantile(0.75) # El valor que marca el 25% més alt
        
        # Acotem l'interval depenent de si la correlació és bona o dolenta
        if corr > 0:
            # Correlació Positiva: Com més gran millor (Interval del 75% al Màxim)
            rang_min, rang_max = q75, maxim
        else:
            # Correlació Negativa: Com més petit millor (Interval del Mínim al 25%)
            rang_min, rang_max = minim, q25
            
        # Generem 5 valors aleatoris dins de l'interval òptim calculat
        valors_generats = []
        es_enter = pd.api.types.is_integer_dtype(dades_reals) # Comprovem si són enters o decimals
        
        for _ in range(5):
            if es_enter:
                valors_generats.append(random.randint(int(rang_min), int(rang_max)))
            else:
                valors_generats.append(round(random.uniform(rang_min, rang_max), 2))
                
        return valors_generats
    
    return []

def omplir_forats_inteligentment(df_entrada, importancias_brutas, X, features, categorical_features):
    """
    Rellena los valores nulos de forma inteligente basándose en la importancia de variables.
    
    Parámetros:
    - df_entrada: DataFrame con los datos a rellenar (ej. df[features + categorical_features]).
    - importancias_brutas: DataFrame con las importancias y correlaciones calculadas por XGBoost.
    - X: DataFrame con el dataset completo original.
    - features: Lista de columnas numéricas.
    - categorical_features: Lista de columnas categóricas.
    
    Retorna:
    - x_input_final: DataFrame con los valores imputados.
    """
    # 1. Fem còpies per no modificar els DataFrames originals de fora de la funció
    df_omplir = df_entrada.copy()
    df_import_corr = importancias_brutas.copy()
    df_sencer = X.copy()
    
    # 2. Recorrem columna a columna (atribut per atribut)
    for col in df_omplir.columns:
        if col not in features and col not in categorical_features:
            continue # Saltem IDs o columnes que no estiguin a les llistes
            
        # Busquem l'índex on hi ha NaNs a aquesta columna
        nans_idx = df_omplir[df_omplir[col].isna()].index.tolist()
        num_forats = len(nans_idx)
        
        if num_forats == 0:
            continue # No hi ha res a omplir aquí
            
        # Cridem la funció per rebre les solucions òptimes, passant-li els nous paràmetres
        possibles_solucions = get_interval(col, df_import_corr, df_sencer, features, categorical_features)
        
        if not possibles_solucions:
            continue # Si alguna cosa falla, saltem
            
        # Eliminem duplicats en cas dels categòrics fent-ho un 'set'
        set_solucions = list(set(possibles_solucions)) if col in categorical_features else possibles_solucions
        
        # La regla màgica de la longitud >= 5
        if len(set_solucions) >= num_forats:
            # Randomitzem sense repetir (garantim que siguin diferents)
            valors_escollits = random.sample(set_solucions, num_forats)
        else:
            # Si hi ha menys opcions que forats, randomitzem deixant que es repeteixin
            valors_escollits = random.choices(set_solucions, k=num_forats)
            
        # Omplim el vector de valors directament dins del dataframe
        for i, idx in enumerate(nans_idx):
            df_omplir.at[idx, col] = valors_escollits[i]
            
    print("✨ Procés d'imputació intel·ligent acabat amb èxit!")
    
    # 3. Retornem el DataFrame ja emplenat
    x_input_final = df_omplir
    return x_input_final

x_input_final = omplir_forats_inteligentment(
     df_entrada=df3[features + categorical_features],
     importancias_brutas=importancias_brutas,
     X=X,
     features=features,
     categorical_features=categorical_features
 )


# ==============================================================================
# 7. PREPARAR DATOS FINALES Y PREDECIR (CORRECCIÓN APLICADA)
# ==============================================================================

# 1. Arreglar el problema de los booleanos (Convertir True/False a 1/0)
cols_has = ['has_price', 'has_discount_badge', 'has_gameplay', 'has_ugc_style']
for col in cols_has:
    if col in x_input_final.columns:
        x_input_final[col] = x_input_final[col].replace({True: 1, False: 0, 'True': 1, 'False': 0})

# 2. Generar las columnas dummies para los nuevos creativos
X_input_encoded = pd.get_dummies(x_input_final, columns=categorical_features)

# 🛠️ EL FILTRO ANTI-DUPLICADOS (La solución al error)
# Esto elimina cualquier columna repetida que haya podido generar get_dummies
X_input_encoded = X_input_encoded.loc[:, ~X_input_encoded.columns.duplicated()]

# Nos aseguramos también de que el entrenamiento original no tuviera duplicados fantasma
columnas_entrenamiento = X_train.columns[~X_train.columns.duplicated()]

# 3. ✨ LA MAGIA: Alinear las columnas exactamente con las de entrenamiento
X_input_encoded = X_input_encoded.reindex(columns=columnas_entrenamiento, fill_value=0)

# 4. Asegurar que todo sea float (Igual que hicimos arriba para XGBoost)
X_input_encoded = X_input_encoded.astype(float)

# 5. Hacemos la predicción sobre los nuevos creativos
pred_input = modelo_xgb.predict(X_input_encoded)

# ==============================================================================
# 8. GUARDAR Y EXPORTAR
# ==============================================================================

# Rellenar nulos originales en df3 por si acaso
df3[features] = df3[features].fillna(1)
df3[categorical_features] = df3[categorical_features].fillna("1")

# Añadimos la columna de predicción
nombre_columna_pred = f'Prediccion_{kpi_solicitado}'
df3[nombre_columna_pred] = pred_input

# Guardamos el archivo final
df3.to_csv('output_creatives.csv', index=False)

print("\n" + "="*40)
print(f"✅ ¡PROCESO COMPLETADO!")
print(f"📂 Archivo generado: 'output_creatives.csv'")
print("-" * 40)
print("🚀 PREDICCIONES:")
for index, pred in enumerate(pred_input):
    print(f" - Creativo {index + 1}: {kpi_solicitado} estimado = {pred:.6f}")
print("="*40)