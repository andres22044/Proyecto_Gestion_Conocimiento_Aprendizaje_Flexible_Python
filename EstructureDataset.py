import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
DATA_PATH = 'datasets'

# Mapeo de condiciones de healing a valores numéricos
HEALING_TIME_MAP = {
    'Undamaged': 0.0,
    '20 min healing': 0.33,
    '1 hr healing': 1.0,
    '3 hr healing': 3.0
}

# Mapeo de archivos a valores HSP
FILE_HSP_MAP = {
    'MA d4ma00289j dataset_Figure 6A.csv': 0.35,  # P35
    'MA d4ma00289j dataset_Figure 6B.csv': 0.40,  # P40
    'MA d4ma00289j dataset_Figure 6C.csv': 0.45   # P45
}

HSP_TO_POLYMER = {
    0.35: 'P35',
    0.40: 'P40',
    0.45: 'P45'
}

POLYMER_TO_HSP = {
    'P35': 0.35,
    'P40': 0.40,
    'P45': 0.45
}

# --- CONSTANTE DE AUMENTO DE DATOS ---
N_MAX_VALUES = 3


# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ============================================================================
def process_stress_strain_file(filepath, hsp_value):
    """
    Procesa un archivo CSV con encabezado doble de stress-strain.
    
    Extrae los N_MAX_VALUES (5) valores máximos de Strain y Stress
    para cada réplica, aumentando el número de filas en el dataset.
    
    Parameters:
    -----------
    filepath : str
        Ruta completa al archivo CSV
    hsp_value : float
        Valor HSP correspondiente al polímero (0.35, 0.40, 0.45)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame procesado con N_MAX_VALUES veces más filas que antes.
    """
    global HEALING_TIME_MAP, N_MAX_VALUES
    
    print(f"\n{'='*70}")
    print(f"Procesando: {os.path.basename(filepath)} | Generando {N_MAX_VALUES} réplicas sintéticas por set original.")
    print(f"HSP: {hsp_value}")
    print(f"{'='*70}")
    
    # 1. Lectura manual de encabezados
    try:
        with open(filepath, 'r') as f:
            # Aseguramos que la lectura del archivo no esté en la definición de la función original
            header_row1 = f.readline().strip().split(',')
            header_row2 = f.readline().strip().split(',')
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {filepath}")
        return pd.DataFrame()

    # 2. Leer el CSV completo saltando las dos primeras filas
    df = pd.read_csv(filepath, skiprows=2, header=None)
    
    results = []
    
    current_condition = None
    current_replica_int = None
    
    # 3. Procesar columnas
    i = 2
    while i < len(header_row1):
        cell_row1 = header_row1[i].strip() if i < len(header_row1) else ''
        cell_row2 = header_row2[i].strip() if i < len(header_row2) else ''
        
        # A. Identificación de la condición (e.g., P35 Undamaged)
        if cell_row1 and not cell_row1.startswith('Unnamed'):
            potential_condition = cell_row1
            if potential_condition in HEALING_TIME_MAP:
                current_condition = potential_condition
                current_replica_int = None # Reset replica when condition changes
            elif cell_row1.isdigit():
                 current_replica_int = int(cell_row1)
        
        # B. Identificación de la réplica (e.g., 1, 2, 3)
        if current_condition and cell_row2.isdigit() and int(cell_row2) >= 1 and int(cell_row2) <= 4:
            current_replica_int = int(cell_row2)
        
        
        # C. Extracción de datos si encontramos "Strain / %"
        if 'Strain' in cell_row2 and current_condition and current_replica_int is not None:
            # Buscar la clave de healing_time más específica
            healing_time_key = next((k for k in HEALING_TIME_MAP if k in current_condition), current_condition)
            
            if healing_time_key in HEALING_TIME_MAP:
                healing_time = HEALING_TIME_MAP[healing_time_key]
                
                # La siguiente columna debería ser "Stress / MPa"
                if i + 1 < len(header_row2) and 'Stress' in header_row2[i + 1].strip():
                    
                    # 4. Extracción de los N_MAX_VALUES máximos 🔑
                    
                    # Asegurar float y quitar NaN
                    strain_data = df.iloc[:, i].dropna().astype(float)
                    stress_data = df.iloc[:, i + 1].dropna().astype(float)
                    
                    # Filtra y ordena los valores POSITIVOS para encontrar los máximos reales
                    # Esto evita el ruido de la máquina cerca del 0 o valores negativos iniciales.
                    strain_positive = strain_data[strain_data > 0].sort_values(ascending=False)
                    stress_positive = stress_data[stress_data > 0].sort_values(ascending=False)

                    # Determinar cuántas filas podemos generar (el mínimo de N y los datos disponibles)
                    num_to_extract = min(N_MAX_VALUES, len(strain_positive), len(stress_positive))

                    # Tomar los N valores superiores
                    top_strains = strain_positive.head(num_to_extract).reset_index(drop=True)
                    top_stresses = stress_positive.head(num_to_extract).reset_index(drop=True)
                    
                    # 5. Generar las N_MAX_VALUES nuevas réplicas (filas)
                    for k in range(num_to_extract):
                        
                        # Usar un ID de réplica compuesto (ej., '1-1', '1-2')
                        replica_id = f"{current_replica_int}-{k+1}" 
                        
                        results.append({
                            'HSP': hsp_value,
                            'Healing_Time': healing_time,
                            'Replica_Set': replica_id,
                            'Max_Strain': top_strains.iloc[k],
                            'Max_Stress': top_stresses.iloc[k]
                        })
                        
                    print(f"✓ Procesado: {current_condition} - Set {current_replica_int} | Generadas {num_to_extract} réplicas sintéticas.")
                    
                    i += 2  # Saltar Strain y Stress
                    continue
        
        i += 1
    
    df_result = pd.DataFrame(results)
    print(f"\n📊 Total de réplicas procesadas: {len(df_result)}")
    
    return df_result

# ============================================================================
# PROCESAMIENTO DE HEALING EFFICIENCY
# ============================================================================
def process_healing_efficiency(filepath):
    """
    Procesa el archivo Figure 6D.csv con datos de eficiencia de healing.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV con datos de healing efficiency
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con columnas: HSP, Healing_Time, HE_Elongation_Mean, HE_UTS_Mean
    """
    print(f"\n{'='*70}")
    print(f"Procesando Healing Efficiency: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Leer el archivo
    df_he_raw = pd.read_csv(filepath)
    
    # Filtrar solo las métricas Mean (ignorar Std)
    df_he_mean = df_he_raw[df_he_raw['Type of HE'].str.contains('Mean', na=False)].copy()
    
    # Renombrar columnas
    df_he_mean.columns = ['Metric_Type', 'HSP', 'Healing_Time', 'Result']
    
    # Crear pivot table
    df_he_pivot = df_he_mean.pivot_table(
        index=['HSP', 'Healing_Time'],
        columns='Metric_Type',
        values='Result',
        aggfunc='first'
    ).reset_index()
    
    # Renombrar columnas para claridad
    df_he_pivot.columns = ['HSP', 'Healing_Time', 'HE_Elongation_Mean', 'HE_UTS_Mean']
    
    print(f"\n✓ Datos de HE procesados: {len(df_he_pivot)} combinaciones HSP × Healing_Time")
    print(df_he_pivot)
    
    return df_he_pivot


# ============================================================================
# FUNCIÓN DE MERGE Y GENERACIÓN DEL DATASET FINAL
# ============================================================================
def create_final_dataset():
    """
    Crea el dataset final combinando todos los archivos y añadiendo HE.
    
    Returns:
    --------
    pd.DataFrame
        Dataset final listo para Machine Learning
    """
    print("\n" + "="*70)
    print("INICIANDO PROCESAMIENTO DE DATASET COMPLETO")
    print("="*70)
    
    # Verificar que existe el directorio
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"El directorio '{DATA_PATH}' no existe. "
                                f"Por favor créalo y coloca los archivos CSV allí.")
    
    # Lista para almacenar DataFrames procesados
    all_dfs = []
    
    # Procesar cada archivo de stress-strain (Figuras 6A, 6B, 6C)
    for filename, hsp_value in FILE_HSP_MAP.items():
        filepath = os.path.join(DATA_PATH, filename)
        
        if os.path.exists(filepath):
            df_processed = process_stress_strain_file(filepath, hsp_value)
            all_dfs.append(df_processed)
        else:
            print(f"⚠️  Archivo no encontrado: {filepath}")
    
    # Concatenar todos los DataFrames
    df_mechanical = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"RESUMEN DE DATOS MECÁNICOS")
    print(f"{'='*70}")
    print(f"Total de muestras: {len(df_mechanical)}")
    print(f"\nDistribución por HSP:")
    print(df_mechanical['HSP'].value_counts().sort_index())
    print(f"\nDistribución por Healing_Time:")
    print(df_mechanical['Healing_Time'].value_counts().sort_index())
    
    # Procesar healing efficiency (Figura 6D)
    he_filepath = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 6D.csv')
    
    if os.path.exists(he_filepath):
        df_he = process_healing_efficiency(he_filepath)
        
        # Merge con datos mecánicos
        df_final = pd.merge(
            df_mechanical,
            df_he,
            on=['HSP', 'Healing_Time'],
            how='left'
        )
        
        # Llenar valores NaN de eficiencia donde Healing_Time = 0.0 con 100.0%
        mask_undamaged = df_final['Healing_Time'] == 0.0
        df_final.loc[mask_undamaged, 'HE_Elongation_Mean'] = df_final.loc[mask_undamaged, 'HE_Elongation_Mean'].fillna(100.0)
        df_final.loc[mask_undamaged, 'HE_UTS_Mean'] = df_final.loc[mask_undamaged, 'HE_UTS_Mean'].fillna(100.0)
        
        print(f"\n{'='*70}")
        print(f"MERGE COMPLETADO CON HEALING EFFICIENCY")
        print(f"{'='*70}")
    else:
        print(f"\n⚠️  Archivo de HE no encontrado: {he_filepath}")
        df_final = df_mechanical
    
    # Reordenar columnas
    column_order = ['HSP', 'Healing_Time', 'Replica_Set', 'Max_Strain', 'Max_Stress']
    if 'HE_Elongation_Mean' in df_final.columns:
        column_order.extend(['HE_Elongation_Mean', 'HE_UTS_Mean'])
    
    df_final = df_final[column_order]
    
    # Ordenar por HSP, Healing_Time, Replica_Set
    df_final = df_final.sort_values(['HSP', 'Healing_Time', 'Replica_Set']).reset_index(drop=True)
    
    return df_final

# ============================================================================
# PROCESAMIENTO DE GPC (FIGURA 1B - PESO MOLECULAR)
# ============================================================================
def process_gpc_data(filepath):
    """
    Procesa el archivo Figure 1B.csv con datos de GPC (Gel Permeation Chromatography).
    
    La GPC mide la distribución de peso molecular de los polímeros.
    Extrae el logM en el pico máximo de dwdlogM para cada polímero.
    
    Estructura esperada del CSV (doble encabezado):
    Fila 1: Machine Name, descripción, P35, , , P40, , , P45, 
    Fila 2: , descripción, logM, dwdlogM, , logM, dwdlogM, , logM, dwdlogM
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV con datos de GPC
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con columnas: HSP, Peak_logM, Max_dwdlogM
        
    Ejemplo de salida:
        HSP  Peak_logM  Max_dwdlogM
        0.35    4.523       0.856
        0.40    4.612       0.923
        0.45    4.498       0.801
    """
    print(f"\n{'='*70}")
    print(f"PROCESANDO GPC (Peso Molecular)")
    print(f"Archivo: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Leer las primeras dos filas manualmente para entender la estructura
    with open(filepath, 'r') as f:
        header_row1 = f.readline().strip().split(',')
        header_row2 = f.readline().strip().split(',')
    
    print(f"Fila 1 (Polímeros): {header_row1}")
    print(f"Fila 2 (Métricas): {header_row2}")
    
    # Leer los datos saltando las dos primeras filas
    df = pd.read_csv(filepath, skiprows=2, header=None)
    print(f"✓ Datos leídos: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    results = []
    
    # Variables para trackear el polímero actual
    current_polymer = None
    logm_col_idx = None
    dwdlogm_col_idx = None
    
    # Procesar columnas (empezar desde la columna 2 para saltar descripción)
    i = 2
    print (len(header_row1))
    while i < (len(header_row1)):
        cell_row1 = header_row1[i].strip() if i <= (len(header_row1)) else ''
        cell_row2 = header_row2[i].strip() if i <= (len(header_row2)) else ''
        
        # Si encontramos un nombre de polímero en row1 (P35, P40, P45)
        if cell_row1 in ['P35', 'P40', 'P45']:
            current_polymer = cell_row1
            logm_col_idx = None
            dwdlogm_col_idx = None
            print(f"\n--- Encontrado polímero: {current_polymer} en columna {i} ---")
            
        
        # Si tenemos un polímero activo, buscar sus columnas logM y dwdlogM
        if current_polymer:
            if 'logM' in cell_row2 and 'dw' not in cell_row2.lower():
                logm_col_idx = i
                print(f"  logM en columna {logm_col_idx}")
            elif 'dwdlogM' in cell_row2 or 'dw' in cell_row2.lower():
                dwdlogm_col_idx = i
                print(f"  dwdlogM en columna {dwdlogm_col_idx}")
            
            # Si ya tenemos ambas columnas, procesar
            if logm_col_idx is not None and dwdlogm_col_idx is not None:
                hsp_value = POLYMER_TO_HSP[current_polymer]
                
                # Extraer datos
                logm_data = pd.to_numeric(df.iloc[:, logm_col_idx], errors='coerce').dropna()
                dwdlogm_data = pd.to_numeric(df.iloc[:, dwdlogm_col_idx], errors='coerce').dropna()
                
                # Crear DataFrame temporal
                df_temp = pd.DataFrame({
                    'logM': logm_data,
                    'dwdlogM': dwdlogm_data
                })
                
                # Asegurar que tienen el mismo índice
                df_temp = df_temp.dropna()
                
                print(f"  Datos válidos: {len(df_temp)} puntos")
                
                if len(df_temp) > 0:
                    # Encontrar el pico máximo
                    max_idx = df_temp['dwdlogM'].idxmax()
                    peak_logm = df_temp.loc[max_idx, 'logM']
                    max_dwdlogm = df_temp.loc[max_idx, 'dwdlogM']
                    
                    # Calcular peso molecular
                    molecular_weight = 10 ** peak_logm
                    
                    results.append({
                        'HSP': hsp_value,
                        'Polymer': current_polymer,
                        'Peak_logM': peak_logm,
                        'Max_dwdlogM': max_dwdlogm,
                        'Molecular_Weight': molecular_weight
                    })
                    
                    print(f"  ✓ Peak logM = {peak_logm:.4f}")
                    print(f"    Peso Molecular = {molecular_weight:.1f} g/mol")
                    print(f"    Max dwdlogM = {max_dwdlogm:.4f}")
                
                # Resetear para el siguiente polímero
                current_polymer = None
        
        i += 1
    
    # Crear DataFrame de resultados
    df_gpc = pd.DataFrame(results)
    
    if len(df_gpc) > 0:
        print(f"\n{'='*70}")
        print(f"RESUMEN GPC")
        print(f"{'='*70}")
        print(df_gpc[['Polymer', 'HSP', 'Peak_logM', 'Molecular_Weight']])
        print(f"\n✓ Datos de GPC procesados exitosamente: {len(df_gpc)} polímeros")
    else:
        print(f"\n⚠️  No se pudieron procesar datos de GPC")
    
    return df_gpc


# ============================================================================
# PROCESAMIENTO DE ÁNGULO DE CONTACTO (FIGURA 8)
# ============================================================================
def process_contact_angle(filepath):
    """
    Procesa el archivo Figure 8.csv con datos de Ángulo de Contacto.
    
    El ángulo de contacto mide la hidrofobicidad de la superficie del polímero.
    Ángulos mayores = más hidrofóbico.
    
    Estructura esperada del CSV:
    - Columnas por polímero con réplicas: P35_1, P35_2, P35_3, etc.
    - Puede tener filas de metadata/unidades al inicio
    - Valores en grados (°)
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV con datos de ángulo de contacto
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con columnas: HSP, Polymer, Contact_Angle_Mean, Contact_Angle_Std, N_Samples
        
    Ejemplo de salida:
        HSP  Polymer  Contact_Angle_Mean  Contact_Angle_Std  N_Samples
        0.35   P35          89.5               2.3              4
        0.40   P40          92.1               1.8              4
        0.45   P45          87.3               3.1              4
    """
    print(f"\n{'='*70}")
    print(f"PROCESANDO ÁNGULO DE CONTACTO")
    print(f"Archivo: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Leer el archivo CSV
    try:
        df_raw = pd.read_csv(filepath)
        print(f"✓ Archivo leído: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")
        print(f"  Columnas disponibles: {df_raw.columns.tolist()}")
    except Exception as e:
        print(f"❌ Error al leer archivo: {e}")
        return pd.DataFrame()
    
    results = []
    
    # Procesar cada polímero (P35, P40, P45)
    for hsp_value, polymer_name in HSP_TO_POLYMER.items():
        print(f"\n--- Procesando {polymer_name} (HSP={hsp_value}) ---")
        
        # Buscar todas las columnas que contengan el nombre del polímero
        # Formato esperado: "P35", "P35_1", "P35 replica 1", etc.
        polymer_cols = [col for col in df_raw.columns 
                        if polymer_name.lower() in str(col).lower()]
        
        print(f"  Columnas encontradas: {polymer_cols}")
        
        if len(polymer_cols) == 0:
            print(f"  ⚠️  No se encontraron columnas para {polymer_name}")
            continue
        
        # Extraer todos los valores numéricos de todas las columnas del polímero
        all_values = []
        
        for col in polymer_cols:
            # Convertir a numérico y limpiar NaN
            values = pd.to_numeric(df_raw[col], errors='coerce').dropna()
            
            # Filtrar valores razonables (ángulos entre 0 y 180 grados)
            values = values[(values >= 0) & (values <= 180)]
            
            if len(values) > 0:
                all_values.extend(values.tolist())
                print(f"    '{col}': {len(values)} valores válidos")
        
        print(f"  Total de mediciones válidas: {len(all_values)}")
        
        if len(all_values) == 0:
            print(f"  ⚠️  No se encontraron datos válidos para {polymer_name}")
            continue
        
        # Calcular estadísticas
        angle_mean = np.mean(all_values)
        angle_std = np.std(all_values, ddof=1) if len(all_values) > 1 else 0.0
        angle_median = np.median(all_values)
        angle_min = np.min(all_values)
        angle_max = np.max(all_values)
        
        results.append({
            'HSP': hsp_value,
            'Polymer': polymer_name,
            'Contact_Angle_Mean': angle_mean,
            'Contact_Angle_Std': angle_std,
            'Contact_Angle_Median': angle_median,
            'Contact_Angle_Min': angle_min,
            'Contact_Angle_Max': angle_max,
            'N_Samples': len(all_values)
        })
        
        print(f"  ✓ Ángulo de Contacto = {angle_mean:.2f}° ± {angle_std:.2f}°")
        print(f"    Rango: [{angle_min:.2f}° - {angle_max:.2f}°]")
        print(f"    Mediana: {angle_median:.2f}°")
    
    # Crear DataFrame de resultados
    df_contact = pd.DataFrame(results)
    
    if len(df_contact) > 0:
        print(f"\n{'='*70}")
        print(f"RESUMEN ÁNGULO DE CONTACTO")
        print(f"{'='*70}")
        print(df_contact[['Polymer', 'HSP', 'Contact_Angle_Mean', 'Contact_Angle_Std', 'N_Samples']])
        print(f"\n✓ Datos de Ángulo de Contacto procesados exitosamente: {len(df_contact)} polímeros")
    else:
        print(f"\n⚠️  No se pudieron procesar datos de Ángulo de Contacto")
    
    return df_contact


# ============================================================================
# FUNCIÓN DE MERGE CON DATASET PRINCIPAL
# ============================================================================
def merge_additional_features(df_final, df_gpc, df_contact):
    """
    Fusiona los datos de GPC y Ángulo de Contacto con el dataset principal.
    
    Parameters:
    -----------
    df_main : pd.DataFrame
        Dataset principal con datos mecánicos
    df_gpc : pd.DataFrame
        Dataset con datos de GPC
    df_contact : pd.DataFrame
        Dataset con datos de Ángulo de Contacto
    
    Returns:
    --------
    pd.DataFrame
        Dataset combinado con todas las features
    """
    print(f"\n{'='*70}")
    print(f"FUSIONANDO FEATURES ADICIONALES")
    print(f"{'='*70}")
    
    df_result = df_final.copy()
    initial_rows = len(df_result)
    
    # Merge con GPC (por HSP)
    if df_gpc is not None and len(df_gpc) > 0:
        df_result = pd.merge(
            df_result,
            df_gpc[['HSP', 'Peak_logM', 'Molecular_Weight']],
            on='HSP',
            how='left'
        )
        print(f"✓ GPC merged: {df_result['Peak_logM'].notna().sum()} filas con datos")
    
    # Merge con Ángulo de Contacto (por HSP)
    if df_contact is not None and len(df_contact) > 0:
        df_result = pd.merge(
            df_result,
            df_contact[['HSP', 'Contact_Angle_Mean', 'Contact_Angle_Std']],
            on='HSP',
            how='left'
        )
        print(f"✓ Ángulo de Contacto merged: {df_result['Contact_Angle_Mean'].notna().sum()} filas con datos")
    
    final_rows = len(df_result)
    
    if initial_rows == final_rows:
        print(f"✓ Merge exitoso sin pérdida de datos ({final_rows} filas)")
    else:
        print(f"⚠️  Cambio en número de filas: {initial_rows} → {final_rows}")
    
    return df_result

# ============================================================================
# FUNCIÓN PARA GUARDAR Y MOSTRAR ESTADÍSTICAS
# ============================================================================
def save_and_show_stats(df, output_filename='dataset_ml_final.csv'):
    """
    Guarda el dataset final y muestra estadísticas descriptivas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset final
    output_filename : str
        Nombre del archivo de salida
    """
    # Guardar dataset
    output_path = os.path.join(DATA_PATH, output_filename)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset guardado en: {output_path}")
    
    # Mostrar estadísticas
    print(f"\n{'='*70}")
    print(f"ESTADÍSTICAS DEL DATASET FINAL")
    print(f"{'='*70}")
    print(f"\nDimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"\nPrimeras 10 filas:")
    print(df.head(10))
    print(f"\nEstadísticas descriptivas:")
    print(df.describe())
    print(f"\nValores faltantes por columna:")
    print(df.isnull().sum())
    print(f"\n{'='*70}")
    print("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print(f"{'='*70}")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    try:
        # Crear dataset final
        df_final = create_final_dataset()

        # Configuración
        DATA_PATH = 'datasets'
        
        # Procesar GPC
        gpc_file = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 1B.csv')
        if os.path.exists(gpc_file):
            df_gpc = process_gpc_data(gpc_file)
            print("\n" + "="*70)
            print("DATASET GPC FINAL:")
            print("="*70)
            print(df_gpc)
        else:
            print(f"⚠️  Archivo no encontrado: {gpc_file}")
        
        # Procesar Ángulo de Contacto
        contact_file = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 8.csv')
        if os.path.exists(contact_file):
            df_contact = process_contact_angle(contact_file)
            print("\n" + "="*70)
            print("DATASET ÁNGULO DE CONTACTO FINAL:")
            print("="*70)
            print(df_contact)
        else:
            print(f"⚠️  Archivo no encontrado: {contact_file}")

        df_result = merge_additional_features(df_final, df_gpc, df_contact)
        
        # Guardar y mostrar estadísticas
        save_and_show_stats(df_result)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()