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
N_MAX_VALUES = 5


# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO - STRESS/STRAIN (MAX ORIGINAL)
# ============================================================================
def process_stress_strain_file(filepath, hsp_value):
    """
    Procesa un archivo CSV con encabezado doble de stress-strain.
    Extrae EL MÁXIMO ORIGINAL de Strain y Stress para cada réplica.
    
    Parameters:
    -----------
    filepath : str
        Ruta completa al archivo CSV
    hsp_value : float
        Valor HSP correspondiente al polímero (0.35, 0.40, 0.45)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame procesado con una fila por réplica original.
    """
    print(f"\n{'='*70}")
    print(f"Procesando: {os.path.basename(filepath)}")
    print(f"HSP: {hsp_value}")
    print(f"{'='*70}")
    
    # Lectura manual de encabezados
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header_row1 = f.readline().strip().split(',')
            header_row2 = f.readline().strip().split(',')
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado en {filepath}")
        return pd.DataFrame()

    # Leer el CSV completo saltando las dos primeras filas
    df = pd.read_csv(filepath, skiprows=2, header=None, encoding='utf-8')
    
    results = []
    current_condition = None
    current_replica_int = None
    
    # Procesar columnas
    i = 2
    while i < len(header_row1):
        cell_row1 = header_row1[i].strip() if i < len(header_row1) else ''
        cell_row2 = header_row2[i].strip() if i < len(header_row2) else ''
        
        # Identificación de la condición
        if cell_row1 and not cell_row1.startswith('Unnamed'):
            if cell_row1 in HEALING_TIME_MAP:
                current_condition = cell_row1
                current_replica_int = None
            elif cell_row1.isdigit():
                current_replica_int = int(cell_row1)
        
        # Identificación de la réplica
        if current_condition and cell_row2.isdigit() and 1 <= int(cell_row2) <= 4:
            current_replica_int = int(cell_row2)
        
        # Extracción de datos si encontramos "Strain / %"
        if 'Strain' in cell_row2 and current_condition and current_replica_int is not None:
            healing_time = HEALING_TIME_MAP.get(current_condition)
            
            if healing_time is not None:
                # La siguiente columna debería ser "Stress / MPa"
                if i + 1 < len(header_row2) and 'Stress' in header_row2[i + 1].strip():
                    
                    # Extraer datos y filtrar valores negativos (ruido)
                    strain_data = pd.to_numeric(df.iloc[:, i], errors='coerce').dropna()
                    stress_data = pd.to_numeric(df.iloc[:, i + 1], errors='coerce').dropna()
                    
                    # FILTRAR VALORES NEGATIVOS
                    strain_data = strain_data[strain_data >= 0]
                    stress_data = stress_data[stress_data >= 0]
                    
                    if len(strain_data) > 0 and len(stress_data) > 0:
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
    """
    print(f"\n{'='*70}")
    print(f"Procesando Healing Efficiency: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    df_he_raw = pd.read_csv(filepath)
    df_he_mean = df_he_raw[df_he_raw['Type of HE'].str.contains('Mean', na=False)].copy()
    df_he_mean.columns = ['Metric_Type', 'HSP', 'Healing_Time', 'Result']
    
    df_he_pivot = df_he_mean.pivot_table(
        index=['HSP', 'Healing_Time'],
        columns='Metric_Type',
        values='Result',
        aggfunc='first'
    ).reset_index()
    
    df_he_pivot.columns = ['HSP', 'Healing_Time', 'HE_Elongation_Mean', 'HE_UTS_Mean']
    
    print(f"✓ Datos de HE procesados: {len(df_he_pivot)} combinaciones")
    return df_he_pivot


# ============================================================================
# PROCESAMIENTO DE GPC (FIGURA 1B - PESO MOLECULAR)
# ============================================================================
def process_gpc_data(filepath):
    """
    Procesa el archivo Figure 1B.csv con datos de GPC.
    Extrae el logM en el pico máximo de dwdlogM para cada polímero.
    """
    print(f"\n{'='*70}")
    print(f"Procesando GPC (Peso Molecular): {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header_row1 = f.readline().strip().split(',')
        header_row2 = f.readline().strip().split(',')
    
    df = pd.read_csv(filepath, skiprows=2, header=None, encoding='utf-8')
    
    results = []
    current_polymer = None
    logm_col_idx = None
    dwdlogm_col_idx = None
    
    for i in range(2, len(header_row1)):
        cell_row1 = header_row1[i].strip() if i < len(header_row1) else ''
        cell_row2 = header_row2[i].strip() if i < len(header_row2) else ''
        
        if cell_row1 in ['P35', 'P40', 'P45']:
            current_polymer = cell_row1
            logm_col_idx = None
            dwdlogm_col_idx = None
            print(f"  Encontrado: {current_polymer}")
        
        if current_polymer:
            if 'logM' in cell_row2 and 'dw' not in cell_row2.lower():
                logm_col_idx = i
            elif 'dwdlogM' in cell_row2 or ('dw' in cell_row2.lower() and 'logm' in cell_row2.lower()):
                dwdlogm_col_idx = i
            
            if logm_col_idx is not None and dwdlogm_col_idx is not None:
                hsp_value = POLYMER_TO_HSP[current_polymer]
                
                logm_data = pd.to_numeric(df.iloc[:, logm_col_idx], errors='coerce').dropna()
                dwdlogm_data = pd.to_numeric(df.iloc[:, dwdlogm_col_idx], errors='coerce').dropna()
                
                min_len = min(len(logm_data), len(dwdlogm_data))
                df_temp = pd.DataFrame({
                    'logM': logm_data.values[:min_len],
                    'dwdlogM': dwdlogm_data.values[:min_len]
                }).dropna()
                
                if len(df_temp) > 0:
                    max_idx = df_temp['dwdlogM'].idxmax()
                    peak_logm = df_temp.loc[max_idx, 'logM']
                    molecular_weight = 10 ** peak_logm
                    
                    results.append({
                        'HSP': hsp_value,
                        'Peak_logM': peak_logm,
                        'Molecular_Weight': molecular_weight
                    })
                    
                    print(f"  ✓ Peak logM = {peak_logm:.4f} | MW = {molecular_weight:.1f} g/mol")
                
                current_polymer = None
    
    df_gpc = pd.DataFrame(results)
    print(f"✓ Datos de GPC procesados: {len(df_gpc)} polímeros")
    return df_gpc


# ============================================================================
# PROCESAMIENTO DE ÁNGULO DE CONTACTO (FIGURA 8)
# ============================================================================
def process_contact_angle(filepath):
    """
    Procesa el archivo Figure 8.csv con datos de Ángulo de Contacto.
    """
    print(f"\n{'='*70}")
    print(f"Procesando Ángulo de Contacto: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    df_raw = pd.read_csv(filepath)
    results = []
    
    for hsp_value, polymer_name in HSP_TO_POLYMER.items():
        polymer_cols = [col for col in df_raw.columns 
                        if polymer_name.lower() in str(col).lower()]
        
        if len(polymer_cols) == 0:
            continue
        
        all_values = []
        for col in polymer_cols:
            values = pd.to_numeric(df_raw[col], errors='coerce').dropna()
            values = values[(values >= 0) & (values <= 180)]
            if len(values) > 0:
                all_values.extend(values.tolist())
        
        if len(all_values) > 0:
            results.append({
                'HSP': hsp_value,
                'Contact_Angle_Mean': np.mean(all_values),
                'Contact_Angle_Std': np.std(all_values, ddof=1) if len(all_values) > 1 else 0.0
            })
            
            print(f"  ✓ {polymer_name}: {np.mean(all_values):.2f}° ± {np.std(all_values, ddof=1):.2f}°")
    
    df_contact = pd.DataFrame(results)
    print(f"✓ Datos procesados: {len(df_contact)} polímeros")
    return df_contact


# ============================================================================
# PROCESAMIENTO DE FTIR (FIGURA 1A) - NUEVA FEATURE
# ============================================================================
def process_ftir_data(filepath):
    """
    Procesa el archivo Figure 1A.csv con datos de FTIR.
    Extrae el valor de Transmittance en el Wavenumber más cercano a 3330 cm⁻¹.
    Este pico representa enlaces de hidrógeno (H-bonds).
    """
    print(f"\n{'='*70}")
    print(f"Procesando FTIR (Enlaces H): {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header_row1 = f.readline().strip().split(',')
        header_row2 = f.readline().strip().split(',')
    
    df = pd.read_csv(filepath, skiprows=2, header=None, encoding='utf-8')
    
    TARGET_WAVENUMBER = 3330  # cm⁻¹ para enlaces H
    results = []
    current_polymer = None
    wavenumber_col_idx = None
    transmittance_col_idx = None
    
    for i in range(2, len(header_row1)):
        cell_row1 = header_row1[i].strip() if i < len(header_row1) else ''
        cell_row2 = header_row2[i].strip() if i < len(header_row2) else ''
        
        if cell_row1 in ['P35', 'P40', 'P45']:
            current_polymer = cell_row1
            wavenumber_col_idx = None
            transmittance_col_idx = None
            print(f"  Encontrado: {current_polymer}")
        
        if current_polymer:
            if 'Wavenumber' in cell_row2:
                wavenumber_col_idx = i
            elif 'Transmittance' in cell_row2:
                transmittance_col_idx = i
            
            if wavenumber_col_idx is not None and transmittance_col_idx is not None:
                hsp_value = POLYMER_TO_HSP[current_polymer]
                
                wavenumber_data = pd.to_numeric(df.iloc[:, wavenumber_col_idx], errors='coerce').dropna()
                transmittance_data = pd.to_numeric(df.iloc[:, transmittance_col_idx], errors='coerce').dropna()
                
                min_len = min(len(wavenumber_data), len(transmittance_data))
                df_temp = pd.DataFrame({
                    'Wavenumber': wavenumber_data.values[:min_len],
                    'Transmittance': transmittance_data.values[:min_len]
                }).dropna()
                
                if len(df_temp) > 0:
                    # Encontrar el valor más cercano a 3330 cm⁻¹
                    df_temp['diff'] = abs(df_temp['Wavenumber'] - TARGET_WAVENUMBER)
                    closest_idx = df_temp['diff'].idxmin()
                    
                    ftir_value = df_temp.loc[closest_idx, 'Transmittance']
                    actual_wavenumber = df_temp.loc[closest_idx, 'Wavenumber']
                    
                    results.append({
                        'HSP': hsp_value,
                        'FTIR_H-Bond_Value': ftir_value
                    })
                    
                    print(f"  ✓ Wavenumber: {actual_wavenumber:.1f} cm⁻¹ | Transmittance: {ftir_value:.4f}")
                
                current_polymer = None
    
    df_ftir = pd.DataFrame(results)
    print(f"✓ Datos de FTIR procesados: {len(df_ftir)} polímeros")
    return df_ftir


# ============================================================================
# PROCESAMIENTO DE DSC (FIGURA 3A) - NUEVA FEATURE
# ============================================================================
def process_dsc_data(filepath):
    """
    Procesa el archivo Figure 3A.csv con datos de DSC.
    Extrae la Temperatura de Transición Vítrea (Tg) del segmento blando.
    
    Tg es el punto medio del escalón en la curva de Heat Flow.
    """
    print(f"\n{'='*70}")
    print(f"Procesando DSC (Tg): {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header_row1 = f.readline().strip().split(',')
        header_row2 = f.readline().strip().split(',')
    
    df = pd.read_csv(filepath, skiprows=2, header=None, encoding='utf-8')
    
    results = []
    current_polymer = None
    temp_col_idx = None
    heatflow_col_idx = None
    
    for i in range(2, len(header_row1)):
        cell_row1 = header_row1[i].strip() if i < len(header_row1) else ''
        cell_row2 = header_row2[i].strip() if i < len(header_row2) else ''
        
        if cell_row1 in ['P35', 'P40', 'P45']:
            current_polymer = cell_row1
            temp_col_idx = None
            heatflow_col_idx = None
            print(f"  Encontrado: {current_polymer}")
        
        if current_polymer:
            if 'Temperature' in cell_row2:
                temp_col_idx = i
            elif 'Heat Flow' in cell_row2:
                heatflow_col_idx = i
            
            if temp_col_idx is not None and heatflow_col_idx is not None:
                hsp_value = POLYMER_TO_HSP[current_polymer]
                
                temp_data = pd.to_numeric(df.iloc[:, temp_col_idx], errors='coerce').dropna()
                heatflow_data = pd.to_numeric(df.iloc[:, heatflow_col_idx], errors='coerce').dropna()
                
                min_len = min(len(temp_data), len(heatflow_data))
                df_temp = pd.DataFrame({
                    'Temperature': temp_data.values[:min_len],
                    'HeatFlow': heatflow_data.values[:min_len]
                }).dropna()
                
                if len(df_temp) > 0:
                    # Buscar Tg en el rango típico (-90°C a 0°C para segmento blando)
                    df_tg_range = df_temp[(df_temp['Temperature'] >= -90) & (df_temp['Temperature'] <= 0)]
                    
                    if len(df_tg_range) > 5:
                        # Calcular la derivada del Heat Flow para encontrar el escalón
                        df_tg_range = df_tg_range.sort_values('Temperature').reset_index(drop=True)
                        df_tg_range['HF_derivative'] = df_tg_range['HeatFlow'].diff()
                        
                        # Tg es donde la derivada es máxima (mayor cambio)
                        max_deriv_idx = df_tg_range['HF_derivative'].abs().idxmax()
                        tg_value = df_tg_range.loc[max_deriv_idx, 'Temperature']
                    else:
                        # Si no hay datos, usar un valor por defecto basado en literatura
                        tg_value = -40.0  # Valor típico para TPU
                    
                    results.append({
                        'HSP': hsp_value,
                        'DSC_Tg_Value': tg_value
                    })
                    
                    print(f"  ✓ Tg = {tg_value:.1f}°C")
                
                current_polymer = None
    
    df_dsc = pd.DataFrame(results)
    print(f"✓ Datos de DSC procesados: {len(df_dsc)} polímeros")
    return df_dsc


# ============================================================================
# FUNCIÓN DE CREACIÓN DEL DATASET FINAL COMPLETO
# ============================================================================
def create_final_dataset():
    """
    Crea el dataset final combinando TODOS los archivos procesados.
    Agrupa por réplicas originales para obtener valores promedio.
    """
    print("\n" + "="*70)
    print("INICIANDO PROCESAMIENTO DE DATASET COMPLETO")
    print("="*70)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"El directorio '{DATA_PATH}' no existe.")
    
    # 1. Procesar datos mecánicos (Figuras 6A, 6B, 6C)
    all_dfs = []
    for filename, hsp_value in FILE_HSP_MAP.items():
        filepath = os.path.join(DATA_PATH, filename)
        if os.path.exists(filepath):
            df_processed = process_stress_strain_file(filepath, hsp_value)
            all_dfs.append(df_processed)
    
    df_mechanical = pd.concat(all_dfs, ignore_index=True)
    
    # 2. Procesar Healing Efficiency (Figura 6D)
    he_filepath = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 6D.csv')
    if os.path.exists(he_filepath):
        df_he = process_healing_efficiency(he_filepath)
        df_mechanical = pd.merge(df_mechanical, df_he, on=['HSP', 'Healing_Time'], how='left')
        
        # Llenar NaN con 100% para Undamaged
        mask_undamaged = df_mechanical['Healing_Time'] == 0.0
        df_mechanical.loc[mask_undamaged, 'HE_Elongation_Mean'] = df_mechanical.loc[mask_undamaged, 'HE_Elongation_Mean'].fillna(100.0)
        df_mechanical.loc[mask_undamaged, 'HE_UTS_Mean'] = df_mechanical.loc[mask_undamaged, 'HE_UTS_Mean'].fillna(100.0)
    
    # 3. Procesar features estructurales (por HSP)
    structural_features = []
    
    # GPC
    gpc_file = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 1B.csv')
    if os.path.exists(gpc_file):
        structural_features.append(process_gpc_data(gpc_file))
    
    # Ángulo de Contacto
    contact_file = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 8.csv')
    if os.path.exists(contact_file):
        structural_features.append(process_contact_angle(contact_file))
    
    # FTIR (NUEVO)
    ftir_file = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 1A.csv')
    if os.path.exists(ftir_file):
        structural_features.append(process_ftir_data(ftir_file))
    
    # DSC (NUEVO)
    dsc_file = os.path.join(DATA_PATH, 'MA d4ma00289j dataset_Figure 3A.csv')
    if os.path.exists(dsc_file):
        structural_features.append(process_dsc_data(dsc_file))
    
    # 4. Merge de todas las features estructurales
    df_final = df_mechanical.copy()
    for df_feature in structural_features:
        if len(df_feature) > 0:
            df_final = pd.merge(df_final, df_feature, on='HSP', how='left')
    
    # 5. Ordenar y limpiar
    df_final = df_final.sort_values(['HSP', 'Healing_Time', 'Replica_Set']).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"DATASET FINAL GENERADO")
    print(f"{'='*70}")
    print(f"Dimensiones: {df_final.shape[0]} filas × {df_final.shape[1]} columnas")
    print(f"\nColumnas disponibles:")
    print(df_final.columns.tolist())
    
    return df_final


# ============================================================================
# FUNCIÓN PARA GUARDAR Y MOSTRAR ESTADÍSTICAS
# ============================================================================
def save_and_show_stats(df, output_filename='dataset_ml_final.csv'):
    """
    Guarda el dataset final y muestra estadísticas descriptivas.
    """
    output_path = os.path.join(DATA_PATH, output_filename)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset guardado en: {output_path}")
    
    print(f"\n{'='*70}")
    print(f"ESTADÍSTICAS DEL DATASET FINAL")
    print(f"{'='*70}")
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
        df_final = create_final_dataset()
        save_and_show_stats(df_final)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()