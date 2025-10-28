import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)

class OptimizedTPUPropertyPredictor:
    """
    Predictor optimizado de propiedades mecánicas de TPU usando Random Forest.
    
    Mejoras implementadas:
    - Normalización (StandardScaler) de todas las entradas
    - Features estructurales como inputs (GPC, Contact Angle, FTIR, DSC)
    - GridSearchCV para optimización de hiperparámetros
    - Leave-One-Out Cross-Validation
    - Enfoque en Max_Stress como variable crítica
    """

    def __init__(self, random_state=42):
        """
        Inicializa el predictor con configuración optimizada.
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.X = None
        self.Y = None
        self.X_scaled = None
        
        # Definición de variables
        self.feature_names = [
            'HSP',
            'Healing_Time',
            'Peak_logM',
            'Molecular_Weight',
            'Contact_Angle_Mean',
            'Contact_Angle_Std',
            'FTIR_H-Bond_Value',
            'DSC_Tg_Value'
        ]
        
        self.target_names = [
            'Max_Stress',
            'Max_Strain',
            'HE_Elongation_Mean',
            'HE_UTS_Mean'
        ]
        
        print("="*70)
        print("OPTIMIZED TPU PROPERTY PREDICTOR")
        print("="*70)
        print(f"Features (X - Inputs): {len(self.feature_names)}")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"  {i}. {feat}")
        print(f"\nTargets (Y - Outputs): {len(self.target_names)}")
        for i, targ in enumerate(self.target_names, 1):
            print(f"  {i}. {targ}")
        print("="*70)

    def load_data(self, filepath='datasets/dataset_ml_final.csv'):
        """
        Carga el dataset y prepara las variables X e Y.
        Aplica normalización a las entradas.
        """
        print(f"\n📂 Cargando dataset desde: {filepath}...")
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo en {filepath}.")
            return False
        
        print(f"✓ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Verificar columnas
        required_cols = self.feature_names + self.target_names
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Error: Faltan columnas en el CSV: {missing_cols}")
            print(f"Columnas disponibles: {df.columns.tolist()}")
            return False
        
        # Separar X e Y
        self.X = df[self.feature_names].copy()
        self.Y = df[self.target_names].copy()
        
        # Manejar valores faltantes
        print("\n🔍 Verificando valores faltantes...")
        missing_X = self.X.isnull().sum()
        missing_Y = self.Y.isnull().sum()
        
        if missing_X.sum() > 0:
            print(f"⚠️  Valores faltantes en X:\n{missing_X[missing_X > 0]}")
            print("   Rellenando con la mediana...")
            self.X = self.X.fillna(self.X.median())
        
        if missing_Y.sum() > 0:
            print(f"⚠️  Valores faltantes en Y:\n{missing_Y[missing_Y > 0]}")
            print("   Rellenando con la mediana...")
            self.Y = self.Y.fillna(self.Y.median())
        
        # NORMALIZACIÓN DE ENTRADAS (CRÍTICO)
        print("\n📊 Aplicando normalización (StandardScaler) a las entradas...")
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.feature_names,
            index=self.X.index
        )
        
        print("✓ Normalización completada")
        print(f"\nEstadísticas de X normalizado:")
        print(self.X_scaled.describe())
        
        print(f"\n✓ Datos preparados: {len(df)} muestras")
        return True

    def optimize_hyperparameters(self, cv_folds=None):
        """
        Optimiza los hiperparámetros del Random Forest usando GridSearchCV.
        
        Parameters:
        -----------
        cv_folds : int or None
            Si None, usa LeaveOneOut. Si int, usa K-Fold.

            Usando mejores parámetros de GridSearchCV:
            - max_depth: 3
            - max_features: sqrt
            - min_samples_leaf: 1
            - min_samples_split: 2
            - n_estimators: 50
        """
        if self.X_scaled is None or self.Y is None:
            print("❌ Error: Primero debes cargar los datos con load_data()")
            return
        
        print("\n" + "="*70)
        print("OPTIMIZACIÓN DE HIPERPARÁMETROS (GridSearchCV)")
        print("="*70)
        
        # Definir el espacio de búsqueda
        # Valores bajos para evitar overfitting en dataset pequeño
        param_grid = {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 3],
            'n_estimators': [50, 100, 150],
            'max_features': ['sqrt', 'log2']
        }
        
        print(f"Espacio de búsqueda: {sum([len(v) for v in param_grid.values()])} combinaciones")
        print(f"Parámetros a optimizar:")
        for key, values in param_grid.items():
            print(f"  - {key}: {values}")
        
        # Configurar validación cruzada
        if cv_folds is None:
            cv = LeaveOneOut()
            cv_name = "Leave-One-Out"
            print(f"\nValidación cruzada: {cv_name} ({len(self.X_scaled)} folds)")
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_name = f"{cv_folds}-Fold"
            print(f"\nValidación cruzada: {cv_name}")
        
        # Crear modelo base
        rf_base = RandomForestRegressor(random_state=self.random_state)
        
        # GridSearchCV
        print("\n⏳ Ejecutando GridSearchCV (esto puede tardar varios minutos)...\n")
        
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',  # Optimizar para R²
            n_jobs=-1,  # Usar todos los cores
            verbose=1
        )
        
        grid_search.fit(self.X_scaled, self.Y)
        
        # Guardar mejores parámetros
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        print("\n" + "="*70)
        print("RESULTADOS DE OPTIMIZACIÓN")
        print("="*70)
        print(f"Mejor R² Score (CV): {grid_search.best_score_:.4f}")
        print(f"\nMejores hiperparámetros:")
        for key, value in self.best_params.items():
            print(f"  - {key}: {value}")
        print("="*70)
        
        return self.best_params

    def train_final_model(self, custom_params=None):
        """
        Entrena el modelo final con todos los datos.
        
        Parameters:
        -----------
        custom_params : dict or None
            Si None, usa los mejores parámetros de GridSearchCV.
            Si dict, usa los parámetros personalizados.
        """
        if self.X_scaled is None or self.Y is None:
            print("❌ Error: Primero debes cargar los datos con load_data()")
            return
        
        print("\n" + "="*70)
        print("ENTRENAMIENTO DEL MODELO FINAL")
        print("="*70)
        
        # Usar parámetros
        if custom_params is not None:
            params = custom_params
            print("Usando parámetros personalizados:")
        elif self.best_params is not None:
            params = self.best_params
            print("Usando mejores parámetros de GridSearchCV:")
        else:
            # Parámetros por defecto conservadores
            params = {
                'max_depth': 3,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'n_estimators': 50,
                'max_features': 'sqrt'
            }
            print("Usando parámetros por defecto:")
        
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        # Crear y entrenar modelo
        self.model = RandomForestRegressor(
            random_state=self.random_state,
            **params
        )
        
        self.model.fit(self.X_scaled, self.Y)
        self.is_trained = True
        
        print(f"\n✓ Modelo entrenado con {len(self.X_scaled)} muestras")
        print("="*70)

    def evaluate_model_loocv(self):
        """
        Evalúa el modelo usando Leave-One-Out Cross-Validation.
        Calcula métricas para cada variable de salida.
        """
        if self.X_scaled is None or self.Y is None:
            print("❌ Error: Primero debes cargar los datos")
            return
        
        print("\n" + "="*70)
        print("EVALUACIÓN DEL MODELO (Leave-One-Out Cross-Validation)")
        print("="*70)
        
        loocv = LeaveOneOut()
        y_true_list = []
        y_pred_list = []
        
        print("⏳ Ejecutando validación cruzada...\n")
        
        # Usar los mejores parámetros si existen
        if self.best_params is not None:
            params = self.best_params
        else:
            params = {
                'max_depth': 5,
                'min_samples_leaf': 2,
                'n_estimators': 100,
                'max_features': 'sqrt',
                'random_state': self.random_state
            }
        
        for train_index, test_index in loocv.split(self.X_scaled):
            X_train, X_test = self.X_scaled.iloc[train_index], self.X_scaled.iloc[test_index]
            y_train, y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]
            
            fold_model = RandomForestRegressor(**params)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            y_true_list.append(y_test.values[0])
            y_pred_list.append(y_pred[0])
        
        y_true_all = np.array(y_true_list)
        y_pred_all = np.array(y_pred_list)
        
        # Calcular métricas por variable
        mae_scores = mean_absolute_error(y_true_all, y_pred_all, multioutput='raw_values')
        r2_scores = r2_score(y_true_all, y_pred_all, multioutput='raw_values')
        rmse_scores = np.sqrt(mean_squared_error(y_true_all, y_pred_all, multioutput='raw_values'))
        
        print("MÉTRICAS DE EVALUACIÓN (LOOCV):")
        print("="*70)
        print(f"{'Propiedad':<22} | {'MAE':<10} | {'RMSE':<10} | {'R² Score':<10}")
        print("-" * 70)
        
        for i, target_name in enumerate(self.target_names):
            print(f"{target_name:<22} | {mae_scores[i]:<10.3f} | {rmse_scores[i]:<10.3f} | {r2_scores[i]:<10.3f}")
        
        print("="*70)
        print("\n📊 Interpretación de R²:")
        print("  • R² > 0.9  : Excelente predicción")
        print("  • R² > 0.7  : Buena predicción")
        print("  • R² > 0.5  : Predicción aceptable")
        print("  • R² < 0.5  : Predicción pobre")
        print("  • R² < 0.0  : Modelo peor que la media")
        
        # Resaltar Max_Stress
        max_stress_idx = self.target_names.index('Max_Stress')
        print(f"\n🎯 OBJETIVO PRINCIPAL - Max_Stress:")
        print(f"   R² Score: {r2_scores[max_stress_idx]:.4f}")
        print(f"   MAE: {mae_scores[max_stress_idx]:.3f} MPa")
        print(f"   RMSE: {rmse_scores[max_stress_idx]:.3f} MPa")
        
        return {
            'MAE': mae_scores,
            'RMSE': rmse_scores,
            'R2': r2_scores
        }

    def get_feature_importance(self):
        """
        Analiza la importancia de cada feature en las predicciones.
        """
        if not self.is_trained:
            print("❌ Error: El modelo debe ser entrenado primero.")
            return
        
        print("\n" + "="*70)
        print("IMPORTANCIA DE LAS VARIABLES DE ENTRADA")
        print("="*70)
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance_df.to_string(index=False))
        
        print("\n📈 Top 3 Features más importantes:")
        for i in range(min(3, len(feature_importance_df))):
            feat = feature_importance_df.iloc[i]
            print(f"  {i+1}. {feat['Feature']}: {feat['Importance']*100:.2f}%")
        
        return feature_importance_df

    def predict(self, hsp, healing_time, peak_logm, molecular_weight, 
                contact_angle_mean, contact_angle_std, ftir_value, dsc_tg):
        """
        Predice las propiedades mecánicas dados los parámetros de entrada.
        
        Parameters:
        -----------
        hsp : float
            Valor HSP (0.35, 0.40, 0.45)
        healing_time : float
            Tiempo de curación en horas (0.0, 0.33, 1.0, 3.0)
        peak_logm : float
            Peak logM de GPC
        molecular_weight : float
            Peso molecular
        contact_angle_mean : float
            Ángulo de contacto promedio (°)
        contact_angle_std : float
            Desviación estándar del ángulo de contacto (°)
        ftir_value : float
            Valor de transmitancia FTIR a 3330 cm⁻¹
        dsc_tg : float
            Temperatura de transición vítrea (°C)
        """
        if not self.is_trained:
            print("❌ Error: El modelo debe ser entrenado primero.")
            return None
        
        print("\n" + "="*70)
        print("PREDICCIÓN DE PROPIEDADES")
        print("="*70)
        print("Parámetros de entrada:")
        print(f"  • HSP: {hsp}")
        print(f"  • Healing Time: {healing_time} hrs")
        print(f"  • Peak logM: {peak_logm}")
        print(f"  • Molecular Weight: {molecular_weight} g/mol")
        print(f"  • Contact Angle: {contact_angle_mean}° ± {contact_angle_std}°")
        print(f"  • FTIR H-Bond: {ftir_value}")
        print(f"  • DSC Tg: {dsc_tg}°C")
        
        # Crear DataFrame de entrada
        input_data = pd.DataFrame([[
            hsp, healing_time, peak_logm, molecular_weight,
            contact_angle_mean, contact_angle_std, ftir_value, dsc_tg
        ]], columns=self.feature_names)
        
        # Normalizar usando el mismo scaler
        input_scaled = self.scaler.transform(input_data)
        
        # Predecir
        prediction = self.model.predict(input_scaled)
        
        # Formatear resultados
        results = dict(zip(self.target_names, prediction[0]))
        
        print("\n" + "="*70)
        print("PROPIEDADES PREDICHAS:")
        print("="*70)
        for key, value in results.items():
            unit = "MPa" if "Stress" in key else "%" if "Strain" in key or "HE" in key else ""
            print(f"  • {key}: {value:.3f} {unit}")
        print("="*70)
        
        return results

    def plot_predictions_vs_actual(self):
        """
        Genera gráficos de valores predichos vs reales para cada target.
        """
        if not self.is_trained:
            print("❌ Error: El modelo debe ser entrenado primero.")
            return
        
        print("\n📊 Generando gráficos de predicciones vs valores reales...")
        
        # Predecir con el modelo entrenado
        y_pred = self.model.predict(self.X_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, target in enumerate(self.target_names):
            ax = axes[i]
            
            y_true = self.Y[target].values
            y_pred_target = y_pred[:, i]
            
            # Scatter plot
            ax.scatter(y_true, y_pred_target, alpha=0.6, edgecolors='k', s=80)
            
            # Línea de predicción perfecta
            min_val = min(y_true.min(), y_pred_target.min())
            max_val = max(y_true.max(), y_pred_target.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción perfecta')
            
            # Calcular R²
            r2 = r2_score(y_true, y_pred_target)
            
            ax.set_xlabel('Valor Real', fontsize=11)
            ax.set_ylabel('Valor Predicho', fontsize=11)
            ax.set_title(f'{target}\nR² = {r2:.3f}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico guardado como 'predictions_vs_actual.png'")
        plt.show()


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    
    print("\n" + "🚀"*35)
    print("OPTIMIZED TPU PROPERTY PREDICTOR - MACHINE LEARNING PIPELINE")
    print("🚀"*35 + "\n")
    
    # 1. Crear instancia del predictor
    predictor = OptimizedTPUPropertyPredictor(random_state=42)
    
    # 2. Cargar datos
    if not predictor.load_data(filepath='datasets/dataset_ml_final.csv'):
        print("\n❌ Error al cargar datos. Verifica que el archivo exista y tenga las columnas correctas.")
        exit(1)
    
    # 3. Optimizar hiperparámetros con GridSearchCV
    print("\n" + "⚙️"*35)
    choice = input("¿Deseas ejecutar GridSearchCV? (puede tardar 10-30 minutos) [y/n]: ")
    
    if choice.lower() == 'y':
        predictor.optimize_hyperparameters(cv_folds=None)  # None = LOOCV
    else:
        print("⏭️  Saltando optimización. Usando parámetros por defecto.")
    
    # 4. Entrenar modelo final
    predictor.train_final_model()
    
    # 5. Evaluar modelo
    metrics = predictor.evaluate_model_loocv()
    
    # 6. Importancia de features
    feature_importance = predictor.get_feature_importance()
    
    # 7. Hacer predicciones de ejemplo
    print("\n" + "🔮"*35)
    print("EJEMPLOS DE PREDICCIÓN")
    print("🔮"*35)
    
    # Ejemplo 1: Valor intermedio
    print("\n--- Ejemplo 1: HSP=0.38, Healing Time=1.0 hrs ---")
    predictor.predict(
        hsp=0.38,
        healing_time=1.0,
        peak_logm=4.5,
        molecular_weight=31623,
        contact_angle_mean=90.0,
        contact_angle_std=2.5,
        ftir_value=0.85,
        dsc_tg=-40.0
    )
    
    # Ejemplo 2: Material undamaged
    print("\n--- Ejemplo 2: HSP=0.40, Undamaged (Healing Time=0.0) ---")
    predictor.predict(
        hsp=0.40,
        healing_time=0.0,
        peak_logm=4.6,
        molecular_weight=39811,
        contact_angle_mean=92.0,
        contact_angle_std=2.0,
        ftir_value=0.88,
        dsc_tg=-38.0
    )
    
    # 8. Generar gráficos (opcional)
    try:
        predictor.plot_predictions_vs_actual()
    except:
        print("\n⚠️  No se pudieron generar los gráficos (matplotlib no disponible o error)")
    
    print("\n" + "✅"*35)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("✅"*35 + "\n")