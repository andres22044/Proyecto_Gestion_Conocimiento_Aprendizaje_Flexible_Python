import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

class XGBoostTPUPropertyPredictor:
    """
    Predictor de propiedades mecánicas de TPU usando XGBoost (Extreme Gradient Boosting).
    
    XGBoost es efectivo para:
    - Datasets pequeños y medianos
    - Captura de relaciones no lineales complejas
    - Manejo de interacciones entre features
    - Regularización para evitar overfitting
    
    Características:
    - Normalización (StandardScaler) de todas las entradas
    - GridSearchCV para optimización de hiperparámetros
    - Leave-One-Out Cross-Validation
    - Multi-output regression
    """

    def __init__(self, random_state=42):
        """
        Inicializa el predictor XGBoost.
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.X = None
        self.Y = None
        self.X_scaled = None
        
        # Métricas de evaluación guardadas
        self.evaluation_metrics = None
        
        # Definición de variables
        self.feature_names = [
            'HSP',
            'Healing_Time',
            'UTS_Original_Mean',
            'Strain_Original_Mean',
            'Peak_logM',
            'Molecular_Weight',
            'Contact_Angle_Mean',
            'Contact_Angle_Std',
            'FTIR_H-Bond_Value',
            'DSC_Tg_Value'
        ]
        
        self.target_names = [
            'HE_Elongation_Mean',
            'HE_UTS_Mean'
        ]
        
        print("="*70)
        print("XGBoost TPU PROPERTY PREDICTOR")
        print("="*70)
        print(f"Features (X - Inputs): {len(self.feature_names)}")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"  {i}. {feat}")
        print(f"\nTargets (Y - Outputs): {len(self.target_names)}")
        for i, targ in enumerate(self.target_names, 1):
            print(f"  {i}. {targ}")
        print("="*70)

    def load_data(self, filepath='dataset_ml_final.csv'):
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
        
        # NORMALIZACIÓN DE ENTRADAS
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

    def train_final_model(self, custom_params=None):
        """
        Entrena el modelo final con todos los datos.
        
        Parameters:
        -----------
        custom_params : dict or None
            Si None, usa los mejores parámetros de GridSearchCV.
        """
        if self.X_scaled is None or self.Y is None:
            print("❌ Error: Primero debes cargar los datos con load_data()")
            return
        
        # Usar parámetros
        if custom_params is not None:
            params = custom_params
            print("Usando parámetros personalizados:")
        elif self.best_params is not None:
            # Extraer parámetros del estimador
            params = {k.replace('estimator__', ''): v for k, v in self.best_params.items()}
            print("Usando mejores parámetros de GridSearchCV:")
        else:
            # Parámetros por defecto conservadores
            params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 2
            }
            print("Usando parámetros por defecto:")
        
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        # Crear y entrenar modelo
        xgb_model = XGBRegressor(
            random_state=self.random_state,
            objective='reg:squarederror',
            tree_method='auto',
            **params
        )
        self.model = MultiOutputRegressor(xgb_model)
        
        self.model.fit(self.X_scaled, self.Y)
        self.is_trained = True
        
        print(f"\n✓ Modelo XGBoost entrenado con {len(self.X_scaled)} muestras")
        print("="*70)

    def evaluate_model_loocv(self):
        """
        Evalúa el modelo usando Leave-One-Out Cross-Validation.
        Guarda las métricas para mostrarlas en la interfaz.
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
            params = {k.replace('estimator__', ''): v for k, v in self.best_params.items()}
        else:
            params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 2
            }
        
        for train_index, test_index in loocv.split(self.X_scaled):
            X_train, X_test = self.X_scaled.iloc[train_index], self.X_scaled.iloc[test_index]
            y_train, y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]
            
            xgb_model = XGBRegressor(
                random_state=self.random_state,
                objective='reg:squarederror',
                tree_method='auto',
                **params
            )
            fold_model = MultiOutputRegressor(xgb_model)
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
        
        # Guardar métricas para la interfaz
        self.evaluation_metrics = {
            'MAE': mae_scores,
            'RMSE': rmse_scores,
            'R2': r2_scores,
            'y_true': y_true_all,
            'y_pred': y_pred_all
        }
        
        return self.evaluation_metrics

    def get_evaluation_metrics(self):
        """
        Retorna las métricas de evaluación del modelo.
        """
        if self.evaluation_metrics is None:
            print("⚠️ No hay métricas disponibles. Ejecuta evaluate_model_loocv() primero.")
            return None
        return self.evaluation_metrics

    def predict(self, hsp, healing_time, UTS_Original_Mean, Strain_Original_Mean, 
                peak_logm, molecular_weight, contact_angle_mean, contact_angle_std, 
                ftir_value, dsc_tg):
        """
        Predice las propiedades mecánicas dados los parámetros de entrada.
        """
        if not self.is_trained:
            print("❌ Error: El modelo debe ser entrenado primero.")
            return None
        
        print("\n" + "="*70)
        print("PREDICCIÓN DE PROPIEDADES (XGBoost)")
        print("="*70)
        print("Parámetros de entrada:")
        print(f"  • HSP: {hsp}")
        print(f"  • Healing Time: {healing_time} hrs")
        print(f"  • UTS Original Mean: {UTS_Original_Mean} MPa")
        print(f"  • Strain Original Mean: {Strain_Original_Mean} %")
        print(f"  • Peak logM: {peak_logm}")
        print(f"  • Molecular Weight: {molecular_weight} g/mol")
        print(f"  • Contact Angle: {contact_angle_mean}° ± {contact_angle_std}°")
        print(f"  • FTIR H-Bond: {ftir_value}")
        print(f"  • DSC Tg: {dsc_tg}°C")
        
        # Crear DataFrame de entrada
        input_data = pd.DataFrame([[
            hsp, healing_time, UTS_Original_Mean, Strain_Original_Mean, peak_logm, molecular_weight,
            contact_angle_mean, contact_angle_std, ftir_value, dsc_tg
        ]], columns=self.feature_names)
        
        # Normalizar
        input_scaled = self.scaler.transform(input_data)
        
        # Predecir
        prediction = self.model.predict(input_scaled)
        
        # Formatear resultados
        results = dict(zip(self.target_names, prediction[0]))
        
        print("\n" + "="*70)
        print("PROPIEDADES PREDICHAS:")
        print("="*70)
        for key, value in results.items():
            unit = "%"
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
        
        y_pred = self.model.predict(self.X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, target in enumerate(self.target_names):
            ax = axes[i]
            
            y_true = self.Y[target].values
            y_pred_target = y_pred[:, i]
            
            ax.scatter(y_true, y_pred_target, alpha=0.6, edgecolors='k', s=80)
            
            min_val = min(y_true.min(), y_pred_target.min())
            max_val = max(y_true.max(), y_pred_target.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción perfecta')
            
            r2 = r2_score(y_true, y_pred_target)
            
            ax.set_xlabel('Valor Real (%)', fontsize=11)
            ax.set_ylabel('Valor Predicho (%)', fontsize=11)
            ax.set_title(f'{target} (XGBoost)\nR² = {r2:.3f}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('xgboost_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico guardado como 'xgboost_predictions_vs_actual.png'")
        plt.show()


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    
    print("\n" + "🚀"*35)
    print("XGBoost TPU PROPERTY PREDICTOR - MACHINE LEARNING PIPELINE")
    print("🚀"*35 + "\n")
    
    # 1. Crear instancia del predictor
    predictor = XGBoostTPUPropertyPredictor(random_state=42)
    
    # 2. Cargar datos
    if not predictor.load_data(filepath='dataset_ml_final.csv'):
        print("\n❌ Error al cargar datos.")
        exit(1)
    
    # 3. Entrenar modelo final
    predictor.train_final_model()
    
    # 4. Evaluar modelo
    metrics = predictor.evaluate_model_loocv()
    
    # 5. Predicciones de ejemplo
    print("\n" + "🔮"*35)
    print("EJEMPLOS DE PREDICCIÓN")
    print("🔮"*35)
    
    # Ejemplo 1: Valor intermedio
    print("\n--- Ejemplo 1: HSP=0.40, Healing Time=1.0 hrs ---")
    predictor.predict(
        hsp=0.40,
        healing_time=1.0,
        UTS_Original_Mean=0.37,
        Strain_Original_Mean=1700.0,
        peak_logm=4.53,
        molecular_weight=33884.42,
        contact_angle_mean=73.61,
        contact_angle_std=40.91,
        ftir_value=142.33,
        dsc_tg=-89.86
    )
    
    print("\n" + "✅"*35)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("✅"*35 + "\n")