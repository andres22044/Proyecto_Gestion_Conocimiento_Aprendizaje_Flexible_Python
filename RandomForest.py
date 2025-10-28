import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut 
from sklearn.metrics import mean_absolute_error, r2_score 
from sklearn.preprocessing import StandardScaler
import warnings

# Ignorar advertencias comunes de scikit-learn para una salida más limpia
warnings.filterwarnings('ignore', category=UserWarning)

class TPUPropertyPredictor:
    """
    Una clase para predecir las propiedades mecánicas y estructurales del TPU
    utilizando un modelo Random Forest de Múltiples Salidas.
    
    Optimizado para datasets pequeños mediante Validación Cruzada Leave-One-Out (LOOCV).
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Inicializa el regresor Random Forest.
        
        Args:
            n_estimators (int): Número de árboles en el bosque.
            random_state (int): Semilla para reproducibilidad.
        """
        # El RF de scikit-learn soporta nativamente la regresión multi-salida
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples_leaf=1, # Importante para datasets pequeños
            max_features='sqrt'   # Robusto para pocas features
        )
        self.is_trained = False
        self.X = None
        self.Y = None
        self.feature_names = []
        self.target_names = []

    def load_data_and_train(self, filepath='datasets/dataset_ml_final.csv'):
        """
        Carga el dataset final, define las variables X (Entrada) e Y (Salida),
        y entrena el modelo final con TODOS los datos.
        """
        print(f"Cargando dataset desde: {filepath}...")
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en {filepath}.")
            print("Asegúrate de que el archivo esté en la ruta correcta.")
            return

        # 1. Definir Variables de Entrada (Features, X)
        # Estas son las variables que TÚ controlas
        self.feature_names = ['HSP', 'Healing_Time']
        
        # 2. Definir Variables de Salida (Targets, Y)
        # Estas son las propiedades que RESULTAN de las entradas
        self.target_names = [
            'Max_Strain', 
            'Max_Stress', 
            'HE_Elongation_Mean', 
            'HE_UTS_Mean',
            'Peak_logM', 
            'Molecular_Weight', 
            'Contact_Angle_Mean', 
            'Contact_Angle_Std'
        ]
        
        # Asegurarse de que todas las columnas existen
        required_cols = self.feature_names + self.target_names
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Error: Faltan columnas en el CSV: {missing_cols}")
            return

        # 3. Separar X e Y
        self.X = df[self.feature_names]
        self.Y = df[self.target_names]
        
        # Manejar NaNs (aunque el dataset debería estar limpio)
        self.Y = self.Y.fillna(self.Y.median())

        # 4. Entrenar el modelo final
        # Entrenamos con TODOS los datos (35 filas) para que el método .predict()
        # tenga el máximo conocimiento posible.
        self.model.fit(self.X, self.Y)
        self.is_trained = True
        
        print(f"Modelo entrenado exitosamente con {len(df)} muestras.")
        print(f"  - Variables de Entrada (X): {self.feature_names}")
        print(f"  - Variables de Salida (Y): {self.target_names}")

    def evaluate_model_loocv(self):
        """
        Implementa el método de validación "especial": Leave-One-Out Cross-Validation (LOOCV).
        Calcula manualmente las métricas para CADA variable de salida.
        """
        if not self.is_trained:
            print("Error: El modelo debe ser entrenado primero (llama a .load_data_and_train())")
            return

        print("\n--- Evaluación del Modelo (Leave-One-Out Cross-Validation) ---")
        
        loocv = LeaveOneOut()
        
        # Listas para guardar los valores reales y predichos de cada "fold" (vuelta)
        y_true_list = []
        y_pred_list = []

        print("Ejecutando validación cruzada (esto puede tardar unos segundos)...")
        
        # 1. Bucle manual de LOOCV
        for train_index, test_index in loocv.split(self.X):
            
            # Separar datos de entrenamiento (34 filas) y prueba (1 fila)
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]
            
            # Crear un modelo temporal solo para este "fold"
            fold_model = RandomForestRegressor(
                n_estimators=self.model.n_estimators,
                random_state=self.model.random_state,
                min_samples_leaf=self.model.min_samples_leaf,
                max_features=self.model.max_features
            )
            
            # Entrenar con las 34 filas
            fold_model.fit(X_train, y_train)
            
            # Predecir la fila única de prueba
            y_pred = fold_model.predict(X_test)
            
            # Guardar los resultados (y_test.values[0] y y_pred[0] son arrays de 8 valores)
            y_true_list.append(y_test.values[0])
            y_pred_list.append(y_pred[0])

        # 2. Convertir las listas en arrays de NumPy
        # Ambas tendrán forma (35 muestras, 8 salidas)
        y_true_all = np.array(y_true_list)
        y_pred_all = np.array(y_pred_list)

        # 3. Calcular métricas por separado para cada salida
        # multioutput='raw_values' nos da un array de 8 valores (uno por salida)
        avg_mae = mean_absolute_error(y_true_all, y_pred_all, multioutput='raw_values')
        avg_r2 = r2_score(y_true_all, y_pred_all, multioutput='raw_values')

        print("Métricas de Evaluación Promedio (LOOCV):")
        print("------------------------------------------")
        print(f"{'Propiedad (Salida)':<22} | {'Error Absoluto Medio (MAE)':<26} | {'R² Score (Precisión)':<20}")
        print("-" * 72)
        
        # Ahora avg_mae y avg_r2 SÍ son arrays de 8 elementos, y el bucle funcionará.
        for i, target_name in enumerate(self.target_names):
            print(f"{target_name:<22} | {avg_mae[i]:<26.3f} | {avg_r2[i]:<20.3f}")
            
        print("------------------------------------------")
        print("MAE: Qué tan lejos (en promedio) está la predicción del valor real.")
        print("R² Score: (1.0 = perfecto, 0.0 = malo). Un R² negativo es muy malo.")
        
    def get_feature_importance(self):
        """
        Muestra qué variable de entrada (HSP o Healing_Time) es más
        importante para predecir las salidas.
        """
        if not self.is_trained:
            print("Error: El modelo debe ser entrenado primero.")
            return

        print("\n--- Importancia de las Variables de Entrada (Features) ---")
        
        # El RF calcula una importancia promedio para todas las 8 salidas
        importances = self.model.feature_importances_
        
        for name, importance in zip(self.feature_names, importances):
            print(f"  - {name}: {importance*100:>6.2f}%")
            
        print("(Esto muestra qué variable usa más el modelo para tomar decisiones)")

    def predict(self, hsp, healing_time):
        """
        Método de predicción principal.
        Predice las 8 propiedades basándose en el HSP y el Tiempo de Curación.
        """
        if not self.is_trained:
            print("Error: El modelo debe ser entrenado primero.")
            return None

        print(f"\n--- Predicción para HSP={hsp} y Healing Time={healing_time}h ---")
        
        # 1. Crear un DataFrame de entrada con los nombres de columna correctos
        input_data = pd.DataFrame([[hsp, healing_time]], columns=self.feature_names)
        
        # 2. Realizar la predicción
        prediction = self.model.predict(input_data)
        
        # 3. Formatear la salida (prediction[0] toma la primera (y única) fila)
        results = dict(zip(self.target_names, prediction[0]))
        
        print("Propiedades Predichas:")
        for key, value in results.items():
            print(f"  - {key}: {value:.3f}")
            
        return results

# --- Ejemplo de cómo usar la clase ---
if __name__ == "__main__":
    
    # 1. Crear una instancia del predictor
    tpu_model = TPUPropertyPredictor(n_estimators=50, random_state=42)
    
    # 2. Cargar datos y entrenar el modelo
    # Asegúrate de que tu CSV esté en 'datasets/dataset_ml_enriquecido.csv'
    tpu_model.load_data_and_train(filepath='datasets/dataset_ml_final.csv')
    
    # 3. Evaluar el rendimiento del modelo usando LOOCV
    tpu_model.evaluate_model_loocv()
    
    # 4. Ver qué variables de entrada son más importantes
    tpu_model.get_feature_importance()
    
    # 5. Realizar predicciones
    # Predicción para un valor intermedio (no en el dataset)
    tpu_model.predict(hsp=0.38, healing_time=1.0)
    
    # Predicción para un valor conocido (para comparar)
    tpu_model.predict(hsp=0.45, healing_time=0.0) # (Tiempo 0 = Undamaged)