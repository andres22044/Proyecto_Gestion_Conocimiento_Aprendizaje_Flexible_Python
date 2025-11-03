# train_and_save.py
import joblib
from XGBoostModel import XGBoostTPUPropertyPredictor
import warnings

# Suprimimos warnings para una salida limpia
warnings.filterwarnings('ignore', category=UserWarning)

print("🚀 Iniciando el entrenamiento del modelo...")

# 1. Crear instancia
predictor = XGBoostTPUPropertyPredictor(random_state=42)

# 2. Cargar datos
if not predictor.load_data(filepath='dataset_ml_final.csv'):
    print("\n❌ Error al cargar datos.")
    exit(1)

# 3. Entrenar modelo final
print("\n🔥 Entrenando modelo final con todos los datos...")
predictor.train_final_model()
print("✓ Modelo entrenado.")

# 4. GUARDAR LOS OBJETOS
print("\n💾 Guardando el modelo y el escalador...")

# Guardar el modelo entrenado
joblib.dump(predictor.model, 'tpu_model.joblib')

# Guardar el escalador ajustado (fit)
joblib.dump(predictor.scaler, 'tpu_scaler.joblib')

print("="*50)
print("✅ Proceso completado.")
print("   Archivos 'tpu_model.joblib' y 'tpu_scaler.joblib' creados.")
print("="*50)