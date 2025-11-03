import mysql.connector
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import warnings
import time 
import qrcode
from io import BytesIO
import base64
import json
from datetime import datetime

# Importamos tu clase de modelo DIRECTAMENTE desde tu archivo
from XGBoostModel import XGBoostTPUPropertyPredictor

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'tu_clave_secreta_aqui_para_el_login') 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================================================================
#                         CONFIGURACIÓN DE LA BASE DE DATOS
# =================================================================
DB_CONFIG = {
    'host': '127.0.0.1',  
    'user': 'root',       
    'password': '', 
    'database': 'usuarios_aula_espejo'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error al conectar a la base de datos: {err}")
        return None

# =================================================================
#                         FUNCIÓN PARA GRÁFICOS
# =================================================================
def create_prediction_vs_actual_chart(predictor, save_path):
    """
    Crea gráficos de Real vs Predicho para validar el modelo
    """
    try:
        if predictor.evaluation_metrics is None:
            # Si no hay métricas, evaluamos el modelo
            predictor.evaluate_model_loocv()
        
        metrics = predictor.evaluation_metrics
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, target_name in enumerate(predictor.target_names):
            ax = axes[i]
            
            y_true_target = y_true[:, i]
            y_pred_target = y_pred[:, i]
            
            # Scatter plot
            ax.scatter(y_true_target, y_pred_target, alpha=0.6, 
                      edgecolors='#004a80', s=60, color='#0d6efd')
            
            # Línea diagonal de referencia (y=x)
            min_val = min(y_true_target.min(), y_pred_target.min())
            max_val = max(y_true_target.max(), y_pred_target.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Predicción Perfecta (y=x)')
            
            # Métricas
            r2 = metrics['R2'][i]
            mae = metrics['MAE'][i]
            
            # Título y etiquetas
            ax.set_xlabel('Valor Real (%)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Valor Predicho (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{target_name}\nR² = {r2:.3f} | MAE = {mae:.2f}%', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"Gráfico de validación guardado en: {save_path}")
    
    except Exception as e:
        logger.error(f"Error al crear el gráfico de validación: {e}")

# =================================================================
#                         CARGA GLOBAL DEL MODELO
# =================================================================
# =================================================================
#                         CARGA GLOBAL DEL MODELO
# =================================================================
print("="*50)
print("INICIANDO SERVIDOR FLASK...")
print("Cargando modelo pre-entrenado...")
warnings.filterwarnings('ignore', category=UserWarning)

# 1. Crear instancia del predictor
predictor = XGBoostTPUPropertyPredictor(random_state=42)

# 2. Cargar el modelo y escalador desde los archivos .joblib
if not predictor.load_trained_model(
    model_path='tpu_model.joblib', 
    scaler_path='tpu_scaler.joblib'
):
    print("!"*50)
    print("ERROR FATAL: No se pudieron cargar los modelos pre-entrenados.")
    print("Asegúrate de haber ejecutado 'train_and_save.py' localmente.")
    print("!"*50)
    exit() 

# 3. Cargar datos SÓLO para las métricas
# (Esto es mucho más ligero porque no entrena)
try:
    # Este método también configura X_scaled y Y
    predictor.load_data(filepath='dataset_ml_final.csv')
    print("\nEvaluando métricas del modelo (LOOCV)...")
    predictor.evaluate_model_loocv()
except Exception as e:
    print(f"Advertencia: No se pudo evaluar el modelo: {e}")

print("="*50)
print("MODELO CARGADO Y LISTO.")
print("="*50)

# =================================================================
#                             RUTAS DE LA APLICACIÓN
# =================================================================

@app.route('/')
def index(): 
    """Ruta de inicio, redirige al login si no hay sesión."""
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Maneja el inicio de sesión del usuario."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if not conn:
            return render_template('login.html', error="Error de conexión con la base de datos.")
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['username'] = user['username']
            session['user_id'] = user['id']
            logger.info(f"Usuario {username} ha iniciado sesión.")
            return redirect(url_for('index')) 
        else:
            logger.warning(f"Intento de login fallido para el usuario {username}.")
            return render_template('login.html', error="Usuario o contraseña incorrectos")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Maneja el registro de nuevos usuarios."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        if not conn:
            return render_template('register.html', error="Error de conexión con la base de datos.")
            
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", 
                         (username, hashed_password))
            conn.commit()
            logger.info(f"Nuevo usuario registrado: {username}")
        except mysql.connector.Error as err:
            logger.error(f"Error al registrar usuario: {err}")
            conn.rollback() 
            return render_template('register.html', error="El usuario ya existe o hubo un error.")
        finally:
            cursor.close()
            conn.close()
            
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Cierra la sesión del usuario."""
    username = session.pop('username', None)
    session.pop('user_id', None)
    if username:
        logger.info(f"Usuario {username} ha cerrado sesión.")
    return redirect(url_for('login'))

@app.route('/proyecto')
def proyecto():
    """Muestra la página de Sostenibilidad."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('sobre_proyecto.html')

# =================================================================
#                  FUNCIONES DE BASE DE DATOS PARA PREDICCIONES
# =================================================================

def save_prediction_to_db(sample_id, username, input_data, prediction_results):
    """
    Guarda una predicción en la base de datos.
    Retorna el ID de la predicción guardada o None si hay error.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("No se pudo conectar a la base de datos para guardar la predicción")
        return None
    
    try:
        cursor = conn.cursor()
        
        query = """
        INSERT INTO tpu_resultados_muestra (
            sample_id, timestamp, user_name,
            HSP, Healing_Time_hrs, UTS_Original_MPa, Strain_Original_percent,
            Peak_logM, Molecular_Weight, Contact_Angle_Mean, Contact_Angle_Std,
            FTIR_H_Bond, DSC_Tg_C,
            HE_UTS_Efficiency_percent, HE_Elongation_Efficiency_percent
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s
        )
        """
        
        values = (
            sample_id,
            datetime.now(),
            username,
            input_data['hsp'],
            input_data['healing_time'],
            input_data['UTS_Original_Mean'],
            input_data['Strain_Original_Mean'],
            input_data['peak_logm'],
            input_data['molecular_weight'],
            input_data['contact_angle_mean'],
            input_data['contact_angle_std'],
            input_data['ftir_value'],
            input_data['dsc_tg'],
            prediction_results['HE_UTS_Mean'],
            prediction_results['HE_Elongation_Mean']
        )
        
        cursor.execute(query, values)
        conn.commit()
        
        prediction_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        logger.info(f"Predicción guardada con ID: {prediction_id}")
        return prediction_id
        
    except mysql.connector.Error as err:
        logger.error(f"Error al guardar predicción en BD: {err}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_all_predictions():
    """
    Obtiene todas las predicciones de la base de datos.
    Retorna una lista de diccionarios con los datos.
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT ID, sample_id, timestamp, user_name,
               HE_UTS_Efficiency_percent, HE_Elongation_Efficiency_percent
        FROM tpu_resultados_muestra
        ORDER BY timestamp DESC
        """
        
        cursor.execute(query)
        predictions = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return predictions
        
    except mysql.connector.Error as err:
        logger.error(f"Error al obtener predicciones: {err}")
        return []
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_prediction_by_id(prediction_id):
    """
    Obtiene una predicción específica por su ID.
    Retorna un diccionario con todos los datos o None si no existe.
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT * FROM tpu_resultados_muestra
        WHERE ID = %s
        """
        
        cursor.execute(query, (prediction_id,))
        prediction = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return prediction
        
    except mysql.connector.Error as err:
        logger.error(f"Error al obtener predicción {prediction_id}: {err}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()


def delete_prediction_by_id(prediction_id):
    """
    Elimina una predicción de la base de datos.
    Retorna True si se eliminó correctamente, False en caso contrario.
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        query = "DELETE FROM tpu_resultados_muestra WHERE ID = %s"
        cursor.execute(query, (prediction_id,))
        conn.commit()
        
        deleted = cursor.rowcount > 0
        
        cursor.close()
        conn.close()
        
        if deleted:
            logger.info(f"Predicción {prediction_id} eliminada correctamente")
        
        return deleted
        
    except mysql.connector.Error as err:
        logger.error(f"Error al eliminar predicción {prediction_id}: {err}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


# =================================================================
#                       RUTAS DEL SIMULADOR TPU
# =================================================================

@app.route('/modelofinal')
def modelo_final():
    """
    Muestra la página del simulador de TPU con información del modelo.
    """
    if 'username' not in session:
        return redirect(url_for('login')) 
    
    # Obtener métricas del modelo
    metrics = predictor.get_evaluation_metrics()
    
    model_info = {
        'r2_uts': round(metrics['R2'][1], 3) if metrics else 0.90,
        'r2_elongation': round(metrics['R2'][0], 3) if metrics else 0.90,
        'mae_uts': round(metrics['MAE'][1], 2) if metrics else 0.00,
        'mae_elongation': round(metrics['MAE'][0], 2) if metrics else 0.00
    }
    
    # Generar gráfico de validación
    chart_path = os.path.join(app.static_folder, 'model_validation_chart.png')
    create_prediction_vs_actual_chart(predictor, chart_path)
    
    return render_template('modelofinal.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API para el simulador. Recibe JSON, ejecuta el modelo, 
    guarda en BD y devuelve predicción con URL para QR.
    """
    if 'username' not in session:
        return jsonify({"error": "No autorizado"}), 401

    try:
        # 1. Obtener los datos de entrada del JavaScript
        data = request.json
        
        input_data = {
            'hsp': float(data['hsp']),
            'healing_time': float(data['healingTime']),
            'UTS_Original_Mean': float(data['utsOriginal']),
            'Strain_Original_Mean': float(data['strainOriginal']),
            'peak_logm': float(data['peakLogM']),
            'molecular_weight': float(data['molecularWeight']),
            'contact_angle_mean': float(data['contactAngleMean']),
            'contact_angle_std': float(data['contactAngleStd']),
            'ftir_value': float(data['ftir']),
            'dsc_tg': float(data['dsc'])
        }

        # 2. Obtener la predicción
        prediction_results = predictor.predict(**input_data)
        
        if prediction_results is None:
            return jsonify({"error": "Error al realizar la predicción"}), 500
        
        # 3. Convertir numpy.float32 a float de Python
        prediction_results_clean = {
            key: float(value) for key, value in prediction_results.items()
        }
        
        # 4. Guardar en la base de datos
        sample_id = data.get('sampleId', f'PLIX-AUTO-{int(time.time())}')
        prediction_id = save_prediction_to_db(
            sample_id=sample_id,
            username=session['username'],
            input_data=input_data,
            prediction_results=prediction_results_clean
        )
        
        if prediction_id is None:
            logger.warning("No se pudo guardar la predicción en la BD, pero se continúa")
        
        # 5. Generar URL de detalle para el QR
        if prediction_id:
            detail_url = url_for('prediction_detail', prediction_id=prediction_id, _external=True)
        else:
            detail_url = url_for('predictions_index', _external=True)
        
        # 6. Generar código QR con la URL
        qr_image_base64 = generate_qr_code_with_url(detail_url)
        
        # 7. Añadir el QR y el ID a los resultados
        prediction_results_clean['qr_code'] = qr_image_base64
        prediction_results_clean['prediction_id'] = prediction_id
        prediction_results_clean['detail_url'] = detail_url
        
        # 8. Devolver todo como JSON
        return jsonify(prediction_results_clean)

    except ValueError as ve:
        logger.error(f"Error de validación en /predict: {ve}")
        return jsonify({"error": f"Datos inválidos: {str(ve)}"}), 400
    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        print(f"Error detallado en /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/predictions')
def predictions_index():
    """
    Página índice con todas las predicciones realizadas.
    """
    if 'username' not in session:
        return redirect(url_for('login'))
    
    predictions = get_all_predictions()
    
    return render_template('predictions_index.html', predictions=predictions)


@app.route('/prediction/<int:prediction_id>')
def prediction_detail(prediction_id):
    """
    Página de detalle de una predicción específica.
    """
    prediction = get_prediction_by_id(prediction_id)
    
    if not prediction:
        return render_template('pagina_error.html',
                             error_title="Predicción no encontrada",
                             error_message=f"No se encontró la predicción con ID {prediction_id}",
                             error_code=404), 404
    
    return render_template('prediction_detail.html', prediction=prediction)


@app.route('/prediction/<int:prediction_id>/delete', methods=['POST'])
def delete_prediction(prediction_id):
    """
    Elimina una predicción de la base de datos.
    """
    if 'username' not in session:
        return jsonify({"error": "No autorizado"}), 401
    
    success = delete_prediction_by_id(prediction_id)
    
    if success:
        return jsonify({"success": True, "message": "Predicción eliminada correctamente"})
    else:
        return jsonify({"success": False, "message": "Error al eliminar la predicción"}), 500


def generate_qr_code_with_url(url):
    """
    Genera un código QR que enlaza a una URL específica.
    """
    try:
        # Crear código QR con la URL
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        # Crear imagen
        img = qr.make_image(fill_color="#004a80", back_color="white")
        
        # Convertir a base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
    except Exception as e:
        logger.error(f"Error al generar QR con URL: {e}")
        return None

# =================================================================
#                             CONTEXT PROCESSOR
# =================================================================

@app.context_processor
def inject_globals():
    """Inyecta variables globales y el estado de la sesión a todos los templates."""
    return {
        'app_name': 'FlexLearn',
        'app_version': '1.0',
        'current_year': 2025,
        'logged_in': 'username' in session,
        'current_user': session.get('username')
    }

# =================================================================
#                                INICIO DE LA APP
# =================================================================

if __name__ == '__main__':
    required_templates = ['index.html', 'pagina_error.html', 'login.html', 
                         'register.html', 'sobre_proyecto.html', 'modelofinal.html',
                         'predictions_index.html', 'prediction_detail.html']
    missing_templates = []
    
    for template in required_templates:
        template_path = os.path.join('templates', template)
        if not os.path.exists(template_path):
            missing_templates.append(template)
    
    if missing_templates:
        logger.warning(f"Plantillas faltantes: {', '.join(missing_templates)}")
        print(f"⚠️  Advertencia: Faltan plantillas: {', '.join(missing_templates)}")
    
    app.run(debug=True)