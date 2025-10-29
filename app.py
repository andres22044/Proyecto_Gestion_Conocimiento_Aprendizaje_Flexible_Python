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
    'database': 'usuarios_aula_espejo' # ¡Correcto!
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
def create_prediction_chart(results, save_path):
    """
    Crea un gráfico de barras horizontal con los resultados de la predicción
    y lo guarda en la ruta especificada.
    """
    try:
        # Extraer los 4 valores clave del diccionario de resultados
        metrics = [
            'Eficiencia de Estrés (HE_UTS)', 
            'Eficiencia de Elongación (HE_Elong)', 
            'Estrés Máximo (Max_Stress)', 
            'Deformación Máx. (Max_Strain)'
        ]
        values = [
            results.get('HE_UTS_Mean', 0),
            results.get('HE_Elongation_Mean', 0),
            results.get('Max_Stress', 0),
            results.get('Max_Strain', 0)
        ]
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(7, 4)) # Tamaño ajustado para la tarjeta
        
        # Gráfico de barras horizontal
        colors = ['#0d6efd', '#0d6efd', '#ffc107', '#ffc107']
        bars = ax.barh(metrics, values, color=colors)
        
        # Añadir etiquetas de valor al final de cada barra
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
        
        # Estilo
        ax.set_title('Resultados de la Predicción', fontsize=14, fontweight='bold')
        ax.set_xlabel('Valor Predicho', fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Invertir eje Y para que la primera métrica esté arriba
        ax.invert_yaxis()
        
        # Ajustar layout y guardar
        plt.tight_layout()
        plt.savefig(save_path, dpi=90, bbox_inches='tight')
        plt.close(fig) # Importante: cerrar la figura para liberar memoria
        
        logger.info(f"Gráfico de predicción guardado en: {save_path}")
    
    except Exception as e:
        logger.error(f"Error al crear el gráfico: {e}")

# =================================================================
#                         CARGA GLOBAL DEL MODELO
# =================================================================
print("="*50)
print("INICIANDO SERVIDOR FLASK...")
print("Cargando y entrenando el modelo XGBoost...")
print("Esto puede tardar unos segundos...")
warnings.filterwarnings('ignore', category=UserWarning)

predictor = XGBoostTPUPropertyPredictor(random_state=42)

try:
    predictor.load_data(filepath='dataset_ml_final.csv')
except FileNotFoundError:
    print("\n" + "!"*50)
    print("ERROR FATAL: No se encontró 'dataset_ml_final.csv'.")
    print("Asegúrate de que 'dataset_ml_final.csv' esté en la misma carpeta que 'app.py'.")
    print("El servidor no puede iniciar sin el modelo.")
    print("!"*50 + "\n")
    exit() 
    
predictor.train_final_model()
print("="*50)
print("MODELO ENTRENADO Y LISTO.")
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
        
        # --- ¡ESTA LÍNEA ES CORRECTA! ---
        # Busca en la tabla 'usuarios' (plural)
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
            # --- ¡ESTA LÍNEA ES CORRECTA! ---
            # Inserta en la tabla 'usuarios' (plural)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password))
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
    
    # --- ¡ESTE ES EL CAMBIO! ---
    # Ahora apunta al nuevo archivo que subiste.
    return render_template('sobre_proyecto.html')

# =================================================================
#                       RUTAS DEL SIMULADOR TPU (NUEVAS)
# =================================================================

@app.route('/modelofinal')
def modelo_final():
    """
    Muestra la página del simulador de TPU (nueva).
    """
    if 'username' not in session:
        return redirect(url_for('login')) 
        
    return render_template('modelofinal.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API para el simulador. Recibe JSON, ejecuta el modelo,
    CREA UN GRÁFICO y devuelve la predicción (JSON) con la URL del gráfico.
    """
    if 'username' not in session:
        return jsonify({"error": "No autorizado"}), 401

    try:
        # 1. Obtener los datos de entrada del JavaScript
        data = request.json
        
        input_data = {
            'hsp': data['hsp'],
            'healing_time': data['healingTime'],         
            'peak_logm': data['peakLogM'],               
            'molecular_weight': data['molecularWeight'],   
            'contact_angle_mean': data['contactAngleMean'], 
            'contact_angle_std': data['contactAngleStd'],     
            'ftir_value': data['ftir'],                  
            'dsc_tg': data['dsc']                        
        }

        # 2. Obtener los NÚMEROS de la predicción
        prediction_results = predictor.predict(**input_data)
        
        # --- ¡CÓDIGO NUEVO PARA EL GRÁFICO! ---
        
        # 3. Definir la ruta completa donde se guardará el gráfico
        #    (Esto busca la carpeta 'static' automáticamente)
        save_path = os.path.join(app.static_folder, 'current_prediction_chart.png')
        
        # 4. Llamar a la función que crea y guarda el gráfico
        #    (Le pasamos los números de la predicción y la ruta)
        create_prediction_chart(prediction_results, save_path)
        
        # --- FIN DEL CÓDIGO NUEVO ---

        # 5. Crear la URL para el gráfico
        #    El "?v={time.time()}" es un truco para evitar que el navegador
        #    use una imagen antigua que tenga guardada en caché.
        image_url = url_for('static', filename='current_prediction_chart.png') + f"?v={time.time()}"
        
        # 6. Añadir la URL de la imagen al diccionario de resultados
        prediction_results['chart_image_url'] = image_url
        
        # 7. Devolver todo como un JSON
        return jsonify(prediction_results)

    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        # Imprimir el error en la consola de Flask ayuda a depurar
        print(f"Error detallado en /predict: {e}") 
        return jsonify({"error": str(e)}), 500

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
    required_templates = ['index.html', 'pagina_error.html', 'login.html', 'register.html', 'proyecto.html', 'modelofinal.html']
    missing_templates = []
    
    for template in required_templates:
        template_path = os.path.join('templates', template)
        if not os.path.exists(template_path):
            missing_templates.append(template)
    
    if missing_templates:
        logger.warning(f"Plantillas faltantes: {', '.join(missing_templates)}")
        print(f"⚠️   Advertencia: Faltan plantillas: {', '.join(missing_templates)}")
    
    app.run(debug=True)

