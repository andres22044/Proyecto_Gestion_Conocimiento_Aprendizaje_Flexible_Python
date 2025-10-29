import mysql.connector
import os
import logging
import pandas as pd
import numpy as np
import warnings
import time 

# Importamos tu clase de modelo
from XGBoostModel import XGBoostTPUPropertyPredictor

# Modificamos la importación de Flask para incluir todo lo necesario
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
# Usamos una variable de entorno para la clave secreta.
app.secret_key = os.environ.get('SECRET_KEY', 'tu_clave_secreta_aqui_para_el_login') 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configurar logging
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
    """Establece y retorna la conexión a la base de datos."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        # No es necesario loggear esto en cada conexión, solo si falla
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error al conectar a la base de datos: {err}")
        return None

# =================================================================
#                         CARGA GLOBAL DEL MODELO (¡NUEVO!)
# =================================================================
# Esto se ejecuta UNA SOLA VEZ cuando inicias el servidor.
# Mantenemos el modelo entrenado en memoria.
print("="*50)
print("INICIANDO SERVIDOR FLASK...")
print("Cargando y entrenando el modelo XGBoost...")
print("Esto puede tardar unos segundos...")
warnings.filterwarnings('ignore', category=UserWarning)

# Crear una instancia de tu predictor
predictor = XGBoostTPUPropertyPredictor(random_state=42)

# Cargar los datos (asegúrate que 'dataset_ml_final.csv' esté en la misma carpeta que 'app.py')
try:
    predictor.load_data(filepath='dataset_ml_final.csv')
except FileNotFoundError:
    print("\n" + "!"*50)
    print("ERROR FATAL: No se encontró 'dataset_ml_final.csv'.")
    print("Asegúrate de que 'dataset_ml_final.csv' esté en la misma carpeta que 'app.py'.")
    print("El servidor no puede iniciar sin el modelo.")
    print("!"*50 + "\n")
    exit() # Hacemos que la app falle si no encuentra el dataset
    
predictor.train_final_model()
print("="*50)
print("MODELO ENTRENADO Y LISTO.")
print("="*50)

# =================================================================
#                             RUTAS DE LA APLICACIÓN
# =================================================================

@app.route('/')
def home():
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
        cursor.execute("SELECT * FROM usuarios WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session['user_id'] = user['id']
            logger.info(f"Usuario {username} ha iniciado sesión.")
            return redirect(url_for('home'))
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
            cursor.execute("INSERT INTO usuarios (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            logger.info(f"Nuevo usuario registrado: {username}")
        except mysql.connector.Error as err:
            logger.error(f"Error al registrar usuario: {err}")
            conn.rollback() # Es buena práctica hacer rollback en caso de error
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
    """Muestra la página del proyecto (existente)."""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('proyecto.html')

# =================================================================
#                       RUTAS DEL SIMULADOR TPU (¡NUEVAS!)
# =================================================================

@app.route('/modelofinal')
def modelo_final():
    """
    Muestra la página del simulador de TPU (nueva).
    """
    if 'username' not in session:
        return redirect(url_for('login')) # Proteger también esta ruta
        
    # Esta plantilla debe extender 'base.html'
    return render_template('modelofinal.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API para el simulador. Recibe JSON, ejecuta el modelo
    y devuelve la predicción (JSON) con la URL del gráfico.
    """
    if 'username' not in session:
        return jsonify({"error": "No autorizado"}), 401 # Proteger la API

    try:
        data = request.json
        
        # 1. Traduce los nombres de JavaScript (healingTime) a Python (healing_time)
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

        # 2. Llama al método 'predict' (que crea el gráfico y arregla el float32)
        prediction_results = predictor.predict(**input_data)
        
        # 3. Generar la URL para el gráfico con "sello de tiempo" para evitar caché
        image_url = url_for('static', filename='current_prediction_chart.png') + f"?v={time.time()}"
        
        # 4. Añadir la URL al diccionario de resultados
        prediction_results['chart_image_url'] = image_url
        
        # 5. Devolver el diccionario completo
        return jsonify(prediction_results)

    except Exception as e:
        logger.error(f"Error en /predict: {e}")
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
    # Verificar que los templates existen
    required_templates = ['index.html', 'pagina_error.html', 'login.html', 'register.html', 'proyecto.html', 'modelofinal.html'] # Añadida 'modelofinal.html'
    missing_templates = []
    
    for template in required_templates:
        template_path = os.path.join('templates', template)
        if not os.path.exists(template_path):
            missing_templates.append(template)
    
    if missing_templates:
        logger.warning(f"Plantillas faltantes: {', '.join(missing_templates)}")
        print(f"⚠️   Advertencia: Faltan plantillas: {', '.join(missing_templates)}")
    
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True) # Es mejor usar el default 127.0.0.1 para desarrollo local

