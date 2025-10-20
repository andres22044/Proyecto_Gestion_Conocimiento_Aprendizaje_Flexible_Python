import mysql.connector
import os
import logging
# Nuevas importaciones necesarias para el login
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash
# Importamos check_password_hash para verificar la contraseña del login

app = Flask(__name__)
# Usamos una variable de entorno para la clave secreta. ¡Reemplázala en producción!
app.secret_key = os.environ.get('SECRET_KEY', 'tu_clave_secreta_aqui_para_el_login') 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================================================================
#                         CONFIGURACIÓN DE LA BASE DE DATOS
# =================================================================

# --- PARÁMETROS DE CONEXIÓN A MySQL (WAMP) ---
DB_CONFIG = {
    'host': '127.0.0.1',  
    'user': 'root',       
    'password': '', 
    'database': 'usuarios_aula_espejo' # El nombre de la BD que creaste
}
# ------------------------------------

def get_db_connection():
    """Establece y retorna la conexión a la base de datos."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error al conectar a MySQL: {err}")
        return None

# =================================================================
#                           MANEJADORES DE ERRORES
# =================================================================
# (Se mantienen tus manejadores de errores 404, 500, RequestEntityTooLarge, Exception)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('pagina_error.html', 
                          error_title="Página no encontrada",
                          error_message="La página que buscas no existe.",
                          error_code=404), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Server Error: {error}')
    return render_template('pagina_error.html',
                          error_title="Error interno del servidor",
                          error_message="Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo.",
                          error_code=500), 500

@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return render_template('pagina_error.html',
                          error_title="Archivo demasiado grande",
                          error_message="El archivo enviado supera el límite de tamaño permitido.",
                          error_code=413), 413

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f'Unhandled Exception: {error}')
    return render_template('pagina_error.html',
                          error_title="Error inesperado",
                          error_message="Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo.",
                          error_code=500), 500

# =================================================================
#                              RUTAS DE AUTENTICACIÓN
# =================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Si el usuario ya está logueado, lo redirigimos a la página principal
    if 'username' in session:
        return redirect(url_for('index'))
        
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if conn is None:
            # Si no hay conexión a la BD, mostramos un error
            error = "Error: No se pudo conectar a la base de datos."
            return render_template('login.html', error=error)

        cursor = conn.cursor(dictionary=True)
        
        # Consultar el usuario en la tabla 'users'
        query = "SELECT username, password_hash FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        
        cursor.close()
        conn.close()

        # Validar credenciales
        if user and check_password_hash(user['password_hash'], password):
            # Login exitoso: Guardamos el usuario en la sesión
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            error = "Usuario o contraseña incorrectos."
            
    # Mostrar el formulario de login (Método GET)
    return render_template('login.html', error=error)


# app.py

# ... (Tus funciones de conexión, manejadores de error, etc.)

# =================================================================
#                              RUTAS DE AUTENTICACIÓN
# =================================================================

# ... (Tu ruta /login)

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Opcional: Si el usuario ya está logueado, no debería ver el formulario de registro.
    if 'username' in session:
        return redirect(url_for('index'))
        
    error = None
    if request.method == 'POST':
        # 1. Obtener datos del formulario
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # 2. Generar el Hash Seguro de la Contraseña (¡CRUCIAL!)
        password_hash = generate_password_hash(password)
        
        conn = get_db_connection()
        if conn is None:
            error = "Error: No se pudo conectar a la base de datos."
            return render_template('register.html', error=error)

        cursor = conn.cursor()
        
        try:
            # 3. Insertar el nuevo usuario en la BD
            # Usamos %s como marcador de posición para evitar inyección SQL
            query = """
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            """
            # El orden de los valores debe coincidir con el orden de las columnas en la consulta
            cursor.execute(query, (username, email, password_hash))
            conn.commit()
            
            # Registro exitoso: Redirigir directamente al login (o a la página de inicio)
            # Nota: Puedes iniciar sesión automáticamente aquí si lo deseas, pero el mejor
            # flujo de seguridad es pedir al usuario que inicie sesión.
            return redirect(url_for('login'))
        
        except mysql.connector.Error as err:
            # Manejo de errores específicos (ej. usuario o email duplicado)
            if err.errno == 1062: # Código de error de MySQL para "Duplicate entry" (clave UNIQUE)
                error = "El nombre de usuario o el correo electrónico ya está registrado."
            else:
                logger.error(f"Error al registrar usuario: {err}")
                error = "Ocurrió un error al intentar registrar el usuario."
        
        finally:
            cursor.close()
            conn.close()
            
    # Mostrar el formulario de registro (Método GET) o mostrar error
    return render_template('register.html', error=error)

# ... (Tu ruta /logout y otras rutas)




@app.route('/logout')
def logout():
    # Cierre de sesión: elimina el usuario de la sesión
    session.pop('username', None) 
    return redirect(url_for('login'))


# =================================================================
#                                RUTAS PRINCIPALES
# =================================================================

@app.route('/')
def index():
    # Restringe el acceso: si no hay 'username' en la sesión, redirige a login
    if 'username' not in session:
        return redirect(url_for('login'))
        
    # Muestra el template index.html al usuario logueado
    return render_template('index.html', username=session['username'])

@app.route('/sobre-proyecto')
def sobre_proyecto():
    # También puedes restringir esta ruta si es solo para usuarios logueados
    if 'username' not in session:
        return redirect(url_for('login'))
        
    return render_template('sobre_proyecto.html')


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
        # Agregamos esta variable para saber si el usuario está logueado en cualquier template
        'logged_in': 'username' in session,
        'current_user': session.get('username')
    }

# =================================================================
#                                INICIO DE LA APP
# =================================================================

if __name__ == '__main__':
    # Verificar que los templates existen
    required_templates = ['index.html', 'pagina_error.html', 'login.html'] # Agregamos 'login.html'
    missing_templates = []
    
    for template in required_templates:
        template_path = os.path.join('templates', template)
        if not os.path.exists(template_path):
            missing_templates.append(template)
    
    if missing_templates:
        logger.warning(f"Plantillas faltantes: {', '.join(missing_templates)}")
        print(f"⚠️  Advertencia: Faltan plantillas: {', '.join(missing_templates)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)