from flask import Flask, render_template
import os
import logging
from werkzeug.exceptions import RequestEntityTooLarge, BadRequest

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manejadores de errores
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

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Funciones de contexto para templates
@app.context_processor
def inject_globals():
    """Inyecta variables globales a todos los templates."""
    return {
        'app_name': 'Aprendizaje Flexible',
        'app_version': '1.0',
        'current_year': 2025
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)