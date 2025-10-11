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

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f'Unhandled Exception: {error}')
    return render_template('pagina_error.html',
                         error_title="Error inesperado",
                         error_message="Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo.",
                         error_code=500), 500

# Rutas principales
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sobre-proyecto')
def sobre_proyecto():
    return render_template('sobre_proyecto.html')

# Funciones de contexto para templates
@app.context_processor
def inject_globals():
    """Inyecta variables globales a todos los templates."""
    return {
        'app_name': 'FlexLearn',
        'app_version': '1.0',
        'current_year': 2025
    }

if __name__ == '__main__':
    # Verificar que los templates existen
    required_templates = ['index.html', 'sobre_proyecto.html', 'pagina_error.html']
    missing_templates = []
    
    for template in required_templates:
        template_path = os.path.join('templates', template)
        if not os.path.exists(template_path):
            missing_templates.append(template)
    
    if missing_templates:
        logger.warning(f"Plantillas faltantes: {', '.join(missing_templates)}")
        print(f"⚠️  Advertencia: Faltan plantillas: {', '.join(missing_templates)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)