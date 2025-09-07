"""
Sistema de logging seguro para APP COLORES
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_secure_logging(app):
    """Configurar sistema de logging seguro"""
    
    # Crear directorio de logs si no existe
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configurar nivel de logging
    log_level = logging.INFO
    if app.config.get('DEBUG'):
        log_level = logging.DEBUG
    
    # Configurar formato de logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar handler para archivo con rotación
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Configurar handler para consola (solo en desarrollo)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Configurar logger principal
    app.logger.setLevel(log_level)
    app.logger.addHandler(file_handler)
    
    # Solo agregar consola en desarrollo
    if app.config.get('DEBUG'):
        app.logger.addHandler(console_handler)
    
    # Configurar loggers específicos
    setup_security_logger()
    setup_error_logger()
    
    app.logger.info("Sistema de logging configurado correctamente")

def setup_security_logger():
    """Configurar logger específico para eventos de seguridad"""
    security_logger = logging.getLogger('security')
    security_logger.setLevel(logging.INFO)
    
    # Handler específico para eventos de seguridad
    security_handler = RotatingFileHandler(
        'logs/security.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=10
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
    )
    security_handler.setFormatter(formatter)
    security_logger.addHandler(security_handler)
    
    # No propagar al logger principal
    security_logger.propagate = False

def setup_error_logger():
    """Configurar logger específico para errores"""
    error_logger = logging.getLogger('errors')
    error_logger.setLevel(logging.ERROR)
    
    # Handler específico para errores
    error_handler = RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=10
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - ERROR - %(levelname)s - %(message)s'
    )
    error_handler.setFormatter(formatter)
    error_logger.addHandler(error_handler)
    
    # No propagar al logger principal
    error_logger.propagate = False

def log_security_event(event_type, details, user_ip=None, user_id=None):
    """Registrar evento de seguridad"""
    security_logger = logging.getLogger('security')
    
    log_data = {
        'event': event_type,
        'details': details,
        'timestamp': datetime.now().isoformat(),
        'user_ip': user_ip,
        'user_id': user_id
    }
    
    security_logger.warning(f"SECURITY_EVENT: {log_data}")

def log_error(error_type, error_message, user_ip=None, user_id=None):
    """Registrar error de forma segura"""
    error_logger = logging.getLogger('errors')
    
    # No incluir información sensible en logs
    safe_message = sanitize_log_message(error_message)
    
    log_data = {
        'error_type': error_type,
        'message': safe_message,
        'timestamp': datetime.now().isoformat(),
        'user_ip': user_ip,
        'user_id': user_id
    }
    
    error_logger.error(f"ERROR: {log_data}")

def sanitize_log_message(message):
    """Sanitizar mensaje de log para evitar exposición de información sensible"""
    if not message:
        return "No message"
    
    # Lista de palabras sensibles a ocultar
    sensitive_words = [
        'password', 'secret', 'key', 'token', 'api_key',
        'stripe', 'sk_live', 'pk_live', 'sk_test', 'pk_test'
    ]
    
    safe_message = str(message)
    
    for word in sensitive_words:
        if word.lower() in safe_message.lower():
            safe_message = safe_message.replace(word, '[REDACTED]')
    
    return safe_message
