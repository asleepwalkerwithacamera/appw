"""
Sistema de validación de archivos seguro para APP COLORES
"""
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Configuración de seguridad
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DIMENSIONS = (4000, 4000)  # Máximo 4000x4000 píxeles

def is_allowed_file(filename):
    """Verificar si la extensión del archivo está permitida"""
    if not filename:
        return False
    
    # Obtener extensión
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return extension in ALLOWED_EXTENSIONS

def validate_image_file(file):
    """Validar archivo de imagen de forma segura"""
    try:
        # Verificar que el archivo existe
        if not file or not file.filename:
            logger.warning("Intento de subir archivo sin nombre")
            return False, "No se proporcionó archivo"
        
        # Verificar extensión
        if not is_allowed_file(file.filename):
            logger.warning(f"Intento de subir archivo con extensión no permitida: {file.filename}")
            return False, "Tipo de archivo no permitido. Solo se permiten: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP"
        
        # Verificar tamaño
        file.seek(0, 2)  # Ir al final del archivo
        file_size = file.tell()
        file.seek(0)  # Volver al inicio
        
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"Intento de subir archivo demasiado grande: {file_size} bytes")
            return False, f"Archivo demasiado grande. Máximo permitido: {MAX_FILE_SIZE // (1024*1024)}MB"
        
        # Verificar que es una imagen válida
        try:
            with Image.open(file) as img:
                # Verificar dimensiones
                if img.size[0] > MAX_DIMENSIONS[0] or img.size[1] > MAX_DIMENSIONS[1]:
                    logger.warning(f"Intento de subir imagen con dimensiones excesivas: {img.size}")
                    return False, f"Imagen demasiado grande. Máximo permitido: {MAX_DIMENSIONS[0]}x{MAX_DIMENSIONS[1]} píxeles"
                
                # Verificar que es una imagen real (no un archivo disfrazado)
                img.verify()
                
        except Exception as e:
            logger.warning(f"Archivo no es una imagen válida: {file.filename}, error: {e}")
            return False, "El archivo no es una imagen válida"
        
        # Si llegamos aquí, el archivo es válido
        logger.info(f"Archivo validado correctamente: {file.filename}")
        return True, "Archivo válido"
        
    except Exception as e:
        logger.error(f"Error inesperado validando archivo: {e}")
        return False, "Error validando archivo"

def generate_secure_filename(original_filename):
    """Generar nombre de archivo seguro"""
    # Obtener extensión
    extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    
    # Generar nombre único y seguro
    secure_name = f"{uuid.uuid4().hex}.{extension}"
    
    logger.info(f"Nombre de archivo generado: {original_filename} -> {secure_name}")
    return secure_name

def sanitize_filename(filename):
    """Sanitizar nombre de archivo"""
    # Usar werkzeug para sanitizar
    sanitized = secure_filename(filename)
    
    # Si el nombre está vacío después de sanitizar, generar uno
    if not sanitized:
        sanitized = f"image_{uuid.uuid4().hex[:8]}"
    
    return sanitized
