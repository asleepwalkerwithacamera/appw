import os
from dotenv import load_dotenv

# Cargar variables de entorno PRIMERO
load_dotenv('config.env')

# Configuración de Email - SOLO desde variables de entorno
# ⚠️ IMPORTANTE: Las credenciales NO deben estar hardcodeadas en el código
MAIL_CONFIG = {
    'MAIL_SERVER': os.getenv('MAIL_SERVER', 'smtp.gmail.com'),
    'MAIL_PORT': int(os.getenv('MAIL_PORT', 587)),
    'MAIL_USE_TLS': os.getenv('MAIL_USE_TLS', 'True').lower() == 'true',
    'MAIL_USERNAME': os.getenv('MAIL_USERNAME'),
    'MAIL_PASSWORD': os.getenv('MAIL_PASSWORD'),
    'MAIL_DEFAULT_SENDER': os.getenv('MAIL_DEFAULT_SENDER')
}

# Verificar que las credenciales críticas estén configuradas
if not MAIL_CONFIG['MAIL_USERNAME']:
    raise ValueError("MAIL_USERNAME no está configurada en las variables de entorno")
if not MAIL_CONFIG['MAIL_PASSWORD']:
    raise ValueError("MAIL_PASSWORD no está configurada en las variables de entorno")
if not MAIL_CONFIG['MAIL_DEFAULT_SENDER']:
    raise ValueError("MAIL_DEFAULT_SENDER no está configurada en las variables de entorno")

# Configuración de Stripe - SOLO PRODUCCIÓN

# Configuración de Stripe - SOLO desde variables de entorno
# ⚠️ IMPORTANTE: Las claves NO deben estar hardcodeadas en el código
STRIPE_CONFIG = {
    'PUBLISHABLE_KEY': os.getenv('STRIPE_PUBLISHABLE_KEY'),
    'SECRET_KEY': os.getenv('STRIPE_SECRET_KEY'),
    'PRODUCT_ID': os.getenv('STRIPE_PRODUCT_ID', 'prod_T0p3Og8mjYPd9t'),
    'PRICE_ID': os.getenv('STRIPE_PRICE_ID', 'price_1S4nPHCCYxyEFt2NixnnwZoD'),
    'WEBHOOK_SECRET': os.getenv('STRIPE_WEBHOOK_SECRET', '')
}

# Verificar que las claves críticas estén configuradas
if not STRIPE_CONFIG['PUBLISHABLE_KEY']:
    raise ValueError("STRIPE_PUBLISHABLE_KEY no está configurada en las variables de entorno")
if not STRIPE_CONFIG['SECRET_KEY']:
    raise ValueError("STRIPE_SECRET_KEY no está configurada en las variables de entorno")

# Configuración de Base de Datos - POSTGRESQL PARA RENDER
def get_database_uri():
    """Obtener URI de base de datos PostgreSQL"""
    # Render siempre proporciona DATABASE_URL
    db_url = os.getenv('DATABASE_URL')
    
    if db_url:
        print("✅ Usando PostgreSQL de Render (producción)")
        return db_url
    
    # Solo para desarrollo local - usar SQLite
    print("⚠️  Usando SQLite temporal para desarrollo local")
    print("💡 En Render usará PostgreSQL automáticamente")
    return 'sqlite:///temp_app.db'

# Configuración de la aplicación - SOLO PRODUCCIÓN
APP_CONFIG = {
    'SECRET_KEY': os.getenv('SECRET_KEY', 'clave_super_secreta_produccion_2024_app_colores_secure_key_xyz789'),
    'DEBUG': False,  # Siempre False en producción
    'DATABASE_URI': get_database_uri(),
    'HOST': os.getenv('HOST', '0.0.0.0'),
    'PORT': int(os.getenv('PORT', 5000)),
    'FLASK_ENV': 'production'
}

# INSTRUCCIONES PASO A PASO:
# 1. Ve a https://myaccount.google.com/
# 2. Seguridad → Verificación en dos pasos → Activar
# 3. Seguridad → Verificación en dos pasos → Contraseñas de aplicación
# 4. Selecciona "Mail" y genera la contraseña
# 5. Copia la contraseña de 16 caracteres
# 6. Cambia 'tu-email@gmail.com' por tu email real
# 7. Cambia 'tu-app-password' por la contraseña generada
