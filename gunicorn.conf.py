# Configuración de Gunicorn para APP COLORES
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv('config.env')

# Configuración del servidor
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', 5000)}"
workers = 4  # Número de workers (ajustar según CPU)
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Configuración de logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Configuración de seguridad
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Configuración de procesos
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# Configuración de SSL (descomentar si usas HTTPS)
# keyfile = "/path/to/your/private.key"
# certfile = "/path/to/your/certificate.crt"
