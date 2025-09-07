#!/usr/bin/env python3
"""
Archivo WSGI para producción
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv('config.env')

from app import app

# Configuración para WSGI
application = app

if __name__ == "__main__":
    # Solo para testing local
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=False)

