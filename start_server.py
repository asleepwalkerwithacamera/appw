#!/usr/bin/env python3
"""
Script para iniciar APP COLORES con Gunicorn en producción
"""
import os
import subprocess
import sys
from pathlib import Path

def create_logs_directory():
    """Crear directorio de logs si no existe"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"📁 Directorio de logs: {logs_dir.absolute()}")

def start_gunicorn():
    """Iniciar la aplicación con Gunicorn"""
    print("🚀 Iniciando APP COLORES con Gunicorn")
    print("💳 Stripe: MODO PRODUCCIÓN (pagos reales)")
    print("🔒 Debug: DESHABILITADO (seguridad)")
    print("🗄️ Base de datos: PostgreSQL")
    print("⚠️  IMPORTANTE: Esta aplicación está configurada para pagos reales")
    
    # Crear directorio de logs
    create_logs_directory()
    
    # Comando de Gunicorn
    cmd = [
        "gunicorn",
        "--config", "gunicorn.conf.py",
        "wsgi:application"
    ]
    
    try:
        print(f"🔧 Ejecutando: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al iniciar Gunicorn: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Gunicorn no está instalado. Instálalo con: pip install gunicorn")
        sys.exit(1)

if __name__ == "__main__":
    start_gunicorn()
