"""
Configuración de seguridad para APP COLORES
"""
from flask_talisman import Talisman
from flask import Flask

def configure_security(app: Flask):
    """Configurar medidas de seguridad adicionales"""
    
    # Configurar Talisman para headers de seguridad
    csp = {
        'default-src': "'self'",
        'script-src': [
            "'self'",
            "'unsafe-inline'",  # Necesario para algunos scripts
            'https://js.stripe.com',
            'https://checkout.stripe.com'
        ],
        'style-src': [
            "'self'",
            "'unsafe-inline'",  # Necesario para estilos inline
            'https://fonts.googleapis.com'
        ],
        'font-src': [
            "'self'",
            'https://fonts.gstatic.com'
        ],
        'img-src': [
            "'self'",
            'data:',
            'blob:'
        ],
        'connect-src': [
            "'self'",
            'https://api.stripe.com'
        ],
        'frame-src': [
            'https://checkout.stripe.com'
        ]
    }
    
    talisman = Talisman(
        app,
        force_https=True,  # Forzar HTTPS en producción
        strict_transport_security=True,
        strict_transport_security_max_age=31536000,  # 1 año
        content_security_policy=csp,
        content_security_policy_nonce_in=['script-src', 'style-src'],
        referrer_policy='strict-origin-when-cross-origin'
    )
    
    return talisman

def configure_rate_limiting(app: Flask):
    """Configurar rate limiting básico"""
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Límites específicos para endpoints sensibles
    limiter.limit("5 per minute")(app.view_functions.get('login'))
    limiter.limit("3 per minute")(app.view_functions.get('signup'))
    limiter.limit("10 per minute")(app.view_functions.get('index'))
    
    return limiter

def configure_csrf_protection(app: Flask):
    """Configurar protección CSRF"""
    from flask_wtf.csrf import CSRFProtect
    
    csrf = CSRFProtect(app)
    
    # Configurar tokens CSRF
    app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hora
    app.config['WTF_CSRF_SSL_STRICT'] = True  # Solo HTTPS en producción
    
    return csrf
