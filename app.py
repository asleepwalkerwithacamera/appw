from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Configurar matplotlib para usar backend no interactivo (evita errores de tkinter)
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from PIL import Image, ExifTags
import io
import base64
from flask import send_from_directory
from io import BytesIO
from flask import send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from forms import LoginForm, SignupForm, ForgotPasswordForm, ResetPasswordForm, ChangePasswordForm
from datetime import datetime, timedelta
import uuid
import secrets
import stripe
import json
from config import STRIPE_CONFIG, APP_CONFIG, MAIL_CONFIG
from file_validation import validate_image_file, generate_secure_filename
from secure_logging import setup_secure_logging, log_security_event, log_error

app = Flask(__name__)

# Configuración de la aplicación
app.config['SECRET_KEY'] = APP_CONFIG['SECRET_KEY']
app.config['SQLALCHEMY_DATABASE_URI'] = APP_CONFIG['DATABASE_URI']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuración de Flask-Mail
app.config['MAIL_SERVER'] = MAIL_CONFIG['MAIL_SERVER']
app.config['MAIL_PORT'] = MAIL_CONFIG['MAIL_PORT']
app.config['MAIL_USE_TLS'] = MAIL_CONFIG['MAIL_USE_TLS']
app.config['MAIL_USERNAME'] = MAIL_CONFIG['MAIL_USERNAME']
app.config['MAIL_PASSWORD'] = MAIL_CONFIG['MAIL_PASSWORD']
app.config['MAIL_DEFAULT_SENDER'] = MAIL_CONFIG['MAIL_DEFAULT_SENDER']

# Configurar Stripe
stripe.api_key = STRIPE_CONFIG['SECRET_KEY']

# Directorio donde se guardarán las imágenes
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inicializar base de datos
db = SQLAlchemy()

# Inicializar Flask-Mail
mail = Mail()

# Configurar logging seguro
setup_secure_logging(app)

# Modelo de Usuario
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime)
    
    # Campos de suscripción
    stripe_customer_id = db.Column(db.String(255), unique=True, nullable=True)
    stripe_subscription_id = db.Column(db.String(255), unique=True, nullable=True)
    is_subscribed = db.Column(db.Boolean, default=False)
    subscription_status = db.Column(db.String(50), default='inactive')  # inactive, active, past_due, canceled
    subscription_end_date = db.Column(db.DateTime, nullable=True)
    
    # Campos para restablecimiento de contraseña
    reset_token = db.Column(db.String(255), unique=True, nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    
    # Campos para contraseña temporal
    temp_password = db.Column(db.String(255), nullable=True)
    temp_password_expires = db.Column(db.DateTime, nullable=True)
    temp_password_used = db.Column(db.Boolean, default=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'is_admin': self.is_admin,
            'is_subscribed': self.is_subscribed,
            'subscription_status': self.subscription_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def generate_reset_token(self):
        """Genera un token único para restablecer la contraseña"""
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.now() + timedelta(hours=1)  # Token válido por 1 hora
        return self.reset_token
    
    def clear_reset_token(self):
        """Limpia el token de restablecimiento"""
        self.reset_token = None
        self.reset_token_expires = None
    
    def is_reset_token_valid(self):
        """Verifica si el token de restablecimiento es válido y no ha expirado"""
        return (self.reset_token is not None and 
                self.reset_token_expires is not None and 
                datetime.now() < self.reset_token_expires)
    
    def set_temp_password(self, temp_password):
        """Establece una contraseña temporal"""
        self.temp_password = generate_password_hash(temp_password)
        self.temp_password_expires = datetime.now() + timedelta(hours=24)  # Expira en 24 horas
        self.temp_password_used = False
    
    def clear_temp_password(self):
        """Limpia la contraseña temporal"""
        self.temp_password = None
        self.temp_password_expires = None
        self.temp_password_used = False
    
    def is_temp_password_valid(self):
        """Verifica si la contraseña temporal es válida y no ha expirado"""
        return (self.temp_password is not None and 
                self.temp_password_expires is not None and 
                datetime.now() < self.temp_password_expires and
                not self.temp_password_used)
    
    def check_temp_password(self, password):
        """Verifica si la contraseña temporal es correcta"""
        if self.is_temp_password_valid():
            return check_password_hash(self.temp_password, password)
        return False
    
    def use_temp_password(self):
        """Marca la contraseña temporal como usada y la convierte en la contraseña principal"""
        self.temp_password_used = True
        # La contraseña temporal se convierte en la contraseña principal
        # (ya está hasheada, solo la movemos)
        self.password_hash = self.temp_password
        # Limpiar los campos de contraseña temporal
        self.temp_password = None
        self.temp_password_expires = None
        # Marcar que la contraseña temporal ya no es válida
        self.temp_password_used = True

# Inicializar extensiones
db.init_app(app)
mail.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'landing'
login_manager.login_message = 'Por favor inicia sesión para acceder a esta página.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

# Crear base de datos y usuario admin
with app.app_context():
    db.create_all()
    
    # Crear usuario admin si no existe
    admin_user = User.query.filter_by(email='admin@admin.com').first()
    if not admin_user:
        admin_user = User(
            email='admin@admin.com',
            is_admin=True
        )
        admin_user.set_password('admin1998')
        db.session.add(admin_user)
        db.session.commit()
        app.logger.info("Usuario admin creado: admin@admin.com / admin1998")

# Función para redimensionar la imagen
def redimensionar_imagen(imagen):
    max_lado = 1000  # Nuevo tamaño máximo para la imagen redimensionada (ancho o alto)
    ancho, alto = imagen.size
    if ancho > max_lado or alto > max_lado:
        proporcion = max_lado / max(ancho, alto)
        nuevo_ancho = int(ancho * proporcion)
        nuevo_alto = int(alto * proporcion)
        return imagen.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
    return imagen

# Función para extraer datos EXIF de la imagen
def extraer_datos_exif(image_path):
    """
    Extrae los datos EXIF de una imagen y retorna un diccionario con la información
    """
    try:
        with Image.open(image_path) as imagen:
            exif_data = {}
            
            # Obtener datos EXIF si existen
            if hasattr(imagen, '_getexif') and imagen._getexif() is not None:
                exif = imagen._getexif()
                app.logger.debug(f"EXIF encontrado: {exif}")
                
                # Mapear códigos EXIF a nombres legibles
                exif_tags = {}
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, str(tag))
                    exif_tags[tag_name] = value
                    app.logger.debug(f"Tag: {tag_name} = {value}")
                
                # Extraer información específica de la cámara
                exif_data = {
                    'camera_make': exif_tags.get('Make', 'No disponible'),
                    'camera_model': exif_tags.get('Model', 'No disponible'),
                    'lens_model': exif_tags.get('LensModel', exif_tags.get('Lens', 'No disponible')),
                    'focal_length': exif_tags.get('FocalLength', 'No disponible'),
                    'f_number': exif_tags.get('FNumber', 'No disponible'),
                    'exposure_time': exif_tags.get('ExposureTime', 'No disponible'),
                    'iso_speed': exif_tags.get('ISOSpeedRatings', exif_tags.get('ISO', 'No disponible')),
                    'flash': exif_tags.get('Flash', 'No disponible'),
                    'date_time': exif_tags.get('DateTime', 'No disponible'),
                    'software': exif_tags.get('Software', 'No disponible')
                }
                
                # Formatear algunos valores para mejor presentación
                if exif_data['focal_length'] != 'No disponible':
                    if isinstance(exif_data['focal_length'], (int, float)):
                        exif_data['focal_length'] = f"{exif_data['focal_length']}mm"
                    else:
                        exif_data['focal_length'] = str(exif_data['focal_length'])
                
                if exif_data['f_number'] != 'No disponible':
                    if isinstance(exif_data['f_number'], (int, float)):
                        exif_data['f_number'] = f"f/{exif_data['f_number']}"
                    else:
                        exif_data['f_number'] = str(exif_data['f_number'])
                
                # Calcular velocidad de obturación a partir del tiempo de exposición
                if exif_data['exposure_time'] != 'No disponible':
                    if isinstance(exif_data['exposure_time'], tuple):
                        # Si es una tupla (numerador/denominador), calcular la velocidad
                        if exif_data['exposure_time'][1] != 0:
                            speed_value = exif_data['exposure_time'][1] / exif_data['exposure_time'][0]
                            exif_data['shutter_speed'] = f"1/{int(speed_value)}"
                        else:
                            exif_data['shutter_speed'] = "Error"
                    elif isinstance(exif_data['exposure_time'], (int, float)):
                        if exif_data['exposure_time'] < 1 and exif_data['exposure_time'] > 0:
                            # Calcular velocidad de obturación
                            speed_value = int(round(1 / exif_data['exposure_time']))
                            exif_data['shutter_speed'] = f"1/{speed_value}"
                        else:
                            exif_data['shutter_speed'] = f"{exif_data['exposure_time']}s"
                    else:
                        exif_data['shutter_speed'] = "Error"
                else:
                    exif_data['shutter_speed'] = "No disponible"
                
                # Mantener el tiempo de exposición original para referencia
                if exif_data['exposure_time'] != 'No disponible':
                    if isinstance(exif_data['exposure_time'], tuple):
                        exif_data['exposure_time'] = f"{exif_data['exposure_time'][0]}/{exif_data['exposure_time'][1]}s"
                    elif isinstance(exif_data['exposure_time'], (int, float)):
                        exif_data['exposure_time'] = f"{exif_data['exposure_time']}s"
                    else:
                        exif_data['exposure_time'] = str(exif_data['exposure_time'])
                
                if exif_data['iso_speed'] != 'No disponible':
                    if isinstance(exif_data['iso_speed'], (int, float)):
                        exif_data['iso_speed'] = f"ISO {exif_data['iso_speed']}"
                    else:
                        exif_data['iso_speed'] = str(exif_data['iso_speed'])
                
                # Formatear información del flash
                if exif_data['flash'] != 'No disponible':
                    if isinstance(exif_data['flash'], int):
                        if exif_data['flash'] == 0:
                            exif_data['flash'] = "No usado"
                        elif exif_data['flash'] == 1:
                            exif_data['flash'] = "Usado"
                        else:
                            exif_data['flash'] = f"Código: {exif_data['flash']}"
                    else:
                        exif_data['flash'] = str(exif_data['flash'])
                
                app.logger.debug(f"EXIF procesado: {exif_data}")
                
            else:
                app.logger.debug("No se encontraron datos EXIF en la imagen")
                # Si no hay datos EXIF, establecer valores por defecto
                exif_data = {
                    'camera_make': 'No disponible',
                    'camera_model': 'No disponible',
                    'lens_model': 'No disponible',
                    'focal_length': 'No disponible',
                    'f_number': 'No disponible',
                    'exposure_time': 'No disponible',
                    'shutter_speed': 'No disponible',
                    'iso_speed': 'No disponible',
                    'flash': 'No disponible',
                    'date_time': 'No disponible',
                    'software': 'No disponible'
                }
            
            return exif_data
            
    except Exception as e:
        app.logger.error(f"Error al extraer datos EXIF: {e}")
        # Retornar valores por defecto en caso de error
        return {
            'camera_make': 'Error al leer',
            'camera_model': 'Error al leer',
            'lens_model': 'Error al leer',
            'focal_length': 'Error al leer',
            'f_number': 'Error al leer',
            'exposure_time': 'Error al leer',
            'shutter_speed': 'Error al leer',
            'iso_speed': 'Error al leer',
            'flash': 'Error al leer',
            'date_time': 'Error al leer',
            'software': 'Error al leer'
        }

# Función para obtener los valores de HUE de los clusters
def plot_hue_distribution(image_path, num_clusters=2):
    # Abrir la imagen
    imagen = Image.open(image_path)

    # Convertir la imagen de PIL a un array numpy
    array = np.array(imagen)

    # Convertir la imagen de RGB a HSV
    imagen_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    # Extraer el canal HUE
    hue = imagen_hsv[:, :, 0]

    # Aplanar la matriz de HUE para facilitar el conteo de frecuencias
    hue_flat = hue.flatten()

    # Calcular las frecuencias de cada HUE (valores entre 0 y 179)
    histograma = cv2.calcHist([hue_flat], [0], None, [180], [0, 180])

    # Normalizar el histograma para obtener frecuencias relativas
    histograma_norm = histograma / np.sum(histograma)

    # Preparar los datos para el gráfico
    hues = np.arange(180)
    frequencies = histograma_norm.flatten() * 100  # Convertir a porcentaje

    # Realizar el clustering K-means sobre los valores de Hue
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(hue_flat.reshape(-1, 1))

    # Obtener los centros de los clusters (los tonos más predominantes)
    hues_predominantes = kmeans.cluster_centers_.flatten()

    # Contar los píxeles en cada cluster para calcular el porcentaje
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    total_pixels = len(hue_flat)
    cluster_percentages = (cluster_counts / total_pixels) * 100

    # Crear la figura del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar el histograma
    ax.bar(hues, frequencies, color=[plt.cm.hsv(h/180) for h in hues], alpha=0.7, label="Todos los Hues")

    # Graficar los centros de los clusters con sus colores reales y porcentajes
    for i, hue_value in enumerate(hues_predominantes):
        # Convertir el hue a un color RGB en el espacio HSV
        cluster_color = np.array([hue_value, 255, 255], dtype=np.uint8)
        cluster_color_rgb = cv2.cvtColor(cluster_color.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0, 0]

        # Normalizar el color RGB para que esté entre 0 y 1
        cluster_color_rgb_normalized = cluster_color_rgb / 255.0

        # Dibujar un círculo con el color correspondiente al centro del cluster
        ax.scatter(hue_value, 1.5, color=cluster_color_rgb_normalized, s=200, label=f"Cluster {i+1}")

        # Mostrar el porcentaje de píxeles en cada cluster
        ax.text(hue_value, 2, f"{cluster_percentages[i]:.1f}%", ha='center', color=cluster_color_rgb_normalized, fontsize=12)

    # Ajustes del gráfico
    ax.set_xlabel('Valor HUE')
    ax.set_ylabel('Frecuencia (%)')
    ax.set_title('Distribución de Frecuencia de HUE con K-means Clustering')
    ax.set_xticks(np.arange(0, 181, 10))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_yticks(np.arange(0, 21, 2))
    ax.set_ylim(0, 20)

    # Añadir leyenda
    ax.legend()

    # Ajustar el diseño
    plt.tight_layout()

    # Guardar el gráfico en un buffer y codificarlo en base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return img_str, hues_predominantes  # Retornar el gráfico y los valores de hues predominantes


def dibujar_dianas_en_imagen(hues_predominantes, radius=20, opacity=0.2, hue_margin=1):  # Cambio de opacity a 0.2
    # Ruta fija de la imagen
    wheel = os.path.join('static', 'uploads', 'js', 'wheel.png')

    # Abrir la imagen
    wheel = Image.open(wheel)

    # Asegurarse de que la imagen esté en formato RGBA (con canal alfa)
    if wheel.mode != 'RGBA':
        wheel = wheel.convert('RGBA')

    # Convertir la imagen de PIL a un array numpy
    array = np.array(wheel)

    # Convertir la imagen de RGB a HSV
    wheel_hsv = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2HSV)

    # Extraer el canal HUE
    hue = wheel_hsv[:, :, 0]

    # Crear una copia de la imagen para modificarla (manteniendo el canal alfa)
    wheel_con_dianas = array.copy()

    # Definir el color gris para la cruz (con canal alfa)
    gray_color = np.array([169, 169, 169, int(255 * opacity)], dtype=np.uint8)  # Opacidad al 20%

    # Recorrer cada valor de HUE de los clusters
    for hue_predominante in hues_predominantes:
        try:
            # Intentar convertir a float, y si no es posible, saltar este valor
            hue_predominante = float(hue_predominante)
        except ValueError:
            print(f"Valor no numérico encontrado en hues_predominantes: {hue_predominante}. Se omitirá.")
            continue  # Salir de la iteración y continuar con el siguiente valor

        print(f"Buscando el HUE: {hue_predominante}")  # Depuración: Ver el valor de HUE que estamos buscando

        # Calcular el rango de HUE usando hue_margin
        min_hue = (hue_predominante - hue_margin) % 180
        max_hue = (hue_predominante + hue_margin) % 180

        # Si el valor mínimo es mayor que el máximo, significa que el rango se cruza en 0, por lo que debemos separar la máscara
        if min_hue > max_hue:
            mask1 = (hue >= min_hue)  # Rango superior
            mask2 = (hue <= max_hue)  # Rango inferior
            mask = mask1 | mask2  # Unimos las dos máscaras
        else:
            mask = (hue >= min_hue) & (hue <= max_hue)  # Máscara normal

        # Verificar cuántos píxeles se están marcando
        print(f"Cantidad de píxeles encontrados con HUE {hue_predominante}: {np.sum(mask)}")

        # Dibujar la cruz en la imagen (vertical y horizontal)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    # Calcular la posición del píxel
                    center = (j, i)

                    # Dibujar la cruz con un tamaño determinado por radius
                    x_size = radius // 2

                    # Verificar que no se salga del rango de la imagen
                    if (0 <= i - x_size < array.shape[0] and 
                        0 <= i + x_size < array.shape[0] and 
                        0 <= j - x_size < array.shape[1] and 
                        0 <= j + x_size < array.shape[1]):
                        # Dibujar la línea vertical (cruz)
                        wheel_con_dianas[i - x_size:i + x_size + 1, j] = gray_color
                        # Dibujar la línea horizontal (cruz)
                        wheel_con_dianas[i, j - x_size:j + x_size + 1] = gray_color

    # Convertir la imagen modificada de nuevo a una imagen de PIL
    imagen_modificada = Image.fromarray(wheel_con_dianas)

    # Guardar la imagen modificada en un archivo temporal
    wheel_image_path = os.path.join('static', 'uploads', 'modified_wheel.png')
    imagen_modificada.save(wheel_image_path)

    # Retornar la ruta de la imagen modificada
    return wheel_image_path



# Función para calcular las intensidades mínima y máxima de la imagen en escala de grises
def calcular_intensidades(imagen_path):
    with Image.open(imagen_path) as imagen:
        imagen_gris = imagen.convert('L')
        array = np.array(imagen_gris)
        intensidad_min = np.min(array)
        intensidad_max = np.max(array)
        intensidad_min_pct = (intensidad_min / 255) * 100
        intensidad_max_pct = (intensidad_max / 255) * 100
        return round(intensidad_min_pct, 2), round(intensidad_max_pct, 2)


def ordenar_por_huesma1(imagen):
     # Abrir la imagen
    imagen = Image.open(imagen)
    
    # Convertir la imagen de PIL a un array numpy
    array = np.array(imagen)

    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    # Extraer componentes de color
    tonos = imagen_hsv[:, :, 0]
    saturacion = 255 * np.ones_like(tonos, dtype=np.uint8)  # Valor constante de saturación máxima

    # Crear máscaras para sombras, medios tonos y altas luces
    sombras_mask = (imagen_hsv[:, :, 2] < 51)  # 0-20% de luminosidad
    medios_tonos_mask = (imagen_hsv[:, :, 2] >= 51) & (imagen_hsv[:, :, 2] <= 204)  # 20-80% de luminosidad
    altas_luces_mask = (imagen_hsv[:, :, 2] > 204)  # 80-100% de luminosidad

    # Crear imágenes reordenadas para cada categoría
    imagen_sombras = np.stack((tonos, saturacion, sombras_mask.astype(np.uint8) * 255), axis=-1)
    imagen_medios_tonos = np.stack((tonos, saturacion, medios_tonos_mask.astype(np.uint8) * 255), axis=-1)
    imagen_altas_luces = np.stack((tonos, saturacion, altas_luces_mask.astype(np.uint8) * 255), axis=-1)

    # Convertir de nuevo a formato BGR para mostrar las imágenes
    imagen_sombras_bgr = cv2.cvtColor(imagen_sombras, cv2.COLOR_HSV2BGR)
    imagen_medios_tonos_bgr = cv2.cvtColor(imagen_medios_tonos, cv2.COLOR_HSV2BGR)
    imagen_altas_luces_bgr = cv2.cvtColor(imagen_altas_luces, cv2.COLOR_HSV2BGR)

    # Mostrar las imágenes reordenadas para sombras, medios tonos y altas luces
    fig, ax = plt.subplots(1, 3, figsize=(18, 10), subplot_kw=dict(xticks=[], yticks=[]))
    
    # Plotear sombras
    ax[0].imshow(cv2.cvtColor(imagen_sombras_bgr, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Sombras (0-20%)')
    
    # Plotear medios tonos
    ax[1].imshow(cv2.cvtColor(imagen_medios_tonos_bgr, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Medios Tonos (20-80%)')
    
    # Plotear altas luces
    ax[2].imshow(cv2.cvtColor(imagen_altas_luces_bgr, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Altas Luces (80-100%)')

    # Ajustar el espacio entre los subplots
    plt.tight_layout(h_pad=1.5)

    # Guardar el gráfico en formato PNG en un objeto BytesIO
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    # Convertir el gráfico a base64 para mostrarlo en la página web
    hues_distribution_combined = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return hues_distribution_combined



def ordenar_por_huesma2(imagen):
    # Abrir la imagen
    imagen = Image.open(imagen)
    
    # Convertir la imagen de PIL a un array numpy
    array = np.array(imagen)

    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    # Extraer componentes de color
    tonos = imagen_hsv[:, :, 0]
    saturacion = 255 * np.ones_like(tonos, dtype=np.uint8)  # Valor constante de saturación máxima

    # Crear máscaras para nuevos rangos de luminosidad
    rango_0_35_mask = (imagen_hsv[:, :, 2] < 25.5)  # 0-10% de luminosidad
    rango_35_50_mask = (imagen_hsv[:, :, 2] >= 26) & (imagen_hsv[:, :, 2] < 128)  # 10-50% de luminosidad
    rango_50_85_mask = (imagen_hsv[:, :, 2] >= 128) & (imagen_hsv[:, :, 2] < 229)  # 50-90% de luminosidad
    rango_85_100_mask = (imagen_hsv[:, :, 2] >= 229.5)  # 90-100% de luminosidad

    # Crear imágenes reordenadas para cada nuevo rango de luminosidad
    imagen_rango_0_35 = np.stack((tonos, saturacion, rango_0_35_mask.astype(np.uint8) * 255), axis=-1)
    imagen_rango_35_50 = np.stack((tonos, saturacion, rango_35_50_mask.astype(np.uint8) * 255), axis=-1)
    imagen_rango_50_85 = np.stack((tonos, saturacion, rango_50_85_mask.astype(np.uint8) * 255), axis=-1)
    imagen_rango_85_100 = np.stack((tonos, saturacion, rango_85_100_mask.astype(np.uint8) * 255), axis=-1)

    # Convertir de nuevo a formato BGR para mostrar las imágenes
    imagen_rango_0_35_bgr = cv2.cvtColor(imagen_rango_0_35, cv2.COLOR_HSV2BGR)
    imagen_rango_35_50_bgr = cv2.cvtColor(imagen_rango_35_50, cv2.COLOR_HSV2BGR)
    imagen_rango_50_85_bgr = cv2.cvtColor(imagen_rango_50_85, cv2.COLOR_HSV2BGR)
    imagen_rango_85_100_bgr = cv2.cvtColor(imagen_rango_85_100, cv2.COLOR_HSV2BGR)

    # Mostrar las imágenes reordenadas para nuevos rangos de luminosidad
    fig, ax = plt.subplots(1, 4, figsize=(18, 10), subplot_kw=dict(xticks=[], yticks=[]))

    ax[0].imshow(cv2.cvtColor(imagen_rango_0_35_bgr, cv2.COLOR_BGR2RGB))
    ax[0].set_title('0-10%')
    
    ax[1].imshow(cv2.cvtColor(imagen_rango_35_50_bgr, cv2.COLOR_BGR2RGB))
    ax[1].set_title('10-50%')
    
    ax[2].imshow(cv2.cvtColor(imagen_rango_50_85_bgr, cv2.COLOR_BGR2RGB))
    ax[2].set_title('50-90%')
    
    ax[3].imshow(cv2.cvtColor(imagen_rango_85_100_bgr, cv2.COLOR_BGR2RGB))
    ax[3].set_title('90-100%')

    # Ajustar el espacio entre los subplots
    plt.tight_layout(h_pad=1.5)

    # Guardar el gráfico en formato PNG en un objeto BytesIO
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    # Convertir el gráfico a base64 para mostrarlo en la página web
    hues_distribution_combined2 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return hues_distribution_combined2



def plot_dominant_colors_piechart(imagen, num_clusters=5):
   # Abrir la imagen
    imagen = Image.open(imagen)
    
    # Convertir la imagen de PIL a un array numpy
    array = np.array(imagen)

    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    # Redimensionar la imagen para facilitar la manipulación
    imagen_hsv_flat = imagen_hsv.reshape((-1, 3))

    # Usar solo el canal H para la clusterización
    hues = imagen_hsv_flat[:, 0].reshape(-1, 1)

    # Aplicar k-means para agrupar los colores en el espacio de Hues
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(hues)

    # Etiquetar los píxeles con el clúster correspondiente
    labels = kmeans.labels_

    # Calcular el número de píxeles en cada clúster
    cluster_counts = np.bincount(labels, minlength=num_clusters)

    # Encontrar los colores más representativos en cada clúster
    centroids = kmeans.cluster_centers_
    sorted_indices = np.argsort(cluster_counts)[::-1][:num_clusters]

    # Preparar datos para el gráfico de sectores
    valores = cluster_counts[sorted_indices]
    porcentajes = 100 * valores / np.sum(valores)  # Calcular el porcentaje de representación
    colores_hue = centroids[sorted_indices].astype(np.uint8)

    # Convertir de Hue a RGB para matplotlib
    colores_rgb = []
    for hue in colores_hue:
        color = np.uint8([[[hue[0], 255, 255]]])
        rgb_color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        colores_rgb.append(rgb_color[0][0])

    # Crear el gráfico de sectores con etiquetas de rango de matiz, porcentajes, saturación y valor medios
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)  # Ajustar los márgenes

    patches, _, _ = plt.pie(porcentajes, colors=[colores_rgb[i] / 255 for i in range(num_clusters)],
                            startangle=90, autopct='%1.1f%%', pctdistance=0.85)
    plt.title('Colores más representativos en HSV (Agrupados)', fontsize=16, fontweight='bold', pad=20)  # Ajustar el tamaño y el espaciado del título
    plt.axis('equal')  # Para que el gráfico sea un círculo en lugar de una elipse

    # Ajustar la leyenda para mostrar el rango de matiz, saturación y valor medios, y desviaciones estándar
    legend_labels = []
    for i in range(num_clusters):
        hue_range = f'Hue: {int(centroids[sorted_indices[i], 0]-10)}-{int(centroids[sorted_indices[i], 0]+10)}'
        saturation_mean = (np.sum(imagen_hsv_flat[labels == sorted_indices[i], 1]) / valores[i]) / 255 * 100
        value_mean = (np.sum(imagen_hsv_flat[labels == sorted_indices[i], 2]) / valores[i]) / 255 * 100
        saturation_std = np.std(imagen_hsv_flat[labels == sorted_indices[i], 1]) / 255 * 100
        value_std = np.std(imagen_hsv_flat[labels == sorted_indices[i], 2]) / 255 * 100
        legend_label = f'{hue_range}\nSat: {saturation_mean:.1f}% ± {saturation_std:.1f}%\nVal: {value_mean:.1f}% ± {value_std:.1f}%'
        legend_labels.append(legend_label)

    # Añadir leyenda
    plt.legend(patches, legend_labels,
               loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title='Hue Ranges', borderaxespad=0., frameon=False)

    # Guardar el gráfico en formato PNG en un objeto BytesIO
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    # Convertir el gráfico a base64 para mostrarlo en la página web
    hues_distribution_pie = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return hues_distribution_pie



def plot_hue_distribution2(imagen, luminosity_range):
    # Abrir la imagen
    imagen = Image.open(imagen)
    
    # Convertir la imagen de PIL a un array numpy
    array = np.array(imagen)

    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    # Extraer el canal HUE
    hue = imagen_hsv[:, :, 0]

    # Calcular la máscara para la luminosidad especificada
    if luminosity_range == 'negros':
        mask = (imagen_hsv[:, :, 2] <= 25)  # 0-10% de luminosidad
        lum_text = "0-10%"
    elif luminosity_range == 'sombras':
        mask = ((imagen_hsv[:, :, 2] > 25) & (imagen_hsv[:, :, 2] <= 102))  # 11-40% de luminosidad
        lum_text = "11-40%"
    elif luminosity_range == 'medios tonos':
        mask = ((imagen_hsv[:, :, 2] > 102) & (imagen_hsv[:, :, 2] <= 153))  # 41-60% de luminosidad
        lum_text = "41-60%"
    elif luminosity_range == 'altas luces':
        mask = ((imagen_hsv[:, :, 2] > 153) & (imagen_hsv[:, :, 2] <= 229))  # 61-89% de luminosidad
        lum_text = "61-89%"
    elif luminosity_range == 'blancos':
        mask = (imagen_hsv[:, :, 2] > 229)  # 90-100% de luminosidad
        lum_text = "90-100%"
    else:
        raise ValueError("Luminosity range must be one of: negros, sombras, medios tonos, altas luces, blancos")

    # Aplicar la máscara al canal HUE
    hue_selected = hue[mask]

    # Calcular las frecuencias de cada HUE (valores entre 0 y 179)
    try:
        histograma = cv2.calcHist([hue_selected], [0], None, [180], [0, 180])
        histograma_norm = histograma / np.sum(histograma)
    except cv2.error as e:
        print(f"No pixels found for luminosity range: {luminosity_range}")
        histograma_norm = np.zeros((180, 1))  # Crear un histograma vacío
        max_frequency = 0
    else:
        # Obtener el máximo valor del histograma normalizado
        max_frequency = np.max(histograma_norm)

    # Preparar los datos para el gráfico
    hues = np.arange(180)
    frequencies = histograma_norm.flatten() * 100  # Convertir a porcentaje

    # Crear el recuadro de colores para el eje x
    hue_colors = [plt.cm.hsv(h/180) for h in hues]

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(hues, frequencies, color=hue_colors, alpha=0.7)
    ax.set_xlabel('HUE Value')
    ax.set_ylabel('Frequency (%)')
    ax.set_title(f'HUE Frequency Distribution for {luminosity_range} ({lum_text})')
    ax.set_xticks(np.arange(0, 181, 10))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar el rango del eje y dinámicamente
    if max_frequency > 0:
        ax.set_yticks(np.arange(0, max_frequency * 100 + 10, 2))
        ax.set_ylim(0, max_frequency * 100 + 10)
    else:
        ax.set_yticks(np.arange(0, 10, 2))  # Valores por defecto si el histograma es vacío
        ax.set_ylim(0, 10)

    plt.tight_layout()

    # Guardar el gráfico en formato PNG en un objeto BytesIO
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    # Convertir el gráfico a base64 para mostrarlo en la página web
    hues_distribution_lumi = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return hues_distribution_lumi





def plot_dominant_colors(imagen, num_clusters=5, luminosity_threshold_min=0, luminosity_threshold_max=10, title=''):
    # Abrir la imagen
    imagen = Image.open(imagen)
    
    # Convertir la imagen de PIL a un array numpy
    array = np.array(imagen)

    # Convertir la imagen de RGB a HSV
    imagen_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    # Redimensionar la imagen para facilitar la manipulación
    imagen_hsv_flat = imagen_hsv.reshape((-1, 3))

    # Filtrar los píxeles con valores de luminosidad en el rango dado
    value_threshold_min = (luminosity_threshold_min / 100) * 255
    value_threshold_max = (luminosity_threshold_max / 100) * 255
    filtered_pixels = imagen_hsv_flat[(imagen_hsv_flat[:, 2] > value_threshold_min) & (imagen_hsv_flat[:, 2] <= value_threshold_max)]

    # Verificar si hay suficientes píxeles para proceder
    if len(filtered_pixels) < num_clusters:
        return None

    # Usar solo el canal H para la clusterización
    hues = filtered_pixels[:, 0].reshape(-1, 1)

    # Aplicar k-means para agrupar los colores en el espacio de Hues
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(hues)

    # Etiquetar los píxeles con el clúster correspondiente
    labels = kmeans.labels_

    # Calcular el número de píxeles en cada clúster
    cluster_counts = np.bincount(labels, minlength=num_clusters)

    # Calcular la suma y la desviación estándar de la saturación y el valor para cada clúster
    cluster_saturation_sum = np.zeros(num_clusters)
    cluster_value_sum = np.zeros(num_clusters)
    cluster_saturation_std = np.zeros(num_clusters)
    cluster_value_std = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_pixels = filtered_pixels[labels == i]
        cluster_saturation_sum[i] = np.sum(cluster_pixels[:, 1])
        cluster_value_sum[i] = np.sum(cluster_pixels[:, 2])
        cluster_saturation_std[i] = np.std(cluster_pixels[:, 1])
        cluster_value_std[i] = np.std(cluster_pixels[:, 2])

    # Calcular la saturación y el valor medios en porcentaje para cada clúster
    cluster_mean_saturation = (cluster_saturation_sum / cluster_counts) / 255 * 100
    cluster_mean_value = (cluster_value_sum / cluster_counts) / 255 * 100
    cluster_saturation_std = (cluster_saturation_std / 255) * 100
    cluster_value_std = (cluster_value_std / 255) * 100

    # Encontrar los colores más representativos en cada clúster
    centroids = kmeans.cluster_centers_
    sorted_indices = np.argsort(cluster_counts)[::-1][:num_clusters]

    # Preparar datos para el gráfico de sectores
    valores = cluster_counts[sorted_indices]
    porcentajes = 100 * valores / np.sum(valores)  # Calcular el porcentaje de representación
    colores_hue = centroids[sorted_indices].astype(np.uint8)

    # Convertir de Hue a RGB para matplotlib
    colores_rgb = []
    for hue in colores_hue:
        color = np.uint8([[[hue[0], 255, 255]]])
        rgb_color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        colores_rgb.append(rgb_color[0][0])

    # Crear el gráfico de sectores con etiquetas de porcentaje
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)  # Ajustar los márgenes

    patches, texts, autotexts = plt.pie(porcentajes, colors=[colores_rgb[i] / 255 for i in range(num_clusters)],
                                        startangle=90, autopct='%1.1f%%', pctdistance=0.85)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)  # Ajustar el tamaño y el espaciado del título
    plt.axis('equal')  # Para que el gráfico sea un círculo en lugar de una elipse

    # Ajustar la leyenda para que no se solape
    legend_labels = [
        f'Hue: {int(centroids[sorted_indices[i], 0]-10)}-{int(centroids[sorted_indices[i], 0]+10)}\n'
        f'Sat: {cluster_mean_saturation[sorted_indices[i]]:.1f}% ± {cluster_saturation_std[sorted_indices[i]]:.1f}%\n'
        f'Val: {cluster_mean_value[sorted_indices[i]]:.1f}% ± {cluster_value_std[sorted_indices[i]]:.1f}%'
        for i in range(num_clusters)
    ]
    plt.legend(patches, legend_labels,
               loc='center left',
               bbox_to_anchor=(1.05, 0.5),  # Ajustar el valor de bbox_to_anchor para separar más la leyenda
               fontsize='small',
               title='Hue Ranges',
               borderaxespad=0.,
               frameon=False)

    # Añadir porcentajes dentro del gráfico
    for autotext in autotexts:
        autotext.set_color('white')  # Hacer que los porcentajes sean visibles sobre los colores

    # Guardar el gráfico en formato PNG en un objeto BytesIO
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    # Convertir el gráfico a base64 para mostrarlo en la página web
    hues_distribution_pie = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return hues_distribution_pie

def plot_dominant_colors_dark_part(imagen, num_clusters=5, luminosity_threshold=10):
    return plot_dominant_colors(imagen, num_clusters, 0, luminosity_threshold, 'Colores más representativos en la parte oscura (0-10% luminosidad)')

def plot_dominant_colors_highlights(imagen, num_clusters=5, luminosity_threshold=90):
    return plot_dominant_colors(imagen, num_clusters, luminosity_threshold, 100, 'Colores más representativos en las altas luces (90-100% luminosidad)')

def plot_dominant_colors_whites(imagen, num_clusters=5, luminosity_threshold_min=95, luminosity_threshold_max=100):
    return plot_dominant_colors(imagen, num_clusters, luminosity_threshold_min, luminosity_threshold_max, 'Colores más representativos en los blancos (95-100% luminosidad)')

def plot_dominant_colors_dark_midtones(imagen, num_clusters=5, luminosity_threshold_min=10, luminosity_threshold_max=50):
    return plot_dominant_colors(imagen, num_clusters, luminosity_threshold_min, luminosity_threshold_max, 'Colores más representativos en los medios tonos oscuros (10-50% luminosidad)')

def plot_dominant_colors_light_midtones(imagen, num_clusters=5, luminosity_threshold_min=50, luminosity_threshold_max=90):
    return plot_dominant_colors(imagen, num_clusters, luminosity_threshold_min, luminosity_threshold_max, 'Colores más representativos en los medios tonos claros (50-90% luminosidad)')

def calcular_hues_por_rango(imagen_path):
    """
    Calcula la saturación y luminancia promedio para diferentes rangos de HUE
    Retorna una lista de diccionarios con la información de cada rango de color
    """
    # Definir los rangos de HUE para cada color
    rangos_hue = {
        'rojo': (0, 30),      # 0-30 grados
        'naranja': (15, 45),  # 15-45 grados (con solapamiento)
        'amarillo': (30, 60), # 30-60 grados
        'verde': (60, 120),   # 60-120 grados
        'magenta': (300, 360), # 300-360 grados (incluye 0)
        'azul': (180, 300)    # 180-300 grados
    }
    
    # Abrir la imagen
    imagen = Image.open(imagen_path)
    
    # Convertir la imagen de PIL a un array numpy
    array = np.array(imagen)
    
    # Convertir la imagen de RGB a HSV
    imagen_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    
    # Extraer canales
    hue = imagen_hsv[:, :, 0]  # HUE (0-179 en OpenCV)
    saturation = imagen_hsv[:, :, 1]  # Saturación (0-255)
    value = imagen_hsv[:, :, 2]  # Valor/Luminancia (0-255)
    
    # Aplanar los arrays para facilitar el procesamiento
    hue_flat = hue.flatten()
    saturation_flat = saturation.flatten()
    value_flat = value.flatten()
    
    resultados = []
    
    for color, (hue_min, hue_max) in rangos_hue.items():
        # Convertir rangos de grados (0-360) a OpenCV (0-179)
        hue_min_cv = int((hue_min * 179) / 360)
        hue_max_cv = int((hue_max * 179) / 360)
        
        # Crear máscara para el rango de HUE
        if hue_min_cv <= hue_max_cv:
            mask = (hue_flat >= hue_min_cv) & (hue_flat <= hue_max_cv)
        else:
            # Para rangos que cruzan el 0 (como magenta)
            mask = (hue_flat >= hue_min_cv) | (hue_flat <= hue_max_cv)
        
        # Filtrar píxeles que están en este rango de HUE
        if np.any(mask):
            hue_pixels = hue_flat[mask]
            saturation_pixels = saturation_flat[mask]
            value_pixels = value_flat[mask]
            
            # Calcular promedios
            hue_promedio = np.mean(hue_pixels)
            saturation_promedio = (np.mean(saturation_pixels) / 255) * 100  # Convertir a porcentaje
            luminancia_promedio = (np.mean(value_pixels) / 255) * 100  # Convertir a porcentaje
            
            # Contar píxeles en este rango
            num_pixeles = len(hue_pixels)
            porcentaje_total = (num_pixeles / len(hue_flat)) * 100
            
            resultados.append({
                'color': color,
                'hue_min': hue_min,
                'hue_max': hue_max,
                'hue_promedio': round(hue_promedio, 1),
                'saturacion_promedio': round(saturation_promedio, 1),
                'luminancia_promedio': round(luminancia_promedio, 1),
                'num_pixeles': num_pixeles,
                'porcentaje_total': round(porcentaje_total, 1)
            })
        else:
            # Si no hay píxeles en este rango
            resultados.append({
                'color': color,
                'hue_min': hue_min,
                'hue_max': hue_max,
                'hue_promedio': 0,
                'saturacion_promedio': 0,
                'luminancia_promedio': 0,
                'num_pixeles': 0,
                'porcentaje_total': 0
            })
    return resultados


# Función para generar contraseñas temporales seguras
def generate_temp_password(length=10):
    """Genera una contraseña temporal segura solo con letras y números"""
    import string
    import random
    
    # Caracteres disponibles (solo letras y números)
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    
    # Asegurar que hay al menos uno de cada tipo
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits)
    ]
    
    # Completar el resto de la contraseña
    all_chars = lowercase + uppercase + digits
    password.extend(random.choice(all_chars) for _ in range(length - 3))
    
    # Mezclar la contraseña
    random.shuffle(password)
    return ''.join(password)

# Función para enviar correos de restablecimiento de contraseña
def send_reset_email(user, reset_url):
    """Envía un correo de restablecimiento de contraseña al usuario"""
    try:
        msg = Message(
            'Restablecimiento de Contraseña - APP COLORES',
            recipients=[user.email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        
        msg.html = render_template('emails/reset_password.html', 
                                 user=user, 
                                 reset_url=reset_url)
        
        mail.send(msg)
        return True
    except Exception as e:
        app.logger.error(f"Error enviando correo: {e}")
        return False

def send_temp_password_email(user, temp_password):
    """Envía un correo con la contraseña temporal"""
    try:
        msg = Message(
            'Nueva Contraseña Temporal - APP COLORES',
            recipients=[user.email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        
        msg.html = render_template('emails/temp_password.html', 
                                 user=user, 
                                 temp_password=temp_password)
        
        mail.send(msg)
        return True
    except Exception as e:
        app.logger.error(f"Error enviando correo con contraseña temporal: {e}")
        return False


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    # Verificar que el usuario tenga una suscripción activa
    if not current_user.is_subscribed:
        flash('Tu suscripción no está activa. Por favor contacta con soporte.', 'error')
        return redirect(url_for('landing'))
    
    if request.method == 'POST':
        # Subir la imagen de forma segura
        image_file = request.files.get('image')
        if image_file:
            # Validar archivo de forma segura
            is_valid, message = validate_image_file(image_file)
            
            if not is_valid:
                flash(f'Error: {message}', 'error')
                log_security_event('INVALID_FILE_UPLOAD', message, 
                                 user_ip=request.remote_addr, 
                                 user_id=current_user.id if current_user.is_authenticated else None)
                return render_template('index.html')
            
            # Generar nombre de archivo seguro
            secure_filename = generate_secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename)
            
            try:
                # Guardar archivo
                image_file.save(image_path)
                session['image_path'] = image_path
                
                # Log del evento
                app.logger.info(f"Imagen subida correctamente: {secure_filename}")
                
                # Limpiar imágenes antiguas después de cada subida
                limpiar_imagenes_antiguas()
                
                return redirect(url_for('show_image'))
                
            except Exception as e:
                flash('Error al guardar la imagen. Intenta de nuevo.', 'error')
                log_error('FILE_SAVE_ERROR', str(e), 
                         user_ip=request.remote_addr,
                         user_id=current_user.id if current_user.is_authenticated else None)
                return render_template('index.html')
        else:
            flash('No se seleccionó ningún archivo.', 'error')
            return render_template('index.html')
    return render_template('index.html')

@app.route('/show_image', methods=['GET', 'POST'])
@login_required
def show_image():
    # Verificar que el usuario tenga una suscripción activa
    if not current_user.is_subscribed:
        flash('Tu suscripción no está activa. Por favor contacta con soporte.', 'error')
        return redirect(url_for('landing'))
    
    image_path = session.get('image_path')  # Obtener la ruta de la imagen desde la sesión

    if not image_path:
        return redirect(url_for('index'))  # Redirigir si no hay imagen cargada

    # Obtener el número de clusters del formulario y validar
    try:
        num_clusters = int(request.form.get('num_clusters', 2))  # Cambiado de 5 a 2
        if num_clusters < 1:
            num_clusters = 2  # Establecer valor predeterminado si el número de clusters es inválido
    except ValueError:
        num_clusters = 2  # Valor por defecto si el input es incorrecto

    # Mostrar el gráfico de distribución de HUE y obtener los hues predominantes
    img_str, hues_predominantes = plot_hue_distribution(image_path, num_clusters)
    # Dibujar las cruces sobre la imagen
    hues_distribution_combined = ordenar_por_huesma1(image_path)
    hues_distribution_combined2 = ordenar_por_huesma2(image_path)
    dominant_colors = plot_dominant_colors_piechart(image_path) 

    modified_image_path = dibujar_dianas_en_imagen(hues_predominantes)
    
    # Gráficos de distribución de HUE por rangos de luminosidad
    negros = plot_hue_distribution2(image_path, 'negros')
    sombras = plot_hue_distribution2(image_path, 'sombras')
    medios_tonos = plot_hue_distribution2(image_path, 'medios tonos')
    altas_luces = plot_hue_distribution2(image_path, 'altas luces')
    blancos = plot_hue_distribution2(image_path, 'blancos')
    
    # Gráficos de colores dominantes (pie charts) por rangos de luminosidad
    negros_pie = plot_dominant_colors_dark_part(image_path)
    sombras_pie = plot_dominant_colors_dark_midtones(image_path)
    medios_tonos_pie = plot_dominant_colors_light_midtones(image_path)
    altas_luces_pie = plot_dominant_colors_highlights(image_path)
    blancos_pie = plot_dominant_colors_whites(image_path)  # Función específica para blancos
    
    #intensidades 
    min_intensity, max_intensity = calcular_intensidades(image_path)
    
    # Calcular rangos de HUE por color
    rangos_hue = calcular_hues_por_rango(image_path)
    
    # Extraer datos EXIF de la imagen
    exif_data = extraer_datos_exif(image_path)
    
    # Extraer paleta de colores por defecto (8 colores)
    palette_data = extract_color_palette(image_path, 8)

    return render_template('show_image.html', 
                           img_str=img_str, 
                           image_path=os.path.basename(image_path), 
                           min_intensity=min_intensity, 
                           max_intensity=max_intensity,
                           hues_distribution_combined=hues_distribution_combined,
                           hues_distribution_combined2=hues_distribution_combined2,
                           dominant_colors=dominant_colors,
                           negros=negros, 
                           negros_pie=negros_pie,
                           sombras=sombras, 
                           sombras_pie=sombras_pie,
                           medios_tonos=medios_tonos, 
                           medios_tonos_pie=medios_tonos_pie,
                           altas_luces=altas_luces, 
                           altas_luces_pie=altas_luces_pie,
                           blancos=blancos,
                           blancos_pie=blancos_pie,
                           num_clusters=num_clusters, 
                           rangos_hue=rangos_hue,
                           exif_data=exif_data,
                           palette_data=palette_data)


@app.route('/update_kmeans', methods=['POST'])
@login_required
def update_kmeans():
    """Ruta para actualizar solo el gráfico de K-means sin recargar toda la página"""
    # Verificar que el usuario tenga una suscripción activa
    if not current_user.is_subscribed:
        return {'error': 'Suscripción no activa'}, 403
    
    image_path = session.get('image_path')
    if not image_path:
        return {'error': 'No hay imagen cargada'}, 400
    
    try:
        num_clusters = int(request.json.get('num_clusters', 2))
        if num_clusters < 1:
            num_clusters = 2
    except ValueError:
        num_clusters = 2
    
    # Generar solo el gráfico de K-means
    img_str, hues_predominantes = plot_hue_distribution(image_path, num_clusters)
    
    # Actualizar la imagen de la rueda de colores
    modified_image_path = dibujar_dianas_en_imagen(hues_predominantes)
    
    return {
        'kmeans_chart': img_str,
        'modified_wheel': url_for('static', filename='uploads/modified_wheel.png'),
        'num_clusters': num_clusters
    }

@app.route('/extract_palette', methods=['POST'])
@login_required
def extract_palette():
    """Ruta para extraer la paleta de colores de una imagen"""
    # Verificar que el usuario tenga una suscripción activa
    if not current_user.is_subscribed:
        return {'error': 'Suscripción no activa'}, 403
    
    image_path = session.get('image_path')
    if not image_path:
        return {'error': 'No hay imagen cargada'}, 400
    
    try:
        num_colors = int(request.json.get('num_colors', 8))
        if num_colors < 3 or num_colors > 20:
            num_colors = 8  # Valor por defecto si está fuera del rango
    except ValueError:
        num_colors = 8
    
    # Extraer la paleta de colores
    palette_data = extract_color_palette(image_path, num_colors)
    
    if palette_data:
        return jsonify(palette_data)
    else:
        return {'error': 'Error al extraer la paleta de colores'}, 500

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    # Verificar que el usuario tenga una suscripción activa
    if not current_user.is_subscribed:
        flash('Tu suscripción no está activa. Por favor contacta con soporte.', 'error')
        return redirect(url_for('landing'))
    
    filepath = os.path.join('uploads', filename)
    
    # Abrir la imagen usando PIL
    imagen = Image.open(filepath)

    # Redimensionar la imagen si es necesario
    imagen_redimensionada = redimensionar_imagen(imagen)

    # Guardar la imagen redimensionada en un objeto BytesIO para enviarla
    img_io = BytesIO()
    imagen_redimensionada.save(img_io, 'JPEG')
    img_io.seek(0)

    # Enviar la imagen redimensionada usando send_file
    return send_file(img_io, mimetype='image/jpeg')

# Rutas de autenticación
@app.route('/')
def landing():
    """Landing page principal con formularios de login/signup"""
    login_form = LoginForm()
    signup_form = SignupForm()
    return render_template('landing.html', login_form=login_form, signup_form=signup_form)

@app.route('/login', methods=['POST'])
def login():
    """Procesar el login del usuario"""
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            # Verificar contraseña normal
            if user.check_password(form.password.data):
                # Verificar si el usuario tiene una suscripción activa
                if not user.is_subscribed:
                    flash('Tu cuenta no está activa. Por favor completa el proceso de pago para acceder.', 'error')
                    return redirect(url_for('landing'))
                
                login_user(user, remember=form.remember_me.data)
                
                # Actualizar último login
                user.last_login = datetime.now()
                db.session.commit()
                
                flash('¡Bienvenido de vuelta!', 'success')
                return redirect(url_for('index'))
            
            # Verificar contraseña temporal
            elif user.check_temp_password(form.password.data):
                # Verificar si el usuario tiene una suscripción activa
                if not user.is_subscribed:
                    flash('Tu cuenta no está activa. Por favor completa el proceso de pago para acceder.', 'error')
                    return redirect(url_for('landing'))
                
                # Marcar la contraseña temporal como usada
                user.use_temp_password()
                db.session.commit()
                
                login_user(user, remember=form.remember_me.data)
                
                # Actualizar último login
                user.last_login = datetime.now()
                db.session.commit()
                
                flash('¡Bienvenido! Tu contraseña temporal ha sido activada y la anterior ha sido reemplazada.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Email o contraseña incorrectos. Por favor intenta de nuevo.', 'error')
        else:
            flash('Email o contraseña incorrectos. Por favor intenta de nuevo.', 'error')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return redirect(url_for('landing'))

@app.route('/initiate_signup', methods=['POST'])
def initiate_signup():
    """Iniciar el proceso de registro - solo email, luego pago con Stripe"""
    email = request.form.get('email')
    
    if not email:
        # Si no hay email, mostrar formulario para solicitarlo
        return render_template('email_prompt.html')
    
    # Verificar si el usuario ya existe
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash('Este email ya está registrado. Por favor usa otro o inicia sesión.', 'error')
        return redirect(url_for('landing'))
    
    try:
        # Crear sesión de checkout de Stripe directamente
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': STRIPE_CONFIG['PRICE_ID'],  # Precio desde configuración
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.host_url + 'subscription-success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=request.host_url + 'subscription-cancel',
            metadata={
                'email': email
            }
        )
        
        # Redirigir a Stripe Checkout
        return redirect(checkout_session.url, code=303)
        
    except stripe.error.StripeError as e:
        flash(f'Error al procesar el pago: {str(e)}', 'error')
        log_error("STRIPE_INITIATE_ERROR", str(e), user_ip=request.remote_addr)
    except Exception as e:
        flash('Error al iniciar el registro. Por favor intenta de nuevo.', 'error')
        log_error("INITIATE_SIGNUP_ERROR", str(e), user_ip=request.remote_addr)
    
    return redirect(url_for('landing'))

@app.route('/logout')
@login_required
def logout():
    """Cerrar sesión del usuario"""
    logout_user()
    flash('Has cerrado sesión exitosamente.', 'success')
    return redirect(url_for('landing'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Página para solicitar restablecimiento de contraseña"""
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generar contraseña temporal
            temp_password = generate_temp_password(10)
            user.set_temp_password(temp_password)
            db.session.commit()
            
            # Enviar email con la contraseña temporal
            if send_temp_password_email(user, temp_password):
                flash(f'Se ha enviado una nueva contraseña temporal a {email}. Revisa tu bandeja de entrada.', 'success')
                flash('La contraseña temporal expira en 24 horas. Tu contraseña anterior seguirá funcionando hasta que uses la temporal.', 'info')
            else:
                flash('Error al enviar el correo. Por favor, intenta de nuevo más tarde.', 'error')
        else:
            # Por seguridad, no revelar si el email existe o no
            flash('Si el email existe en nuestra base de datos, recibirás una nueva contraseña temporal.', 'success')
        
        return redirect(url_for('landing'))
    
    return render_template('forgot_password.html', form=form)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Página para restablecer la contraseña con el token"""
    # Buscar usuario con el token válido
    user = User.query.filter_by(reset_token=token).first()
    
    if not user or not user.is_reset_token_valid():
        flash('El enlace de restablecimiento es inválido o ha expirado.', 'error')
        return redirect(url_for('landing'))
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        # Actualizar contraseña
        user.set_password(form.password.data)
        user.clear_reset_token()
        db.session.commit()
        
        flash('Tu contraseña ha sido restablecida exitosamente. Ahora puedes iniciar sesión.', 'success')
        return redirect(url_for('landing'))
    
    return render_template('reset_password.html', form=form)

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Página para cambiar la contraseña del usuario logueado"""
    form = ChangePasswordForm()
    if form.validate_on_submit():
        # Verificar que la contraseña actual sea correcta
        if current_user.check_password(form.current_password.data):
            # Cambiar la contraseña
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash('Tu contraseña ha sido cambiada exitosamente.', 'success')
            return redirect(url_for('index'))
        else:
            flash('La contraseña actual es incorrecta.', 'error')
    
    return render_template('change_password.html', form=form)

@app.route('/subscription-success')
def subscription_success():
    """Página de éxito después del pago exitoso - redirige a sign_up.html"""
    session_id = request.args.get('session_id')
    
    if not session_id:
        flash('Error: No se pudo verificar la sesión de pago.', 'error')
        return redirect(url_for('landing'))
    
    try:
        # Verificar la sesión de checkout
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        if checkout_session.payment_status == 'paid':
            # Obtener el email de los metadatos
            email = checkout_session.metadata.get('email')
            
            if email:
                # Guardar información en la sesión para el registro
                session['pending_registration'] = {
                    'email': email,
                    'stripe_subscription_id': checkout_session.subscription if checkout_session.subscription else None
                }
                
                # Redirigir a la página de registro estática
                return render_template('sign_up.html', email=email)
            else:
                flash('Error: Información de registro no encontrada.', 'error')
        else:
            flash('Error: El pago no se completó correctamente.', 'error')
            
    except stripe.error.StripeError as e:
        flash(f'Error al verificar el pago: {str(e)}', 'error')
        log_error("STRIPE_SUCCESS_ERROR", str(e), user_ip=request.remote_addr)
    except Exception as e:
        flash('Error inesperado al procesar el pago.', 'error')
        log_error("SUBSCRIPTION_SUCCESS_ERROR", str(e), user_ip=request.remote_addr)
    
    return redirect(url_for('landing'))

@app.route('/complete_registration', methods=['POST'])
def complete_registration():
    """Completar el registro después del pago exitoso"""
    # Verificar que hay información de registro pendiente
    pending_data = session.get('pending_registration')
    if not pending_data:
        flash('Error: No hay información de registro pendiente.', 'error')
        return redirect(url_for('landing'))
    
    email = pending_data['email']
    stripe_subscription_id = pending_data['stripe_subscription_id']
    
    # Obtener datos del formulario
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    
    # Validaciones
    if not password or not confirm_password:
        flash('Por favor completa todos los campos.', 'error')
        return render_template('sign_up.html', email=email)
    
    if password != confirm_password:
        flash('Las contraseñas no coinciden.', 'error')
        return render_template('sign_up.html', email=email)
    
    if len(password) < 6:
        flash('La contraseña debe tener al menos 6 caracteres.', 'error')
        return render_template('sign_up.html', email=email)
    
    try:
        # Obtener información de la suscripción de Stripe
        if stripe_subscription_id:
            subscription = stripe.Subscription.retrieve(stripe_subscription_id)
            stripe_customer_id = subscription.customer
        else:
            # Si no hay subscription_id, crear un customer
            customer = stripe.Customer.create(email=email)
            stripe_customer_id = customer.id
        
        # Crear el usuario en la base de datos
        new_user = User(
            email=email,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_subscription_id,
            is_subscribed=True,
            subscription_status='active'
        )
        new_user.set_password(password)
        
        # Guardar usuario en la base de datos
        db.session.add(new_user)
        db.session.commit()
        
        # Limpiar la sesión
        session.pop('pending_registration', None)
        
        flash('¡Registro completado exitosamente! Ahora puedes iniciar sesión.', 'success')
        return redirect(url_for('landing'))
        
    except Exception as e:
        db.session.rollback()
        flash('Error al completar el registro. Por favor intenta de nuevo.', 'error')
        log_error("COMPLETE_REGISTRATION_ERROR", str(e), user_ip=request.remote_addr)
        return render_template('sign_up.html', email=email)

@app.route('/subscription-cancel')
def subscription_cancel():
    """Página de cancelación del pago"""
    flash('El proceso de pago fue cancelado. Tu cuenta no ha sido activada.', 'error')
    return redirect(url_for('landing'))

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook de Stripe para manejar eventos de suscripción"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        # Verificar la firma del webhook (recomendado para producción)
        if STRIPE_CONFIG.get('WEBHOOK_SECRET'):
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_CONFIG['WEBHOOK_SECRET'])
        else:
            # Solo para desarrollo - NO usar en producción
            event = stripe.Event.construct_from(json.loads(payload), sig_header)
        
        # Manejar eventos de suscripción
        if event.type == 'customer.subscription.updated':
            subscription = event.data.object
            handle_subscription_updated(subscription)
        elif event.type == 'customer.subscription.deleted':
            subscription = event.data.object
            handle_subscription_deleted(subscription)
        elif event.type == 'invoice.payment_failed':
            invoice = event.data.object
            handle_payment_failed(invoice)
        
        return jsonify({'status': 'success'}), 200
        
    except ValueError as e:
        # Payload inválido
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        # Firma inválida
        return jsonify({'error': 'Invalid signature'}), 400
    except Exception as e:
        log_error("WEBHOOK_ERROR", str(e), user_ip=request.remote_addr)
        return jsonify({'error': 'Internal error'}), 500

def handle_subscription_updated(subscription):
    """Manejar actualización de suscripción"""
    try:
        user = User.query.filter_by(stripe_subscription_id=subscription.id).first()
        if user:
            user.subscription_status = subscription.status
            if subscription.current_period_end:
                from datetime import datetime
                user.subscription_end_date = datetime.fromtimestamp(subscription.current_period_end)
            
            user.is_subscribed = subscription.status == 'active'
            db.session.commit()
            print(f"Usuario {user.email} - Suscripción actualizada: {subscription.status}")
    except Exception as e:
        print(f"Error al actualizar suscripción: {e}")

def handle_subscription_deleted(subscription):
    """Manejar eliminación de suscripción"""
    try:
        user = User.query.filter_by(stripe_subscription_id=subscription.id).first()
        if user:
            user.subscription_status = 'canceled'
            user.is_subscribed = False
            db.session.commit()
            print(f"Usuario {user.email} - Suscripción cancelada")
    except Exception as e:
        print(f"Error al cancelar suscripción: {e}")

def handle_payment_failed(invoice):
    """Manejar fallo de pago"""
    try:
        user = User.query.filter_by(stripe_customer_id=invoice.customer).first()
        if user:
            user.subscription_status = 'past_due'
            user.is_subscribed = False
            db.session.commit()
            print(f"Usuario {user.email} - Pago fallido")
    except Exception as e:
        print(f"Error al manejar pago fallido: {e}")

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard principal después del login"""
    return redirect(url_for('index'))

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    """Ruta de administrador simple con contraseña"""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == 'admin1998':
            # Obtener todos los usuarios
            users = User.query.all()
            return render_template('admin.html', users=users)
        else:
            flash('Contraseña incorrecta', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/delete_user/<user_id>', methods=['POST'])
def delete_user(user_id):
    """Eliminar usuario de la base de datos"""
    try:
        user = User.query.get(user_id)
        if user:
            # No permitir eliminar al usuario admin principal
            if user.email == 'admin@admin.com':
                flash('No se puede eliminar al administrador principal', 'error')
                return redirect(url_for('admin'))
            
            # Eliminar el usuario
            db.session.delete(user)
            db.session.commit()
            flash(f'Usuario {user.email} eliminado exitosamente', 'success')
        else:
            flash('Usuario no encontrado', 'error')
    except Exception as e:
        db.session.rollback()
        flash(f'Error al eliminar usuario: {str(e)}', 'error')
        print(f"Error eliminando usuario: {e}")
    
    return redirect(url_for('admin'))

# Función para limpiar imágenes antiguas (más de 5 minutos)
def limpiar_imagenes_antiguas():
    """Elimina todas las imágenes del directorio uploads que tengan más de 5 minutos de antigüedad"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        tiempo_limite = datetime.now() - timedelta(minutes=5)
        archivos_eliminados = 0
        archivos_revisados = 0
        
        print(f"Iniciando limpieza de imágenes antiguas en: {upload_folder}")
        print(f"Tiempo límite: {tiempo_limite.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Verificar que el directorio existe
        if not os.path.exists(upload_folder):
            print(f"El directorio {upload_folder} no existe")
            return
        
        # Recorrer todos los archivos en el directorio uploads
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            archivos_revisados += 1
            
            # Verificar que es un archivo (no un directorio)
            if os.path.isfile(file_path):
                # Obtener el tiempo de modificación del archivo
                tiempo_modificacion = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                print(f"Archivo: {filename} - Modificado: {tiempo_modificacion.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Si el archivo es más antiguo que 5 minutos, eliminarlo
                if tiempo_modificacion < tiempo_limite:
                    try:
                        os.remove(file_path)
                        archivos_eliminados += 1
                        print(f"✓ Imagen antigua eliminada: {filename}")
                    except Exception as e:
                        print(f"✗ Error al eliminar {filename}: {e}")
                else:
                    print(f"  Imagen reciente mantenida: {filename}")
        
        print(f"\nResumen de limpieza:")
        print(f"- Archivos revisados: {archivos_revisados}")
        print(f"- Archivos eliminados: {archivos_eliminados}")
        print(f"- Archivos mantenidos: {archivos_revisados - archivos_eliminados}")
        
        if archivos_eliminados > 0:
            print(f"✓ Limpieza completada: {archivos_eliminados} imágenes antiguas eliminadas")
        else:
            print("✓ No se encontraron imágenes antiguas para eliminar")
            
    except Exception as e:
        print(f"✗ Error durante la limpieza de imágenes: {e}")
        import traceback
        traceback.print_exc()

def extract_color_palette(imagen, num_colors=8):
    """
    Extrae una paleta de colores dominantes de una imagen.
    
    Args:
        imagen: Ruta de la imagen
        num_colors: Número de colores a extraer (por defecto 8)
    
    Returns:
        dict: Diccionario con la paleta de colores y visualizaciones
    """
    try:
        # Abrir la imagen
        img = Image.open(imagen)
        
        # Convertir la imagen a RGB si no lo está
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convertir a array numpy
        img_array = np.array(img)
        
        # Redimensionar para procesamiento más rápido
        height, width = img_array.shape[:2]
        if height * width > 1000000:  # Si la imagen es muy grande
            scale_factor = np.sqrt(1000000 / (height * width))
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            img_array = cv2.resize(img_array, (new_width, new_height))
        
        # Reshape para clustering
        pixels = img_array.reshape(-1, 3)
        
        # Aplicar K-means para agrupar colores
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Obtener colores y etiquetas
        colors = kmeans.cluster_centers_.astype(np.uint8)
        labels = kmeans.labels_
        
        # Calcular porcentajes de cada color
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels)) * 100
        
        # Ordenar por porcentaje (más dominante primero)
        sorted_indices = np.argsort(percentages)[::-1]
        sorted_colors = colors[sorted_indices]
        sorted_percentages = percentages[sorted_indices]
        
        # Color principal (más dominante)
        dominant_color = sorted_colors[0]
        dominant_percentage = sorted_percentages[0]
        
        # Colores secundarios
        secondary_colors = sorted_colors[1:]
        secondary_percentages = sorted_percentages[1:]
        
        # Crear visualización de paleta uniforme
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Paleta uniforme
        ax1.set_title('Paleta Uniforme', fontsize=16, fontweight='bold', pad=20)
        for i, (color, percentage) in enumerate(zip(sorted_colors, sorted_percentages)):
            rect = plt.Rectangle((i, 0), 1, 1, facecolor=color/255, edgecolor='white', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(i + 0.5, 0.5, f'{percentage:.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white' if np.mean(color) < 128 else 'black')
        
        ax1.set_xlim(0, num_colors)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(num_colors))
        ax1.set_xticklabels([f'C{i+1}' for i in range(num_colors)])
        ax1.set_yticks([])
        
        # Paleta proporcional
        ax2.set_title('Paleta Proporcional', fontsize=16, fontweight='bold', pad=20)
        current_x = 0
        for i, (color, percentage) in enumerate(zip(sorted_colors, sorted_percentages)):
            width = percentage / 100
            rect = plt.Rectangle((current_x, 0), width, 1, facecolor=color/255, edgecolor='white', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(current_x + width/2, 0.5, f'{percentage:.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white' if np.mean(color) < 128 else 'black')
            current_x += width
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.tight_layout()
        
        # Guardar visualización
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
        img_io.seek(0)
        plt.close()
        
        # Convertir a base64
        palette_visualization = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Preparar datos de la paleta
        palette_data = {
            'dominant_color': {
                'rgb': dominant_color.tolist(),
                'hex': '#{:02x}{:02x}{:02x}'.format(dominant_color[0], dominant_color[1], dominant_color[2]),
                'percentage': dominant_percentage
            },
            'secondary_colors': [
                {
                    'rgb': color.tolist(),
                    'hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]),
                    'percentage': percentage
                }
                for color, percentage in zip(secondary_colors, secondary_percentages)
            ],
            'total_colors': num_colors,
            'visualization': palette_visualization
        }
        
        return palette_data
        
    except Exception as e:
        print(f"Error al extraer paleta de colores: {e}")
        return None

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Configuración para producción
    app.run(
        host=APP_CONFIG['HOST'],
        port=APP_CONFIG['PORT'],
        debug=APP_CONFIG['DEBUG']
    )