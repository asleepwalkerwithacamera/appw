# APP COLORES - Análisis de Colores con Python

Una aplicación web desarrollada en Flask para analizar y visualizar la distribución de colores en imágenes utilizando técnicas de procesamiento de imagen y machine learning.

## Características

- **Análisis de HUE**: Distribución de frecuencias de colores en el espacio HSV
- **Clustering K-means**: Agrupación automática de colores predominantes
- **Análisis por luminosidad**: Separación de colores por rangos de luminosidad (sombras, medios tonos, altas luces)
- **Visualización interactiva**: Gráficos de barras, gráficos de sectores y rueda de colores
- **Análisis por rangos de color**: Estadísticas detalladas por rangos de HUE (rojo, verde, azul, etc.)

## Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Procesamiento de Imágenes**: OpenCV, PIL (Pillow)
- **Machine Learning**: Scikit-learn (K-means clustering)
- **Visualización**: Matplotlib
- **Frontend**: HTML, CSS, JavaScript

## Instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd APP-COLORES
```

2. Crea un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecuta la aplicación:
```bash
python app.py
```

5. Abre tu navegador en `http://localhost:5000`

## Uso

1. **Subir imagen**: Selecciona una imagen desde tu computadora
2. **Configurar clusters**: Ajusta el número de clusters para el análisis K-means
3. **Ver resultados**: Explora los diferentes análisis de colores:
   - Distribución de HUE
   - Colores predominantes
   - Análisis por luminosidad
   - Rangos de color específicos

## Estructura del Proyecto

```
APP-COLORES/
├── app.py                 # Aplicación principal Flask
├── models.py             # Modelos de datos
├── forms.py              # Formularios
├── templates/            # Plantillas HTML
├── static/               # Archivos estáticos (CSS, JS, imágenes)
├── uploads/              # Directorio de imágenes subidas
├── requirements.txt      # Dependencias del proyecto
└── README.md            # Documentación
```

## Funcionalidades Principales

### Análisis de HUE
- Histograma de distribución de frecuencias de colores
- Clustering K-means para identificar colores predominantes
- Visualización con colores reales en el eje X

### Análisis por Luminosidad
- **Negros**: 0-10% de luminosidad
- **Sombras**: 11-40% de luminosidad
- **Medios tonos**: 41-60% de luminosidad
- **Altas luces**: 61-89% de luminosidad
- **Blancos**: 90-100% de luminosidad

### Rangos de Color
- Análisis estadístico por rangos de HUE
- Cálculo de saturación y luminancia promedio
- Porcentaje de píxeles por rango de color

## Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Autor

Desarrollado como proyecto de análisis de colores con Python y Flask.
