from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# === 1. Cargar y rotar la imagen ===
# Se abre la imagen 
image_path = "C:/Users/maria/miniproyectos-metodos-numericos/imagen_Chocolatera.png"// pon su ruta de acceso al abrir el archivo clonado la imagen_Chocolatera.png
image = Image.open(image_path)

# Se rota la imagen 90° en sentido antihorario.
# expand=True evita que la imagen se recorte tras la rotación.
rotated_image = image.rotate(90, expand=True)

# === 2. Conversión de píxeles a centímetros ===
# Se obtiene el ancho y alto de la imagen en píxeles.
width_px, height_px = rotated_image.size

# Diámetro real del objeto en centímetros (dato de referencia conocido).
diametro_real_cm = 12.3

# Radio en píxeles (mitad del ancho de la imagen).
radio_px = width_px / 2

# Factor de conversión: cuántos cm reales corresponde cada píxel.
px_to_cm = diametro_real_cm / radio_px

# === 3. Corrección del eje Y ===
# Posición vertical (en píxeles) medida manualmente que corresponde al centro real del objeto.
obj_mid_y = 550  

# Desplazamiento en cm para ajustar el sistema de coordenadas con respecto al centro real.
y_offset_cm = (obj_mid_y - height_px / 2) * px_to_cm

# === 4. Visualización con Matplotlib ===
fig, ax = plt.subplots()

# Se muestra la imagen escalada a centímetros.
# extent define los límites en X y Y en el sistema de coordenadas físicas.
ax.imshow(
    rotated_image,
    extent=[
        0, width_px * px_to_cm,  # Eje X desde 0 hasta el ancho en cm
        -height_px / 2 * px_to_cm - y_offset_cm,  # Límite inferior en Y (ajustado)
        height_px / 2 * px_to_cm - y_offset_cm    # Límite superior en Y (ajustado)
    ]
)

# === 5. Referencia visual ===
# Línea horizontal roja en Y=0 para indicar el eje de referencia.
ax.axhline(y=0, color="red", linestyle="--")

# === 6. Configuración del gráfico ===
ax.set_xlabel("X (cm)")   # Etiqueta del eje X
ax.set_ylabel("Y (cm)")   # Etiqueta del eje Y
ax.grid(True)             # Mostrar cuadrícula para referencia

# Mostrar la figura final
plt.show()


