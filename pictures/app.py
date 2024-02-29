import cv2
from matplotlib import pyplot as plt

# Cargar la imagen en escala de grises
image_path = 'pictures/1.pgm'  # Asegúrate de cambiar esto por la ruta correcta de tu imagen
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Mejorar el contraste con CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image)

# Aplicar la umbralización adaptativa de la media a la imagen con contraste mejorado
img_adaptive_mean_clahe = cv2.adaptiveThreshold(image_clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)

# Invertir la imagen para que el fondo sea negro y el reconocimiento sea blanco
img_inverted_clahe = cv2.bitwise_not(img_adaptive_mean_clahe)

# Mostrar la imagen original y la imagen final invertida
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(img_inverted_clahe, cmap='gray')
axs[1].set_title('Final Invertida')
axs[1].axis('off')

plt.show()
