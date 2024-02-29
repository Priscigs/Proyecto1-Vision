import cv2
from matplotlib import pyplot as plt

# Cargar la imagen en escala de grises
image_path = 'pictures/1.pgm'  # Asegúrate de cambiar esto por la ruta correcta de tu imagen
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Aplicar la umbralización adaptativa de la media
img_adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)

# Invertir la imagen para que el fondo sea negro y el reconocimiento sea blanco
img_inverted = cv2.bitwise_not(img_adaptive_mean)

# Mostrar la imagen original y la imagen invertida
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(img_inverted, cmap='gray')
axs[1].set_title('Umbralización Adaptativa Media')
axs[1].axis('off')

plt.show()
