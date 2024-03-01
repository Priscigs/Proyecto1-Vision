import cv2
from matplotlib import pyplot as plt
import numpy as np

# Función para calcular métricas
def calculate_metrics(true_mask, predicted_mask):
    true_positive = np.sum(np.logical_and(predicted_mask == 255, true_mask == 255))
    false_positive = np.sum(np.logical_and(predicted_mask == 255, true_mask == 0))
    true_negative = np.sum(np.logical_and(predicted_mask == 0, true_mask == 0))
    false_negative = np.sum(np.logical_and(predicted_mask == 0, true_mask == 255))

    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    recall = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = true_positive / (true_positive + false_positive)
    f1_score = (2 * precision * recall) / (precision + recall)

    return accuracy, recall, specificity, precision, f1_score

# Cargar la imagen en escala de grises
image_path = 'pictures/2.pgm'  # Cambia esto por la ruta correcta de tu imagen
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocesamiento: Reducir ruido con un filtro mediano
image_filtered = cv2.medianBlur(image, 5)

# Mejorar el contraste con CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image_filtered)

# Aplicar la umbralización adaptativa de la media a la imagen con contraste mejorado
# Ajusta el tamaño de bloque y C según las necesidades específicas de tu imagen
img_adaptive_mean_clahe = cv2.adaptiveThreshold(image_clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 19, 5)

# Invertir la imagen para que el fondo sea negro y el reconocimiento sea blanco
img_inverted_clahe = cv2.bitwise_not(img_adaptive_mean_clahe)

# Post-procesamiento: Aplicar operaciones morfológicas para limpiar la imagen
# Crear un kernel para las operaciones morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Utilizar apertura morfológica para eliminar ruido pequeño
img_morph = cv2.morphologyEx(img_inverted_clahe, cv2.MORPH_OPEN, kernel)

# Cargar la máscara de referencia
true_mask_path = 'pictures/_gt.pgm'  # Ruta de la máscara de referencia
true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)

# Calcular métricas
accuracy, recall, specificity, precision, f1_score = calculate_metrics(true_mask, img_morph)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Specificity:", specificity)
print("Precision:", precision)
print("F1 Score:", f1_score)

# Mostrar la imagen original y la imagen final procesada
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(img_morph, cmap='gray')
axs[1].set_title('Final Procesada')
axs[1].axis('off')

plt.show()
