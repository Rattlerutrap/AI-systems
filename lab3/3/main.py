import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

def compress_image(image_path, n_colors=16):
    """
    Сжимает изображение, уменьшая количество цветов до n_colors
    
    Параметры:
    - image_path: путь к изображению
    - n_colors: количество цветов в новой палитре (8, 16, 32, 64)
    """
    
    # 1. Загружаем изображение
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # 2. Получаем размеры
    height, width, channels = image_array.shape
    
    # 3. Преобразуем в список пикселей
    pixels = image_array.reshape(-1, channels)
    
    # 4. Обучаем K-means
    print(f"Обработка {len(pixels)} пикселей...")
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # 5. Получаем новые цвета (центры кластеров)
    new_colors = kmeans.cluster_centers_.astype(int)
    
    # 6. Заменяем пиксели
    labels = kmeans.predict(pixels)
    compressed_pixels = new_colors[labels]
    
    # 7. Возвращаем форму изображения
    compressed_image = compressed_pixels.reshape(height, width, channels)
    
    # 8. Считаем исходное количество цветов
    original_colors = len(np.unique(pixels, axis=0))
    
    # 9. Визуализируем
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image_array)
    axes[0].set_title(f'Исходное\n{original_colors} цветов')
    axes[0].axis('off')
    
    axes[1].imshow(compressed_image.astype(np.uint8))
    axes[1].set_title(f'Сжатое (k={n_colors})\n{n_colors} цветов')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('compress')
    plt.show()
    
    # 10. Показываем новую палитру
    print(f"\nНовая палитра из {n_colors} цветов:")
    show_palette(new_colors)
    
    return compressed_image

def show_palette(colors):
    """Показывает палитру цветов"""
    palette_height = 50
    palette_width = len(colors) * 20
    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        palette[:, i*20:(i+1)*20] = color
    
    plt.figure(figsize=(6, 2))
    plt.imshow(palette)
    plt.axis('off')
    plt.title(f'Палитра из {len(colors)} цветов')
    plt.savefig('pallete')
    plt.show()

# Использование
compressed = compress_image('image.jpg', n_colors=8)