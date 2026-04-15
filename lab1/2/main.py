import numpy as np

# Устанавливаем seed для воспроизводимости
np.random.seed(42)

# Параметры
mean = [10, 14]  # мат. ожидание для X1 и X2
std = 3          # среднеквадратическое отклонение

# Генерируем 100 точек с двумя признаками
data = np.random.normal(mean, std, size=(100, 2))

# Выводим результат
print(data)
print(f"\nФорма массива: {data.shape}")