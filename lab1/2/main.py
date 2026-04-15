import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


mean = [10, 14]  
std = 3          

data = np.random.normal(mean, std, size=(100, 2))

print(data)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data[:, 0], bins=10, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.title('Гистограмма X1')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(data[:, 1], bins=10, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.title('Гистограмма X2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()