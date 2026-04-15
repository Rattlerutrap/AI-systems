import numpy as np
import random
import matplotlib.pyplot as plt



n_neg = 20
mean_neg = [10, 14]
std_neg = 3

n_pos = 80
mean_pos = [19, 16]
std_pos = 4

np.random.seed(42)
X_neg = np.random.normal(mean_neg, std_neg, size=(n_neg, 2))
X_pos = np.random.normal(mean_pos, std_pos, size=(n_pos, 2))


dataSet = []

for i in range(len(X_neg)):
    dataSet.append([X_neg[i][0], X_neg[1], -1])

for i in range(len(X_pos)):
    dataSet.append([X_pos[i][0], X_pos[1], 1])

random.shuffle(dataSet)
target = []
for i in dataSet:
    target.append(i[-1])
    i.pop()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', alpha=0.7)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Класс -1 (20 точек)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', alpha=0.7)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Класс 1 (80 точек)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

