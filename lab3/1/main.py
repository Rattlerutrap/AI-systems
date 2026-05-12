from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importCSV('pluton.csv', 1, delimiter=',')

data = np.array(data, dtype=float)

inS = []
dbS = []
sS = []

inN = []
sN = []
dbN = []

for i in ['Без стандартизации', 'С стандартизацией']:
    if i == 'С стандартизацией':
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
    for j in range(1, 10):
        kmeans = KMeans(3, random_state=0, max_iter=j).fit(data)
        if i == 'С стандартизацией':
            inS.append(kmeans.inertia_)
            dbS.append(silhouette_score(data, kmeans.labels_))
            sS.append(davies_bouldin_score(data, kmeans.labels_))
        else:
            inN.append(kmeans.inertia_)
            dbN.append(silhouette_score(data, kmeans.labels_))
            sN.append(davies_bouldin_score(data, kmeans.labels_))
        # print(f'kmeans.inertia_ {kmeans.inertia_} silhouette_score(data, kmeans.labels_) {silhouette_score(data, kmeans.labels_)} davies_bouldin_score(data, kmeans.labels_) {davies_bouldin_score(data, kmeans.labels_)}')

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(range(1, 10), inS, color='b', linewidth=2, label='inertia + standart')
plt.plot(range(1, 10), inN, color='r', linewidth=2, label='inertia + no standart')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(1, 10), dbS, color='b', linewidth=2, label='davies bouldin + standart')
plt.plot(range(1, 10), dbN, color='r', linewidth=2, label='davies bouldin + no standart')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(1, 10), sS, color='b', linewidth=2, label='silhouette + standart')
plt.plot(range(1, 10), sN, color='r', linewidth=2, label='silhouette + no standart')
plt.legend()

plt.savefig('t1result')
# plt.show()

