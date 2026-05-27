from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import sys
sys.path.append(r'F:\6 семестр\Системы ИИ')
import utility

data1 = utility.importCSV('clustering_1.csv', delimiter='\t')
data2 = utility.importCSV('clustering_2.csv', delimiter='\t')
data3 = utility.importCSV('clustering_3.csv', delimiter='\t')

data1 = utility.toNpArray(data1)
data2 = utility.toNpArray(data2)
data3 = utility.toNpArray(data3)

def get_optimal_eps(data, min_samples=5):
    """
    Возвращает оптимальный eps для DBSCAN
    
    Параметры:
    - data: массив данных (numpy array)
    - min_samples: параметр DBSCAN (обычно 4-10)
    
    Возвращает:
    - eps: оптимальное значение (одно число)
    """
    # Вычисляем расстояния до min_samples-го соседа
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    
    # Берем расстояния до min_samples-го соседа
    k_distances = distances[:, min_samples-1]
    
    # Сортируем
    k_distances_sorted = np.sort(k_distances)
    
    # Берем 85-й перцентиль (оптимальный eps)
    eps = np.percentile(k_distances_sorted, 85)
    
    return eps

def makeThreeClusters(data, title='default'):  
    best_kmeans = 0
    best_debscan = 0
    best_aggl = 0
    
    n_clusters_kmeans = 0
    n_clusters_aggl = 0
    n_clusters_dbscan = 0
    for i in range(2, 10):
        kmeans = KMeans(i).fit(data)
        aggl = AgglomerativeClustering(i).fit(data)
        if best_kmeans < silhouette_score(data, kmeans.labels_):
            n_clusters_kmeans = i
            best_kmeans = silhouette_score(data, kmeans.labels_)
        if best_aggl < silhouette_score(data, aggl.labels_):
            n_clusters_aggl = i
            best_aggl = silhouette_score(data, aggl.labels_)

    for i in range(0, 200):
        dbscan = DBSCAN(eps=(0.1 + i * 0.01)).fit(data)
        if (len(set(dbscan.labels_)) < 2):
            continue
        if best_debscan < silhouette_score(data, dbscan.labels_):
            best_debscan = silhouette_score(data, dbscan.labels_)
            n_clusters_dbscan = i

    kmeans = KMeans(n_clusters_kmeans).fit(data)
    dbscan = DBSCAN(eps=(0.1 + n_clusters_dbscan * 0.01)).fit(data)
    aggl = AgglomerativeClustering(n_clusters_aggl).fit(data)
    silh_kmeans = silhouette_score(data, kmeans.labels_)
    silh_dbscan = silhouette_score(data, dbscan.labels_)
    silh_aggl = silhouette_score(data, aggl.labels_)

    dav_bol_kmeans = davies_bouldin_score(data, kmeans.labels_)
    dav_bol_dbscan = davies_bouldin_score(data, dbscan.labels_)
    dav_bol_aggl = davies_bouldin_score(data, aggl.labels_)

    n_clusters_dbscan = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)

    plt.figure(figsize=(12, 6))
    colors = []
    for i in range(500):
        # Используем золотое сечение для равномерного распределения
        hue = (i * 0.618033988749895) % 1.0  # золотое сечение
        # Преобразуем HSV в RGB (упрощенно)
        r = abs(np.sin(hue * 2 * np.pi))
        g = abs(np.sin((hue + 0.333) * 2 * np.pi))
        b = abs(np.sin((hue + 0.667) * 2 * np.pi))
        colors.append((r, g, b))

    plt.subplot(1, 3, 1)
    plt.title('KMeans')
    counter = 0
    for i in set(kmeans.labels_):
        plt.scatter(data[kmeans.labels_ == i][:, 0], data[kmeans.labels_ == i][:, 1], color=colors[counter])
        counter += 1

    plt.subplot(1, 3, 2)
    plt.title('DBSCAN')
    counter = 0
    for i in set(dbscan.labels_):
        if i == -1:
            plt.scatter(data[dbscan.labels_ == i][:, 0], data[dbscan.labels_ == i][:, 1], color='black', label="class -1")
            plt.legend()
        else:
            plt.scatter(data[dbscan.labels_ == i][:, 0], data[dbscan.labels_ == i][:, 1], color=colors[counter])
        counter += 1

    plt.subplot(1, 3, 3)
    plt.title('AgglomerativeClustering')
    counter = 0
    for i in set(aggl.labels_):
        plt.scatter(data[aggl.labels_ == i][:, 0], data[aggl.labels_ == i][:, 1], color=colors[counter])
        counter += 1
    
    plt.suptitle(title)
    plt.savefig(title)
    # plt.show()

    return silh_kmeans, silh_dbscan, silh_aggl, dav_bol_kmeans, dav_bol_dbscan, dav_bol_aggl, n_clusters_kmeans, n_clusters_dbscan, n_clusters_aggl

silh_kmeans1, silh_dbscan1, silh_aggl1, dav_bol_kmeans1, dav_bol_dbscan1, dav_bol_aggl1, n_clusters_kmeans1, n_clusters_dbscan1, n_clusters_aggl1 =  makeThreeClusters(data1, 'data1 clusterisation')
silh_kmeans2, silh_dbscan2, silh_aggl2, dav_bol_kmeans2, dav_bol_dbscan2, dav_bol_aggl2, n_clusters_kmeans2, n_clusters_dbscan2, n_clusters_aggl2 =  makeThreeClusters(data2, 'data2 clusterisation')
silh_kmeans3, silh_dbscan3, silh_aggl3, dav_bol_kmeans3, dav_bol_dbscan3, dav_bol_aggl3, n_clusters_kmeans3, n_clusters_dbscan3, n_clusters_aggl3 =  makeThreeClusters(data3, 'data3 clusterisation')

print(f'silhouette_score for data1(KMeans, DBSCAN, AgglomerativeClustering):\n{silh_kmeans1}, {silh_dbscan1}, {silh_aggl1}')
print(f'davies_bouldin_score for data1(KMeans, DBSCAN, AgglomerativeClustering):\n{dav_bol_kmeans1}, {dav_bol_dbscan1}, {dav_bol_aggl1}')
print(f'n_of_clusters for data1(KMeans, DBSCAN, AgglomerativeClustering):\n{n_clusters_kmeans1}, {n_clusters_dbscan1}, {n_clusters_aggl1}')

print(f'silhouette_score for data2(KMeans, DBSCAN, AgglomerativeClustering):\n{silh_kmeans2}, {silh_dbscan2}, {silh_aggl2}')
print(f'davies_bouldin_score for data2(KMeans, DBSCAN, AgglomerativeClustering):\n{dav_bol_kmeans2}, {dav_bol_dbscan2}, {dav_bol_aggl2}')
print(f'n_of_clusters for data2(KMeans, DBSCAN, AgglomerativeClustering):\n{n_clusters_kmeans2}, {n_clusters_dbscan2}, {n_clusters_aggl2}')

print(f'silhouette_score for data3(KMeans, DBSCAN, AgglomerativeClustering):\n{silh_kmeans3}, {silh_dbscan3}, {silh_aggl3}')
print(f'davies_bouldin_score for data3(KMeans, DBSCAN, AgglomerativeClustering):\n{dav_bol_kmeans3}, {dav_bol_dbscan3}, {dav_bol_aggl3}')
print(f'n_of_clusters for data3(KMeans, DBSCAN, AgglomerativeClustering):\n{n_clusters_kmeans3}, {n_clusters_dbscan3}, {n_clusters_aggl3}')