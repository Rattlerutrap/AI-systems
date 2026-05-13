import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
df = pd.read_csv('votes.csv')

print("="*60)
print("АНАЛИЗ ДАННЫХ VOTES.CSV")
print("="*60)
print(f"Форма данных: {df.shape}")
print(f"\nПервые 5 строк:")
print(df.head())
print(f"\nНазвания колонок (годы):")
print(df.columns.tolist())

# ============================================
# 2. ПОДГОТОВКА ДАННЫХ
# ============================================
# Берем только числовые данные (без названий штатов)
data = df.values.astype(float)

# Получаем количество объектов (строк)
n_objects = data.shape[0]
print(f"\nКоличество объектов (строк): {n_objects}")
print(f"Количество признаков (годы): {data.shape[1]}")

# Создаем простые метки (номера строк)
labels_names = [str(i) for i in range(n_objects)]

# Заполняем пропуски средним значением по столбцу (году)
for col in range(data.shape[1]):
    col_mean = np.nanmean(data[:, col])
    data[np.isnan(data[:, col]), col] = col_mean

print(f"Пропуски заполнены средними значениями")

# ============================================
# 3. СТАНДАРТИЗАЦИЯ (ВАЖНО!)
# ============================================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print(f"Данные стандартизированы (среднее=0, стд=1)")

# ============================================
# 4. ПОСТРОЕНИЕ ДЕНДРОГРАММЫ
# ============================================
# Вычисляем матрицу связей (метод Ward)
Z = linkage(data_scaled, method='ward')

plt.figure(figsize=(20, 12))

# Строим дендрограмму с номерами строк
dendrogram(Z,
           labels=labels_names,  # номера строк 0, 1, 2, ...
           orientation='top',
           leaf_rotation=90,
           leaf_font_size=8,
           color_threshold=np.max(Z[:, 2]) * 0.3,
           above_threshold_color='gray')

plt.title('Дендрограмма объектов (строк) по голосованию за республиканцев\n(1856-1976)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Номер объекта (строки)', fontsize=12)
plt.ylabel('Расстояние (метод Ward)', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('dendro')
plt.show()

# ============================================
# 5. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ
# ============================================
print("\n" + "="*60)
print("ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ")
print("="*60)

# Метод 1: Силуэтный коэффициент
silhouette_scores = []
k_range = range(2, 50)

for k in k_range:
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels_temp = clustering.fit_predict(data_scaled)
    sil_score = silhouette_score(data_scaled, labels_temp)
    silhouette_scores.append(sil_score)
    print(f"k={k:2d}: силуэт = {sil_score:.4f}")

best_k_by_silhouette = k_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)
print(f"\n✅ По силуэту: оптимальное k = {best_k_by_silhouette} (силуэт={best_silhouette:.4f})")

# Выбираем лучшее k
best_k = best_k_by_silhouette

# ============================================
# 6. ФИНАЛЬНАЯ КЛАСТЕРИЗАЦИЯ
# ============================================
final_clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
labels = final_clustering.fit_predict(data_scaled)

# ============================================
# 7. ВЫВОД РЕЗУЛЬТАТОВ ПО КЛАСТЕРАМ
# ============================================
print("\n" + "="*60)
print(f"РАЗБИЕНИЕ НА {best_k} КЛАСТЕРОВ")
print("="*60)

# Группируем объекты по кластерам
clusters = {}
for obj_id, label in zip(range(n_objects), labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(obj_id)

# Выводим кластеры
for label in sorted(clusters.keys()):
    cluster_objects = sorted(clusters[label])
    print(f"\n📌 КЛАСТЕР {label + 1} ({len(cluster_objects)} объектов):")
    # Выводим по 10 номеров в строке
    for i in range(0, len(cluster_objects), 10):
        print("   " + ", ".join(str(s) for s in cluster_objects[i:i+10]))

# ============================================
# 8. ХАРАКТЕРИСТИКИ КЛАСТЕРОВ
# ============================================
print("\n" + "="*60)
print("ХАРАКТЕРИСТИКИ КЛАСТЕРОВ")
print("="*60)

years = range(1856, 1977, 4)

for label in sorted(clusters.keys()):
    indices = [i for i, l in enumerate(labels) if l == label]
    cluster_data = data[indices]  # исходные данные
    
    # Средний профиль голосования
    avg_profile = np.mean(cluster_data, axis=0)
    std_profile = np.std(cluster_data, axis=0)
    
    # Общий средний процент
    avg_overall = np.mean(avg_profile)
    
    # Годы с экстремальными значениями
    extreme_years = []
    for i, val in enumerate(avg_profile):
        if val > 60 or val < 30:  # экстремальные проценты
            year = years[i] if i < len(years) else 1856 + i*4
            extreme_years.append(f"{year}({val:.1f}%)")
    
    print(f"\n📊 КЛАСТЕР {label + 1}:")
    print(f"   Количество объектов: {len(clusters[label])}")
    print(f"   Средний % за республиканцев: {avg_overall:.1f}%")
    print(f"   Экстремальные годы: {', '.join(extreme_years[:5]) if extreme_years else 'нет'}")

# ============================================
# 9. ВИЗУАЛИЗАЦИЯ ПРОФИЛЕЙ КЛАСТЕРОВ
# ============================================
plt.figure(figsize=(14, 8))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

for label in sorted(clusters.keys()):
    indices = [i for i, l in enumerate(labels) if l == label]
    cluster_data = data[indices]
    avg_profile = np.mean(cluster_data, axis=0)
    std_profile = np.std(cluster_data, axis=0)
    
    plt.plot(years[:len(avg_profile)], avg_profile, 
             color=colors[label % len(colors)], 
             linewidth=2, 
             label=f'Кластер {label + 1} ({len(indices)} объектов)')
    plt.fill_between(years[:len(avg_profile)], 
                     avg_profile - std_profile, 
                     avg_profile + std_profile, 
                     color=colors[label % len(colors)], 
                     alpha=0.2)

plt.xlabel('Год выборов', fontsize=12)
plt.ylabel('Процент голосов за республиканцев', fontsize=12)
plt.title('Средние профили голосования по кластерам', fontsize=14)
plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50%')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2clusters')
plt.show()

# ============================================
# 10. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*60)
print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
print("="*60)

print(f"""
1. ОБЩАЯ СТРУКТУРА:
   - Всего объектов: {n_objects}
   - Оптимальное число кластеров: {best_k}
   - Кластеры имеют разный размер

2. ХАРАКТЕРИСТИКИ КЛАСТЕРОВ:
""")

for label in sorted(clusters.keys()):
    indices = [i for i, l in enumerate(labels) if l == label]
    cluster_data = data[indices]
    avg_profile = np.mean(cluster_data, axis=0)
    avg_overall = np.mean(avg_profile)
    
    trend = "выше" if avg_overall > 50 else "ниже"
    print(f"   Кластер {label+1}: {len(clusters[label])} объектов, "
          f"средний % = {avg_overall:.1f}% ({trend} 50%)")

print(f"""
3. ВРЕМЕННЫЕ ЗАКОНОМЕРНОСТИ:
   - На графике видно, как менялось голосование во времени
   - Переломные годы: 1932 (Великая депрессия), 1964 (Акт о гражданских правах)
   - Некоторые кластеры стабильны, другие сильно колеблются

4. ВЫВОД:
   Кластеризация объектов (строк) показывает наличие устойчивых групп 
   со схожим электоральным поведением на протяжении 120 лет (1856-1976).
   Оптимальное количество кластеров = {best_k}.
""")

print(f"\nДендрограмма позволяет наглядно увидеть иерархическую структуру")
print(f"   сходства электорального поведения объектов на протяжении 120 лет.")