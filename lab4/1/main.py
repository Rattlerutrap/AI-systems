import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import random

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importCSV('glass.csv', 1, 1, ',')
data = utility.toNpArray(data)



scaler = StandardScaler()
random.seed(42)
random.shuffle(data)
x = data[:, :-1]
y = data[:, -1]
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

base_estimators = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF)": SVC(probability=False, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

n_estimators_range = [1, 5, 10, 15, 20, 30, 40, 50, 75, 100]

results = {name: [] for name in base_estimators.keys()}

# 4. Проводим эксперимент
print("Эксперимент 1: Бэггинг на датасете Glass")
print("="*60)

for name, clf in base_estimators.items():
    print(f"\nБазовый классификатор: {name}")
    print("-" * 30)
    
    for n in n_estimators_range:
        # Создаем ансамбль бэггинга
        bagging = BaggingClassifier(
            estimator=clf,
            n_estimators=n,
            n_jobs=-1,  # используем все ядра процессора
            random_state=42
        )
        
        # Обучаем
        bagging.fit(x_train, y_train)
        
        # Предсказываем
        y_pred = bagging.predict(x_test)
        
        # Оцениваем качество
        acc = accuracy_score(y_test, y_pred)
        results[name].append(acc)
        
        print(f"  n={n:3d}: Accuracy = {acc:.4f}")

# 5. Построение графика
plt.figure(figsize=(10, 6))

for name, scores in results.items():
    plt.plot(n_estimators_range, scores, marker='o', linewidth=2, markersize=6, label=name)

plt.xlabel('Количество классификаторов в ансамбле (n_estimators)', fontsize=12)
plt.ylabel('Точность классификации (Accuracy)', fontsize=12)
plt.title('Зависимость качества бэггинга от числа классификаторов\n(датасет Glass)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.savefig('1')
plt.show()

# 6. Анализ результатов
print("\n" + "="*60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*60)

for name, scores in results.items():
    best_score = max(scores)
    best_n = n_estimators_range[scores.index(best_score)]
    first_score = scores[0]
    improvement = (best_score - first_score) / first_score * 100
    
    print(f"\n{name}:")
    print(f"  - Качество при 1 классификаторе: {first_score:.4f}")
    print(f"  - Максимальное качество: {best_score:.4f} (при n={best_n})")
    print(f"  - Улучшение: {improvement:.1f}%")
    
    # Оценка стабильности
    if len(scores) > 5:
        last_scores = scores[-3:]
        if max(last_scores) - min(last_scores) < 0.01:
            print(f"  - Качество стабилизировалось после n={n_estimators_range[len(scores)-4]}")

