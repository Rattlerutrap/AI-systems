from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importTxt('reglab1.txt', 1, separator='\t')

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])

for i in data:
    print(i)

col0 = [row[0] for row in data]
col1 = [row[1] for row in data]
col2 = [row[2] for row in data]


n = len(data) 
results = []
for y_idx in range(3):
    Y = [row[y_idx] for row in data]
    X = [[row[j] for j in range(3) if j != y_idx] for row in data]
    
    # Обучаем модель
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    
    # Считаем метрики
    rss = sum((Y[i] - Y_pred[i])**2 for i in range(n))
    r2 = r2_score(Y, Y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2 - 1)
    
    results.append((f'col{y_idx}', r2, adj_r2, rss, model.intercept_, model.coef_))
    
    # Вывод
    print(f"\nМодель {y_idx+1}: Y = col{y_idx}, X = остальные")
    print(f"  Уравнение: col{y_idx} = {model.intercept_:.4f}", end="")
    coef_idx = 0
    for j in range(3):
        if j != y_idx:
            print(f" + ({model.coef_[coef_idx]:.4f})*col{j}", end="")
            coef_idx += 1
    print(f"\n  R² = {r2:.6f}, Adj. R² = {adj_r2:.6f}, RSS = {rss:.6f}")

import matplotlib.pyplot as plt

# Настройка графиков
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Сравнение моделей линейной регрессии', fontsize=14, fontweight='bold')

# Цвета для разных моделей
colors = ['blue', 'green', 'red']
titles = ['Модель 1: Y = col0', 'Модель 2: Y = col1', 'Модель 3: Y = col2']

for y_idx in range(3):
    Y = [row[y_idx] for row in data]
    X = [[row[j] for j in range(3) if j != y_idx] for row in data]
    
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    
    # График: реальные vs предсказанные значения
    ax = axes[y_idx]
    
    # Точки (реальные значения)
    ax.scatter(Y, Y_pred, color=colors[y_idx], alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # Линия идеального предсказания (y = x)
    min_val = min(min(Y), min(Y_pred))
    max_val = max(max(Y), max(Y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Идеальная линия (y=x)')
    
    # Настройки
    ax.set_xlabel('Фактические значения', fontsize=10)
    ax.set_ylabel('Предсказанные значения', fontsize=10)
    ax.set_title(f'{titles[y_idx]}\nR² = {r2_score(Y, Y_pred):.4f}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Добавляем текстовую информацию
    rss_val = sum((Y[i] - Y_pred[i])**2 for i in range(len(Y)))
    ax.text(0.05, 0.95, f'RSS = {rss_val:.4f}', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Дополнительный график: сравнение метрик
fig2, ax2 = plt.subplots(figsize=(10, 6))

model_names = [f'col{i}' for i in range(3)]
r2_values = [results[i][1] for i in range(3)]
adj_r2_values = [results[i][2] for i in range(3)]
rss_values = [results[i][3] for i in range(3)]

# Нормализуем RSS для визуализации (делим на максимальное значение)
rss_normalized = [rss / max(rss_values) for rss in rss_values]

x = range(len(model_names))
width = 0.25

bars1 = ax2.bar([i - width for i in x], r2_values, width, label='R²', color='skyblue', edgecolor='black')
bars2 = ax2.bar(x, adj_r2_values, width, label='Adj. R²', color='lightgreen', edgecolor='black')
bars3 = ax2.bar([i + width for i in x], rss_normalized, width, label='RSS (нормализованный)', color='salmon', edgecolor='black')

ax2.set_xlabel('Зависимая переменная (Y)', fontsize=12)
ax2.set_ylabel('Значение метрики', fontsize=12)
ax2.set_title('Сравнение метрик качества моделей', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# График остатков для лучшей модели
best_idx = max(range(3), key=lambda i: results[i][2])  # индекс лучшей модели по Adj. R²
print(f"\nАнализ остатков для лучшей модели: col{best_idx}")

Y_best = [row[best_idx] for row in data]
X_best = [[row[j] for j in range(3) if j != best_idx] for row in data]
model_best = LinearRegression()
model_best.fit(X_best, Y_best)
Y_best_pred = model_best.predict(X_best)
residuals = [Y_best[i] - Y_best_pred[i] for i in range(len(Y_best))]

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))

# График остатков
axes3[0].scatter(Y_best_pred, residuals, color='purple', alpha=0.7, s=50, edgecolors='black')
axes3[0].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
axes3[0].set_xlabel('Предсказанные значения', fontsize=10)
axes3[0].set_ylabel('Остатки', fontsize=10)
axes3[0].set_title(f'График остатков (модель col{best_idx})', fontsize=11)
axes3[0].grid(True, alpha=0.3)

# Гистограмма остатков
axes3[1].hist(residuals, bins=5, color='orange', edgecolor='black', alpha=0.7)
axes3[1].set_xlabel('Остатки', fontsize=10)
axes3[1].set_ylabel('Частота', fontsize=10)
axes3[1].set_title(f'Распределение остатков (модель col{best_idx})', fontsize=11)
axes3[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Вывод итогового выбора
print("\n" + "="*60)
print("ИТОГОВЫЙ ВЫБОР МОДЕЛИ")
print("="*60)
best_model = max(results, key=lambda x: x[2])
print(f"✅ Наиболее подходящая модель: Y = {best_model[0]}")
print(f"   R² = {best_model[1]:.6f}")
print(f"   Adj. R² = {best_model[2]:.6f} (основной критерий)")
print(f"   RSS = {best_model[3]:.6f}")