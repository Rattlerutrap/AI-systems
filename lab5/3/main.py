from sklearn.linear_model import LinearRegression

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importTxt('cygage.txt', 1, separator='\t')

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])

age = [row[0] for row in data]
depth = [row[1] for row in data]
weights = [row[2] for row in data]

n = len(data)

X = [[d] for d in depth]

model_ols = LinearRegression()
model_ols.fit(X, age)
age_pred_ols = model_ols.predict(X)

# Метрики для обычной модели
rss_ols = sum((age[i] - age_pred_ols[i])**2 for i in range(n))
mean_age = sum(age) / n
tss_ols = sum((age[i] - mean_age)**2 for i in range(n))
r2_ols = 1 - rss_ols / tss_ols

print("\n" + "="*60)
print("МОДЕЛЬ 1: ОБЫЧНАЯ РЕГРЕССИЯ (без весов)")
print("="*60)
print(f"Уравнение: age = {model_ols.intercept_:.4f} + {model_ols.coef_[0]:.4f} * depth")
print(f"R² = {r2_ols:.6f}")
print(f"RSS = {rss_ols:.6f}")

sum_weights = sum(weights)
weights_norm = [w / sum_weights for w in weights]

def weighted_regression(X, Y, weights):
    sum_w = sum(weights)
    sum_wx = sum(weights[i] * X[i] for i in range(len(X)))
    sum_wy = sum(weights[i] * Y[i] for i in range(len(X)))
    sum_wx2 = sum(weights[i] * X[i] * X[i] for i in range(len(X)))
    sum_wxy = sum(weights[i] * X[i] * Y[i] for i in range(len(X)))
    
    
    det = sum_w * sum_wx2 - sum_wx * sum_wx
    
    if det == 0:
        raise ValueError("Определитель равен нулю")
    
    intercept = (sum_wy * sum_wx2 - sum_wx * sum_wxy) / det
    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / det
    
    return intercept, slope

# Строим взвешенную регрессию
intercept_w, slope_w = weighted_regression(depth, age, weights_norm)
age_pred_w = [intercept_w + slope_w * d for d in depth]

# Взвешенные метрики
rss_w = sum(weights_norm[i] * (age[i] - age_pred_w[i])**2 for i in range(n))
weighted_mean_y = sum(weights_norm[i] * age[i] for i in range(n))
tss_w = sum(weights_norm[i] * (age[i] - weighted_mean_y)**2 for i in range(n))
r2_w = 1 - rss_w / tss_w

print("\n" + "="*60)
print("МОДЕЛЬ 2: ВЗВЕШЕННАЯ РЕГРЕССИЯ (с весами из файла)")
print("="*60)
print(f"Уравнение: age = {intercept_w:.4f} + {slope_w:.4f} * depth")
print(f"R² (взвешенный) = {r2_w:.6f}")
print(f"RSS (взвешенный) = {rss_w:.6f}")
