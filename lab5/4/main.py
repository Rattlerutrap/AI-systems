from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import sys
sys.path.append(r'F:\Системы ИИ')
import utility
import random

data = utility.importCSV('longley.csv', 1, delimiter=',')

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])
    
random.seed(0)
random.shuffle(data)

y = [row[-1] for row in data]
x = [row[0:-1] for row in data]

y_train = y[0:8]
y_test = y[8:16]
x_train = x[0:8]
x_test = x[8:16]

model_lin = LinearRegression()
model_lin.fit(x_train, y_train)
predict_lin = model_lin.predict(x_test)
r2_score_lin = r2_score(y_test, predict_lin)
print(r2_score_lin)

best_i = 0
best_r2 = 0
r2_scores = []
for i in range(26):
    model_ridge = Ridge(alpha=(10**(-3 + 0.2 * i)))
    model_ridge.fit(x_train, y_train)
    predict_ridge = model_ridge.predict(x_test)
    r2_score_ridge = r2_score(y_test, predict_ridge)
    if (best_r2 < r2_score_ridge):
        best_i = i
        best_r2 = r2_score_ridge
    print(f'lambda = {10**(-3 + 0.2 * i)}: {r2_score_ridge}')    
    r2_scores.append(r2_score_ridge)

print(f'best r2 for ridge: {best_r2}')
print(f'best i for ridge: {best_i}, lambda = {10**(-3 + 0.2 * best_i)}')

plt.figure(figsize=(12, 6))

plt.plot(range(0, 26), r2_scores)
plt.xticks(range(0, 26))
plt.xlabel('I')
plt.ylabel('R2 score')
plt.title('Зависимость метрики r2 от i')
plt.savefig('t4')
plt.grid(True)
plt.show()

