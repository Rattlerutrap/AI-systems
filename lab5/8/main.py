from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importTxt('svmdata6.txt', 1, separator='\t')


x=[]
y=[]
for i in range(len(data)):
    x.append([float(data[i][0]), float(data[i][1])])
    y.append(float(data[i][2]))


epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
mse_values = []

for eps in epsilon_values:
    model = SVR(kernel='rbf', C=1, epsilon=eps)
    model.fit(x, y)
    Y_pred = model.predict(x)
    mse = mean_squared_error(y, Y_pred)
    mse_values.append(mse)
    print(f"ε = {eps:.2f}, MSE = {mse:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, mse_values, 'bo-', markersize=6)
plt.xlabel('ε (epsilon)')
plt.ylabel('MSE')
plt.title('Зависимость MSE от ε для SVR (kernel=rbf, C=1)')
plt.grid(True, alpha=0.3)
plt.savefig('t8')
plt.show()

best_eps = epsilon_values[mse_values.index(min(mse_values))]
print(f"\nОптимальное ε = {best_eps}, MSE = {min(mse_values):.6f}")
