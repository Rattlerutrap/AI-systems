from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importTxt('reglab.txt', 1, separator='\t')

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])

Y = [row[0] for row in data]
x1 = [row[1] for row in data]
x2 = [row[2] for row in data]
x3 = [row[3] for row in data]
x4 = [row[4] for row in data]

X_full = [x1, x2, x3, x4]
model = LinearRegression()

mean_y = sum(Y) / len(Y)
Y_pred = [mean_y] * len(Y)

rss_k0 = sum((Y[i] - Y_pred[i])**2 for i in range(len(Y)))
print(f'rss_k0: {rss_k0}')

rss_k1 = []
for i in range(len(X_full)):
    X = []
    for j in X_full[i]:
        X.append([j])
    model.fit(X, Y)
    Y_pred = model.predict(X)
    rss = sum((Y[i] - Y_pred[i])**2 for i in range(len(X)))
    rss_k1.append(rss)

print(f'rss_k1: {rss_k1}')

rss_k2 = []
for i in range(len(X_full)):
    for j in range(i + 1, len(X_full)):
        X = []
        for k in range(len(Y)):
            X.append([X_full[i][k], X_full[j][k]])
        model.fit(X, Y)
        Y_pred = model.predict(X)
        rss = sum((Y[i] - Y_pred[i])**2 for i in range(len(X)))
        rss_k2.append(rss)

print(f'rss_k2: {rss_k2}')

rss_k3 = []
for i in range(len(X_full)):
    for j in range(i + 1, len(X_full)):
        for k in range(j + 1, len(X_full)):
            X = []
            for n in range(len(Y)):
                X.append([X_full[i][n], X_full[j][n], X_full[k][n]])
            model.fit(X, Y)
            Y_pred = model.predict(X)
            rss = sum((Y[i] - Y_pred[i])**2 for i in range(len(X)))
            rss_k3.append(rss)

print(f'rss_k3: {rss_k3}')