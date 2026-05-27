from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append(r'F:\6 семестр\Системы ИИ')
import utility
data = utility.importCSV('nsw74psid1.csv', 1, delimiter=',')

x = []
y = []
for i in range(len(data)):
    x.append([float(el) for el in data[i][0:-1]])
    y.append(float(data[i][-1]))

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, 
    test_size=0.3, 
    random_state=42
)

model_tree = DecisionTreeRegressor(
    max_depth=5,           
    random_state=42
)
model_tree.fit(X_train, Y_train)
Y_pred_tree = model_tree.predict(X_test)


model_lin = LinearRegression()
model_lin.fit(X_train, Y_train)
Y_pred_lin = model_lin.predict(X_test)

model_svr = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1
)
model_svr.fit(X_train, Y_train)
Y_pred_svr = model_svr.predict(X_test)

results = []

for name, Y_pred in [("Дерево решений", Y_pred_tree), 
                      ("Линейная регрессия", Y_pred_lin), 
                      ("SVR", Y_pred_svr)]:
    
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    results.append({
        'model': name,
        'MSE': mse,
        'R²': r2
    })
    
    print(f"\n{name}:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  R²:   {r2:.4f}")


best_r2 = max(results, key=lambda x: x['R²'])
best_mse = min(results, key=lambda x: x['MSE'])

print(f"\nПо R² лучшая модель: {best_r2['model']} (R² = {best_r2['R²']:.4f})")
print(f"По MSE лучшая модель: {best_mse['model']} (MSE = {best_mse['MSE']:.2f})")