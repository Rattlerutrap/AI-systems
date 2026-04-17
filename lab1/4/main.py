import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, title, support_vectors=None):
    """
    Визуализация границ решений SVM
    """

    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap_light = ListedColormap(["#680000", '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                         edgecolor='black', s=50, alpha=0.8)
    
    if support_vectors is not None:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           marker='s', s=200, facecolors='none', 
           edgecolors='blue', linewidth=2)
    
    if model.kernel == 'linear':
        w = model.coef_[0]
        b = model.intercept_[0]
        
        x_line = np.array([x_min, x_max])
        y_line = -(w[0] * x_line + b) / w[1]
        plt.plot(x_line, y_line, 'k-', linewidth=2, label='Разделяющая линия')
        
        y_margin_up = -(w[0] * x_line + b - 1) / w[1]
        y_margin_down = -(w[0] * x_line + b + 1) / w[1]
        plt.plot(x_line, y_margin_up, 'k--', linewidth=1, alpha=0.7)
        plt.plot(x_line, y_margin_down, 'k--', linewidth=1, alpha=0.7)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()


data_train= utility.importTxt('svmdata_a.txt', 1, 1, '\t')

data_test = utility.importTxt('svmdata_a_test.txt', 1, 1, '\t')

X_train = np.array([[float(x[0]), float(x[1])] for x in data_train])
Y_train = np.array([1 if x[2] == 'green' else 0 for x in data_train])

X_test = np.array([[float(x[0]), float(x[1])] for x in data_test])
Y_test = np.array([1 if x[2] == 'green' else 0 for x in data_test])

svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, Y_train)

y_train_pred = svm_linear.predict(X_train)
y_test_pred = svm_linear.predict(X_test)

train_accuracy = accuracy_score(Y_train, y_train_pred)
test_accuracy = accuracy_score(Y_test, y_test_pred)

print("=== SVM с линейным ядром ===")
print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
print(f"Количество опорных векторов: {len(svm_linear.support_vectors_)}")

cm_train = confusion_matrix(Y_train, y_train_pred)
print("\nМатрица ошибок (обучение):")
print("          Предсказано")
print("          Класс 0  Класс 1")
print(f"Класс 0   {cm_train[0,0]:6d}  {cm_train[0,1]:6d}")
print(f"Класс 1   {cm_train[1,0]:6d}  {cm_train[1,1]:6d}")

cm_test = confusion_matrix(Y_test, y_test_pred)
print("\nМатрица ошибок (тест):")
print("          Предсказано")
print("          Класс 0  Класс 1")
print(f"Класс 0   {cm_test[0,0]:6d}  {cm_test[0,1]:6d}")
print(f"Класс 1   {cm_test[1,0]:6d}  {cm_test[1,1]:6d}")


plot_decision_boundary(
    svm_linear, 
    X_train, 
    Y_train, 
    f'SVM с линейным ядром (обучение)\nОпорных векторов: {len(svm_linear.support_vectors_)}',
    support_vectors=svm_linear.support_vectors_
)

plot_decision_boundary(
    svm_linear, 
    X_test, 
    Y_test, 
    f'SVM с линейным ядром (тест)\nОпорных векторов: {len(svm_linear.support_vectors_)}',
    support_vectors=svm_linear.support_vectors_
)